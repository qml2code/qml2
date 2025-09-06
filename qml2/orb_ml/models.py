# KK: Everything is composed in terms of OML_Compound and OML_Slater_pair instances since
# they include references to files where ab initio results are stored for later re-use,
# decreasing CPU time required to test a model many times.

from ..jit_interfaces import LinAlgError_, Module_, empty_, exp_
from ..models.hyperparameter_optimization import BOMain, KRRLeaveOneOutL2regOpt
from ..models.krr import KRRLocalModel, KRRModel
from ..utils import flatten_to_scalar
from .kernels import (
    OML_KernelInput,
    kernel_from_processed_input,
    renormed_smoothened_sigmas,
    rep_stddevs_from_kernel_input,
)
from .oml_compound import OML_Compound
from .oml_compound_list import OML_CompoundList
from .representations import OML_rep_params


class FJKModel(KRRModel):
    def __init__(
        self,
        sigmas=None,
        global_sigma=None,
        l2reg=None,
        l2reg_diag_ratio=None,
        orb_product="gaussian",
        norm="l2",
        training_reps_suppress_openmp=True,
        rep_params: OML_rep_params = OML_rep_params(),
        base_compound_class=OML_Compound,
        compound_kwargs={},
        shift_quantities_intensive=False,
        shift_quantities_extensive=False,
        possible_nuclear_charges=None,
        relatively_small_val=1.0e-3,
        ln_l2reg_diag_ratio_bounds=[-24.0, 0.0],
        ln_sigmas_mult_shift_bounds=[-10.0, 10.0],
        ln_global_sigma_bounds=[-3.0, 3.0],
        l2reg_total_iterpts=32,
        all_sigmas_total_iterpts=32,
    ):
        """
        KRR model using FJK representation.

        While optimization of l2reg is done similarly to KRRModel, sigma values are optimized in two dimensions. The first one is the `global_sigma`, whose considered logarithm range is defined in `ln_global_sigma_bounds`. The second one are the `sigmas` array which is obtained by an initial guess multiplied by a constant (same as in DOI:10.1063/5.0083301); the logarithm range of the multiplier is defined by `ln_sigmas_mult_shift_bounds`.

        NOTE (K.Karan.): Only added as an easy way to reproduce the results from the original FJK paper (DOI:10.1063/5.0083301). MSORF is much more CPU-efficient to use, and I suspect that the numerical noise associated with FJK representation depending on SCF and orbital localization makes SORF's random noise less of an issue.

        Args:
            sigmas (np.ndarray or None): array of sigma hyperparameters applied to atomic contribution vectors.
            global_sigma (float or None): sigma hyperparameter applied to localized orbitals representations.
            l2reg (float or None): l2 regularization parameter.
            l2reg_diag_ratio (float or None): ratio of l2 regularization parameter and average diagonal element of the training kernel matrix.
            orb_product (str): "gaussian" or "linear", corresponding to using as orbital kernel functions $k^{\\text{Gauss}}_{\\text{FJK}}$ or $k^{\\text{Gauss}}_{\\text{FJK}}$ (following terminology of arXiv:2505.21247).
            norm (str): "l2" or "l1"; use Gaussian or Laplacian kernel for calculating kernel function for atomic contributions.
            training_reps_suppress_openmp (bool): if True suppress OpenMP parallelization while calculating representations of the training set.
            rep_params (OML_rep_params): parameters of the orbital representation.
            base_compound_class : i.e. choose between OML_Compound, OML_Slater_pair, or OML_Slater_pairs.
            compound_kwargs : keyword arguments used to generate compound from coordinates and nuclear charges during prediction.
            shift_quantities_intensive (bool): shift quantities by their mean (as in KRRModel).
            shift_quantities_extensive (bool): use "dressed atoms" approach (as in KRRLocalModel), cannot be used together with shift_quantities_intensive.
            possible_nuclear_charges (np.ndarray or None): if not None defines nuclear charges present in chemical space of interest. Only relevant if shift_quantities_extensive=True.
            relatively_small_val (float): used to assign dummy sigmas for components of atomic contribution vectors that exhibit no variance.
            ln_l2reg_diag_ratio_bounds : range in which l2reg_diag_ratio is optimized (as in KRRModel).
            ln_sigmas_mult_shift_bounds : range in which the sigmas multiplier is optimized.
            ln_global_sigma_bounds : range in which the global sigma is optimized.
            l2reg_total_iterpts : number of optimized function calls for l2reg optimization.
            all_sigmas_total_iterpts : number of optimized function calls for optimizing sigmas and global_sigma.
        """
        Module_.__init__(self)
        # hyperparameters
        self.sigmas = sigmas
        self.global_sigma = global_sigma
        self.l2reg = l2reg
        self.l2reg_diag_ratio = l2reg_diag_ratio
        # representation parameters
        self.rep_params = rep_params
        self.compound_kwargs = compound_kwargs
        self.base_compound_class = base_compound_class
        # kernel parameters
        self.orb_product = orb_product
        self.norm = norm
        # basic arrays.
        self.init_basic_arrays()
        self.init_shifts()
        # for smoothing sigmas
        self.relatively_small_val = relatively_small_val
        # other
        self.shift_quantities_intensive = shift_quantities_intensive
        self.shift_quantities_extensive = shift_quantities_extensive
        self.shift_quantities = self.shift_quantities_intensive or self.shift_quantities_extensive
        assert not (
            self.shift_quantities_extensive and self.shift_quantities_intensive
        ), "Using both 'dressed atoms' approach and shifting an intensive quantity by its mean is not supported."
        self.possible_nuclear_charges = possible_nuclear_charges

        self.training_reps_suppress_openmp = training_reps_suppress_openmp
        # hyperparameter optimization parameters
        self.ln_l2reg_diag_ratio_bounds = ln_l2reg_diag_ratio_bounds
        self.ln_sigmas_mult_shift_bounds = ln_sigmas_mult_shift_bounds
        self.ln_global_sigma_bounds = ln_global_sigma_bounds
        self.l2reg_total_iterpts = l2reg_total_iterpts
        self.all_sigmas_total_iterpts = all_sigmas_total_iterpts

        self.sigmas_guess = None

    def init_shifts(self):
        KRRModel.init_shifts(self)
        KRRLocalModel.init_shifts(self)

    def adjust_init_quant_shift(self):
        if self.shift_quantities_intensive:
            KRRModel.adjust_init_quant_shift(self)
        if self.shift_quantities_extensive:
            KRRLocalModel.adjust_init_quant_shift(self)

    def check_ntrain_consistency(self):
        assert self.ntrain == len(self.training_compounds)

    def get_initialized_compound(self, compound: OML_Compound):
        compound.generate_orb_reps(rep_params=self.rep_params)
        return compound

    def get_all_initialized_compounds(
        self, oml_compound_list, suppress_openmp=False, num_procs=None
    ):
        if suppress_openmp:
            fixed_num_threads = 1
        else:
            fixed_num_threads = None
        all_compounds = OML_CompoundList(oml_compound_list)
        all_compounds.generate_orb_reps(
            rep_params=self.rep_params, fixed_num_threads=fixed_num_threads, num_procs=num_procs
        )
        return all_compounds

    def assign_training_set(
        self,
        training_compounds=None,
        training_quantities=None,
        suppress_openmp=None,
        num_procs=None,
        training_kernel_input=None,
    ):
        if suppress_openmp is None:
            suppress_openmp = self.training_reps_suppress_openmp
        self.training_compounds = self.get_all_initialized_compounds(
            training_compounds, suppress_openmp=suppress_openmp, num_procs=num_procs
        )
        self.training_quantities = training_quantities
        if self.shift_quantities_intensive or self.shift_quantities_extensive:
            self.adjust_init_quant_shift()
        self.ntrain = len(self.training_quantities)
        if training_kernel_input is None:
            training_kernel_input = OML_KernelInput(self.training_compounds)
        self.training_kernel_input = training_kernel_input

        self.readjust_training_temp_arrays()
        self.check_ntrain_consistency()

    def get_training_kernel(self, sigmas=None, global_sigma=None, out=None):
        if sigmas is None:
            sigmas = self.sigmas
        if global_sigma is None:
            global_sigma = self.global_sigma
        assert sigmas is not None
        assert global_sigma is not None
        return kernel_from_processed_input(
            self.training_kernel_input, None, sigmas, global_sigma, norm=self.norm, out=out
        )

    def get_all_sigmas(self, sigma_hyperparams):
        assert self.sigmas_guess is not None
        return self.sigmas_guess * sigma_hyperparams[0], sigma_hyperparams[1]

    def get_sigma_opt_func(self):
        """
        Define the function minimized by BOSS to optimize sigmas.
        """
        temp_kernel = empty_((self.ntrain, self.ntrain))

        def opt_func_kernel(ln_sigma_hyperparams):
            # TODO: make a short function?
            ln_sigma_hyperparams = flatten_to_scalar(ln_sigma_hyperparams)
            print("Testing ln sigma hyperparams:", ln_sigma_hyperparams)
            sigma_hyperparams = exp_(ln_sigma_hyperparams)
            sigmas, global_sigma = self.get_all_sigmas(sigma_hyperparams)
            return self.get_training_kernel(
                sigmas=sigmas, global_sigma=global_sigma, out=temp_kernel
            )

        return KRRLeaveOneOutL2regOpt(
            opt_func_kernel,
            self.training_quantities,
            ln_l2reg_diag_ratio_bounds=self.ln_l2reg_diag_ratio_bounds,
            total_iterpts=self.l2reg_total_iterpts,
        )

    def optimize_hyperparameters(self):
        print("Optimizing hyperparameters")
        raw_sigmas = rep_stddevs_from_kernel_input(self.training_kernel_input)
        self.sigmas_guess = renormed_smoothened_sigmas(
            raw_sigmas, relatively_small_val=self.relatively_small_val, norm=self.norm
        )
        sigma_opt_func = self.get_sigma_opt_func()
        bo = BOMain(
            sigma_opt_func,
            bounds=[self.ln_sigmas_mult_shift_bounds, self.ln_global_sigma_bounds],
            initpts=self.all_sigmas_total_iterpts // 2,
            iterpts=self.all_sigmas_total_iterpts // 2,
            minfreq=0,
            outfile="boss_sigmas_opt.out",
            rstfile="boss_sigmas_opt.rst",
        )
        res = bo.run()
        min_ln_sigma_hyperparams = res.select("x_glmin", -1)
        min_ln_l2reg_rel_diag, minimized_l2reg_loss = sigma_opt_func(
            min_ln_sigma_hyperparams, min_output=True
        )
        min_l2reg_rel_diag = exp_(min_ln_l2reg_rel_diag)
        # just in case, check the l2reg BOSS found actually corresponds to a numerically stable kernel matrix
        while True:
            try:
                minimized_l2reg_loss.calculate_for_l2reg_rel_diag(min_l2reg_rel_diag)
                break
            except LinAlgError_:
                min_l2reg_rel_diag *= 2
        self.sigmas, self.global_sigma = self.get_all_sigmas(exp_(min_ln_sigma_hyperparams))
        self.l2reg_diag_ratio = min_l2reg_rel_diag

    def train(self, training_quantities=None, **kwargs):
        self.assign_training_set(training_quantities=training_quantities, **kwargs)
        self.optimize_hyperparameters()
        self.fit(
            training_compounds=self.training_compounds,
            training_quantities=training_quantities,
            suppress_openmp=False,
            num_procs=None,
            training_kernel_input=self.training_kernel_input,
        )

    def get_shifted_prediction(self, prediction, **kwargs):
        if self.shift_quantities_extensive:
            return KRRLocalModel.get_shifted_prediction(self, prediction, **kwargs)
        if self.shift_quantities_intensive:
            return KRRModel.get_shifted_prediction(self, prediction, **kwargs)
        return prediction

    def predict_from_kernel_input(self, kernel_input: OML_KernelInput, **kwargs):
        self.temp_kernel = kernel_from_processed_input(
            self.training_kernel_input,
            kernel_input,
            self.sigmas,
            self.global_sigma,
            norm=self.norm,
        )
        return self.predict_from_kernel(nmols=kernel_input.num_mols, **kwargs)

    def predict_from_compounds(self, compounds, suppress_openmp=True, num_procs=None):
        init_compounds = self.get_all_initialized_compounds(
            compounds, suppress_openmp=suppress_openmp, num_procs=num_procs
        )
        kernel_input = OML_KernelInput(init_compounds)
        return self.predict_from_kernel_input(
            kernel_input,
            all_nuclear_charges=[comp.nuclear_charges for comp in compounds],
        )

    def predict_from_compound(self, input_compound):
        init_compound = self.get_initialized_compound(input_compound)
        kernel_input = OML_KernelInput([init_compound])
        return self.predict_from_kernel_input(
            kernel_input, all_nuclear_charges=[init_compound.nuclear_charges]
        )[0]

    def __call__(self, nuclear_charges, coords):
        compound = self.base_compound_class(
            nuclear_charges=nuclear_charges, coordinates=coords, **self.compound_kwargs
        )
        return self.predict_from_compound(compound)
