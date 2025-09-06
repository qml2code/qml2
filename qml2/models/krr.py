from ..basic_utils import checked_dict_entry, dump2pkl
from ..compound import Compound
from ..data import nCartDim
from ..jit_interfaces import (
    LinAlgError_,
    Module_,
    array_,
    concatenate_,
    default_rng_,
    diag_indices_from_,
    dot_,
    empty_,
    exp_,
    log_,
    mean_,
)
from ..kernels.kernels import construct_gaussian_kernel
from ..math import cho_solve
from ..optimizers import global_optimize_1D
from ..parallelization import embarrassingly_parallel
from ..representations import generate_coulomb_matrix, generate_fchl19
from ..utils import check_allocation, flatten_to_scalar, get_numba_list
from .hyperparameter_init_guesses import vector_std
from .hyperparameter_optimization import KRRLeaveOneOutL2regOpt
from .learning_curves import learning_curve_from_predictions
from .loss_functions import MAE
from .property_shifts import get_optimal_shift_guesses, get_shift_coeffs


class KRRModel(Module_):
    def __init__(
        self,
        kernel_function=None,
        kernel_function_symmetric=None,
        kernel_constructor=construct_gaussian_kernel,
        additional_kernel_constructor_kwargs={},
        kernel_kwargs={},
        representation_function=generate_coulomb_matrix,
        rep_kwargs={},
        sigma=None,
        l2reg=None,
        l2reg_diag_ratio=None,
        training_reps_suppress_openmp=True,
        shift_quantities=False,
        ln_l2reg_diag_ratio_bounds=[-24.0, 0.0],
        ln_sigma_shift_bounds=[-10.0, 10.0],
        l2reg_total_iterpts=32,
        sigma_total_iterpts=32,
        test_mode=False,
    ):
        """
        KRR model using global kernels.

        Args:
            kernel_function : function calculating kernel matrix, e.g. `qml2.kernels.gaussian_kernel`. If `None` construction from kernel_constructor will be attempted.
            kernel_function_symmetric : function calculating symmetric (training) kernel matrix, e.g. `qml2.kernels.gaussian_kernel_symmetric`. If `None` training kernel will be evaluated with `kernel_function`.
            kernel_constructor : if kernel_function is None, it (along with kernel_function_symmetric) is constructed with kernel_constructor, e.g. `qml2.kernels.construct_gaussian_kernel`. (K.Karan.: was introduced to prevent accidentally using kernel_function and kernel_function_symmetric that do not match; might be excessive.)
            additional_kernel_constructor_kwargs (dict): keyword arguments used by kernel_constructor for kernel functions.
            kernel_kwargs (dict): keyword arguments used by kernel_function and kernel_function_symmetric.
            representation_function : function for calculating global representation.
            rep_kwargs : keyword arguments used by the representation function.
            sigma (float or None): sigma hyperparameter.
            l2reg (float or None): l2 regularization hypermarameter (also known as lambda).
            l2reg_diag_ratio (float or None): ratio of l2reg and average diagonal matrix of the training kernel matrix, used to recover l2reg if the latter is None. (Introduced mainly for KRRLocalModel.)
            training_reps_suppress_openmp (bool): if True suppress Numba parallelization inside representation function when training compounds' representations are calculated for training.
            shift_quantities (bool): if True shift quantities by their mean for training and prediction.
            ln_l2reg_diag_ratio_bounds : bounds in which logarithm of l2reg_diag_ratio is optimized.
            ln_sigma_shift_bounds : bounds in which logarithm of sigma is optimized, shifted by logarithm of initial guess.
            l2reg_total_iterpts (int): total number of function calls as l2reg is optimized (e.g. with BOSS).
            sigma_total_iterpts (int): total number of function calls as sigma is optimized (e.g. with BOSS).
            test_mode (bool): disable parts of hyperparameter optimization that are random (used for tests).
        """
        Module_.__init__(self)
        self.basic_init(
            representation_function=representation_function,
            rep_kwargs=rep_kwargs,
            sigma=sigma,
            l2reg=l2reg,
            l2reg_diag_ratio=l2reg_diag_ratio,
            training_reps_suppress_openmp=training_reps_suppress_openmp,
            shift_quantities=shift_quantities,
            ln_l2reg_diag_ratio_bounds=ln_l2reg_diag_ratio_bounds,
            ln_sigma_shift_bounds=ln_sigma_shift_bounds,
            l2reg_total_iterpts=l2reg_total_iterpts,
            sigma_total_iterpts=sigma_total_iterpts,
            test_mode=test_mode,
        )
        # kernel function
        self.init_kernel_functions(
            kernel_constructor=kernel_constructor,
            kernel_function=kernel_function,
            kernel_function_symmetric=kernel_function_symmetric,
            kernel_kwargs=kernel_kwargs,
            **additional_kernel_constructor_kwargs,
        )
        self.readjust_training_temp_arrays()

    def basic_init(
        self,
        representation_function=generate_coulomb_matrix,
        rep_kwargs={},
        sigma=None,
        l2reg=None,
        l2reg_diag_ratio=None,
        training_reps_suppress_openmp=True,
        shift_quantities=False,
        ln_l2reg_diag_ratio_bounds=[-24.0, 0.0],
        ln_sigma_shift_bounds=[-10.0, 10.0],
        l2reg_total_iterpts=32,
        sigma_total_iterpts=32,
        test_mode=False,
    ):
        # hyperparameters
        self.sigma = sigma
        self.l2reg = l2reg
        self.l2reg_diag_ratio = l2reg_diag_ratio
        # parameters of hyperparameter optimization
        self.ln_l2reg_diag_ratio_bounds = ln_l2reg_diag_ratio_bounds
        self.l2reg_total_iterpts = l2reg_total_iterpts
        self.sigma_total_iterpts = sigma_total_iterpts
        self.ln_sigma_shift_bounds = ln_sigma_shift_bounds
        # representation parameters
        self.representation_function = representation_function
        self.rep_kwargs = rep_kwargs
        self.training_reps_suppress_openmp = training_reps_suppress_openmp
        self.test_mode = test_mode

        self.init_basic_arrays()

        # related to training set
        self.init_training_set_attrs()
        # related to shifts
        self.init_shifts()
        self.shift_quantities = shift_quantities

    def init_kernel_functions(
        self,
        kernel_constructor=None,
        kernel_function=None,
        kernel_function_symmetric=None,
        kernel_kwargs={},
        **additional_kernel_constructor_kwargs,
    ):
        if kernel_function is None:
            assert (
                kernel_constructor is not None
            ), "Kernel function should be defined through the `kernel_function` or `kernel_constructor` keyword arguments."
            kernel_function = kernel_constructor(
                symmetric=False, **additional_kernel_constructor_kwargs
            )
        if (kernel_function_symmetric is None) and (kernel_constructor is not None):
            kernel_function_symmetric = kernel_constructor(
                symmetric=True, **additional_kernel_constructor_kwargs
            )

        self.kernel_function_asym = kernel_function
        self.kernel_function_sym = kernel_function_symmetric
        self.kernel_kwargs = kernel_kwargs

    def init_basic_arrays(self):
        self.alphas = None

    def init_training_set_attrs(self):
        self.training_compounds = None
        self.training_quantities = None
        self.training_representations = None
        self.training_nuclear_charges = None
        self.training_representations = None

    def init_shifts(self):
        self.intensive_shift_val = None

    def get_rep(self, nuclear_charges, coords):
        assert nuclear_charges.shape[0] == coords.shape[0]
        assert coords.shape[1] == nCartDim
        return self.representation_function(nuclear_charges, coords, **self.rep_kwargs)

    def get_rep_tuple(self, ncharges_coords_tuple):
        return self.get_rep(*ncharges_coords_tuple)

    def compound2rep(self, comp: Compound):
        return self.get_rep(comp.nuclear_charges, comp.coordinates)

    def get_training_kernel(self, sigma=None, out=None):
        if sigma is None:
            sigma = self.sigma

        assert sigma is not None
        if out is None:
            out = empty_((self.ntrain, self.ntrain))
        if self.kernel_function_sym is None:
            self.kernel_function_asym(
                self.training_representations,
                self.training_representations,
                sigma,
                out,
                **self.kernel_kwargs,
            )
        else:
            self.kernel_function_sym(
                self.training_representations, sigma, out, **self.kernel_kwargs
            )
        return out

    def get_sigma_init_guess(self):
        return vector_std(self.training_representations)

    def combine_representations(self, representations_list):
        return array_(representations_list)

    def calc_all_representations(self, compounds=None, suppress_openmp=True, num_procs=None):
        if compounds is None:
            compounds = self.training_compounds
        if suppress_openmp:
            fixed_num_threads = 1
        else:
            fixed_num_threads = None
        all_representations_list = embarrassingly_parallel(
            self.compound2rep,
            compounds,
            (),
            fixed_num_threads=fixed_num_threads,
            num_procs=num_procs,
        )
        return self.combine_representations(all_representations_list)

    def adjust_init_quant_shift(self):
        self.intensive_shift_val = mean_(self.training_quantities)
        self.training_quantities = self.training_quantities - self.intensive_shift_val

    def check_ntrain_consistency(self):
        assert self.ntrain == self.training_representations.shape[0]

    def readjust_training_temp_arrays(self):
        # to ensure it corresponds to the correct ntrain
        self.temp_kernel = None

    def assign_training_set(
        self,
        training_compounds=None,
        training_quantities=None,
        training_nuclear_charges=None,
        training_coordinates=None,
        training_representations=None,
        suppress_openmp=None,
        num_procs=None,
    ):
        if (training_compounds is None) and (
            training_nuclear_charges is not None and training_coordinates is not None
        ):
            training_compounds = [
                Compound(coordinates=coordinates, nuclear_charges=nuclear_charges)
                for nuclear_charges, coordinates in zip(
                    training_nuclear_charges, training_coordinates
                )
            ]
        self.training_compounds = training_compounds
        self.training_quantities = training_quantities
        if self.shift_quantities:
            self.adjust_init_quant_shift()
        if training_representations is None:
            assert self.training_compounds is not None
            if suppress_openmp is None:
                suppress_openmp = self.training_reps_suppress_openmp
            training_representations = self.calc_all_representations(
                suppress_openmp=suppress_openmp, num_procs=num_procs
            )
        self.training_representations = training_representations
        self.ntrain = len(self.training_quantities)
        self.readjust_training_temp_arrays()
        self.check_ntrain_consistency()

    def get_sigma_opt_func(self):
        """
        Define the function minimized by BOSS to optimize sigma.
        """
        # ensure the training kernel is not re-allocated at each call
        temp_kernel = empty_((self.ntrain, self.ntrain))

        def opt_func_kernel(ln_sigma):
            # TODO: make a short function?
            ln_sigma = flatten_to_scalar(ln_sigma)
            print("Testing ln sigma:", ln_sigma)
            sigma = exp_(ln_sigma)
            return self.get_training_kernel(sigma=sigma, out=temp_kernel)

        return KRRLeaveOneOutL2regOpt(
            opt_func_kernel,
            self.training_quantities,
            ln_l2reg_diag_ratio_bounds=self.ln_l2reg_diag_ratio_bounds,
            total_iterpts=self.l2reg_total_iterpts,
            test_mode=self.test_mode,
        )

    def optimize_hyperparameters(self):
        """
        Use BOSS to optimize the sigma hyperparameter.
        """
        sigma_opt_func = self.get_sigma_opt_func()
        sigma_init_guess = self.get_sigma_init_guess()
        min_ln_sigma = global_optimize_1D(
            sigma_opt_func,
            self.ln_sigma_shift_bounds + log_(sigma_init_guess),
            total_iterpts=self.sigma_total_iterpts,
            test_mode=self.test_mode,
            opt_name="sigma_opt",
        )
        min_ln_l2reg_rel_diag, minimized_l2reg_loss = sigma_opt_func(min_ln_sigma, min_output=True)
        min_l2reg_rel_diag = exp_(min_ln_l2reg_rel_diag)
        # just in case, check the l2reg BOSS found actually corresponds to a numerically stable kernel matrix
        while True:
            try:
                minimized_l2reg_loss.calculate_for_l2reg_rel_diag(min_l2reg_rel_diag)
                break
            except LinAlgError_:
                min_l2reg_rel_diag *= 2
        self.sigma = exp_(min_ln_sigma)
        self.l2reg_diag_ratio = min_l2reg_rel_diag

    def get_alphas(self, train_kernel, rhs=None, solver=cho_solve, **solver_kwargs):
        l2reg = self.l2reg
        if rhs is None:
            rhs = self.training_quantities
        if l2reg is None:
            assert self.l2reg_diag_ratio is not None
            l2reg = self.l2reg_diag_ratio * mean_(train_kernel[diag_indices_from_(train_kernel)])
        self.alphas = solver(train_kernel, rhs, l2reg=l2reg, **solver_kwargs)
        return self.alphas

    def fit(self, **kwargs):
        self.assign_training_set(**kwargs)
        train_kernel = self.get_training_kernel()
        self.get_alphas(train_kernel)

    def train(self, **kwargs):
        self.assign_training_set(**kwargs)
        self.optimize_hyperparameters()
        self.fit(**kwargs)

    def get_shifted_prediction(self, prediction, **kwargs):
        if self.shift_quantities:
            return prediction + self.intensive_shift_val
        return prediction

    def predict_from_kernel(self, nmols=1, **kwargs):
        predictions = dot_(self.alphas, self.temp_kernel[:, :nmols])
        return self.get_shifted_prediction(predictions, **kwargs)

    def predict_from_representations(self, representations, **kwargs):
        nmols = representations.shape[0]
        self.temp_kernel = check_allocation((self.ntrain, nmols), output=self.temp_kernel)
        self.kernel_function_asym(
            self.training_representations,
            representations,
            self.sigma,
            self.temp_kernel[:, :nmols],
            **self.kernel_kwargs,
        )
        return self.predict_from_kernel(nmols=nmols)

    def predict_from_compound(self, compound: Compound):
        return self(compound.nuclear_charges, compound.coordinates)

    def __call__(self, nuclear_charges, coords):
        representation = self.get_rep(nuclear_charges, coords)
        return self.predict_from_representations(
            representation.reshape((1, representation.shape[0]))
        )[0]

    def predict_from_compounds(self, compounds, suppress_openmp=True, num_procs=None):
        representations = self.calc_all_representations(
            compounds, suppress_openmp=suppress_openmp, num_procs=num_procs
        )
        return self.predict_from_representations(representations)

    def save(self, filename):
        """
        Save the trained model to a file.
        filename: string
        """

        dump2pkl(self, filename)

    # TODO can be combined with qml2.multilevel_sorf.models code too?
    def learning_curve_predictions(
        self,
        training_compounds,
        training_quantities,
        test_compounds,
        training_set_sizes,
        max_subset_num=8,
        rng=None,
        hyperparameter_reoptimization=True,
        num_procs=None,
        suppress_openmp=True,
    ):
        """
        For current hyperparameter values calculate all predictions necessary to build a learning curve.

        NOTE: not really optimized, since it's more about function testing than efficient learning curve generation.

        Args:
            training_compounds (List[Compound]): Compound instances representing training set molecules.
            training_quantities (numpy.array): quantities of interest for training set molecules.
            test_objects (List[Compound]): Compound instances representing test set molecules.
            test_quantities (numpy.array): quantities of interest for test set molecules.
            training_set_sizes (List[int]): list of sizes of the subsets of the training set used to build the learning curve.
            max_subset_num (int): for a given training set subset size, how many subsets are used at most. Default is 8.
            training_nuclear_charges (list): if not None, contains list of training molecules' nuclear charges.
            test_nuclear_charges (list): if not None, contains list of test molecules' nuclear charges.
            rng (numpy.random RNG or None): RNG used during shuffling of training set. If None (default), an RNG is generated by the code.
            hyperparameter_reoptimization (bool): if True perform a separate hyperparameter optimization for each training set subset. If False (default), hyperparameters currently initialized in the model are used.
            num_procs (int or None): if not None defines the number of processes spawned to calculate molecular representations.
            suppress_openmp (bool): whether to suppress OpenMP in child processes calculating representations.

        Returns:
            For each training set subset numpy.array of predictions is generated, which is combined into lists by subset size. A list of these lists is returned.
        """

        tot_ntrain = len(training_compounds)
        if rng is None:
            rng = default_rng_()
        all_training_indices = list(range(tot_ntrain))

        all_predictions = []

        if not hyperparameter_reoptimization:
            assert (self.sigma is not None) and (
                self.l2reg is not None or self.l2reg_diag_ratio is not None
            ), "hyperparameters need to be initialized"

        for training_set_size in training_set_sizes:
            rng.shuffle(all_training_indices)
            nsubsets = min(tot_ntrain // training_set_size, max_subset_num)
            assert nsubsets != 0
            subset_predictions = []
            for isubset in range(nsubsets):
                training_indices = array_(
                    all_training_indices[
                        isubset * training_set_size : (isubset + 1) * training_set_size
                    ]
                )
                subset_training_compounds = [training_compounds[ti] for ti in training_indices]
                subset_training_quantities = training_quantities[training_indices]
                if hyperparameter_reoptimization:
                    self.train(
                        training_compounds=subset_training_compounds,
                        training_quantities=subset_training_quantities,
                        num_procs=num_procs,
                    )
                else:
                    self.fit(
                        training_compounds=subset_training_compounds,
                        training_quantities=subset_training_quantities,
                        num_procs=num_procs,
                    )
                predictions = self.predict_from_compounds(
                    test_compounds, num_procs=num_procs, suppress_openmp=suppress_openmp
                )
                subset_predictions.append(predictions)
            all_predictions.append(subset_predictions)
        return all_predictions

    def learning_curve(
        self,
        training_compounds,
        training_quantities,
        test_compounds,
        test_quantities,
        training_set_sizes,
        max_subset_num=8,
        rng=None,
        lc_error_loss_function=MAE(),
        hyperparameter_reoptimization=True,
        num_procs=None,
        suppress_openmp=True,
    ):
        """
        Make a learning curve with errors.

        Args:
            training_compounds (List[Compound]): Compound instances representing training set molecules.
            training_quantities (numpy.array): quantities of interest for training set molecules.
            test_objects (List[Compound]): Compound instances representing test set molecules.
            test_quantities (numpy.array): quantities of interest for test set molecules.
            training_set_sizes (List[int]): list of sizes of the subsets of the training set used to build the learning curve.
            max_subset_num (int): for a given training set subset size, how many subsets are used at most. Default is 8.
            training_nuclear_charges (list): if not None, contains list of training molecules' nuclear charges.
            test_nuclear_charges (list): if not None, contains list of test molecules' nuclear charges.
            rng (numpy.random RNG or None): RNG used during shuffling of training set. If None (default), an RNG is generated by the code.
            lc_error_loss_function (function): loss function of prediction errors used in the learning curve. Default is Mean Absolute Error.
            hyperparameter_reoptimization (bool): if True perform a separate hyperparameter optimization for each training set subset. If False (default), hyperparameters currently initialized in the model are used.
            num_procs (int or None): if not None defines the number of processes spawned to calculate molecular representations.
            suppress_openmp (bool): whether to suppress OpenMP in child processes calculating representations.

        Returns:
            Tuple of `(means, stds)`, both members Numpy arrays, `means` containing mean values of loss function for a given training set size, `stds` containing standard deviations of the loss function among subsets of the same size.
        """
        all_predictions = self.learning_curve_predictions(
            training_compounds,
            training_quantities,
            test_compounds,
            training_set_sizes,
            max_subset_num=max_subset_num,
            rng=rng,
            hyperparameter_reoptimization=hyperparameter_reoptimization,
            num_procs=num_procs,
            suppress_openmp=suppress_openmp,
        )
        return learning_curve_from_predictions(
            all_predictions,
            test_quantities,
            error_loss_function=lc_error_loss_function,
        )


class KRRLocalModel(KRRModel):
    def __init__(
        self,
        representation_function=generate_fchl19,
        local_dn=True,
        possible_nuclear_charges=None,
        **other_kwargs,
    ):
        """
        Models using local/local(dn) kernels.

        Args (also see KRRModel):
            local_dn (bool): whether the kernel has delta w.r.t. nuclear charges in the expression.
            shift_quantities (bool): if True use "dressed atoms" approach.
            possible_nuclear_charges (ndarray or None): if not None lists nuclear charges present in training and test sets. Only required if dressed atoms are used.
        """
        add_kernel_constr_kwargs = checked_dict_entry(
            other_kwargs, "additional_kernel_constructor_kwargs", {}
        )
        other_kwargs["additional_kernel_constructor_kwargs"] = {
            **add_kernel_constr_kwargs,
            "local": True,
            "local_dn": local_dn,
        }
        self.local_dn = True
        self.possible_nuclear_charges = possible_nuclear_charges
        super().__init__(**other_kwargs, representation_function=representation_function)

    def adjust_init_quant_shift(self):
        assert self.possible_nuclear_charges is not None
        nb_training_nuclear_charges = get_numba_list(
            [comp.nuclear_charges for comp in self.training_compounds]
        )
        training_shift_coeffs = get_shift_coeffs(
            len(self.training_quantities),
            intensive_shift=False,
            extensive_shift=True,
            possible_nuclear_charges=self.possible_nuclear_charges,
            all_nuclear_charges=nb_training_nuclear_charges,
        )
        self.extensive_shift_vals = get_optimal_shift_guesses(
            self.training_quantities, shift_coeffs=training_shift_coeffs
        )
        self.training_quantities = self.training_quantities - dot_(
            training_shift_coeffs, self.extensive_shift_vals
        )

    def check_ntrain_consistency(self):
        assert self.ntrain == len(self.training_compounds)

    def assign_training_set(self, **kwargs):
        super().assign_training_set(**kwargs)
        self.training_natoms = array_(
            [len(comp.nuclear_charges) for comp in self.training_compounds]
        )
        if self.local_dn:
            self.combined_training_nuclear_charges = concatenate_(
                [comp.nuclear_charges for comp in self.training_compounds]
            )
        else:
            self.combined_training_nuclear_charges = None

    def init_shifts(self):
        self.extensive_shift_vals = None

    def get_training_kernel(self, sigma=None, out=None):
        if sigma is None:
            sigma = self.sigma

        assert sigma is not None
        if out is None:
            out = empty_((self.ntrain, self.ntrain))
        common_args = (sigma, out)
        if self.kernel_function_sym is None:
            args = (
                self.training_representations,
                self.training_representations,
                self.training_natoms,
                self.training_natoms,
            )
            if self.local_dn:
                args = (
                    *args,
                    self.combined_training_nuclear_charges,
                    self.combined_training_nuclear_charges,
                )
            self.kernel_function_asym(*args, *common_args, **self.kernel_kwargs)
        else:
            args = (self.training_representations, self.training_natoms)
            if self.local_dn:
                args = (*args, self.combined_training_nuclear_charges)
            self.kernel_function_sym(*args, *common_args, **self.kernel_kwargs)
        return out

    def combine_representations(self, representations_list):
        return concatenate_(representations_list)

    def get_shifted_prediction(self, prediction, all_nuclear_charges=None, **kwargs):
        if not self.shift_quantities:
            return prediction
        assert all_nuclear_charges is not None
        shift_coeffs = get_shift_coeffs(
            len(all_nuclear_charges),
            intensive_shift=False,
            extensive_shift=True,
            possible_nuclear_charges=self.possible_nuclear_charges,
            all_nuclear_charges=get_numba_list(all_nuclear_charges),
        )
        return prediction + dot_(shift_coeffs, self.extensive_shift_vals)

    def predict_from_representations(self, representations, all_nuclear_charges=None):
        assert all_nuclear_charges is not None
        nmols = len(all_nuclear_charges)
        query_natoms = array_([len(nuclear_charges) for nuclear_charges in all_nuclear_charges])
        self.temp_kernel = check_allocation((self.ntrain, nmols), output=self.temp_kernel)
        args = (self.training_representations, representations, self.training_natoms, query_natoms)
        if self.local_dn:
            combined_query_nuclear_charges = concatenate_(all_nuclear_charges)
            args = (*args, self.combined_training_nuclear_charges, combined_query_nuclear_charges)
        self.kernel_function_asym(*args, self.sigma, self.temp_kernel, **self.kernel_kwargs)
        return self.predict_from_kernel(nmols=nmols, all_nuclear_charges=all_nuclear_charges)

    def __call__(self, nuclear_charges, coords):
        representation = self.get_rep(nuclear_charges, coords)
        return self.predict_from_representations(
            representation, all_nuclear_charges=[nuclear_charges]
        )[0]

    def predict_from_compounds(self, compounds, **kwargs):
        representations = self.calc_all_representations(compounds, **kwargs)
        return self.predict_from_representations(
            representations,
            all_nuclear_charges=[comp.nuclear_charges for comp in compounds],
        )
