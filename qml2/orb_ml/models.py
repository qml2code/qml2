# KK: Everything is composed in terms of OML_Compound and OML_Slater_pair instances since
# they include references to files where ab initio results are stored for later re-use,
# decreasing CPU time required to test a model many times.

from ..jit_interfaces import Module_, array_, empty_
from ..models.krr import KRRModel
from .kernels import (
    OML_KernelInput,
    kernel_from_processed_input,
    renormed_smoothened_sigmas,
    rep_stddevs,
)
from .oml_compound import OML_Compound, OML_pyscf_calc_params
from .oml_compound_list import OML_Compound_list
from .representations import OML_rep_params


class FJKModel(KRRModel):
    def __init__(
        self,
        sigmas=None,
        global_sigma=None,
        orb_product="gaussian",
        norm="l2",
        l2reg_diag_ratio=array_(1.0e-6),
        num_consistency_check=array_(1.0e-6),
        training_reps_suppress_openmp=False,
        rep_params: OML_rep_params = OML_rep_params(),
        pyscf_calc_params: OML_pyscf_calc_params = OML_pyscf_calc_params(),
        apply_shift=False,
        relatively_small_val=1.0e-3,
    ):
        Module_.__init__(self)
        # hyperparameters
        self.sigmas = sigmas
        self.global_sigma = global_sigma
        self.l2reg_diag_ratio = l2reg_diag_ratio
        # representation parameters
        self.rep_params = rep_params
        self.pyscf_calc_params = pyscf_calc_params
        # kernel parameters
        self.orb_product = orb_product
        self.norm = norm
        # basic arrays.
        self.init_basic_arrays()
        # For checking model's numerical stability.
        self.num_consistency_check = num_consistency_check
        self.relatively_small_val = relatively_small_val
        # other
        self.apply_shift = apply_shift
        self.training_reps_suppress_openmp = training_reps_suppress_openmp

    def init_basic_arrays(self):
        # fitted model parameters
        self.alphas = None
        # where the query-training set kernel is stored.
        self.temp_kernel = None
        # For shifting predicted quantities.
        self.val_shift = None

    def get_initialized_compound(self, compound: OML_Compound):
        compound.generate_orb_reps(rep_params=self.rep_params)
        return compound

    def optimize_hyperparameters(self, all_compounds):
        """
        For now just initial guess.
        TODO: make better
        """
        if self.sigmas is not None:
            return
        print("Optimizing hyperparameters")
        raw_sigmas = rep_stddevs(all_compounds)
        self.sigmas = renormed_smoothened_sigmas(
            raw_sigmas, relatively_small_val=self.relatively_small_val, norm=self.norm
        )
        # KK: I am not sure how to get a reasonable guess for this one TBH.
        self.global_sigma = 0.5
        print("Done")

    def fit(self, all_compounds, training_set_values):
        self.training_set_kernel_input = OML_KernelInput(all_compounds)
        train_kernel = kernel_from_processed_input(
            self.training_set_kernel_input, None, self.sigmas, self.global_sigma, norm=self.norm
        )
        shifted_training_set_values = self.init_apply_shift(training_set_values)
        self.get_alphas_w_lambda(train_kernel, shifted_training_set_values)

    def combine_representations(self, representations_list):
        return array_(representations_list)

    def get_all_initialized_compounds(self, oml_compound_list, suppress_openmp=False):
        if suppress_openmp:
            fixed_num_threads = 1
        else:
            fixed_num_threads = None
        all_compounds = OML_Compound_list(oml_compound_list)
        all_compounds.generate_orb_reps(
            rep_params=self.rep_params, fixed_num_threads=fixed_num_threads
        )
        return all_compounds

    def train(
        self,
        training_set_compounds,
        training_set_values,
    ):
        print("Calculating representations.")
        init_compounds = self.get_all_initialized_compounds(
            training_set_compounds, suppress_openmp=self.training_reps_suppress_openmp
        )
        print("Done")
        self.ntrain = len(init_compounds)
        assert self.ntrain == len(training_set_values)
        self.optimize_hyperparameters(init_compounds)
        self.fit(init_compounds, training_set_values)
        self.temp_kernel = empty_((self.ntrain, 1))

    def predict_from_kernel_input(self, kernel_input: OML_KernelInput):
        self.temp_kernel = kernel_from_processed_input(
            self.training_set_kernel_input,
            kernel_input,
            self.sigmas,
            self.global_sigma,
            norm=self.norm,
        )
        return self.predict_from_kernel(nmols=kernel_input.num_mols)

    def predict_from_compound(self, input_compound):
        init_compound = self.get_initialized_compound(input_compound)
        kernel_input = OML_KernelInput([init_compound])
        return self.predict_from_kernel_input(kernel_input)[0]

    def forward(self, nuclear_charges, coordinates, store_filename=None):
        comp = OML_Compound(
            coordinates=coordinates, nuclear_charges=nuclear_charges, mats_savefile=store_filename
        )
        return self.predict_from_compound(comp)
