from ..basic_utils import dump2pkl, loadpkl
from ..dimensionality_reduction import (
    fix_reductor_signs,
    fix_reductors_signs,
    get_rand_reductor,
    get_reductor,
    get_reductors_diff_species,
    project_scale_local_representations,
    project_scale_representations,
)
from ..jit_interfaces import array_, concatenate_, dot_, empty_, exp_, mean_
from ..kernels.sorf import (
    create_sorf_matrices,
    create_sorf_matrices_diff_species,
    generate_local_sorf_processed_input,
    generate_sorf_processed_input,
)
from ..math import regression_using_Z_SVD, svd_aligned
from ..representations import generate_fchl19
from ..utils import (
    check_allocation,
    flatten_to_scalar,
    get_atom_environment_ranges,
    get_element_ids_from_sorted,
    int_in_sorted,
    is_power2,
    searchsorted_wexception,
)
from .krr import KRRLocalModel, KRRModel
from .sorf_hyperparameter_optimization import SORFLeaveOneOutL2regOpt


class SORFModel(KRRModel):
    def __init__(
        self,
        nfeatures=32768,
        ntransforms=3,
        use_rep_reduction=False,
        npcas=None,
        rng=None,
        fixed_reductor_signs=False,
        **other_kwargs,
    ):
        """
        Model using (global) SORF.

        Args (also see KRRModel):
            nfeatures (int): number of features.
            ntransforms (int): number of fast Walsh-Hadamard transforms performed to obtain SORF.
            npcas (int or None): if use_rep_reduction==True defines number of principle components; otherwise defines to which size input representation vectors are padded with zeros. If None taken to equal nfeatures; otherwise should be a power of 2.
            rng (numpy.random._generator.Generator): the random number generator used to generate random components of SORF.
            fixed_reductor_signs (bool): if True sign of reductor matrices are fixed to make the first row's values positive. Does not affect numerical performance, but prevents occasional "flipping" of eigenvectors signs which obstructs test writing.
        """
        # feature size
        self.basic_init(**other_kwargs)
        self.nfeatures = int(nfeatures)
        if npcas is None:
            npcas = self.nfeatures
        assert is_power2(npcas)
        self.npcas = int(npcas)
        assert self.nfeatures % self.npcas == 0
        self.nfeature_stacks = self.nfeatures // self.npcas
        self.ntransforms = ntransforms
        self.use_rep_reduction = use_rep_reduction
        self.fixed_reductor_signs = fixed_reductor_signs
        # some temp arrays
        self.temp_reduced_scaled_reps = None
        self.temp_feature_vectors = None

        # rng used for feature (and sometimes random reductor) generation
        self.rng = rng

        self.init_reductors()
        self.init_features()

    def readjust_training_temp_arrays(self):
        pass

    # To make reproducability easier.
    def dump_randomized_parameters(self, filename):
        dump2pkl((self.reductor, self.biases, self.sorf_diags), filename)

    def load_randomized_parameters(self, filename):
        self.reductor, self.biases, self.sorf_diags = loadpkl(filename)

    def init_reductors(self):
        self.reductor = None

    def calc_reductors(self, representative_sample_num=1024):
        if not self.use_rep_reduction:
            return
        self.reductor = get_reductor(
            self.training_representations,
            self.npcas,
            num_samples=representative_sample_num,
            rng=self.rng,
        )
        if self.fixed_reductor_signs:
            fix_reductor_signs(self.reductor)

    def init_features(self):
        self.biases, self.sorf_diags = create_sorf_matrices(
            self.nfeature_stacks, self.ntransforms, self.npcas, rng=self.rng
        )

    def get_sigma_opt_func(self):
        """
        Define the function minimized by BOSS to optimize sigma.
        """
        # ensure the training kernel is not re-allocated at each call
        temp_feature_vectors = empty_((self.ntrain, self.nfeatures))
        if self.use_rep_reduction:
            temp_red_representations_shape = (self.training_representations.shape[0], self.npcas)
        else:
            temp_red_representations_shape = self.training_representations.shape
        temp_red_representations = empty_(temp_red_representations_shape)

        def opt_func_feature_vectors(ln_sigma):
            # TODO: make a short function?
            ln_sigma = flatten_to_scalar(ln_sigma)
            print("Testing ln sigma:", ln_sigma)
            sigma = exp_(ln_sigma)
            return self.get_training_feature_vectors(
                sigma=sigma,
                out=temp_feature_vectors,
                temp_red_representations=temp_red_representations,
            )

        return SORFLeaveOneOutL2regOpt(
            opt_func_feature_vectors,
            self.training_quantities,
            ln_l2reg_diag_ratio_bounds=self.ln_l2reg_diag_ratio_bounds,
            total_iterpts=self.l2reg_total_iterpts,
            test_mode=self.test_mode,
        )

    def get_alphas(self, train_feature_vectors, training_quantities=None):
        if training_quantities is None:
            training_quantities = self.training_quantities
        Z_U, Z_singular_values, Z_Vh = svd_aligned(train_feature_vectors)
        l2reg = self.l2reg
        if l2reg is None:
            assert self.l2reg_diag_ratio is not None
            l2reg = self.l2reg_diag_ratio * mean_(Z_singular_values)

        self.alphas = regression_using_Z_SVD(
            None,
            l2reg,
            training_quantities,
            Z_U=Z_U,
            Z_singular_values=Z_singular_values,
            Z_Vh=Z_Vh,
        )

    def get_training_feature_vectors(self, sigma=None, temp_red_representations=None, out=None):
        if sigma is None:
            sigma = self.sigma
        if out is None:
            out = empty_((self.ntrain, self.nfeatures))
        all_red_representations = project_scale_representations(
            self.training_representations, self.reductor, sigma, output=temp_red_representations
        )
        generate_sorf_processed_input(
            all_red_representations,
            self.sorf_diags,
            self.biases,
            out,
            self.nfeature_stacks,
            self.npcas,
        )
        return out

    def assign_training_set(self, representative_sample_num=1024, **kwargs):
        super().assign_training_set(**kwargs)
        self.calc_reductors(representative_sample_num=representative_sample_num)

    def fit(self, **kwargs):
        self.assign_training_set(**kwargs)
        train_feature_vectors = self.get_training_feature_vectors()
        self.get_alphas(train_feature_vectors)

    def predict_from_kernel(self, nmols=1, **kwargs):
        predictions = dot_(self.temp_feature_vectors[:nmols, :], self.alphas)
        return self.get_shifted_prediction(predictions, **kwargs)

    def predict_from_representations(self, representations, nmols=None):
        if nmols is None:
            nmols = representations.shape[0]
        self.temp_reduced_scaled_reps = project_scale_representations(
            representations,
            self.reductor,
            self.sigma,
            output=self.temp_reduced_scaled_reps,
            nmols=nmols,
        )
        self.temp_feature_vectors = check_allocation(
            (nmols, self.nfeatures), output=self.temp_feature_vectors
        )
        generate_sorf_processed_input(
            self.temp_reduced_scaled_reps,
            self.sorf_diags,
            self.biases,
            self.temp_feature_vectors,
            self.nfeature_stacks,
            self.npcas,
        )
        return self.predict_from_kernel(nmols=nmols)


class SORFLocalModel(SORFModel, KRRLocalModel):
    def __init__(
        self,
        representation_function=generate_fchl19,
        possible_nuclear_charges=None,
        **other_kwargs,
    ):
        """
        Model using (local-dn) SORF. Keyword arguments are shared with SORFModel and KRRLocalModel classes.
        """
        assert possible_nuclear_charges is not None
        self.possible_nuclear_charges = possible_nuclear_charges
        self.sorted_elements = array_(sorted(possible_nuclear_charges))
        self.nelements = len(self.sorted_elements)
        self.temp_element_ids = None
        self.local_dn = True  # mostly for compatibility with some KRRLocalModel procedures; though also we do use SORF reproducing local_dn kernel here.
        SORFModel.__init__(self, representation_function=representation_function, **other_kwargs)

    def readjust_training_temp_arrays(self):
        pass

    def init_shifts(self):
        KRRLocalModel.init_shifts(self)

    def adjust_init_quant_shift(self):
        KRRLocalModel.adjust_init_quant_shift(self)

    def dump_randomized_parameters(self, filename):
        dump2pkl(
            (self.all_reductors, self.all_biases, self.all_sorf_diags, self.sorted_elements),
            filename,
        )

    def load_randomized_parameters(self, filename):
        self.all_reductors, self.all_biases, self.all_sorf_diags, self.sorted_elements = loadpkl(
            filename
        )

    def fill_missing_reductors(self, present_elements):
        """
        Sometimes not all elements are found in the training set. In such cases missing reductors need to be initialized randomly.
        """
        old_reductors = self.all_reductors
        rep_size = old_reductors.shape[1]
        self.all_reductors = empty_((self.nelements, rep_size, self.npcas))
        for el_id, el in enumerate(self.sorted_elements):
            if el in present_elements:
                present_el_id = searchsorted_wexception(self.sorted_elements, el)
                self.all_reductors[el_id, :, :] = old_reductors[present_el_id, :, :]
            else:
                self.all_reductors[el_id, :, :] = get_rand_reductor(rep_size, self.npcas)

    def check_sorted_elements_reductors(self, found_sorted_elements):
        for el in found_sorted_elements:
            assert int_in_sorted(el, self.sorted_elements)
        if found_sorted_elements.shape[0] < self.sorted_elements.shape[0]:
            self.fill_missing_reductors(found_sorted_elements)

    def init_reductors(self):
        self.all_reductors = None

    def calc_reductors(self, representative_atom_num=1024):
        if not self.use_rep_reduction:
            return
        self.all_reductors, found_sorted_elements = get_reductors_diff_species(
            self.training_representations,
            self.combined_training_nuclear_charges,
            self.npcas,
            representative_atom_num,
            rng=self.rng,
        )
        self.check_sorted_elements_reductors(found_sorted_elements)
        if self.fixed_reductor_signs:
            fix_reductors_signs(self.all_reductors)

    def assign_training_set(self, representative_atom_num=1024, **kwargs):
        KRRLocalModel.assign_training_set(self, **kwargs)
        self.calc_reductors(representative_atom_num=representative_atom_num)

    def init_features(self):
        self.all_biases, self.all_sorf_diags = create_sorf_matrices_diff_species(
            self.nfeature_stacks, self.nelements, self.ntransforms, self.npcas, rng=self.rng
        )

    def get_training_feature_vectors(self, sigma=None, temp_red_representations=None, out=None):
        if sigma is None:
            sigma = self.sigma
        if out is None:
            out = empty_((self.ntrain, self.nfeatures))
        element_ids = get_element_ids_from_sorted(
            self.combined_training_nuclear_charges, self.sorted_elements
        )
        all_red_representations = project_scale_local_representations(
            self.training_representations,
            element_ids,
            self.all_reductors,
            sigma,
            output=temp_red_representations,
        )
        ubound_arr = get_atom_environment_ranges(self.training_natoms)
        generate_local_sorf_processed_input(
            all_red_representations,
            element_ids,
            self.all_sorf_diags,
            self.all_biases,
            out,
            ubound_arr,
            self.nfeature_stacks,
            self.npcas,
        )
        return out

    def get_shifted_prediction(self, prediction, **kwargs):
        return KRRLocalModel.get_shifted_prediction(self, prediction, **kwargs)

    def predict_from_kernel(self, nmols=1, **kwargs):
        return SORFModel.predict_from_kernel(self, nmols, **kwargs)

    def predict_from_representations(self, representations, all_nuclear_charges=None):
        assert all_nuclear_charges is not None
        atom_nums = array_([nuclear_charges.shape[0] for nuclear_charges in all_nuclear_charges])
        ubound_arr = get_atom_environment_ranges(atom_nums)
        self.temp_element_ids = get_element_ids_from_sorted(
            concatenate_(all_nuclear_charges), self.sorted_elements, output=self.temp_element_ids
        )
        self.temp_reduced_scaled_reps = project_scale_local_representations(
            representations,
            self.temp_element_ids,
            self.all_reductors,
            self.sigma,
            output=self.temp_reduced_scaled_reps,
            natoms=ubound_arr[-1],
        )
        nmols = ubound_arr.shape[0] - 1
        self.temp_feature_vectors = check_allocation(
            (nmols, self.nfeatures), output=self.temp_feature_vectors
        )
        tot_natoms = ubound_arr[-1]
        generate_local_sorf_processed_input(
            self.temp_reduced_scaled_reps[:tot_natoms],
            self.temp_element_ids[:tot_natoms],
            self.all_sorf_diags,
            self.all_biases,
            self.temp_feature_vectors[:nmols],
            ubound_arr,
            self.nfeature_stacks,
            self.npcas,
        )
        return self.predict_from_kernel(nmols=nmols, all_nuclear_charges=all_nuclear_charges)

    def predict_from_compounds(self, *args, **kwargs):
        return KRRLocalModel.predict_from_compounds(self, *args, **kwargs)

    def __call__(self, *args):
        return KRRLocalModel.__call__(self, *args)
