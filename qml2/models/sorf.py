from ..basic_utils import dump2pkl, loadpkl
from ..dimensionality_reduction import (
    get_rand_reductor,
    get_reductor,
    get_reductors_diff_species,
    project_local_representations,
    project_representation,
    project_scale_local_representations,
    project_scale_representations,
)
from ..jit_interfaces import array_, concatenate_, dint_, dot_, empty_, matmul_
from ..kernels.sorf import (
    create_sorf_matrices,
    create_sorf_matrices_diff_species,
    generate_local_sorf_processed_input,
    generate_sorf_processed_input,
)
from ..math import lu_solve
from ..representations import generate_fchl19
from ..utils import (
    check_allocation,
    get_atom_environment_ranges,
    get_element_ids_from_sorted,
    int_in_sorted,
    searchsorted_wexception,
)
from .krr import KRRLocalModel, KRRModel


def is_power2(n: int):
    return (n & (n - 1) == 0) and n != 0


class SORFModel(KRRModel):
    def __init__(
        self, npcas=128, nfeatures=None, ntransforms=2, use_rep_reduction=False, **other_kwargs
    ):
        # feature size
        KRRModel.__init__(self, **other_kwargs)
        assert is_power2(npcas)
        self.npcas = int(npcas)
        self.nfeatures = int(nfeatures)
        if self.nfeatures is not None:
            assert self.nfeatures % self.npcas == 0
            self.nfeature_stacks = self.nfeatures // self.npcas
        self.ntransforms = ntransforms
        self.temp_reduced_scaled_reps = None
        self.use_rep_reduction = use_rep_reduction
        self.init_internal_feature_parameters()

    def init_kernel_functions(self, *args, **kwargs):
        pass

    def init_internal_feature_parameters(self):
        self.reductor = None
        self.biases = None
        self.sorf_diags = None

    # To make reproducability easier.
    def dump_randomized_parameters(self, filename):
        dump2pkl((self.reductor, self.biases, self.sorf_diags), filename)

    def load_randomized_parameters(self, filename):
        self.reductor, self.biases, self.sorf_diags = loadpkl(filename)

    def init_nfeatures_from_train_size(self, training_set_size):
        """
        If the number of random Fourier Features is not yet set just pick one to fit training set size.
        TODO: KK: the choice is actually very bad, should be replaced.
        """
        if self.nfeatures is not None:
            return  # was already initialized.
        self.nfeature_stacks = training_set_size // self.npcas
        if training_set_size % self.npcas == 0:
            self.nfeatures = training_set_size
        else:
            self.nfeature_stacks += 1
            self.nfeatures = self.nfeature_stacks * self.ncpas

    def init_reductor(self, training_representations, representative_atom_num=1024):
        if self.use_rep_reduction:
            self.reductor = get_reductor(
                training_representations, self.npcas, num_samples=representative_atom_num
            )

    def reference_diag_ids(self):
        r = array_(list(range(self.nfeatures)))
        return (r, r)

    def init_features(self):
        self.biases, self.sorf_diags = create_sorf_matrices(
            self.nfeature_stacks, self.ntransforms, self.npcas
        )
        self.temp_kernel = empty_((1, self.nfeatures))

    def get_reduced_representations(self, all_representations):
        return project_representation(all_representations, self.reductor)

    def fit(self, all_representations, training_set_values, ntrain):
        Z = empty_((ntrain, self.nfeatures))
        all_red_representations = project_representation(all_representations, self.reductor)
        self.optimize_hyperparameters(all_red_representations)
        all_red_representations[:, :] /= self.sigma

        generate_sorf_processed_input(
            all_red_representations,
            self.sorf_diags,
            self.biases,
            Z,
            self.nfeature_stacks,
            self.npcas,
        )
        shifted_training_set_values = self.init_apply_shift(training_set_values)
        rhs = matmul_(Z.T, shifted_training_set_values)
        train_kernel = matmul_(Z.T, Z)
        self.get_alphas_w_lambda(train_kernel, rhs, solver=lu_solve)

    def train(
        self,
        training_set_nuclear_charges,
        training_set_coords,
        training_set_values,
        representative_atom_num=1024,
    ):
        print("Calculating representations.")
        all_representations = self.get_all_representations(
            training_set_nuclear_charges,
            training_set_coords,
            suppress_openmp=self.training_reps_suppress_openmp,
        )
        print("Done")
        ntrain = len(training_set_nuclear_charges)
        assert ntrain == len(training_set_values)
        self.init_reductor(
            all_representations,
            representative_atom_num=representative_atom_num,
        )
        self.init_features()
        self.fit(all_representations, training_set_values, ntrain)

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
        self.temp_kernel = check_allocation((nmols, self.nfeatures), output=self.temp_kernel)
        generate_sorf_processed_input(
            self.temp_reduced_scaled_reps,
            self.sorf_diags,
            self.biases,
            self.temp_kernel,
            self.nfeature_stacks,
            self.npcas,
        )
        return self.predict_from_kernel(nmols=nmols)

    def predict_from_kernel(self, nmols=1):
        return dot_(self.temp_kernel[:nmols], self.alphas) + self.val_shift

    def forward(self, nuclear_charges, coords):
        self.temp_reps = check_allocation((1, self.reductor.shape[0]), output=self.temp_reps)
        self.temp_reps[0, :] = self.get_rep(nuclear_charges, coords)
        return self.predict_from_representations(self.temp_reps)[0]


class SORFLocalModel(SORFModel, KRRLocalModel):
    def __init__(
        self, representation_function=generate_fchl19, sorted_elements=None, **other_kwargs
    ):
        # feature size
        SORFModel.__init__(self, representation_function=representation_function, **other_kwargs)
        self.sorted_elements = sorted_elements
        if self.sorted_elements is None:
            self.nelements = None
        else:
            self.nelements = len(self.sorted_elements)
        self.default_mol_ubound_arr = empty_((2,), dtype=dint_)
        self.temp_element_ids = None

    def init_internal_feature_parameters(self):
        self.all_reductors = None
        self.all_biases = None
        self.all_sorf_diags = None

    def dump_randomized_parameters(self, filename):
        dump2pkl(
            (self.all_reductors, self.all_biases, self.all_sorf_diags, self.sorted_elements),
            filename,
        )

    def load_randomized_parameters(self, filename):
        self.all_reductors, self.all_biases, self.all_sorf_diags, self.sorted_elements = loadpkl(
            filename
        )

    def get_rep(self, *args, **kwargs):
        return KRRLocalModel.get_rep(self, *args)

    def get_all_representations(self, *args, **kwargs):
        return KRRLocalModel.get_all_representations(*args)

    def fill_missing_reductors(self, present_elements):
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
        if self.sorted_elements is None:
            self.sorted_elements = found_sorted_elements
            self.nelements = self.sorted_elements.shape[0]
        else:
            for el in found_sorted_elements:
                assert int_in_sorted(el, self.sorted_elements)
            if found_sorted_elements.shape[0] < self.sorted_elements.shape[0]:
                self.fill_missing_reductors(found_sorted_elements)

    def init_reductors_elements(
        self, training_representations, training_nuclear_charges, representative_atom_num=1024
    ):
        if (self.all_reductors is not None) or (not self.use_rep_reduction):
            return
        self.all_reductors, found_sorted_elements = get_reductors_diff_species(
            training_representations, training_nuclear_charges, self.npcas, representative_atom_num
        )
        self.check_sorted_elements_reductors(found_sorted_elements)

    def init_features(self):
        if self.all_biases is not None:
            return
        self.all_biases, self.all_sorf_diags = create_sorf_matrices_diff_species(
            self.nfeature_stacks, self.nelements, self.ntransforms, self.npcas
        )
        self.temp_kernel = empty_((1, self.nfeatures))

    def fit(
        self, all_representations, all_nuclear_charges, training_set_values, atom_nums, ntrain
    ):
        Z = empty_((ntrain, self.nfeatures))
        element_ids = get_element_ids_from_sorted(all_nuclear_charges, self.sorted_elements)
        all_red_representations = project_local_representations(
            all_representations, element_ids, self.all_reductors
        )
        self.optimize_hyperparameters(all_red_representations)
        all_red_representations[:, :] /= self.sigma
        ubound_arr = get_atom_environment_ranges(atom_nums)
        generate_local_sorf_processed_input(
            all_red_representations,
            element_ids,
            self.all_sorf_diags,
            self.all_biases,
            Z,
            ubound_arr,
            self.nfeature_stacks,
            self.npcas,
        )
        shifted_training_set_values = self.init_apply_shift(training_set_values)
        rhs = matmul_(Z.T, shifted_training_set_values)
        train_kernel = matmul_(Z.T, Z)
        self.get_alphas_w_lambda(train_kernel, rhs)

    def train(
        self,
        training_set_nuclear_charges,
        training_set_coords,
        training_set_values,
        representative_atom_num=1024,
    ):
        print("Calculating representations.")
        all_representations = self.get_all_representations(
            self, training_set_nuclear_charges, training_set_coords
        )
        print("Done")
        all_nuclear_charges = concatenate_(training_set_nuclear_charges)
        ntrain = len(training_set_nuclear_charges)
        assert ntrain == len(training_set_values)
        atom_nums = empty_((len(training_set_nuclear_charges),), dtype=dint_)
        for i, nuclear_charges in enumerate(training_set_nuclear_charges):
            atom_nums[i] = len(nuclear_charges)
        self.init_reductors_elements(
            all_representations,
            all_nuclear_charges,
            representative_atom_num=representative_atom_num,
        )
        self.init_features()
        self.fit(all_representations, all_nuclear_charges, training_set_values, atom_nums, ntrain)

    def predict_from_kernel(self, nmols=1):
        return SORFModel.predict_from_kernel(self, nmols)

    def predict_from_representations(self, representations, nuclear_charges, atom_nums=None):
        if atom_nums is None:
            # we are doing calculations for just one molecule
            self.default_mol_ubound_arr[1] = representations.shape[0]
            self.default_mol_ubound_arr[0] = 0
            ubound_arr = self.default_mol_ubound_arr
        else:
            ubound_arr = get_atom_environment_ranges(atom_nums)
        self.temp_element_ids = get_element_ids_from_sorted(
            nuclear_charges, self.sorted_elements, output=self.temp_element_ids
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
        self.temp_kernel = check_allocation((nmols, self.nfeatures), output=self.temp_kernel)
        generate_local_sorf_processed_input(
            self.temp_reduced_scaled_reps,
            self.temp_element_ids,
            self.all_sorf_diags,
            self.all_biases,
            self.temp_kernel,
            ubound_arr,
            self.nfeature_stacks,
            self.npcas,
        )
        return self.predict_from_kernel(nmols=nmols)

    def forward(self, *args, **kwargs):
        return KRRLocalModel.forward(self, *args, **kwargs)
