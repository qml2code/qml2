# NOTE: KK: Nick Browning didn't use multipliers for training energies and forces. I implemented such an option anyway
# because I think that it should affect how regularization affects how accurately training set energies and forces are reproduced.
from ..basic_utils import divided_by_parents, dump2pkl, loadpkl, mktmpdir, rmdir
from ..dimensionality_reduction import (
    project_local_representations,
    project_scale_local_representations,
)
from ..jit_interfaces import concatenate_, dint_, dot_, empty_, zeros_
from ..kernels.gradient_hadamard import (
    local_hadamard_force_kernel_processed_input,
    local_hadamard_product_force_kernel_processed_input,
)
from ..kernels.gradient_kernels import (
    get_energy_force_ranges,
    prediction_vector_to_forces_energies,
)
from ..math import lu_solve
from ..utils import check_allocation, get_atom_environment_ranges, get_element_ids_from_sorted
from .hadamard import HadamardFeaturesLocalModel
from .krr_forces import OQMLModel


# KK: the class includes some trash from both parents, but I don't think getting rid of it is worth the effort.
class HadamardFeaturesLocalForcesModel(HadamardFeaturesLocalModel, OQMLModel):
    def __init__(
        self,
        num_Z_matrix_dumps=None,
        energy_importance=None,
        clean_Z_matrix=True,
        use_lstsq=False,
        **kwargs
    ):
        # Divide kwargs by what is relevant for the two parents and
        # initialize according to it.
        keyword_relevance = {
            HadamardFeaturesLocalModel: [
                "sorted_elements",
                "npcas",
                "nfeatures",
                "ntransforms",
                "l2reg_diag_ratio",
            ]
        }
        # all other keywords will go into OQMLModel init.
        parent_init_order = [HadamardFeaturesLocalModel, OQMLModel]
        # ensuring that by default energy importance is not used.
        kwargs["energy_importance"] = energy_importance
        divided_by_parents(self, parent_init_order, "__init__", keyword_relevance, kwargs)
        self.num_Z_matrix_dumps = num_Z_matrix_dumps
        self.dumped_Z_pkl_files = None
        self.dump_Z_dir = None
        self.temp_Z_matrix = None
        self.clean_Z_matrix = clean_Z_matrix
        self.default_en_force_ranges_arr = empty_((2,), dtype=dint_)
        self.use_lstsq = use_lstsq

    def Z_matrix_dump_bounds(self):
        """
        Bounds of RFF feature stacks and RFF features that are stored in separate pkl files.
        """
        if self.num_Z_matrix_dumps is None:
            num_dumps = 1
        else:
            num_dumps = self.num_Z_matrix_dumps
        dump_size = self.nfeature_stacks // num_dumps
        dump_size_remainder = self.nfeature_stacks % num_dumps
        output = []
        feature_stack_lb = 0
        for dump_id in range(num_dumps):
            cur_dump_size = dump_size
            if dump_id < dump_size_remainder:
                cur_dump_size += 1
            feature_stack_ub = feature_stack_lb + cur_dump_size
            feature_lb = feature_stack_lb * self.npcas
            feature_ub = feature_stack_ub * self.npcas
            output.append((feature_stack_lb, feature_stack_ub, feature_lb, feature_ub))
            feature_stack_lb = feature_stack_ub
        return output

    def calculate_dump_weighted_Z_matrix(
        self,
        all_red_representations,
        element_ids,
        atom_nums,
        all_representation_grads,
        all_rel_neighbors,
        all_rel_neighbor_nums,
        mult_vals=None,
    ):
        self.dump_Z_dir = mktmpdir(template="Z_matrix_dump.XXXXXX")
        self.dumped_Z_pkl_files = []
        en_force_ranges_arr = get_energy_force_ranges(atom_nums)
        nvals = en_force_ranges_arr[-1]
        mol_ubound_arr = get_atom_environment_ranges(atom_nums)
        for dump_id, (feature_stack_lb, feature_stack_ub, feature_lb, feature_ub) in enumerate(
            self.Z_matrix_dump_bounds()
        ):
            temp_Z = empty_((nvals, feature_ub - feature_lb))
            dump_nfeature_stacks = feature_stack_ub - feature_stack_lb
            local_hadamard_force_kernel_processed_input(
                all_red_representations,
                all_representation_grads,
                all_rel_neighbors,
                all_rel_neighbor_nums,
                element_ids,
                self.all_sorf_diags[feature_stack_lb:feature_stack_ub],
                self.all_biases[feature_stack_lb:feature_stack_ub],
                self.all_reductors,
                self.sigma,
                temp_Z,
                en_force_ranges_arr,
                mol_ubound_arr,
                dump_nfeature_stacks,
                self.npcas,
                true_nfeatures=self.nfeatures,
            )
            if mult_vals is None:
                used_Z = temp_Z
            else:
                used_Z = (temp_Z.T * mult_vals).T

            if self.num_Z_matrix_dumps is None:
                self.temp_Z_matrix = used_Z
                return

            dump_filename = self.dump_Z_dir + "/Z_dump_" + str(dump_id) + ".pkl"

            dump2pkl(
                (feature_stack_lb, feature_stack_ub, feature_lb, feature_ub, used_Z),
                dump_filename,
            )
            self.dumped_Z_pkl_files.append(dump_filename)

    def train_kernel_rhs_from_dumped_Z(self, weighted_fitted_vals):
        if self.num_Z_matrix_dumps is None:
            train_kernel = dot_(self.temp_Z_matrix.T, self.temp_Z_matrix)
            rhs = dot_(self.temp_Z_matrix.T, weighted_fitted_vals)
        else:
            train_kernel = zeros_((self.nfeatures, self.nfeatures))
            rhs = zeros_((self.nfeatures,))
            for dump_id1, dump_filename1 in enumerate(self.dumped_Z_pkl_files):
                _, _, feature_lb1, feature_ub1, temp_Z1 = loadpkl(dump_filename1)
                rhs[feature_lb1:feature_ub1] = dot_(temp_Z1.T, weighted_fitted_vals)
                for dump_id2, dump_filename2 in enumerate(self.dumped_Z_pkl_files[: dump_id1 + 1]):
                    _, _, feature_lb2, feature_ub2, temp_Z2 = loadpkl(dump_filename2)
                    train_kernel[feature_lb1:feature_ub1, feature_lb2:feature_ub2] = dot_(
                        temp_Z1.T, temp_Z2
                    )
                    if dump_id1 != dump_id2:
                        train_kernel[
                            feature_lb2:feature_ub2, feature_lb1:feature_ub1
                        ] = train_kernel[feature_lb1:feature_ub1, feature_lb2:feature_ub2].T
        return train_kernel, rhs

    def run_clean_Z_matrix(self):
        if not self.clean_Z_matrix:
            return
        if self.num_Z_matrix_dumps is None:
            self.temp_Z_matrix = None
        else:
            rmdir(self.dump_Z_dir)
            self.dump_Z_dir = None
            self.dumped_Z_pkl_files = None

    def fit(
        self,
        all_nuclear_charges,
        all_representations,
        atom_nums,
        all_representation_grads,
        all_rel_neighbors,
        all_rel_neighbor_nums,
        training_energies,
        training_forces,
    ):
        fitted_vals = self.combine_energy_forces_rhs(training_energies, training_forces)
        if self.energy_importance is None:
            mult_vals = None
            used_fitted_vals = fitted_vals
        else:
            mult_vals = self.get_importance_multipliers(atom_nums)
            used_fitted_vals = fitted_vals * mult_vals
        element_ids = get_element_ids_from_sorted(all_nuclear_charges, self.sorted_elements)
        all_red_representations = project_local_representations(
            all_representations, element_ids, self.all_reductors
        )
        self.optimize_hyperparameters(all_red_representations)
        all_red_representations[:, :] /= self.sigma
        self.calculate_dump_weighted_Z_matrix(
            all_red_representations,
            element_ids,
            atom_nums,
            all_representation_grads,
            all_rel_neighbors,
            all_rel_neighbor_nums,
            mult_vals=mult_vals,
        )
        if self.use_lstsq:
            assert self.num_Z_matrix_dumps is None
            # Just as OQML use lstsq_ subroutine.
            self.get_alphas(self.temp_Z_matrix.T, used_fitted_vals)
        else:
            train_kernel, rhs = self.train_kernel_rhs_from_dumped_Z(used_fitted_vals)
            self.get_alphas_w_lambda(train_kernel, rhs, solver=lu_solve)
        self.run_clean_Z_matrix()

    def train(
        self,
        training_set_nuclear_charges,
        training_set_coords,
        training_energies,
        training_forces,
        representative_atom_num=1024,
    ):
        print("Calculating representations.")
        (
            all_reps,
            atom_nums,
            all_rep_grads,
            all_rel_neighbors,
            all_rel_neighbor_nums,
        ) = self.get_all_representations_wgrads(
            training_set_nuclear_charges,
            training_set_coords,
            suppress_openmp=self.training_reps_suppress_openmp,
        )
        print("Done")
        all_nuclear_charges = concatenate_(training_set_nuclear_charges)
        atom_nums = empty_((len(training_set_nuclear_charges),), dtype=dint_)
        for i, nuclear_charges in enumerate(training_set_nuclear_charges):
            atom_nums[i] = len(nuclear_charges)
        self.init_reductors_elements(
            all_reps,
            all_nuclear_charges,
            representative_atom_num=representative_atom_num,
        )
        self.init_features()
        self.fit(
            all_nuclear_charges,
            all_reps,
            atom_nums,
            all_rep_grads,
            all_rel_neighbors,
            all_rel_neighbor_nums,
            training_energies,
            training_forces,
        )

    def predict_from_representations(self, nmols):
        atom_nums = self.temp_atom_nums[:nmols]
        ubound_arr = get_atom_environment_ranges(atom_nums)
        tot_natoms = ubound_arr[nmols]
        energy_force_ranges = get_energy_force_ranges(atom_nums)

        self.temp_element_ids = get_element_ids_from_sorted(
            self.temp_nuclear_charges[:tot_natoms],
            self.sorted_elements,
            output=self.temp_element_ids,
        )
        self.temp_reduced_scaled_reps = project_scale_local_representations(
            self.temp_reps,
            self.temp_element_ids,
            self.all_reductors,
            self.sigma,
            output=self.temp_reduced_scaled_reps,
            natoms=ubound_arr[nmols],
        )
        self.temp_prediction_vector = check_allocation(
            (energy_force_ranges[nmols],), output=self.temp_prediction_vector
        )
        local_hadamard_product_force_kernel_processed_input(
            self.temp_reduced_scaled_reps,
            self.temp_rep_grads,
            self.temp_relevant_neighbor_arr,
            self.temp_relevant_neighbor_nums,
            self.temp_element_ids,
            self.all_sorf_diags,
            self.all_biases,
            self.all_reductors,
            self.sigma,
            self.temp_prediction_vector,
            self.alphas,
            energy_force_ranges,
            ubound_arr,
            self.nfeature_stacks,
            self.npcas,
        )
        return prediction_vector_to_forces_energies(
            self.temp_prediction_vector,
            atom_nums,
            nmols,
            energy_output=self.temp_energies,
            forces_output=self.temp_forces,
        )

    def forward(self, *args, **kwargs):
        return OQMLModel.forward(self, *args, **kwargs)
