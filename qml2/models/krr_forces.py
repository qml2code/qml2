# TODO: Perhaps use energy_forces_ids from ..kernels.gradient_kernels more?
from ..data import nCartDim
from ..jit_interfaces import (
    Module_,
    array_,
    concatenate_,
    dint_,
    dot_,
    empty_,
    jit_,
    lstsq_,
    prange_,
    sqrt_,
    sum_,
)
from ..kernels.gradient_kernels import (
    energy_forces_ids,
    get_derivative_kernel,
    nen_force_vals,
    prediction_vector_to_forces_energies,
)
from ..math import svd_solve
from ..parallelization import embarrassingly_parallel
from ..test_utils.toy_representation import generate_toy_representation_with_gradients
from ..utils import check_allocation
from .krr import KRRLocalModel


# For fast copying
@jit_
def copy_grads_to_merged(
    destination_rep_grads,
    destination_rel_neighbors,
    copied_rep_grads,
    copied_rel_neighbors,
    rel_neighbor_nums,
):
    for i in prange_(rel_neighbor_nums.shape[0]):
        nneighbor_num = rel_neighbor_nums[i]
        destination_rep_grads[i, :, :nneighbor_num, :] = copied_rep_grads[i, :, :nneighbor_num, :]
        destination_rel_neighbors[i, :nneighbor_num] = copied_rel_neighbors[i, :nneighbor_num]


class OQMLModel(KRRLocalModel):
    def __init__(
        self,
        representation_function_wgrads=generate_toy_representation_with_gradients,
        sigma=None,
        rep_kwargs={},
        lstsq_rcond=None,
        training_reps_suppress_openmp=False,
        energy_importance=0.01,
    ):
        Module_.__init__(self)
        # hyperparameters
        self.sigma = sigma
        self.lstsq_rcond = lstsq_rcond
        # representation parameters
        self.representation_function_wgrads = representation_function_wgrads
        self.rep_kwargs = rep_kwargs
        self.energy_importance = energy_importance
        # kernel
        self.init_basic_arrays()
        self.init_kernel_functions()
        self.training_reps_suppress_openmp = training_reps_suppress_openmp

    def init_kernel_functions(self):
        self.kernel_function = get_derivative_kernel(symmetric=False, derivatives="oqml")

    def init_basic_arrays(self):
        # temporary arrays
        # fitted model parameters
        self.alphas = None
        self.temp_kernel = None
        # will be useful if (hopefully) we start doing representation padding.
        self.temp_reps = None
        self.temp_rep_grads = None
        self.temp_relevant_neighbor_arr = None
        self.temp_nuclear_charges = None
        self.temp_atom_nums = empty_((1,), dtype=dint_)
        self.temp_relevant_neighbor_nums = None
        # Where prediction is written
        self.temp_prediction_vector = None
        self.temp_forces = None
        self.temp_energies = None

    def get_rep_wgrads(self, nuclear_charges, coords):
        assert nuclear_charges.shape[0] == coords.shape[0]
        assert coords.shape[1] == 3
        return self.representation_function_wgrads(nuclear_charges, coords, **self.rep_kwargs)

    def get_rep_wgrads_tuple(self, ncharges_coords_tuple):
        return self.get_rep_wgrads(*ncharges_coords_tuple)

    def get_all_representations_wgrads(
        self, all_nuclear_charges, all_coords, suppress_openmp=False
    ):
        if suppress_openmp:
            fixed_num_threads = 1
        else:
            fixed_num_threads = None
        all_reps_wgrads_list = embarrassingly_parallel(
            self.get_rep_wgrads_tuple,
            zip(all_nuclear_charges, all_coords),
            (),
            fixed_num_threads=fixed_num_threads,
        )
        # First determine dimensionalities of the merged arrays.
        max_num_rel_neighbors = 0
        atom_nums = empty_((len(all_nuclear_charges),), dtype=dint_)
        for i, (_, _, _, rel_neighbor_nums) in enumerate(all_reps_wgrads_list):
            max_num_rel_neighbors = max(max_num_rel_neighbors, max(rel_neighbor_nums))
            atom_nums[i] = len(rel_neighbor_nums)

        tot_natoms = sum_(atom_nums)
        rep_size = all_reps_wgrads_list[0][0].shape[1]
        all_reps = empty_((tot_natoms, rep_size))
        all_rep_grads = empty_((tot_natoms, rep_size, max_num_rel_neighbors, nCartDim))
        all_rel_neighbors = empty_((tot_natoms, max_num_rel_neighbors), dtype=dint_)
        all_rel_neighbor_nums = empty_((tot_natoms,), dtype=dint_)

        cur_lb = 0
        for i, (reps, rep_grads, rel_neighbors, rel_neighbor_nums) in enumerate(
            all_reps_wgrads_list
        ):
            cur_ub = cur_lb + atom_nums[i]
            all_reps[cur_lb:cur_ub, :] = reps[:, :]
            all_rel_neighbor_nums[cur_lb:cur_ub] = rel_neighbor_nums[:]
            copy_grads_to_merged(
                all_rep_grads[cur_lb:cur_ub],
                all_rel_neighbors[cur_lb:cur_ub],
                rep_grads,
                rel_neighbors,
                rel_neighbor_nums,
            )
            cur_lb = cur_ub

        return all_reps, atom_nums, all_rep_grads, all_rel_neighbors, all_rel_neighbor_nums

    def get_alphas(self, train_kernel, rhs):
        self.alphas = lstsq_(train_kernel.T, rhs, rcond=self.lstsq_rcond)[0]

    def combine_energy_forces_rhs(self, energies, forces):
        tot_natoms = sum(f.shape[0] for f in forces)
        nvals = len(energies) + tot_natoms * nCartDim
        output = empty_((nvals,))
        en_id = 0
        for en, f in zip(energies, forces):
            output[en_id] = en
            force_lb = en_id + 1
            force_ub = force_lb + f.shape[0] * nCartDim
            output[force_lb:force_ub] = f.flatten()
            en_id = force_ub
        return output

    def get_importance_multipliers(
        self,
        atom_nums,
    ):
        energy_importance_multiplier = sqrt_(array_(self.energy_importance))
        all_importance_multipliers = empty_((atom_nums.shape[0] + nCartDim * sum_(atom_nums),))
        lb = 0
        for atom_num in atom_nums:
            all_importance_multipliers[lb] = energy_importance_multiplier
            lb += 1
            ub = lb + atom_num * nCartDim
            all_importance_multipliers[lb:ub] = 1 / sqrt_(array_(float(atom_num)))
            lb = ub

        return all_importance_multipliers

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
        self.ntrain = all_representations.shape[0]
        tot_natoms = sum_(atom_nums)
        train_kernel = empty_((tot_natoms, nen_force_vals(atom_nums)))
        self.training_set_representations = all_representations
        self.training_set_nuclear_charges = concatenate_(all_nuclear_charges)
        self.kernel_function(
            self.training_set_representations,
            self.training_set_representations,
            all_representation_grads,
            atom_nums,
            self.training_set_nuclear_charges,
            self.training_set_nuclear_charges,
            all_rel_neighbors,
            all_rel_neighbor_nums,
            self.sigma,
            train_kernel,
        )
        rhs = self.combine_energy_forces_rhs(training_energies, training_forces)
        # Account for energy and force components having different importance factors.
        imp_mult = self.get_importance_multipliers(atom_nums)
        rhs *= imp_mult
        train_kernel *= imp_mult

        self.get_alphas(train_kernel, rhs)

    def train(
        self, training_set_nuclear_charges, training_set_coords, training_energies, training_forces
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
        self.optimize_hyperparameters(all_reps)
        self.fit(
            training_set_nuclear_charges,
            all_reps,
            atom_nums,
            all_rep_grads,
            all_rel_neighbors,
            all_rel_neighbor_nums,
            training_energies,
            training_forces,
        )

    def predict_from_kernel(self, nmols):
        pred_length = nen_force_vals(self.temp_atom_nums[:nmols])
        self.temp_prediction_vector = check_allocation(
            (pred_length,), output=self.temp_prediction_vector
        )
        # KK: made that way to check with torch.autograd
        dot_(
            self.alphas,
            self.temp_kernel[:, :pred_length],
            out=self.temp_prediction_vector[:pred_length],
        )
        # Reshape prediction vector into energy and force vectors and return.
        return prediction_vector_to_forces_energies(
            self.temp_prediction_vector,
            self.temp_atom_nums,
            nmols,
            energy_output=self.temp_energies,
            forces_output=self.temp_forces,
        )

    def predict_from_representations(self, nmols):
        self.temp_kernel = check_allocation(
            (self.ntrain, nen_force_vals(self.temp_atom_nums[:nmols])), output=self.temp_kernel
        )
        self.kernel_function(
            self.training_set_representations,
            self.temp_reps,
            self.temp_rep_grads,
            self.temp_atom_nums,
            self.training_set_nuclear_charges,
            self.temp_nuclear_charges,
            self.temp_relevant_neighbor_arr,
            self.temp_relevant_neighbor_nums,
            self.sigma,
            self.temp_kernel,
        )
        return self.predict_from_kernel(nmols)

    def forward(self, nuclear_charges, coords):
        natoms = nuclear_charges.shape[0]
        (
            self.temp_reps,
            self.temp_rep_grads,
            self.temp_relevant_neighbor_arr,
            self.temp_relevant_neighbor_nums,
        ) = self.get_rep_wgrads(nuclear_charges, coords)
        self.temp_atom_nums[0] = natoms
        self.temp_nuclear_charges = check_allocation((natoms,), output=self.temp_nuclear_charges)
        self.temp_nuclear_charges[:natoms] = nuclear_charges[:]

        en_arr, force_arr = self.predict_from_representations(1)
        return en_arr[0], force_arr[:natoms]


# KK: The inheritance could be optimized a bit more.
class GPRForceModel(OQMLModel):
    def __init__(
        self,
        l2reg_diag_ratio=array_(1.0e-6),
        num_consistency_check=array_(1.0e-6),
        rcond=array_(0.0),
        **other_kwargs
    ):
        super().__init__(**other_kwargs)
        self.init_stability_checks(
            l2_reg_diag_ratio=l2reg_diag_ratio, num_consistency_check=num_consistency_check
        )
        self.rcond = rcond

    def init_kernel_functions(self):
        self.kernel_function_asym = get_derivative_kernel(
            symmetric=False, derivatives="gaussian_process"
        )
        self.kernel_function_sym = get_derivative_kernel(
            symmetric=True, derivatives="gaussian_process"
        )

    def reference_diag_ids(self):
        """
        l2 regularization is estimated based on diagonal elements that correspond to energies.
        """
        energy_ids, _ = energy_forces_ids(self.training_set_natoms)
        return (energy_ids, energy_ids)

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
        self.ntrain = nen_force_vals(atom_nums)
        train_kernel = empty_((self.ntrain, self.ntrain))
        self.training_set_representations = all_representations
        self.training_set_nuclear_charges = concatenate_(all_nuclear_charges)
        self.training_set_rep_grads = all_representation_grads
        self.training_set_rel_neighbors = all_rel_neighbors
        self.training_set_rel_neighbor_nums = all_rel_neighbor_nums
        self.training_set_natoms = atom_nums
        self.kernel_function_sym(
            self.training_set_representations,
            self.training_set_rep_grads,
            atom_nums,
            self.training_set_nuclear_charges,
            self.training_set_rel_neighbors,
            self.training_set_rel_neighbor_nums,
            self.sigma,
            train_kernel,
        )
        rhs = self.combine_energy_forces_rhs(training_energies, training_forces)
        self.get_alphas_w_lambda(train_kernel, rhs, solver=svd_solve, rcond=self.rcond)

    def predict_from_representations(self, nmols):
        self.temp_kernel = check_allocation(
            (self.ntrain, nen_force_vals(self.temp_atom_nums[:nmols])), output=self.temp_kernel
        )
        self.kernel_function_asym(
            self.training_set_representations,
            self.temp_reps,
            self.training_set_rep_grads,
            self.temp_rep_grads,
            self.training_set_natoms,
            self.temp_atom_nums,
            self.training_set_nuclear_charges,
            self.temp_nuclear_charges,
            self.training_set_rel_neighbors,
            self.temp_relevant_neighbor_arr,
            self.training_set_rel_neighbor_nums,
            self.temp_relevant_neighbor_nums,
            self.sigma,
            self.temp_kernel,
        )
        return self.predict_from_kernel(nmols)
