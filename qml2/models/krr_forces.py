# TODO: Perhaps use energy_forces_ids from ..kernels.gradient_kernels more?
from ..jit_interfaces import Module_, array_, concatenate_, dint_, dot_, empty_, lstsq_, sum_
from ..kernels.gradient_kernels import get_derivative_kernel
from ..math import svd_solve
from ..parallelization import embarrassingly_parallel
from ..representations import generate_fchl19
from ..utils import check_allocation
from .forces_utils import (
    combine_energy_forces_rhs,
    energy_forces_ids,
    get_importance_multipliers,
    merge_grad_lists,
    nen_force_vals,
    prediction_vector_to_forces_energies,
)
from .krr import KRRLocalModel


class OQMLModel(KRRLocalModel):
    def __init__(
        self,
        representation_function=generate_fchl19,
        sigma=None,
        rep_kwargs={"gradients": True},
        lstsq_rcond=None,
        training_reps_suppress_openmp=False,
        energy_importance=0.01,
    ):
        Module_.__init__(self)
        # hyperparameters
        self.sigma = sigma
        self.lstsq_rcond = lstsq_rcond
        # representation parameters
        self.representation_function = representation_function
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
        return self.representation_function(nuclear_charges, coords, **self.rep_kwargs)

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
        all_reps_list = []
        all_rep_grads_list = []
        all_rel_atoms_list = []
        all_rel_atom_nums_list = []
        for reps, rep_grads, rel_atoms, rel_atom_nums in all_reps_wgrads_list:
            all_reps_list.append(reps)
            all_rep_grads_list.append(rep_grads)
            all_rel_atoms_list.append(rel_atoms)
            all_rel_atom_nums_list.append(rel_atom_nums)

        return all_reps_list, all_rep_grads_list, all_rel_atoms_list, all_rel_atom_nums_list

    def get_alphas(self, train_kernel, rhs):
        self.alphas = lstsq_(train_kernel.T, rhs, rcond=self.lstsq_rcond)[0]

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
        rhs = combine_energy_forces_rhs(training_energies, training_forces)
        # Account for energy and force components having different importance factors.
        imp_mult = get_importance_multipliers(atom_nums, self.energy_importance)
        rhs *= imp_mult
        train_kernel *= imp_mult

        self.get_alphas(train_kernel, rhs)

    def train_from_rep_lists(
        self,
        all_reps_list,
        all_rep_grads_list,
        all_rel_atoms_list,
        all_rel_atom_nums_list,
        training_set_nuclear_charges,
        training_energies,
        training_forces,
        **kwargs
    ):
        all_reps = concatenate_(all_reps_list)
        atom_nums = array_([len(reps) for reps in all_reps_list])
        self.optimize_hyperparameters(all_reps)
        all_rep_grads, all_rel_atoms, all_rel_atom_nums = merge_grad_lists(
            all_rep_grads_list, all_rel_atoms_list, all_rel_atom_nums_list
        )
        self.fit(
            training_set_nuclear_charges,
            all_reps,
            atom_nums,
            all_rep_grads,
            all_rel_atoms,
            all_rel_atom_nums,
            training_energies,
            training_forces,
        )

    def train(
        self,
        training_set_nuclear_charges,
        training_set_coords,
        training_energies,
        training_forces,
        **kwargs
    ):
        print("Calculating representations.")
        (
            all_reps_list,
            all_rep_grads_list,
            all_rel_atoms_list,
            all_rel_atom_nums_list,
        ) = self.get_all_representations_wgrads(
            training_set_nuclear_charges,
            training_set_coords,
            suppress_openmp=self.training_reps_suppress_openmp,
        )
        print("Done")
        self.train_from_rep_lists(
            all_reps_list,
            all_rep_grads_list,
            all_rel_atoms_list,
            all_rel_atom_nums_list,
            training_set_nuclear_charges,
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

    def predict_from_rep_lists(
        self,
        all_nuclear_charges,
        all_reps_list,
        all_rep_grads_list,
        all_rel_atoms_list,
        all_rel_atom_nums_list,
    ):
        self.temp_nuclear_charges = concatenate_(all_nuclear_charges)
        self.temp_reps = concatenate_(all_reps_list)
        (
            self.temp_rep_grads,
            self.temp_relevant_neighbor_arr,
            self.temp_relevant_neighbor_nums,
        ) = merge_grad_lists(all_rep_grads_list, all_rel_atoms_list, all_rel_atom_nums_list)
        self.temp_atom_nums = array_([len(ncharges) for ncharges in all_nuclear_charges])
        return self.predict_from_representations(len(all_nuclear_charges))

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
            l2reg_diag_ratio=l2reg_diag_ratio, num_consistency_check=num_consistency_check
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
        rhs = combine_energy_forces_rhs(training_energies, training_forces)
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
