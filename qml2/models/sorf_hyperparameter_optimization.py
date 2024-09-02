from ..basic_utils import now
from ..data import nCartDim
from ..dimensionality_reduction import get_reductors_diff_species
from ..jit_interfaces import (
    LinAlgError_,
    abs_,
    array_,
    concatenate_,
    diag_indices_from_,
    dint_,
    dot_,
    empty_,
    exp_,
    lstsq_,
    max_,
    mean_,
)
from ..kernels.gradient_kernels import prediction_vector_length
from ..kernels.gradient_sorf import generate_local_force_sorf
from ..kernels.sorf import create_sorf_matrices_diff_species, generate_local_sorf
from ..math import cho_solve
from ..utils import get_sorted_elements, multiply_transposed
from .forces_utils import combine_energy_forces_rhs, get_importance_multipliers
from .hyperparameter_optimization import KFoldsMultipleObservables, callable_ninv_MAE_local_dn


# For BO in space of lambda and sigma for local_dn SORF and SORF with forces.
class callable_ninv_MAE_local_dn_SORF(callable_ninv_MAE_local_dn):
    def __init__(
        self,
        training_representations_list,
        training_nuclear_charges_list,
        training_quantities,
        nfeature_stacks,
        init_size,
        ntransforms=3,
        use_reductors=False,
        pca_num_samples=1024,
        use_lstsq=False,
        lstsq_rcond=0.0,
        **kfold_kwargs,
    ):
        self.use_lstsq = use_lstsq
        self.lstsq_rcond = lstsq_rcond
        self.init_training_set(
            training_representations_list, training_nuclear_charges_list, training_quantities
        )
        self.init_sorf(
            nfeature_stacks,
            init_size,
            ntransforms=ntransforms,
            use_reductors=use_reductors,
            pca_num_samples=pca_num_samples,
        )
        self.init_temp_arrays()
        self.init_kfolds(**kfold_kwargs)

    def init_training_set(
        self, training_representations_list, training_nuclear_charges_list, training_quantities
    ):
        super().init_training_set(
            training_representations_list, training_nuclear_charges_list, training_quantities
        )
        self.sorted_elements = get_sorted_elements(self.training_nuclear_charges)
        self.nspecies = self.sorted_elements.shape[0]

    def init_sorf(
        self, nfeature_stacks, init_size, ntransforms=3, use_reductors=False, pca_num_samples=1024
    ):
        self.ntransforms = ntransforms
        self.nfeature_stacks = nfeature_stacks
        self.init_size = init_size
        self.nfeatures = self.nfeature_stacks * self.init_size
        self.all_biases, self.all_sorf_diags = create_sorf_matrices_diff_species(
            self.nfeature_stacks, self.nspecies, self.ntransforms, self.init_size
        )
        self.use_reductors = use_reductors
        if self.use_reductors:
            self.pca_num_samples = pca_num_samples
            self.all_reductors = get_reductors_diff_species(
                self.training_representations,
                self.training_nuclear_charges,
                self.init_size,
                num_samples=self.pca_num_samples,
                sorted_elements=self.sorted_elements,
            )
        else:
            self.all_reductors = None
            self.pca_num_samples = None

    def init_temp_arrays(self):
        self.temp_Z_matrix = empty_((self.num_observables, self.nfeatures))
        if not self.use_lstsq:
            self.temp_K_matrix = empty_((self.nfeatures, self.nfeatures))

    def get_sorf(self, sigma):
        return generate_local_sorf(
            self.training_representations,
            self.training_nuclear_charges,
            self.training_natoms,
            self.sorted_elements,
            self.all_sorf_diags,
            self.all_biases,
            sigma,
            self.nfeature_stacks,
            self.init_size,
            out=self.temp_Z_matrix,
            reductors=self.all_reductors,
        )

    def kfold_train_test_sorf_matrices(self, full_sorf_matrix, kfold_id):
        train_indices, test_indices = self.kfolds.train_test_indices(kfold_id)
        return full_sorf_matrix[train_indices], full_sorf_matrix[test_indices]

    def kfold_train_test_quantities(self, kfold_id):
        train_indices, test_indices = self.kfolds.train_test_indices(kfold_id)
        return self.training_quantities[train_indices], self.training_quantities[test_indices]

    def __call__(self, ln_parameters):
        print("started MAE calculation for:", ln_parameters, now())
        parameters = exp_(ln_parameters[0])
        sigma = parameters[-1]
        if not self.use_lstsq:
            normalized_lambda = parameters[0]
        full_sorf_matrix = self.get_sorf(sigma)
        tot_MAE = 0.0
        for kfold_id in range(self.nkfolds):
            train_sorf_matrix, test_sorf_matrix = self.kfold_train_test_sorf_matrices(
                full_sorf_matrix, kfold_id
            )
            train_quantities, test_quantities = self.kfold_train_test_quantities(kfold_id)

            if self.use_lstsq:
                alphas = lstsq_(train_sorf_matrix, train_quantities, rcond=self.lstsq_rcond)[0]
            else:
                K = dot_(train_sorf_matrix.T, train_sorf_matrix, out=self.temp_K_matrix)
                diag_ids = diag_indices_from_(K)
                lambda_val = mean_(K[diag_ids]) * normalized_lambda

                alpha_rhs = dot_(train_quantities, train_sorf_matrix)
                try:
                    alphas = cho_solve(K, alpha_rhs, l2reg=lambda_val)
                except LinAlgError_:
                    return 0.0
            predictions = dot_(test_sorf_matrix, alphas)
            tot_MAE += mean_(abs_(predictions - test_quantities))
        av_MAE = tot_MAE / self.nkfolds
        print("finished calculations:", av_MAE, now())
        return -1 / av_MAE


class callable_ninv_MAE_local_dn_forces_SORF(callable_ninv_MAE_local_dn_SORF):
    def __init__(
        self,
        training_representations_list,
        training_nuclear_charges_list,
        training_grad_representations_list,
        training_relevant_atom_ids_list,
        training_relevant_atom_nums_list,
        training_energies,
        training_forces_list,
        nfeature_stacks,
        init_size,
        energy_importance=0.0,
        **callable_ninv_MAE_local_dn_SORF_kwargs,
    ):
        self.energy_importance = energy_importance
        self.init_num_en_force_observables(training_forces_list)
        super().__init__(
            training_representations_list,
            training_nuclear_charges_list,
            None,
            nfeature_stacks,
            init_size,
            **callable_ninv_MAE_local_dn_SORF_kwargs,
        )
        self.init_training_energy_forces(training_energies, training_forces_list)
        self.init_training_set_grad_reps(
            training_grad_representations_list,
            training_relevant_atom_ids_list,
            training_relevant_atom_nums_list,
        )

    def init_num_en_force_observables(self, training_forces_list):
        self.num_observables = len(training_forces_list) + nCartDim * sum(
            [force.shape[0] for force in training_forces_list]
        )

    def init_prediction_vector_lengths(self):
        self.prediction_vector_lengths = array_(
            [prediction_vector_length(na) for na in self.training_natoms]
        )

    def init_kfolds(self, nkfolds=8, training_set_ratio=0.5, mol_kfolds=None):
        self.init_prediction_vector_lengths()
        self.kfolds = KFoldsMultipleObservables(
            self.prediction_vector_lengths,
            nkfolds=nkfolds,
            training_set_ratio=training_set_ratio,
            mol_kfolds=mol_kfolds,
        )
        self.nkfolds = self.kfolds.nkfolds

    def init_training_energy_forces(self, training_energies, training_forces_list):
        self.training_quantities = combine_energy_forces_rhs(
            training_energies, training_forces_list
        )
        self.importance_multipliers = get_importance_multipliers(
            self.training_natoms, self.energy_importance
        )
        self.training_quantities *= self.importance_multipliers

    def get_sorf(self, sigma):
        sorf_matrix = generate_local_force_sorf(
            self.training_representations,
            self.training_grad_representations,
            self.training_nuclear_charges,
            self.training_natoms,
            self.training_relevant_atom_ids,
            self.training_relevant_atom_nums,
            self.sorted_elements,
            self.all_sorf_diags,
            self.all_biases,
            sigma,
            self.nfeature_stacks,
            self.init_size,
            reductors=self.all_reductors,
            out=self.temp_Z_matrix,
        )
        multiply_transposed(sorf_matrix, self.importance_multipliers)
        return sorf_matrix

    def init_training_set_grad_reps(
        self,
        training_grad_representations_list,
        training_relevant_atom_ids_list,
        training_relevant_atom_nums_list,
    ):
        self.training_relevant_atom_nums = concatenate_(training_relevant_atom_nums_list)
        self.max_relevant_atom_num = max_(self.training_relevant_atom_nums)
        self.training_grad_representations = empty_(
            (*self.training_representations.shape, self.max_relevant_atom_num, nCartDim)
        )
        self.training_relevant_atom_ids = empty_(
            (self.training_representations.shape[0], self.max_relevant_atom_num), dtype=dint_
        )

        lbound = 0
        for grad_rep, rel_atom_ids, rel_atom_num in zip(
            training_grad_representations_list,
            training_relevant_atom_ids_list,
            self.training_relevant_atom_nums,
        ):
            ubound = lbound + grad_rep.shape[0]
            self.training_grad_representations[lbound:ubound, :, :rel_atom_num, :] = grad_rep[
                :, :, :rel_atom_num, :
            ]
            self.training_relevant_atom_ids[lbound:ubound, :rel_atom_num] = rel_atom_ids[
                :, :rel_atom_num
            ]
            lbound = ubound
