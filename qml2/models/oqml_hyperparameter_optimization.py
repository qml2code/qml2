from ..jit_interfaces import sum_
from ..kernels.gradient_kernels import local_dn_oqml_gaussian_kernel_symmetric
from ..utils import multiply_transposed
from .hyperparameter_optimization import KFolds, KFoldsMultipleObservables
from .sorf_hyperparameter_optimization import callable_ninv_MAE_local_dn_forces_SORF


# For optimizing hyperparameters for Qperator Quantum Machine Learning (OQML).
# Based on SORF due to formula similarities.
class callable_ninv_MAE_local_dn_OQML(callable_ninv_MAE_local_dn_forces_SORF):
    def __init__(
        self,
        training_representations_list,
        training_nuclear_charges_list,
        training_grad_representations_list,
        training_relevant_atom_ids_list,
        training_relevant_atom_nums_list,
        training_energies,
        training_forces_list,
        energy_importance=0.0,
        use_lstsq=True,
        lstsq_rcond=0.0,
        **kfold_kwargs,
    ):
        self.energy_importance = energy_importance
        self.use_lstsq = use_lstsq
        self.lstsq_rcond = lstsq_rcond
        self.init_num_en_force_observables(training_forces_list)
        self.init_training_set(training_representations_list, training_nuclear_charges_list, None)
        self.init_training_energy_forces(training_energies, training_forces_list)
        self.nfeatures = sum_(self.training_natoms)
        self.init_temp_arrays()
        self.init_kfolds(**kfold_kwargs)
        self.init_training_set_grad_reps(
            training_grad_representations_list,
            training_relevant_atom_ids_list,
            training_relevant_atom_nums_list,
        )

    def get_sorf(self, sigma):
        output_matrix = local_dn_oqml_gaussian_kernel_symmetric(
            self.training_representations,
            self.training_grad_representations,
            self.training_natoms,
            self.training_nuclear_charges,
            self.training_relevant_atom_ids,
            self.training_relevant_atom_nums,
            sigma,
        ).T
        multiply_transposed(output_matrix, self.importance_multipliers)
        return output_matrix

    def init_kfolds(self, nkfolds=8, training_set_ratio=0.5):
        mol_kfolds = KFolds(self.training_set_size, nkfolds, training_set_ratio=training_set_ratio)
        super().init_kfolds(mol_kfolds=mol_kfolds)
        self.atom_kfolds = KFoldsMultipleObservables(self.training_natoms, mol_kfolds=mol_kfolds)

    def kfold_train_test_sorf_matrices(self, full_sorf_matrix, kfold_id):
        train_indices, test_indices = self.kfolds.train_test_indices(kfold_id)
        train_atom_indices, _ = self.atom_kfolds.train_test_indices(kfold_id)
        return (
            full_sorf_matrix[train_indices][:, train_atom_indices],
            full_sorf_matrix[test_indices][:, train_atom_indices],
        )
