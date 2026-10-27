from ..jit_interfaces import sum_
from ..kernels.gradient_kernels import local_dn_oqml_gaussian_kernel_symmetric
from ..utils import multiply_transposed
from .kfolds import GroupKFoldsMultipleObservables, conformer_folds
from .sorf_hyperparameter_optimization import callable_ninv_MAE_local_dn_forces_SORF


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
        """
        For optimizing hyperparameters for Qperator Quantum Machine Learning (OQML). Based on SORF due to formula similarities.
        """
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

    def init_kfolds(self, nkfolds=8, training_set_ratio=0.5, conformer_group_ids=None):
        """One conformer-level split, expanded twice.

        The observables index the ROWS of the OQML matrix and the atoms index its COLUMNS,
        so both expansions have to come from the same conformer split. Building them
        independently would let them disagree about which conformer went where -- the
        greedy fold fill balances on block size, and a conformer's observable block
        (3*natoms + 1) and atom block (natoms) are different sizes -- and the model would
        then be validated on rows it had partly trained on.

        ``training_set_ratio`` is ignored; see the parent's init_kfolds.
        """
        mol_kfolds = conformer_folds(
            self.training_set_size, nkfolds, conformer_group_ids=conformer_group_ids
        )
        super().init_kfolds(mol_kfolds=mol_kfolds)
        self.atom_kfolds = GroupKFoldsMultipleObservables(self.training_natoms, mol_kfolds)

    def kfold_train_test_sorf_matrices(self, full_sorf_matrix, kfold_id):
        train_indices, test_indices = self.kfolds.train_test_indices(kfold_id)
        train_atom_indices, _ = self.atom_kfolds.train_test_indices(kfold_id)
        return (
            full_sorf_matrix[train_indices][:, train_atom_indices],
            full_sorf_matrix[test_indices][:, train_atom_indices],
        )
