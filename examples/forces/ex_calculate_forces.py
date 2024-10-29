# The script will generate a trial dataset of Lennard-Jones particles (2 possible elements,
# between 2 and 4 atoms), then apply SORF and OQML approaches to try ML their forces and energies,
# and then compare the force and energy MAEs.
import numpy as np
from boss.bo.bo_main import BOMain

from qml2 import Compound
from qml2.compound_list import CompoundList
from qml2.models.hyperparameter_init_guesses import vector_std
from qml2.models.krr_forces import OQMLModel
from qml2.models.oqml_hyperparameter_optimization import callable_ninv_MAE_local_dn_OQML
from qml2.models.sorf_forces import SORFLocalForcesModel
from qml2.models.sorf_hyperparameter_optimization import callable_ninv_MAE_local_dn_forces_SORF
from qml2.test_utils.lennard_jones_potential import lj_energy_force, random_lj_molecule

mol_natoms_min = 3
mol_natoms_max = 3
# possible_nuclear_charges=np.array([1, 3])
possible_nuclear_charges = np.array([1])


def generate_dataset(num_mols):
    nuclear_charges_list = []
    compounds = CompoundList()
    energies = np.empty((num_mols,))
    forces_list = []
    for mol_id in range(num_mols):
        nuclear_charges, coordinates = random_lj_molecule(
            mol_natoms_min, mol_natoms_max, possible_nuclear_charges=possible_nuclear_charges
        )
        nuclear_charges_list.append(nuclear_charges)
        compounds.append(Compound(nuclear_charges=nuclear_charges, coordinates=coordinates))
        energies[mol_id], force = lj_energy_force(nuclear_charges, coordinates)
        forces_list.append(force)
    compounds.generate_fchl19(gradients=True, elements=possible_nuclear_charges)

    representation_list = []
    grad_representation_list = []
    relevant_atom_ids_list = []
    relevant_atom_nums_list = []

    for comp in compounds:
        representation_list.append(comp.representation)
        grad_representation_list.append(comp.grad_representation)
        relevant_atom_ids_list.append(comp.grad_relevant_atom_ids)
        relevant_atom_nums_list.append(comp.grad_relevant_atom_nums)

    return (
        nuclear_charges_list,
        representation_list,
        grad_representation_list,
        relevant_atom_ids_list,
        relevant_atom_nums_list,
        energies,
        forces_list,
    )


training_set_size = 200

energy_importance = 0.01

# Create training dataset.
(
    train_ncharges_list,
    train_rep_list,
    train_grad_rep_list,
    train_rel_atoms_list,
    train_rel_atom_nums_list,
    train_ens,
    train_forces_list,
) = generate_dataset(training_set_size)

# BO of hyperparameters

# get initial guess of the sigma parameter
train_reps = np.concatenate(train_rep_list)
init_sigma_guess = vector_std(train_reps)

print("Initial sigma guess:")
print(init_sigma_guess)

log_init_sigma_guess = np.log(init_sigma_guess)
log_sigma_search_bounds = [log_init_sigma_guess - 2.5, log_init_sigma_guess + 2.5]
log_relative_lambda_search_bounds = [np.log(1.0e-14), np.log(1.0e-5)]
nkfolds = 4
training_set_ratio = 0.5

# SORF hyperparameters.
ntransforms = 3
init_size = 256
nfeature_stacks = 16


ninv_MAE = callable_ninv_MAE_local_dn_forces_SORF(
    train_rep_list,
    train_ncharges_list,
    train_grad_rep_list,
    train_rel_atoms_list,
    train_rel_atom_nums_list,
    train_ens,
    train_forces_list,
    nfeature_stacks,
    init_size,
    energy_importance=energy_importance,
    ntransforms=ntransforms,
    nkfolds=nkfolds,
    training_set_ratio=training_set_ratio,
)

bo_kwargs = {"initpts": 1, "iterpts": 32, "kernel": "rbf", "minfreq": 0}
bo = BOMain(ninv_MAE, [log_relative_lambda_search_bounds, log_sigma_search_bounds], **bo_kwargs)
res = bo.run()

sorf_ln_param_min = res.select("x_glmin", -1)

sorf_relative_lambda_val, sorf_sigma = np.exp(sorf_ln_param_min)
print("Optimized SORF hyperparameters:")
print(sorf_relative_lambda_val, sorf_sigma)

# OQML hyperparameters
ninv_MAE = callable_ninv_MAE_local_dn_OQML(
    train_rep_list,
    train_ncharges_list,
    train_grad_rep_list,
    train_rel_atoms_list,
    train_rel_atom_nums_list,
    train_ens,
    train_forces_list,
    energy_importance=energy_importance,
    nkfolds=nkfolds,
    training_set_ratio=training_set_ratio,
)
bo = BOMain(ninv_MAE, [log_sigma_search_bounds], **bo_kwargs)
res = bo.run()
oqml_ln_param_min = res.select("x_glmin", -1)
oqml_sigma = np.exp(oqml_ln_param_min)[0]
print("Optimized OQML sigma:", oqml_sigma)

# Create test dataset.
test_set_size = 20

(
    test_ncharges_list,
    test_rep_list,
    test_grad_rep_list,
    test_rel_atoms_list,
    test_rel_atom_nums_list,
    test_ens,
    test_forces_list,
) = generate_dataset(training_set_size)


def en_forces_MAE(energy_predictions, force_predictions):
    en_MAE = 0.0
    force_MAE = 0.0
    nmols = len(energy_predictions)
    for en_pred, force_pred, ref_en, ref_force in zip(
        energy_predictions, force_predictions, test_ens, test_forces_list
    ):
        en_MAE += abs(en_pred - ref_en)
        force_MAE += np.sum(np.abs(force_pred - ref_force)) / len(force_pred)
    return en_MAE / nmols, force_MAE / nmols


# Train and test SORF model.
sorf_model = SORFLocalForcesModel(
    npcas=init_size,
    nfeatures=nfeature_stacks * init_size,
    ntransforms=ntransforms,
    l2reg_diag_ratio=sorf_relative_lambda_val,
    sigma=sorf_sigma,
    sorted_elements=possible_nuclear_charges,
    energy_importance=energy_importance,
)
sorf_model.train_from_rep_lists(
    train_rep_list,
    train_grad_rep_list,
    train_rel_atoms_list,
    train_rel_atom_nums_list,
    train_ncharges_list,
    train_ens,
    train_forces_list,
)

sorf_energy_predictions, sorf_force_predictions = sorf_model.predict_from_rep_lists(
    test_ncharges_list,
    test_rep_list,
    test_grad_rep_list,
    test_rel_atoms_list,
    test_rel_atom_nums_list,
)

print("SORF energy and force MAEs:")
print(*en_forces_MAE(sorf_energy_predictions, sorf_force_predictions))

# Train and test OQML model.
oqml_model = OQMLModel(sigma=oqml_sigma, energy_importance=energy_importance)
oqml_model.train_from_rep_lists(
    train_rep_list,
    train_grad_rep_list,
    train_rel_atoms_list,
    train_rel_atom_nums_list,
    train_ncharges_list,
    train_ens,
    train_forces_list,
)

oqml_energy_predictions, oqml_force_predictions = oqml_model.predict_from_rep_lists(
    test_ncharges_list,
    test_rep_list,
    test_grad_rep_list,
    test_rel_atoms_list,
    test_rel_atom_nums_list,
)
print("OQML energy and force MAEs:")
print(*en_forces_MAE(oqml_energy_predictions, oqml_force_predictions))
