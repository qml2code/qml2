# A script demonstrating kfold optimization for SORF which is analogous to
# what is demonstrated in examples/hyperparameter_optimization
import csv
import random
import tarfile
from datetime import datetime

import numpy as np
from boss.bo.bo_main import BOMain

from qml2 import Compound
from qml2.compound_list import CompoundList
from qml2.kernels.sorf import create_sorf_matrices_diff_species, generate_local_sorf
from qml2.math import cho_solve
from qml2.models.hyperparameter_init_guesses import vector_std
from qml2.models.sorf_hyperparameter_optimization import callable_ninv_MAE_local_dn_SORF


def now():
    return datetime.now()


xyzs = []
energies = []

training_set_size = 4001
test_set_size = 3171
num_mols = training_set_size + test_set_size

with open("../../tests/test_data/hof_qm7.txt") as csvfile:
    reader = csv.reader(csvfile, delimiter=" ")
    all_rows = list(reader)
    random.shuffle(all_rows)
    for row in all_rows[:num_mols]:
        xyzs.append(row[0])
        energies.append(float(row[1]))

compounds = CompoundList()
with tarfile.open("../../tests/test_data/qm7.tar.gz") as tar:
    for xyz_name in xyzs:
        xyz = tar.extractfile(xyz_name)
        comp = Compound(xyz=xyz)
        compounds.append(comp)

energies = np.array(energies)

ntransforms = 3
nfeature_stacks = 8  # 32
init_size = 1024

print("Calculating FCHL19:", now())
compounds.generate_fchl19()
print("Finished")
representations_list = compounds.all_representations()
# divide into training and test_set
training_representations_list = representations_list[:training_set_size]
test_representations_list = representations_list[training_set_size:]

training_energies = energies[:training_set_size]
test_energies = energies[training_set_size:]

nuclear_charges_list = compounds.all_nuclear_charges()
training_nuclear_charges_list = nuclear_charges_list[:training_set_size]
test_nuclear_charges_list = nuclear_charges_list[training_set_size:]

# Set up the loss function

# number of k-folds and what portion of training set is dedicated to k-fold's training set.
nkfolds = 8
training_set_ratio = 0.5

ninv_MAE = callable_ninv_MAE_local_dn_SORF(
    training_representations_list,
    training_nuclear_charges_list,
    training_energies,
    nfeature_stacks,
    init_size,
    ntransforms=ntransforms,
)

# Get initial guess for optimal sigma.
training_representations = np.concatenate(training_representations_list)
init_sigma_guess = vector_std(training_representations)
print("initial sigma guess:", init_sigma_guess)
log_init_sigma_guess = np.log(init_sigma_guess)
log_sigma_search_bounds = [log_init_sigma_guess - 2.5, log_init_sigma_guess + 2.5]
log_relative_lambda_search_bounds = [np.log(1.0e-14), np.log(1.0e-5)]

bo_kwargs = {"initpts": 1, "iterpts": 32, "kernel": "rbf"}
bo = BOMain(ninv_MAE, [log_relative_lambda_search_bounds, log_sigma_search_bounds], **bo_kwargs)
res = bo.run()

ln_param_min = res.select("x_glmin", -1)

relative_lambda_val, sigma = np.exp(ln_param_min)
print("optimized relative lambda and sigma:", relative_lambda_val, sigma)
# Train model.
nfeatures = nfeature_stacks * init_size
sorted_elements = np.array([1, 6, 7, 8, 16])
all_biases, all_sorf_diags = create_sorf_matrices_diff_species(
    nfeature_stacks, sorted_elements.shape[0], ntransforms, init_size
)

training_nuclear_charges = np.concatenate(training_nuclear_charges_list)
training_num_atoms = np.array(
    [len(nuclear_charges) for nuclear_charges in training_nuclear_charges_list]
)
training_sorf_matrix = generate_local_sorf(
    training_representations,
    training_nuclear_charges,
    training_num_atoms,
    sorted_elements,
    all_sorf_diags,
    all_biases,
    sigma,
    nfeature_stacks,
    init_size,
)

training_kernel = np.dot(training_sorf_matrix.T, training_sorf_matrix)

lambda_val = relative_lambda_val * np.mean(np.diagonal(training_kernel))

alphas_rhs = np.dot(training_energies, training_sorf_matrix)

alphas = cho_solve(training_kernel, alphas_rhs, l2reg=lambda_val)

# Calculate MAE.
test_representations = np.concatenate(test_representations_list)
test_nuclear_charges = np.concatenate(test_nuclear_charges_list)
test_num_atoms = np.array([len(nuclear_charges) for nuclear_charges in test_nuclear_charges_list])

test_sorf_matrix = generate_local_sorf(
    test_representations,
    test_nuclear_charges,
    test_num_atoms,
    sorted_elements,
    all_sorf_diags,
    all_biases,
    sigma,
    nfeature_stacks,
    init_size,
)

predictions = np.dot(test_sorf_matrix, alphas)


def MAE(predictions, values):
    return np.mean(np.abs(predictions - values))


print("final MAE:")
print(MAE(predictions, test_energies))
