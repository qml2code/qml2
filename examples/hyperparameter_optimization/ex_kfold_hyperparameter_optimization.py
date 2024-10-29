# The script demonstrates how kfold-based loss function implemented in QML2
# can be used to optimize hyperparameters for machine learning.

import csv
import random
import tarfile

# Miscellanious imports
import numpy as np

# We will use BOSS optimizer in this example; to install it run
# `pip install aalto-boss`
# WARNING 2024.08.10: if you encounter problems running BOSS optimizer try
# pip install "numpy<2.0"
# or use another global optimizer
from boss.bo.bo_main import BOMain

# For conveniently transforming xyz's into representations.
from qml2 import Compound
from qml2.compound_list import CompoundList

# We will use Matern kernels in this example
from qml2.kernels import (
    local_dn_matern_kernel,
    local_dn_matern_kernel_symmetric,
    matern_kernel,
    matern_kernel_symmetric,
)

# For solving the regression equation
from qml2.math import cho_solve

# For providing initial guess of sigma.
from qml2.models.hyperparameter_init_guesses import vector_std

# We will find the optimal sigma and lambda by minimizing negative inverse of
# MAE - it is more convenient to use than MAE since the latter can go to infinity in
# some situations.
from qml2.models.hyperparameter_optimization import callable_ninv_MAE, callable_ninv_MAE_local_dn

# Both callable classes take as input logarithms of lambda value divided by the average
# training kernel diagonal element (which is always 1 for global kernels, but can vary
# dramatically for local kernels) and the logarithm of sigma.


xyzs = []
energies = []

training_set_size = 2001
test_set_size = 1000
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


# Throughout this work we'll use Matern kernel of order 0 with l2 metric.
matern_kwargs = {"order": 0, "metric": "l2"}

# First try global kernels and global representation (Coulomb Matrix).
print("Coulomb Matrix:")
compounds.generate_coulomb_matrix()
representations = np.array([comp.representation for comp in compounds])
# divide into training and test_set
training_representations = representations[:training_set_size]
test_representations = representations[training_set_size:]

training_energies = energies[:training_set_size]
test_energies = energies[training_set_size:]

# Set up the loss function

# number of k-folds and what portion of training set is dedicated to k-fold's training set.
nkfolds = 8
training_set_ratio = 0.5

ninv_MAE_kwargs = {
    "nkfolds": nkfolds,
    "training_set_ratio": training_set_ratio,
    "kernel_function_kwargs": matern_kwargs,
}

ninv_MAE = callable_ninv_MAE(
    matern_kernel_symmetric, training_representations, training_energies, **ninv_MAE_kwargs
)

# Get initial guess for optimal sigma.
init_sigma_guess = vector_std(training_representations)
print("initial sigma guess:", init_sigma_guess)
log_init_sigma_guess = np.log(init_sigma_guess)
log_sigma_search_bounds = [log_init_sigma_guess - 2.5, log_init_sigma_guess + 2.5]
log_relative_lambda_search_bounds = [np.log(1.0e-14), np.log(1.0e-5)]

bo_kwargs = {"initpts": 1, "iterpts": 32, "kernel": "rbf", "minfreq": 0}
bo = BOMain(ninv_MAE, [log_relative_lambda_search_bounds, log_sigma_search_bounds], **bo_kwargs)
res = bo.run()

ln_param_min = res.select("x_glmin", -1)
# Note that relative lambda equals lambda for global kernels.
lambda_val, sigma = np.exp(ln_param_min)
print("optimized lambda and sigma:", lambda_val, sigma)

# Train model and calculate the MAE.
train_kernel = matern_kernel_symmetric(training_representations, sigma, **matern_kwargs)
train_kernel[np.diag_indices_from(train_kernel)] += lambda_val
alphas = cho_solve(train_kernel, training_energies)
test_kernel = matern_kernel(test_representations, training_representations, sigma, **matern_kwargs)
predictions = np.dot(test_kernel, alphas)


def MAE(predictions, values):
    return np.mean(np.abs(predictions - values))


print("final MAE:")
print(MAE(predictions, test_energies))

# Let's do the same for a local representation.
print("FCHL19:")
compounds.generate_fchl19()
all_nuclear_charges = compounds.all_nuclear_charges()
representation_list = [comp.representation for comp in compounds]

training_representations_list = representation_list[:training_set_size]
test_representations_list = representation_list[training_set_size:]
training_nuclear_charges_list = all_nuclear_charges[:training_set_size]

ninv_MAE_local_dn = callable_ninv_MAE_local_dn(
    local_dn_matern_kernel_symmetric,
    training_representations_list,
    training_nuclear_charges_list,
    training_energies,
    **ninv_MAE_kwargs
)

training_representations = np.concatenate(training_representations_list)
init_sigma_guess = vector_std(training_representations)
print("initial sigma guess:", init_sigma_guess)
# perform Bayesian optimization
log_init_sigma_guess = np.log(init_sigma_guess)
log_sigma_search_bounds = [log_init_sigma_guess - 2.5, log_init_sigma_guess + 2.5]

bo = BOMain(
    ninv_MAE_local_dn, [log_relative_lambda_search_bounds, log_sigma_search_bounds], **bo_kwargs
)
res = bo.run()

ln_param_min = res.select("x_glmin", -1)
relative_lambda_val, sigma = np.exp(ln_param_min)
print("optimized relative lambda and sigma:")
print(relative_lambda_val, sigma)

# Create the model.
training_natoms = np.array([len(rep) for rep in training_representations_list])
test_nuclear_charges_list = all_nuclear_charges[training_set_size:]
training_nuclear_charges = np.concatenate(training_nuclear_charges_list)
training_kernel = local_dn_matern_kernel_symmetric(
    training_representations, training_natoms, training_nuclear_charges, sigma, **matern_kwargs
)
lambda_val = np.mean(np.diagonal(training_kernel)) * relative_lambda_val
training_kernel[np.diag_indices_from(training_kernel)] += lambda_val
alphas = cho_solve(training_kernel, training_energies)
# Calculate MAE.
test_representations = np.concatenate(test_representations_list)
test_natoms = np.array([len(rep) for rep in test_representations_list])
test_nuclear_charges = np.concatenate(all_nuclear_charges[training_set_size:])

test_kernel = local_dn_matern_kernel(
    test_representations,
    training_representations,
    test_natoms,
    training_natoms,
    test_nuclear_charges,
    training_nuclear_charges,
    sigma,
    **matern_kwargs
)
predictions = np.dot(test_kernel, alphas)
print("final")
print(MAE(predictions, test_energies))
