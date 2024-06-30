# How to use Structured Orthogonal Random Features (DOI:10.1063/5.0108967).
import csv
import random
import tarfile

import numpy as np

from qml2 import Compound
from qml2.dimensionality_reduction import get_reductors_diff_species
from qml2.kernels.hadamard import create_SORF_matrices_diff_species, local_hadamard_kernel
from qml2.math import cho_solve
from qml2.models.hyperparameter_init_guesses import vector_std

# Import energies.
xyzs = []
energies = []
with open("../../tests/test_data/hof_qm7.txt") as csvfile:
    reader = csv.reader(csvfile, delimiter=" ")
    all_rows = list(reader)
    random.shuffle(all_rows)
    for row in all_rows:
        xyzs.append(row[0])
        energies.append(float(row[1]))
nmols = len(xyzs)
# Get compounds from the compressed xyz files and calculate a local representation for each.
representations = []
num_atoms = []
ncharges = []
print("Calculating representations.")
with tarfile.open("../../tests/test_data/qm7.tar.gz") as tar:
    for xyz_name in xyzs:
        xyz = tar.extractfile(xyz_name)
        comp = Compound(xyz=xyz)
        # calculate FCHL representation
        comp.generate_fchl19()

        representations += list(comp.representation)
        num_atoms.append(comp.nuclear_charges.shape[0])
        ncharges += list(comp.nuclear_charges)

representations = np.array(representations)
num_atoms = np.array(num_atoms)
ncharges = np.array(ncharges)

# Use PCA to get reductors. If npcas is larger than representation size a random projector will be generated.
npcas = 512
reductors, sorted_elements = get_reductors_diff_species(representations, ncharges, npcas)
# Generate SORF diagonals and biases.
ntransforms = 2
nfeature_stacks = 16
nfeatures = nfeature_stacks * npcas
all_biases, all_sorf_diags = create_SORF_matrices_diff_species(
    nfeature_stacks, sorted_elements.shape[0], ntransforms, npcas
)
# NOTE: for a global representation we would use qml2.kernels.hadamard_kernels.create_SORF_matrices

# Get some reasonable sigma parameter for demonstration purposes.
sigma = vector_std(representations)

# Calculate the Z-matrix.
Z_matrix = np.empty((nmols, nfeatures))
print("Calculating Z matrix.")
local_hadamard_kernel(
    representations,
    ncharges,
    num_atoms,
    sorted_elements,
    all_sorf_diags,
    all_biases,
    Z_matrix,
    sigma,
    nfeature_stacks,
    npcas,
    reductors=reductors,
)
# NOTE: for a global representation we would use qml2.kernels.hadamard_kernels.hadamard_kernel

# Separate Z_matrix and values into training and test sets.
ntrain = 4000
Z_train = Z_matrix[:ntrain]
en_train = energies[:ntrain]

Z_test = Z_matrix[ntrain:]
en_test = energies[ntrain:]

# get regression coefficients from the training set.
print("Getting regression coefficients.")
K = np.dot(Z_train.T, Z_train)
# l2 regularization
K[np.diag_indices_from(K)] += 1.0e-5 * np.mean(K[np.diag_indices_from(K)])

rhs = np.dot(Z_train.T, en_train)
alphas = cho_solve(K, rhs)

# Evaluate model for the test set and check MAE.
predictions = np.dot(Z_test, alphas)
print("MAE:", np.mean(np.abs(predictions - en_test)))
