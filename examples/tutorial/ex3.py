import numpy as np
from tutorial_data import compounds, energy_pbe0

from qml2.kernels import gaussian_kernel
from qml2.math import cho_solve

# For every compound generate a coulomb matrix or BoB
for mol in compounds:
    mol.generate_coulomb_matrix(size=23)
    # mol.generate_bob(size=23, asize={"O":3, "C":7, "N":3, "H":16, "S":1})

# Make a big 2D array with all the representations
X = np.array([mol.representation for mol in compounds])

# Print all representations
print(X)

# Run on only a subset of the first 1000 (for speed)
X = X[:1000]

# Define the kernel width
sigma = 1000.0

# K is also a Numpy array
K = gaussian_kernel(X, X, sigma)

# Print the kernel
print(K)

# Assign 1000 first molecules to the training set
X_training = X[:1000]
Y_training = energy_pbe0[:1000]

sigma = 4000.0
K = gaussian_kernel(X_training, X_training, sigma)
print(K)

# Add a small lambda to the diagonal of the kernel matrix
K[np.diag_indices_from(K)] += 1e-8

print(len(K))
print(len(Y_training))


# Use the built-in Cholesky-decomposition to solve
alpha = cho_solve(K, Y_training)

print(alpha)
