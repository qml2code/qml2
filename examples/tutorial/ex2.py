import numpy as np
from tutorial_data import compounds

from qml2.kernels import gaussian_kernel

# For every compound generate a coulomb matrix or BoB
for mol in compounds:
    mol.generate_coulomb_matrix(size=23)

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
