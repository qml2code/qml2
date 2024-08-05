# WARNING: the example requires CuPy installation.
from datetime import datetime

import numpy as np
from tutorial_data import compounds

from qml2.cupy import gaussian_kernel as gaussian_kernel_gpu1
from qml2.cupy.raw_kernels import gaussian_kernel as gaussian_kernel_gpu2

# The code has several implementations of kernel functions that
# are numerically equivalent but differ in computational time
# and can be more suitable depending on the problem and hardware at hand.
from qml2.kernels import gaussian_kernel
from qml2.representations import get_slatm_mbtypes


def now():
    return datetime.now()


if __name__ == "__main__":
    # As in ex4.py, we calculate a representation for every compound, but instead of Coulomb Matrix
    # we calculate SLATM. We also do this in an embarassingly parallel way using qml2.CompoundList
    # functionality.
    slatm_mbtypes = get_slatm_mbtypes(compounds.all_nuclear_charges())
    compounds.generate_slatm(slatm_mbtypes)
    # Create 2D array of representations as in ex4.
    X = np.array(compounds.all_representations())

    # Let's calculate kernel elements for these two sets of molecules.
    X_training = X[:4000]
    X_test = X[4000:]
    sigma = 700.0

    print("Calculation times:")
    # with CPUs.
    cpu_start = now()
    K_cpu = gaussian_kernel(X_training, X_test, sigma)
    cpu_end = now()
    print("CPU:", cpu_end - cpu_start)
    # with GPUs (using CuPy)
    gpu1_start = now()
    K_gpu1 = gaussian_kernel_gpu1(X_training, X_test, sigma)
    gpu1_end = now()
    print("GPU (CuPy):", gpu1_end - gpu1_start)
    # with GPUs (using CUDA code precompiled with CuPy)
    # NOTE: blocks_per_grid and threads_per_block are optional arguments,
    # but we recommend defining them and playing with them for optimal efficiency.
    gpu2_start = now()
    K_gpu2 = gaussian_kernel_gpu2(
        X_training, X_test, sigma, blocks_per_grid=(64,), threads_per_block=(64,)
    )
    gpu2_end = now()
    print("CPU (CUDA source+CuPy):", gpu2_end - gpu2_start)
    # Check that the matrices agree.
    print("Kernel matrices CPU vs GPU evaluation difference")
    print(np.mean(np.abs(K_gpu1 - K_cpu)), np.mean(np.abs(K_gpu2 - K_cpu)))
