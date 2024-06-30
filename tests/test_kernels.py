import importlib
import random

from conftest import add_checksum_to_dict, compare_or_create

from qml2.jit_interfaces import randint_, random_, seed_
from qml2.models.hyperparameter_init_guesses import vector_std

interface_submodule = importlib.import_module("qml2.kernels")
imported_kernels = interface_submodule.__dict__

repsize = 64

nA = 128
nB = 256

kernel_functions = ["matern", "laplacian", "gaussian"]


def test_global_kernels():
    test_name = "global_kernels"
    checksums_storage = {}
    seed_(1)
    checksums_rng = random.Random(1)
    A = random_((nA, repsize))
    B = random_((nB, repsize))
    # get some reasonable sigma value
    sigma = vector_std(B)
    for kernel_function_name in kernel_functions:
        # import the asymmetric kernel implementation
        kernel_asym_name = kernel_function_name + "_kernel"
        kernel_asym = imported_kernels[kernel_asym_name]
        kernel_matrix_asym = kernel_asym(A, B, sigma)
        add_checksum_to_dict(
            checksums_storage,
            kernel_function_name + "_asym",
            kernel_matrix_asym,
            checksums_rng,
            stacks=8,
            nstack_checksums=4,
        )
        kernel_sym_name = kernel_asym_name + "_symmetric"
        kernel_sym = imported_kernels[kernel_sym_name]
        kernel_matrix_sym = kernel_sym(B, sigma)
        add_checksum_to_dict(
            checksums_storage,
            kernel_function_name + "_sym",
            kernel_matrix_sym,
            checksums_rng,
            stacks=8,
            nstack_checksums=4,
        )

    compare_or_create(checksums_storage, test_name, max_rel_difference=1.0e-10)


def test_local_kernels():
    test_name = "local_kernels"
    checksums_storage = {}
    seed_(1)
    checksums_rng = random.Random(1)
    nA = randint_(1, 16, (24,))
    nB = randint_(1, 32, (32,))
    tot_nA = sum(nA)
    tot_nB = sum(nB)

    ncharges_A = randint_(1, 3, (tot_nA,))
    ncharges_B = randint_(1, 3, (tot_nB,))

    A = random_((tot_nA, repsize))
    B = random_((tot_nB, repsize))
    # get some reasonable sigma value
    sigma = vector_std(B)
    for kernel_function_name in kernel_functions:
        # test simple local kernels
        kernel_asym_name = "local_" + kernel_function_name + "_kernel"
        kernel_asym = imported_kernels[kernel_asym_name]
        kernel_matrix_asym = kernel_asym(A, B, nA, nB, sigma)
        add_checksum_to_dict(
            checksums_storage,
            kernel_function_name + "_asym",
            kernel_matrix_asym,
            checksums_rng,
            stacks=4,
            nstack_checksums=4,
        )
        kernel_sym_name = kernel_asym_name + "_symmetric"
        kernel_sym = imported_kernels[kernel_sym_name]
        kernel_matrix_sym = kernel_sym(A, nA, sigma)
        add_checksum_to_dict(
            checksums_storage,
            kernel_function_name + "_sym",
            kernel_matrix_sym,
            checksums_rng,
            stacks=4,
            nstack_checksums=4,
        )
        # test dn kernels.
        kernel_dn_asym_name = "local_dn_" + kernel_function_name + "_kernel"
        kernel_dn_asym = imported_kernels[kernel_dn_asym_name]
        kernel_dn_matrix_asym = kernel_dn_asym(A, B, nA, nB, ncharges_A, ncharges_B, sigma)
        add_checksum_to_dict(
            checksums_storage,
            kernel_function_name + "_dn_asym",
            kernel_dn_matrix_asym,
            checksums_rng,
            stacks=4,
            nstack_checksums=4,
        )
        kernel_dn_sym_name = kernel_dn_asym_name + "_symmetric"
        kernel_dn_sym = imported_kernels[kernel_dn_sym_name]
        kernel_dn_matrix_sym = kernel_dn_sym(A, nA, ncharges_A, sigma)
        add_checksum_to_dict(
            checksums_storage,
            kernel_function_name + "_dn_sym",
            kernel_dn_matrix_sym,
            checksums_rng,
            stacks=4,
            nstack_checksums=4,
        )

    compare_or_create(checksums_storage, test_name, max_rel_difference=1.0e-10)


if __name__ == "__main__":
    test_global_kernels()
    test_local_kernels()
