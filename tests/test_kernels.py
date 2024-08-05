import importlib
import random

from conftest import add_checksum_to_dict, compare_or_create, randint_arr, random_arr, str2rng

from qml2.models.hyperparameter_init_guesses import vector_std

repsize = 64

nA = 128
nB = 256

kernel_functions = ["laplacian", "gaussian"]

for matern_order in [0, 1, 2]:
    for matern_metric in ["l1", "l2"]:
        kernel_functions.append(
            (
                "matern",
                {"metric": matern_metric, "order": matern_order},
                "matern_" + matern_metric + "_" + str(matern_order),
            )
        )


def get_imported_kernels(module_name):
    interface_submodule = importlib.import_module(module_name)
    return interface_submodule.__dict__


def run_global_kernels_test(
    module_name,
    test_name="global_kernels",
    kernel_functions=kernel_functions,
    partial_comparison=False,
):
    imported_kernels = get_imported_kernels(module_name)
    checksums_storage = {}
    test_input_rng = random.Random(1)
    A = random_arr((nA, repsize), test_input_rng)
    B = random_arr((nB, repsize), test_input_rng)
    # get some reasonable sigma value
    sigma = vector_std(B)
    for kernel_function_tuple in kernel_functions:
        if isinstance(kernel_function_tuple, tuple):
            kernel_function_name = kernel_function_tuple[0]
            kernel_kwargs = kernel_function_tuple[1]
            test_component_name = kernel_function_tuple[2]
        else:
            kernel_function_name = kernel_function_tuple
            kernel_kwargs = {}
            test_component_name = kernel_function_name
        checksums_rng = str2rng(test_component_name)
        # import the asymmetric kernel implementation
        kernel_asym_name = kernel_function_name + "_kernel"
        kernel_asym = imported_kernels[kernel_asym_name]
        kernel_matrix_asym = kernel_asym(A, B, sigma, **kernel_kwargs)
        add_checksum_to_dict(
            checksums_storage,
            test_component_name + "_asym",
            kernel_matrix_asym,
            checksums_rng,
            stacks=8,
            nstack_checksums=4,
        )
        kernel_sym_name = kernel_asym_name + "_symmetric"
        kernel_sym = imported_kernels[kernel_sym_name]
        kernel_matrix_sym = kernel_sym(B, sigma, **kernel_kwargs)
        add_checksum_to_dict(
            checksums_storage,
            test_component_name + "_sym",
            kernel_matrix_sym,
            checksums_rng,
            stacks=8,
            nstack_checksums=4,
        )

    compare_or_create(
        checksums_storage,
        test_name,
        max_rel_difference=1.0e-10,
        partial_comparison=partial_comparison,
    )


def run_local_kernels_test(
    module_name,
    test_name="local_kernels",
    kernel_functions=kernel_functions,
    partial_comparison=False,
):
    imported_kernels = get_imported_kernels(module_name)
    checksums_storage = {}
    test_input_rng = random.Random(1)
    nA = randint_arr(1, 16, (24,), test_input_rng)
    nB = randint_arr(1, 32, (32,), test_input_rng)
    tot_nA = sum(nA)
    tot_nB = sum(nB)

    ncharges_A = randint_arr(1, 3, (tot_nA,), test_input_rng)
    ncharges_B = randint_arr(1, 3, (tot_nB,), test_input_rng)

    A = random_arr((tot_nA, repsize), test_input_rng)
    B = random_arr((tot_nB, repsize), test_input_rng)
    # get some reasonable sigma value
    sigma = vector_std(B)
    for kernel_function_tuple in kernel_functions:
        if isinstance(kernel_function_tuple, tuple):
            kernel_function_name = kernel_function_tuple[0]
            kernel_kwargs = kernel_function_tuple[1]
            test_component_name = kernel_function_tuple[2]
        else:
            kernel_function_name = kernel_function_tuple
            kernel_kwargs = {}
            test_component_name = kernel_function_name

        checksums_rng = str2rng(test_component_name)
        # test simple local kernels
        kernel_asym_name = "local_" + kernel_function_name + "_kernel"
        kernel_asym = imported_kernels[kernel_asym_name]
        kernel_matrix_asym = kernel_asym(A, B, nA, nB, sigma, **kernel_kwargs)
        add_checksum_to_dict(
            checksums_storage,
            test_component_name + "_asym",
            kernel_matrix_asym,
            checksums_rng,
            stacks=4,
            nstack_checksums=4,
        )
        kernel_sym_name = kernel_asym_name + "_symmetric"
        kernel_sym = imported_kernels[kernel_sym_name]
        kernel_matrix_sym = kernel_sym(A, nA, sigma, **kernel_kwargs)
        add_checksum_to_dict(
            checksums_storage,
            test_component_name + "_sym",
            kernel_matrix_sym,
            checksums_rng,
            stacks=4,
            nstack_checksums=4,
        )
        # test dn kernels.
        kernel_dn_asym_name = "local_dn_" + kernel_function_name + "_kernel"
        kernel_dn_asym = imported_kernels[kernel_dn_asym_name]
        kernel_dn_matrix_asym = kernel_dn_asym(
            A, B, nA, nB, ncharges_A, ncharges_B, sigma, **kernel_kwargs
        )
        add_checksum_to_dict(
            checksums_storage,
            test_component_name + "_dn_asym",
            kernel_dn_matrix_asym,
            checksums_rng,
            stacks=4,
            nstack_checksums=4,
        )
        kernel_dn_sym_name = kernel_dn_asym_name + "_symmetric"
        kernel_dn_sym = imported_kernels[kernel_dn_sym_name]
        kernel_dn_matrix_sym = kernel_dn_sym(A, nA, ncharges_A, sigma, **kernel_kwargs)
        add_checksum_to_dict(
            checksums_storage,
            test_component_name + "_dn_sym",
            kernel_dn_matrix_sym,
            checksums_rng,
            stacks=4,
            nstack_checksums=4,
        )

    compare_or_create(
        checksums_storage,
        test_name,
        max_rel_difference=1.0e-10,
        partial_comparison=partial_comparison,
    )


def test_global_kernels():
    run_global_kernels_test("qml2.kernels")


def test_local_kernels():
    run_local_kernels_test("qml2.kernels")


if __name__ == "__main__":
    test_global_kernels()
    test_local_kernels()
