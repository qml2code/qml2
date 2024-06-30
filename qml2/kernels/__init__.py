import importlib

implemented_kernel_functions = ["matern", "laplacian", "gaussian"]


def import_standard_kernels():
    """
    Import all version of implemented kernels
    """
    interface_submodule = importlib.import_module(".kernels", package=__name__)
    imported_funcs = interface_submodule.__dict__
    for type_prefix in ["local_", "local_dn_", ""]:
        for sym_suffix in ["_symmetric", ""]:
            for kernel_function in implemented_kernel_functions:
                kernel_name = type_prefix + kernel_function + "_kernel" + sym_suffix
                globals()[kernel_name] = imported_funcs[kernel_name]


if __name__ != "main":
    import_standard_kernels()
