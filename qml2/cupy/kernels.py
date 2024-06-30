import cupy as cp

from ..basic_utils import fetch_package_data

available_kernels = ["local_dn_gaussian_kernel", "local_gaussian_kernel", "gaussian_kernel"]

kernel_source = fetch_package_data("/cupy/kernels.cu")

kernel_module = cp.RawModule(code=kernel_source)

for kernel_name in available_kernels:
    globals()[kernel_name] = kernel_module.get_function(kernel_name)
