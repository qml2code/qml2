"""
Calculate kernels via CuPy's functionality for using raw CUDA kernels.

Each kernel defined in `qml2.kernels.kernels` can be calculated here as well using completely analogous arguments and
keyword arguments. The differences are: 1. defining keyword arguments `blocks_per_grid` and `threads_per_block`
for the function call will define numbers of blocks and threads according to those values 2. `out` keyword should
be either `None` or a CuPy array 3. sigma can be only float (multiple sigmas not supported).

For both blocks and threads, x-dimension corresponds to first kernel matrix index, y-dimension - second. z-dimension
does not affect parallelization.

While routines analogous to the ones from `qml2.kernels.kernels` (e.g. `gaussian_kernel`) operate on NumPy arrays,
each has an analogous routine with an `inside_` prefix (e.g. `inside_gaussian_kernel`) that operates on CuPy arrays
and supports the `out` keyword analogously to functions in `qml2.kernels.kernels`.

For example:
gaussian_kernel(A, B, sigma, out=kernel) is analogous to inside_gaussian_kernel(cp.asarray(A), cp.asarray(B), cp.float64(sigma), out=cp.asarray(kernel))

NOTE: the underlying algorithm assumes arrays are stored congruously, which might yield weird behavior with inside_*
routines if the user is not careful.


K.Karan.: I achieve implementation of different kernel functions in an admittedly very nasty way, but it works.
Symmetric kernel implementation could be optimized if needed, with main goal being fast predictions of model results.
If someone knows a better way to automatically choose numbers of threads and blocks please let me know.
"""


import cupy as cp

from ..basic_utils import fetch_package_data
from ..jit_interfaces import all_, dfloat_, dint_, empty_, sum_
from ..kernels.kernels import half_inv_sq_sigma, inv_sigma

sigma_to_multiplier = {"gaussian": half_inv_sq_sigma, "matern": inv_sigma}

sym_suffix = "_symmetric"


# K.Karan: I vaguely remember that doing cupy.asarray instead of this might cause problems with creating arrays that are unaligned.
# If that's not the case please correct & test this.
def safe_copy(nparray, dtype=dfloat_):
    temp_array = empty_(nparray.shape, dtype=dtype)
    match len(nparray.shape):
        case 1:
            temp_array[:] = nparray[:]
        case 2:
            temp_array[:, :] = nparray[:, :]
        case _:
            raise Exception
    return cp.asarray(temp_array)


max_nthreadings = 256


def default_nthreads_nblocks_1D(arr_dim):
    nblocks = min(arr_dim, max_nthreadings)
    if arr_dim > max_nthreadings:
        nthreads = min(arr_dim // max_nthreadings, max_nthreadings)
    else:
        nthreads = 1
    return [nblocks, nthreads]


def default_nthreads_nblocks(calculated_arr_shape):
    output = []
    for l in calculated_arr_shape:
        output.append(default_nthreads_nblocks_1D(l))
    return list(zip(*output))


class _inside_kernel:
    def __init__(self, raw_kernel, sigma2mult):
        self.raw_kernel = raw_kernel
        self.sigma2mult = sigma2mult

    def __call__(self, A, B, sigma, out=None, blocks_per_grid=None, threads_per_block=None):
        features = A.shape[1]
        assert features == B.shape[1]
        out_shape = (A.shape[0], B.shape[0])
        if out is None:
            out = cp.empty(out_shape)
        else:
            assert out.shape == out_shape
        if (blocks_per_grid is None) or (threads_per_block is None):
            blocks_per_grid, threads_per_block = default_nthreads_nblocks(out_shape)
        mult = self.sigma2mult(sigma)
        self.raw_kernel(
            blocks_per_grid, threads_per_block, (A, B, mult, *out.shape, features, out)
        )
        return out


class _kernel:
    def __init__(self, inside_kernel):
        self.inside_kernel = inside_kernel

    def __call__(self, A, B, sigma, **kwargs):
        A_cp = safe_copy(A)
        B_cp = safe_copy(B)

        kernel_cp = self.inside_kernel(A_cp, B_cp, cp.float64(sigma), **kwargs)

        return cp.asnumpy(kernel_cp)


class _kernel_sym:
    def __init__(self, kernel):
        self.kernel = kernel

    def __call__(self, A, *args, **kwargs):
        return self.kernel(A, A, *args, **kwargs)


class _inside_local_kernel(_inside_kernel):
    def __call__(
        self,
        A,
        B,
        A_natoms,
        B_natoms,
        sigma,
        out=None,
        blocks_per_grid=None,
        threads_per_block=None,
    ):
        features = A.shape[1]
        assert (features, (A.shape[0], B.shape[0])) == (
            B.shape[1],
            (sum_(A_natoms), sum_(B_natoms)),
        )
        assert (all_(A_natoms > 0)) and (all_(B_natoms > 0))
        out_shape = (A_natoms.shape[0], B_natoms.shape[0])
        if out is None:
            out = cp.empty(out_shape)
        else:
            assert out.shape == out_shape

        if (blocks_per_grid is None) or (threads_per_block is None):
            blocks_per_grid, threads_per_block = default_nthreads_nblocks(out_shape)

        mult = self.sigma2mult(sigma)
        self.raw_kernel(
            blocks_per_grid,
            threads_per_block,
            (A, B, A_natoms, B_natoms, mult, *out_shape, features, out),
        )
        return out


class _local_kernel(_kernel):
    def __call__(self, A, B, A_natoms, B_natoms, sigma, **kwargs):
        # K.Karan: I vaguely remember that doing cupy.asarray instead of this might cause problems with creating arrays that are unaligned.
        A_cp = safe_copy(A)
        B_cp = safe_copy(B)

        na_cp = safe_copy(A_natoms, dtype=dint_)
        nb_cp = safe_copy(B_natoms, dtype=dint_)

        kernel_cp = self.inside_kernel(A_cp, B_cp, na_cp, nb_cp, cp.float64(sigma), **kwargs)

        return cp.asnumpy(kernel_cp)


class _local_kernel_sym(_kernel_sym):
    def __call__(self, A, A_natoms, *args, **kwargs):
        return self.kernel(A, A, A_natoms, A_natoms, *args, **kwargs)


class _inside_local_dn_kernel(_inside_kernel):
    def __call__(
        self,
        A,
        B,
        A_natoms,
        B_natoms,
        A_ncharges,
        B_ncharges,
        sigma,
        out=None,
        blocks_per_grid=None,
        threads_per_block=None,
    ):
        features = A.shape[1]
        out_shape = (A_natoms.shape[0], B_natoms.shape[0])
        if out is None:
            out = cp.empty(out_shape)
        else:
            assert out.shape == out_shape
        assert features == B.shape[1]
        assert (A.shape[0], B.shape[0]) == (A_ncharges.shape[0], B_ncharges.shape[0])
        assert (A.shape[0], B.shape[0]) == (sum_(A_natoms), sum_(B_natoms))
        assert (all_(A_natoms > 0)) and (all_(B_natoms > 0))
        if (blocks_per_grid is None) or (threads_per_block is None):
            blocks_per_grid, threads_per_block = default_nthreads_nblocks(out_shape)
        mult = self.sigma2mult(sigma)
        self.raw_kernel(
            blocks_per_grid,
            threads_per_block,
            (
                A,
                B,
                A_natoms,
                B_natoms,
                A_ncharges,
                B_ncharges,
                mult,
                *out_shape,
                features,
                out,
            ),
        )
        return out


class _local_dn_kernel(_kernel):
    def __call__(
        self, A, B, A_natoms, B_natoms, A_ncharges, B_ncharges, sigma, out=None, **kwargs
    ):
        # K.Karan: I vaguely remember that doing cupy.asarray instead of this might cause problems with creating arrays that are discontinuously stored.
        A_cp = safe_copy(A)
        B_cp = safe_copy(B)

        na_cp = safe_copy(A_natoms, dtype=dint_)
        nb_cp = safe_copy(B_natoms, dtype=dint_)

        A_ncharges_cp = safe_copy(A_ncharges, dtype=dint_)
        B_ncharges_cp = safe_copy(B_ncharges, dtype=dint_)

        kernel_cp = self.inside_kernel(
            A_cp,
            B_cp,
            na_cp,
            nb_cp,
            A_ncharges_cp,
            B_ncharges_cp,
            cp.float64(sigma),
            out=out,
            **kwargs,
        )

        return cp.asnumpy(kernel_cp)


class _local_dn_kernel_sym(_kernel_sym):
    def __call__(self, A, A_natoms, A_ncharges, *args, **kwargs):
        return self.kernel(A, A, A_natoms, A_natoms, A_ncharges, A_ncharges, *args, **kwargs)


def get_kernel_type_additional_sources(kernel_definition_tuple):
    kernel_type = kernel_definition_tuple[0]
    match kernel_type:
        case "gaussian":
            add_sources = ["l2_metric", "gaussian_kernel"]
        case "matern":
            order = kernel_definition_tuple[1]
            metric = kernel_definition_tuple[2]
            add_sources = [metric + "_metric", "matern_" + str(order) + "_kernel"]
        case _:
            raise Exception
    return kernel_type, add_sources


precompiled_kernels = {}


def underscore_join(t):
    return "_".join([str(s) for s in t])


def setup_raw_kernels(kernel_definition_tuple):
    kernel_type, additional_sources = get_kernel_type_additional_sources(kernel_definition_tuple)
    sigma2mult = sigma_to_multiplier[kernel_type]
    kernel_source = ""
    for add_source in [*additional_sources, "raw_kernel_common"]:
        kernel_source += fetch_package_data("cupy/" + add_source + ".cu")

    kernel_module = cp.RawModule(code=kernel_source)

    full_kernel_type = "_".join([str(s) for s in kernel_definition_tuple])

    for kernel_name_prefix, [inside_kernel_cl, kernel_cl, kernel_sym_cl] in [
        ("", [_inside_kernel, _kernel, _kernel_sym]),
        ("local_", [_inside_local_kernel, _local_kernel, _local_kernel_sym]),
        ("local_dn_", [_inside_local_dn_kernel, _local_dn_kernel, _local_dn_kernel_sym]),
    ]:
        kernel_name = kernel_name_prefix + full_kernel_type + "_kernel"
        interface_kernel_name = "interface_" + kernel_name
        inside_kernel_name = "inside_" + kernel_name
        raw_kernel_name = "raw_" + kernel_name
        globals()[raw_kernel_name] = kernel_module.get_function(kernel_name_prefix + "kernel")

        globals()[inside_kernel_name] = inside_kernel_cl(globals()[raw_kernel_name], sigma2mult)
        globals()[interface_kernel_name] = kernel_cl(globals()[inside_kernel_name])
        globals()[interface_kernel_name + sym_suffix] = kernel_sym_cl(
            globals()[interface_kernel_name]
        )


def full_kernel_name(full_kernel_definition_tuple):
    return (
        full_kernel_definition_tuple[0]
        + underscore_join(full_kernel_definition_tuple[1:])
        + "_kernel"
    )


def calc_kernel(full_kernel_definition_tuple, *args, symmetric=False, **kwargs):
    global precompiled_kernels
    interface_name = "interface_" + full_kernel_name(full_kernel_definition_tuple)
    if symmetric:
        interface_name = interface_name + "_symmetric"
    if interface_name not in globals():
        setup_raw_kernels(full_kernel_definition_tuple[1:])
    return globals()[interface_name](*args, **kwargs)


def get_inside_kernel_routine(
    kernel_type="gaussian", order=0, metric="l1", local=False, local_dn=False, symmetric=False
):
    if local_dn:
        prefix = "local_dn_"
    else:
        if local:
            prefix = "local_"
        else:
            prefix = ""
    if kernel_type == "matern":
        full_kernel_definition_tuple = (prefix, kernel_type, order, metric)
    else:
        full_kernel_definition_tuple = (
            prefix,
            kernel_type,
        )
    inside_name = "inside_" + full_kernel_name(full_kernel_definition_tuple)
    if symmetric:
        inside_name = inside_name + sym_suffix
    if inside_name not in globals():
        setup_raw_kernels(full_kernel_definition_tuple[1:])
    return globals()[inside_name]


# Gaussian kernels.
def gaussian_kernel(*args, **kwargs):
    return calc_kernel(("", "gaussian"), *args, **kwargs)


def gaussian_kernel_symmetric(*args, **kwargs):
    return calc_kernel(("", "gaussian"), *args, symmetric=True, **kwargs)


def local_gaussian_kernel(*args, **kwargs):
    return calc_kernel(("local_", "gaussian"), *args, **kwargs)


def local_gaussian_kernel_symmetric(*args, **kwargs):
    return calc_kernel(("local_", "gaussian"), *args, symmetric=True, **kwargs)


def local_dn_gaussian_kernel(*args, **kwargs):
    return calc_kernel(("local_dn_", "gaussian"), *args, **kwargs)


def local_dn_gaussian_kernel_symmetric(*args, **kwargs):
    return calc_kernel(("local_dn_", "gaussian"), *args, symmetric=True, **kwargs)


# Matern kernels
def matern_kernel(*args, order=0, metric="l1", **kwargs):
    return calc_kernel(("", "matern", order, metric), *args, **kwargs)


def matern_kernel_symmetric(*args, order=0, metric="l1", **kwargs):
    return calc_kernel(("", "matern", order, metric), *args, symmetric=True, **kwargs)


def local_matern_kernel(*args, order=0, metric="l1", **kwargs):
    return calc_kernel(("local_", "matern", order, metric), *args, **kwargs)


def local_matern_kernel_symmetric(*args, order=0, metric="l1", **kwargs):
    return calc_kernel(("local_", "matern", order, metric), *args, symmetric=True, **kwargs)


def local_dn_matern_kernel(*args, order=0, metric="l1", **kwargs):
    return calc_kernel(("local_dn_", "matern", order, metric), *args, **kwargs)


def local_dn_matern_kernel_symmetric(*args, order=0, metric="l1", **kwargs):
    return calc_kernel(("local_dn_", "matern", order, metric), *args, symmetric=True, **kwargs)


# Laplacian kernels (Matern with order 0 and l1 metric)
def laplacian_kernel(*args, **kwargs):
    return matern_kernel(*args, **kwargs)


def laplacian_kernel_symmetric(*args, **kwargs):
    return matern_kernel_symmetric(*args, **kwargs)


def local_laplacian_kernel(*args, **kwargs):
    return local_matern_kernel(*args, **kwargs)


def local_laplacian_kernel_symmetric(*args, **kwargs):
    return local_matern_kernel_symmetric(*args, **kwargs)


def local_dn_laplacian_kernel(*args, **kwargs):
    return local_dn_matern_kernel(*args, **kwargs)


def local_dn_laplacian_kernel_symmetric(*args, **kwargs):
    return local_dn_matern_kernel_symmetric(*args, **kwargs)
