"""
Expressions for calculating Gaussian, Laplacian, and Matern kernels, both local and global.

# Kernel types.

For a kernel function named ${kf_name} (\f$\in\f$ ["gaussian", "laplacian", "matern"]) corresponding to function \f$F\f$ we have defined:

```${kf_name}_kernel(A, B, sigma, out=None, **kwargs):```
calculate matrix \f$K\f$ such that

\f[
K_{ij}=F(A_{i}, B_{j}, \sigma)
\f]

Arguments:

- A, B - arrays of *global* representation vectors of length corresponding to number of molecules in A and B (A_nmols and B_nmols).
- sigma (\f$\sigma\f$) - float of sigma value or 1D array of sigma values length nsigmas.
- out - if not None the array will be used for kernel output.
- **kwargs - see "Implemented $F$ options"

Output:
- If \f$\sigma\f$ is float return matrix K of dimensionality A_nmols x B_nmols.
- If \f$\sigma\f$ is 1D array return tensor K of dimensionality A_nmols x B_nmols x nsigmas, with \f$K_{ijk}\f$ corresponding to \f$K_{ij}\f$ calculated for \f$\sigma_{k}\f$.


```${kf_name}_kernel_symmetric(A, sigma, **kwargs):```
returns ${kf_name}_kernel(A, A, sigma, **kwargs)

local_${kf_name}_kernel(A, B, A_natoms, B_natoms, sigma, out=None, **kwargs): calculate matrix K such that

\f[
K_{ij}=\sum_{ij}\sum_{k=1}^{Na_{i}}\sum_{l=1}^{Nb_{j}}F(A_{ik}, B_{jl}, \sigma),
\f]

where \f$A_{ik}\f$, \f$B_{jl}\f$ are representations vectors of atoms \f$k\f$ and \f$l\f$ in molecules \f$A_{i}\f$ and \f$B_{j}\f$, \f$Na_{i}\f$ and \f$Nb_{j}\f$ are numbers of atoms in molecules \f$A_{i}\f$ and \f$B_{j}\f$

Arguments:

- A, B - 2D arrays of *local* representation vectors of molecules concatenated (e.g. via numpy.concatenate) together.
- A_natoms, B_natoms - 1D integer arrays of number of atoms for each molecule.
-sigma, out - same as in ${kf_name}_kernel

Output:
- same as in ${kf_name}_kernel

```
local_${kf_name}_kernel_symmetric(A, A_natoms, sigma, **kwargs):
```
returns local_${kf_name}_kernel(A, A, A_natoms, A_natoms, sigma, **kwargs)

```
local_dn_${kf_name}_kernel(A, B, A_natoms, B_natoms, A_ncharges, B_ncharges, out=None, **kwargs):
```
calculate matrix \f$K\f$ such that

\f[
K_{ij}=\sum_{ij}\sum_{k=1}^{Na_{i}}\sum_{l=1}^{Nb_{j}}F(A_{ik}, B_{jl}, \sigma) \delta(Nca_{ik}-Ncb_{jl}),
\f]
where \f$\delta\f$ is the delta function, \f$Nca_{ik}\f$ and \f$Ncb_{jl}\f$ are nuclear charges of atoms \f$k\f$ and \f$l\f$ in molecules \f$A_i\f$ and \f$B_j\f$.

Arguments:

- A, B, A_natoms, B_natoms, sigma, out - same as local_${kf_name}_kernel.
- A_ncharges, B_ncharges - concatenated (e.g. via numpy.concatenate) nuclear charges arrays of individual molecules.

```
local_dn_${kf_name}_kernel_symmetric(A, A_natoms, A_ncharges, sigma, **kwargs):
```
returns local_dn_${kf_name}_kernel(A, A, A_natoms, A_natoms, A_ncharges, A_ncharges, sigma, **kwargs)

# Implemented \f$F\f$ options

- *Gaussian:*
\f[
F(X^1, X^2, \sigma)=\exp\left[\frac{(X^1-X^2)^2}{2\sigma^2}\right]
\f]
- *Laplacian:*
\f[
F(X^1, X^2, \sigma)=\exp\left(\frac{\sum_{k}|X^1_{k}-X^2_{k}|}{\sigma}\right)
\f]
- *Matern:* \f$F(X^{1},X^{2})\f$ depends on the distance \f$r(X^{1},X^{2})\f$, which, depending on the `metric` keyword set during the kernel calculation, can be L1 ("l1") or L2 ("l2"). The `order` keyword for each kernel calculation determines order of the Matern kernel. The implemented order values are (writing \f$r_s=r(X^{1},X^{2})/\sigma\f$ for short):

    0:
\f[
F(X^1, X^2, \sigma)=\exp(-r_s)
\f]
    1:
\f[
F(X^1, X^2, \sigma)=\exp(-r_s*\sqrt{3})*(1+\sqrt{3} r_s)
\f]
    2:
\f[
F(X^1, X^2, \sigma)=\exp(-r_s*\sqrt{5})*(1+\sqrt{5}*r_s+5 r_s^2/3)
\f]

"""
# COMMENT: Konstantin Karandashev:
# The precompiled function variables are seeminly the best way to capitalize on Python's polymorphism while avoiding
# calls on functions as arguments (they seem to slow the code down a bit). There would be better ways to write this when jitclass moves out of
# experimental and starts supporting __call__.


from ..jit_interfaces import (
    constr_dfloat_,
    dim0float_array_,
    empty_,
    exp_,
    float_,
    is_scalar_,
    jit_,
    l2_norm_,
    ndarray_,
    prange_,
    sqrt_,
    zero_scalar_,
    zeros_,
)
from ..utils import get_atom_environment_ranges, l1_norm_dist, l2_norm_sq_dist

available_metrics = ["l1", "l2"]

available_matern_orders = [0, 1, 2]


@jit_
def l2_norm_dist(vec1, vec2):
    return l2_norm_(vec1 - vec2)


def construct_exp_kernel_function(metric="l1"):
    match metric:
        case "l1":
            dist_func = l1_norm_dist
        case "l2":
            dist_func = l2_norm_sq_dist
        case _:
            raise Exception

    @jit_
    def exp_kernel_function(vec1, vec2, sigma_param):
        return exp_(-dist_func(vec1, vec2) * sigma_param)

    return exp_kernel_function


# For Matern kernels. The expression is taken from qmlcode.org.
sqrt3 = sqrt_(constr_dfloat_(3.0))
sqrt5 = sqrt_(constr_dfloat_(5.0))


# Bessel functions of different orders.
@jit_
def bf0(d: ndarray_):
    return exp_(-d)


@jit_
def bf1(d: ndarray_, sqrt3: dim0float_array_ = sqrt3):
    return exp_(-d * sqrt3) * (1 + sqrt3 * d)


@jit_
def bf2(d: ndarray_, sqrt5: dim0float_array_ = sqrt5):
    return exp_(-d * sqrt5) * (1 + sqrt5 * d + 5.0 * d**2 / 3.0)


def construct_bessel_function(order):
    assert order in available_matern_orders
    match order:
        case 0:
            return bf0
        case 1:
            return bf1
        case 2:
            return bf2
        case _:
            raise Exception


def construct_matern_kernel_function(order=0, metric="l1"):
    assert metric in available_metrics
    if metric == "l1":
        dist_func = l1_norm_dist
    if metric == "l2":
        dist_func = l2_norm_dist
    used_bessel_func = construct_bessel_function(order)

    @jit_
    def matern_kernel_function(vec1, vec2, sigma_param):
        d = dist_func(vec1, vec2) * sigma_param
        return used_bessel_func(d)

    return matern_kernel_function


@jit_
def inv_sigma(sigma):
    return sigma ** (-1)


@jit_
def half_inv_sq_sigma(sigma):
    return sigma ** (-2) * 0.5


def construct_kernel_symmetric(kernel_function, sigma_to_param=inv_sigma):
    @jit_(numba_parallel=True)
    def kernel_symmetric(A, sigma, output_kernel):
        sigma_param = sigma_to_param(sigma)
        for i in prange_(A.shape[0]):
            for j in range(i + 1):
                output_kernel[i, j] = kernel_function(A[i], A[j], sigma_param)
                output_kernel[j, i] = output_kernel[i, j]

    return kernel_symmetric


def construct_kernel_asymmetric(kernel_function, sigma_to_param=inv_sigma):
    @jit_(numba_parallel=True)
    def kernel_asymmetric(A, B, sigma, output_kernel):
        sigma_param = sigma_to_param(sigma)
        for i in prange_(A.shape[0]):
            for j in range(B.shape[0]):
                output_kernel[i, j] = kernel_function(A[i], B[j], sigma_param)

    return kernel_asymmetric


# TODO: this and zero_scalar_ are introduced to account for cases when kernel element is a scalar (for standard kernels)
# or an array (e.g. when kernels for different sigmas are calculated).
def construct_init_kernel_element(many_sigmas=None):
    if many_sigmas is None:
        return zero_scalar_

    @jit_
    def init_kernel_element():
        return zeros_((many_sigmas,))

    return init_kernel_element


def construct_local_kernel_function(kernel_function, init_kernel_element=zero_scalar_):
    @jit_
    def local_kernel_function(A_reps, B_reps, sigma_param):
        output_kernel = init_kernel_element()
        for A_rep in A_reps:
            for B_rep in B_reps:
                output_kernel += kernel_function(A_rep, B_rep, sigma_param)
        return output_kernel

    return local_kernel_function


def construct_local_kernel_asymmetric(
    kernel_function, sigma_to_param=inv_sigma, init_kernel_element=zero_scalar_
):
    lc_func = construct_local_kernel_function(
        kernel_function, init_kernel_element=init_kernel_element
    )

    @jit_(numba_parallel=True)
    def local_kernel_asymmetric(A, B, A_natoms, B_natoms, sigma, output_kernel):
        sigma_param = sigma_to_param(sigma)
        nmols_A = A_natoms.shape[0]
        nmols_B = B_natoms.shape[0]
        ubound_arr_A = get_atom_environment_ranges(A_natoms)
        ubound_arr_B = get_atom_environment_ranges(B_natoms)

        for i in prange_(nmols_A):
            A_rep_subarray = A[ubound_arr_A[i] : ubound_arr_A[i + 1]]
            for j in range(nmols_B):
                output_kernel[i, j] = lc_func(
                    A_rep_subarray, B[ubound_arr_B[j] : ubound_arr_B[j + 1]], sigma_param
                )

    return local_kernel_asymmetric


def construct_local_kernel_symmetric(
    kernel_function, sigma_to_param=inv_sigma, init_kernel_element=zero_scalar_
):
    lc_func = construct_local_kernel_function(
        kernel_function, init_kernel_element=init_kernel_element
    )

    @jit_(numba_parallel=True)
    def local_kernel_symmetric(A, A_natoms, sigma, output_kernel):
        sigma_param = sigma_to_param(sigma)
        nmols_A = A_natoms.shape[0]
        ubound_arr_A = get_atom_environment_ranges(A_natoms)

        for i in prange_(nmols_A):
            A_rep_subarray = A[ubound_arr_A[i] : ubound_arr_A[i + 1]]
            for j in range(i + 1):
                output_kernel[i, j] = lc_func(
                    A_rep_subarray,
                    A[ubound_arr_A[j] : ubound_arr_A[j + 1]],
                    sigma_param,
                )
                output_kernel[j, i] = output_kernel[i, j]

    return local_kernel_symmetric


# For kernels with delta-functions of nuclear charge difference.
def construct_local_dn_kernel_function(kernel_function, init_kernel_element=zero_scalar_):
    @jit_
    def local_dn_kernel_function(A_reps, B_reps, A_ncharges, B_ncharges, sigma_param):
        output = init_kernel_element()
        for A_rep_id in range(A_reps.shape[0]):
            for B_rep_id in range(B_reps.shape[0]):
                if A_ncharges[A_rep_id] != B_ncharges[B_rep_id]:
                    continue
                output += kernel_function(A_reps[A_rep_id], B_reps[B_rep_id], sigma_param)
        return output

    return local_dn_kernel_function


def construct_local_dn_kernel_asymmetric(
    kernel_function, sigma_to_param=inv_sigma, init_kernel_element=zero_scalar_
):
    lc_func = construct_local_dn_kernel_function(
        kernel_function, init_kernel_element=init_kernel_element
    )

    @jit_(numba_parallel=True)
    def local_dn_kernel_asymmetric(
        A, B, A_natoms, B_natoms, A_ncharges, B_ncharges, sigma, output_kernel
    ):
        sigma_param = sigma_to_param(sigma)
        nmols_A = A_natoms.shape[0]
        nmols_B = B_natoms.shape[0]
        ubound_arr_A = get_atom_environment_ranges(A_natoms)
        ubound_arr_B = get_atom_environment_ranges(B_natoms)

        for i in prange_(nmols_A):
            A_rep_subarray = A[ubound_arr_A[i] : ubound_arr_A[i + 1]]
            A_ncharges_subarray = A_ncharges[ubound_arr_A[i] : ubound_arr_A[i + 1]]
            for j in range(nmols_B):
                output_kernel[i, j] = lc_func(
                    A_rep_subarray,
                    B[ubound_arr_B[j] : ubound_arr_B[j + 1]],
                    A_ncharges_subarray,
                    B_ncharges[ubound_arr_B[j] : ubound_arr_B[j + 1]],
                    sigma_param,
                )

    return local_dn_kernel_asymmetric


def construct_local_dn_kernel_symmetric(
    kernel_function, sigma_to_param=inv_sigma, init_kernel_element=zero_scalar_
):
    lc_func = construct_local_dn_kernel_function(
        kernel_function, init_kernel_element=init_kernel_element
    )

    @jit_(numba_parallel=True)
    def local_dn_kernel_symmetric(A, A_natoms, A_ncharges, sigma, output_kernel):
        sigma_param = sigma_to_param(sigma)
        nmols_A = A_natoms.shape[0]
        ubound_arr_A = get_atom_environment_ranges(A_natoms)

        for i in prange_(nmols_A):
            A_rep_subarray = A[ubound_arr_A[i] : ubound_arr_A[i + 1]]
            A_ncharges_subarray = A_ncharges[ubound_arr_A[i] : ubound_arr_A[i + 1]]
            for j in range(i + 1):
                output_kernel[i, j] = lc_func(
                    A_rep_subarray,
                    A[ubound_arr_A[j] : ubound_arr_A[j + 1]],
                    A_ncharges_subarray,
                    A_ncharges[ubound_arr_A[j] : ubound_arr_A[j + 1]],
                    sigma_param,
                )
                output_kernel[j, i] = output_kernel[i, j]

    return local_dn_kernel_symmetric


# Precompiling different kernel functions
precompiled_exp_kernel_functions = {}


def get_exp_kernel_function(metric="l1"):
    if metric not in precompiled_exp_kernel_functions:
        precompiled_exp_kernel_functions[metric] = construct_exp_kernel_function(metric=metric)
    return precompiled_exp_kernel_functions[metric]


precompiled_matern_kernel_functions = {}


def get_matern_kernel_function(metric="l1", order=0):
    if metric not in precompiled_matern_kernel_functions:
        precompiled_matern_kernel_functions[metric] = {}
    if order not in precompiled_matern_kernel_functions[metric]:
        precompiled_matern_kernel_functions[metric][order] = construct_matern_kernel_function(
            order=order, metric=metric
        )
    return precompiled_matern_kernel_functions[metric][order]


def get_kernel_function(type="exp", **kwargs):
    match type:
        case "exp":
            return get_exp_kernel_function(**kwargs)
        case "Matern":
            return get_matern_kernel_function(**kwargs)
        case _:
            raise Exception


def construct_kernel(
    symmetric=False,
    local=False,
    local_dn=False,
    sigma_to_param=inv_sigma,
    many_sigmas=None,
    **kernel_function_kwargs,
):
    kernel_function = get_kernel_function(**kernel_function_kwargs)
    if local:
        init_kernel_element = construct_init_kernel_element(many_sigmas)
        if local_dn:
            if symmetric:
                kernel_constructor = construct_local_dn_kernel_symmetric
            else:
                kernel_constructor = construct_local_dn_kernel_asymmetric
        else:
            if symmetric:
                kernel_constructor = construct_local_kernel_symmetric
            else:
                kernel_constructor = construct_local_kernel_asymmetric
        return kernel_constructor(
            kernel_function, sigma_to_param=sigma_to_param, init_kernel_element=init_kernel_element
        )
    else:
        if symmetric:
            kernel_constructor = construct_kernel_symmetric
        else:
            kernel_constructor = construct_kernel_asymmetric
        return kernel_constructor(kernel_function, sigma_to_param=sigma_to_param)


# TODO: There should be a way to do this in a more python-esque way, should be investigated once it becomes clearer which kernels we need.
def get_matern_kernel(precomp_kernel_dict, metric="l1", order=0, **other_kwargs):
    if order not in precomp_kernel_dict:
        precomp_kernel_dict[order] = {}
    if metric not in precomp_kernel_dict[order]:
        precomp_kernel_dict[order][metric] = construct_kernel(
            order=order, metric=metric, **other_kwargs
        )  # we assume type="Matern" is in **other_kwargs
    return precomp_kernel_dict[order][metric]


# TODO: KK: there should be a more Python-esque way to write this.
def get_kernel(
    precomp_kernel_dict,
    symmetric=False,
    local=False,
    local_dn=False,
    many_sigmas=None,
    **other_construct_kernel_kwargs,
):
    matern = ("type" in other_construct_kernel_kwargs) and (
        other_construct_kernel_kwargs["type"] == "Matern"
    )
    if many_sigmas not in precomp_kernel_dict:
        precomp_kernel_dict[many_sigmas] = {}
    if symmetric not in precomp_kernel_dict[many_sigmas]:
        precomp_kernel_dict[many_sigmas][symmetric] = {}
    if local:
        if local not in precomp_kernel_dict[many_sigmas][symmetric]:
            precomp_kernel_dict[many_sigmas][symmetric][local] = {}
        if matern:
            if local_dn not in precomp_kernel_dict[many_sigmas][symmetric][local]:
                precomp_kernel_dict[many_sigmas][symmetric][local][local_dn] = {}
            return get_matern_kernel(
                precomp_kernel_dict[many_sigmas][symmetric][local][local_dn],
                symmetric=symmetric,
                local=local,
                local_dn=local_dn,
                **other_construct_kernel_kwargs,
            )
        else:
            if local_dn not in precomp_kernel_dict[many_sigmas][symmetric][local]:
                precomp_kernel_dict[many_sigmas][symmetric][local][local_dn] = construct_kernel(
                    symmetric=symmetric,
                    local=local,
                    local_dn=local_dn,
                    **other_construct_kernel_kwargs,
                )
            return precomp_kernel_dict[many_sigmas][symmetric][local][local_dn]
    else:
        if matern:
            if local not in precomp_kernel_dict[many_sigmas][symmetric]:
                precomp_kernel_dict[many_sigmas][symmetric][local] = {}
            return get_matern_kernel(
                precomp_kernel_dict[many_sigmas][symmetric][local],
                symmetric=symmetric,
                local=local,
                local_dn=local_dn,
                **other_construct_kernel_kwargs,
            )
        else:
            if local not in precomp_kernel_dict[many_sigmas][symmetric]:
                precomp_kernel_dict[many_sigmas][symmetric][local] = construct_kernel(
                    symmetric=symmetric, local=local, **other_construct_kernel_kwargs
                )
            return precomp_kernel_dict[many_sigmas][symmetric][local]


def allocate_kernel(
    A: ndarray_, B: ndarray_, sigma: ndarray_ | float_, out: ndarray_ | None = None
):
    nA = A.shape[0]
    if B is None:
        nB = nA
    else:
        nB = B.shape[0]
    new_shape = (nA, nB)
    if is_scalar_(sigma):
        many_sigmas = None
    else:
        many_sigmas = sigma.shape[0]
        new_shape = (*new_shape, sigma.shape[0])
    if out is None:
        return empty_(new_shape), many_sigmas
    else:
        assert (
            new_shape == out.shape
        ), "ERROR: mismatch in dimensions of desired kernel output and the one provided in the `out` keyword."
        return out, many_sigmas


def calculate_global_kernel(
    A, B, sigma, precompiled_kernel_dictionnary, out=None, **get_kernel_kwargs
):
    symmetric = B is None
    kernel_output, many_sigmas = allocate_kernel(A, B, sigma, out=out)
    kernel = get_kernel(
        precompiled_kernel_dictionnary,
        symmetric=symmetric,
        many_sigmas=many_sigmas,
        **get_kernel_kwargs,
    )
    if symmetric:
        kernel(A, sigma, kernel_output)
    else:
        kernel(A, B, sigma, kernel_output)
    return kernel_output


def calculate_local_kernel(
    A, B, A_natoms, B_natoms, sigma, precompiled_kernel_dictionnary, out=None, **get_kernel_kwargs
):
    symmetric = B is None
    kernel_output, many_sigmas = allocate_kernel(A_natoms, B_natoms, sigma, out=out)
    kernel = get_kernel(
        precompiled_kernel_dictionnary,
        symmetric=symmetric,
        local=True,
        many_sigmas=many_sigmas,
        **get_kernel_kwargs,
    )
    if symmetric:
        kernel(A, A_natoms, sigma, kernel_output)
    else:
        kernel(A, B, A_natoms, B_natoms, sigma, kernel_output)
    return kernel_output


def calculate_local_dn_kernel(
    A,
    B,
    A_natoms,
    B_natoms,
    A_ncharges,
    B_ncharges,
    sigma,
    precompiled_kernel_dictionnary,
    out=None,
    **get_kernel_kwargs,
):
    symmetric = B is None
    kernel_output, many_sigmas = allocate_kernel(A_natoms, B_natoms, sigma, out=out)
    kernel = get_kernel(
        precompiled_kernel_dictionnary,
        symmetric=symmetric,
        local=True,
        local_dn=True,
        many_sigmas=many_sigmas,
        **get_kernel_kwargs,
    )
    if symmetric:
        kernel(A, A_natoms, A_ncharges, sigma, kernel_output)
    else:
        kernel(A, B, A_natoms, B_natoms, A_ncharges, B_ncharges, sigma, kernel_output)
    return kernel_output


# TODO: KK: Should probably be rewritten using the naming convention similarly to how __init__.py
# avoids writing out all functions that should be imported.

precompiled_laplacian_kernels = {}
laplacian_kernel_specifics = {}  # this is the default kernel


def laplacian_kernel(A, B, sigma, out=None):
    return calculate_global_kernel(A, B, sigma, precompiled_laplacian_kernels, out=out)


def laplacian_kernel_symmetric(A, sigma, out=None):
    return laplacian_kernel(A, None, sigma, out=out)


def local_laplacian_kernel(A, B, A_natoms, B_natoms, sigma, out=None):
    return calculate_local_kernel(
        A, B, A_natoms, B_natoms, sigma, precompiled_laplacian_kernels, out=out
    )


def local_laplacian_kernel_symmetric(A, A_natoms, sigma, out=None):
    return local_laplacian_kernel(A, None, A_natoms, None, sigma, out=out)


precompiled_gaussian_kernels = {}
gaussian_kernel_specifics = {"metric": "l2", "sigma_to_param": half_inv_sq_sigma}


def gaussian_kernel(A, B, sigma, out=None):
    return calculate_global_kernel(
        A, B, sigma, precompiled_gaussian_kernels, out=out, **gaussian_kernel_specifics
    )


def gaussian_kernel_symmetric(A, sigma, out=None):
    return gaussian_kernel(A, None, sigma, out=out)


def local_gaussian_kernel(A, B, A_natoms, B_natoms, sigma, out=None):
    return calculate_local_kernel(
        A,
        B,
        A_natoms,
        B_natoms,
        sigma,
        precompiled_gaussian_kernels,
        out=out,
        **gaussian_kernel_specifics,
    )


def local_gaussian_kernel_symmetric(A, A_natoms, sigma, out=None):
    return local_gaussian_kernel(A, None, A_natoms, None, sigma, out=out)


precompiled_matern_kernels = {}
matern_kernel_specifics = {"type": "Matern"}


def matern_kernel(A, B, sigma, out=None, order=0, metric="l1"):
    return calculate_global_kernel(
        A,
        B,
        sigma,
        precompiled_matern_kernels,
        order=order,
        metric=metric,
        out=out,
        **matern_kernel_specifics,
    )


def matern_kernel_symmetric(A, sigma, **kwargs):
    return matern_kernel(A, None, sigma, **kwargs)


def local_matern_kernel(A, B, A_natoms, B_natoms, sigma, out=None, order=0, metric="l1"):
    return calculate_local_kernel(
        A,
        B,
        A_natoms,
        B_natoms,
        sigma,
        precompiled_matern_kernels,
        order=order,
        metric=metric,
        out=out,
        **matern_kernel_specifics,
    )


def local_matern_kernel_symmetric(A, A_natoms, sigma, **kwargs):
    return local_matern_kernel(A, None, A_natoms, None, sigma, **kwargs)


# Local kernels with delta(n1-n2).
def local_dn_laplacian_kernel(A, B, A_natoms, B_natoms, A_ncharges, B_ncharges, sigma, out=None):
    return calculate_local_dn_kernel(
        A,
        B,
        A_natoms,
        B_natoms,
        A_ncharges,
        B_ncharges,
        sigma,
        precompiled_laplacian_kernels,
        out=out,
    )


def local_dn_laplacian_kernel_symmetric(A, A_natoms, A_ncharges, sigma, out=None):
    return local_dn_laplacian_kernel(A, None, A_natoms, None, A_ncharges, None, sigma, out=out)


def local_dn_gaussian_kernel(A, B, A_natoms, B_natoms, A_ncharges, B_ncharges, sigma, out=None):
    return calculate_local_dn_kernel(
        A,
        B,
        A_natoms,
        B_natoms,
        A_ncharges,
        B_ncharges,
        sigma,
        precompiled_gaussian_kernels,
        out=out,
        **gaussian_kernel_specifics,
    )


def local_dn_gaussian_kernel_symmetric(A, A_natoms, A_ncharges, sigma, out=None):
    return local_dn_gaussian_kernel(A, None, A_natoms, None, A_ncharges, None, sigma, out=out)


def local_dn_matern_kernel(
    A, B, A_natoms, B_natoms, A_ncharges, B_ncharges, sigma, out=None, order=0, metric="l1"
):
    return calculate_local_dn_kernel(
        A,
        B,
        A_natoms,
        B_natoms,
        A_ncharges,
        B_ncharges,
        sigma,
        precompiled_matern_kernels,
        out=out,
        order=order,
        metric=metric,
        **matern_kernel_specifics,
    )


def local_dn_matern_kernel_symmetric(A, A_natoms, A_ncharges, sigma, out=None, **kwargs):
    return local_dn_matern_kernel(
        A, None, A_natoms, None, A_ncharges, None, sigma, out=out, **kwargs
    )


# KK: introduced for making it easier to get symmetric and asymmetric versions of the same kernel
# in qml2.models. Probably should be revised to make it more intuitive and user-friendly.
def construct_laplacian_kernel(**kwargs):
    return get_kernel(precompiled_laplacian_kernels, **kwargs, **laplacian_kernel_specifics)


def construct_gaussian_kernel(**kwargs):
    return get_kernel(precompiled_gaussian_kernels, **kwargs, **gaussian_kernel_specifics)


def construct_matern_kernel(**kwargs):
    return get_kernel(precompiled_matern_kernels, **kwargs, **matern_kernel_specifics)
