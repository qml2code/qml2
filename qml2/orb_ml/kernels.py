import itertools
from typing import Tuple

from numpy import ndarray

from ..jit_interfaces import (
    abs_,
    array_,
    copy_,
    copy_detached_,
    dint_,
    dtype_,
    empty_,
    exp_,
    int_,
    jit_,
    mean_,
    prange_,
    sqrt_,
    sum_,
    where_,
    zeros_,
)
from ..kernels.kernels import l1_norm_dist, l2_norm_sq_dist

dist_function_dict = {"l1": l1_norm_dist, "l2": l2_norm_sq_dist}


def construct_orb_linear_kernel_function(dist_func):
    @jit_
    def orb_linear_kernel_function(scalar_reps1, scalar_reps2, weights1, weights2):
        product = 0.0
        nreps1 = scalar_reps1.shape[0]
        nreps2 = scalar_reps2.shape[0]
        for i1 in range(nreps1):
            for i2 in range(nreps2):
                product += (
                    exp_(-dist_func(scalar_reps1[i1], scalar_reps2[i2]))
                    * weights1[i1]
                    * weights2[i2]
                )
        return product

    return orb_linear_kernel_function


def construct_orb_gaussian_kernel_function(dist_func):
    orb_linear_kernel_function = construct_orb_linear_kernel_function(dist_func)

    @jit_
    def orb_gaussian_kernel_function(
        scalar_reps1, scalar_reps2, weights1, weights2, inv_sq_global_sigma
    ):
        linear_product = orb_linear_kernel_function(
            scalar_reps1,
            scalar_reps2,
            weights1,
            weights2,
        )
        return exp_(-(1.0 - linear_product) * inv_sq_global_sigma)

    return orb_gaussian_kernel_function


@jit_
def get_lubound_from_ubounds(i: int_, ubounds_arr) -> Tuple[int_, int_]:
    if i == 0:
        lbound = 0
    else:
        lbound = int(ubounds_arr[i - 1])
    return lbound, ubounds_arr[i]


def construct_mol_mol_kernel_function(orb_kernel_function):
    @jit_
    def mol_mol_dot_product(
        mol_scalar_reps1,
        mol_scalar_reps2,
        mol_orb_weights1,
        mol_orb_weights2,
        rep_weights1,
        rep_weights2,
        orb_rep_ubounds1,
        orb_rep_ubounds2,
        *orb_kern_args,
    ):
        mol_prod = 0.0
        num_orbs1 = mol_orb_weights1.shape[0]
        num_orbs2 = mol_orb_weights2.shape[0]
        for i_orb1 in range(num_orbs1):
            lbound1, ubound1 = get_lubound_from_ubounds(i_orb1, orb_rep_ubounds1)
            for i_orb2 in range(num_orbs2):
                lbound2, ubound2 = get_lubound_from_ubounds(i_orb2, orb_rep_ubounds2)
                mol_prod += (
                    orb_kernel_function(
                        mol_scalar_reps1[lbound1:ubound1],
                        mol_scalar_reps2[lbound2:ubound2],
                        rep_weights1[lbound1:ubound1],
                        rep_weights2[lbound2:ubound2],
                        *orb_kern_args,
                    )
                    * mol_orb_weights1[i_orb1]
                    * mol_orb_weights2[i_orb2]
                )
        return mol_prod

    return mol_mol_dot_product


def construct_mol_mol_kernel_asymmetric(orb_kernel_function):
    mol_prod = construct_mol_mol_kernel_function(orb_kernel_function)

    @jit_(numba_parallel=True)
    def mol_mol_kernel_asymmetric(
        mol_scalar_reps1,
        mol_scalar_reps2,
        mol_orb_weights1,
        mol_orb_weights2,
        rep_weights1,
        rep_weights2,
        mol_orb_ubounds1,
        mol_orb_ubounds2,
        mol_rep_ubounds1,
        mol_rep_ubounds2,
        orb_rep_ubounds1,
        orb_rep_ubounds2,
        *orb_kern_args,
    ):
        nmols_A = mol_rep_ubounds1.shape[0]
        nmols_B = mol_rep_ubounds2.shape[0]
        kernel_matrix = empty_((nmols_A, nmols_B))
        for i_A in prange_(nmols_A):
            lbound_orb1, ubound_orb1 = get_lubound_from_ubounds(i_A, mol_orb_ubounds1)
            lbound_rep1, ubound_rep1 = get_lubound_from_ubounds(i_A, mol_rep_ubounds1)
            for i_B in range(nmols_B):
                lbound_orb2, ubound_orb2 = get_lubound_from_ubounds(i_B, mol_orb_ubounds2)
                lbound_rep2, ubound_rep2 = get_lubound_from_ubounds(i_B, mol_rep_ubounds2)
                kernel_matrix[i_A, i_B] = mol_prod(
                    mol_scalar_reps1[lbound_rep1:ubound_rep1],
                    mol_scalar_reps2[lbound_rep2:ubound_rep2],
                    mol_orb_weights1[lbound_orb1:ubound_orb1],
                    mol_orb_weights2[lbound_orb2:ubound_orb2],
                    rep_weights1[lbound_rep1:ubound_rep1],
                    rep_weights2[lbound_rep2:ubound_rep2],
                    orb_rep_ubounds1[lbound_orb1:ubound_orb1],
                    orb_rep_ubounds2[lbound_orb2:ubound_orb2],
                    *orb_kern_args,
                )
        return kernel_matrix

    return mol_mol_kernel_asymmetric


def construct_mol_mol_kernel_symmetric(orb_kernel_function):
    mol_prod = construct_mol_mol_kernel_function(orb_kernel_function)

    @jit_(numba_parallel=True)
    def mol_mol_kernel_symmetric(
        mol_scalar_reps,
        mol_orb_weights,
        rep_weights,
        mol_orb_ubounds,
        mol_rep_ubounds,
        orb_rep_ubounds,
        *orb_kern_args,
    ):
        nmols = mol_rep_ubounds.shape[0]
        kernel_matrix = empty_((nmols, nmols))
        for i1 in prange_(nmols):
            lbound_orb1, ubound_orb1 = get_lubound_from_ubounds(i1, mol_orb_ubounds)
            lbound_rep1, ubound_rep1 = get_lubound_from_ubounds(i1, mol_rep_ubounds)
            for i2 in range(i1 + 1):
                lbound_orb2, ubound_orb2 = get_lubound_from_ubounds(i2, mol_orb_ubounds)
                lbound_rep2, ubound_rep2 = get_lubound_from_ubounds(i2, mol_rep_ubounds)
                kernel_matrix[i1, i2] = mol_prod(
                    mol_scalar_reps[lbound_rep1:ubound_rep1],
                    mol_scalar_reps[lbound_rep2:ubound_rep2],
                    mol_orb_weights[lbound_orb1:ubound_orb1],
                    mol_orb_weights[lbound_orb2:ubound_orb2],
                    rep_weights[lbound_rep1:ubound_rep1],
                    rep_weights[lbound_rep2:ubound_rep2],
                    orb_rep_ubounds[lbound_orb1:ubound_orb1],
                    orb_rep_ubounds[lbound_orb2:ubound_orb2],
                    *orb_kern_args,
                )
                kernel_matrix[i2, i1] = kernel_matrix[i1, i2]
        return kernel_matrix

    return mol_mol_kernel_symmetric


orb_kernel_contructor_dict = {
    "gaussian": construct_orb_gaussian_kernel_function,
    "linear": construct_orb_linear_kernel_function,
}
kernel_constructor_dict = {
    False: construct_mol_mol_kernel_asymmetric,
    True: construct_mol_mol_kernel_symmetric,
}


def construct_kernel(orb_product="gaussian", symmetric=False, norm="l2"):
    orb_kernel = orb_kernel_contructor_dict[orb_product](dist_function_dict[norm])
    kernel_constructor = kernel_constructor_dict[symmetric]
    return kernel_constructor(orb_kernel)


precompiled_kernels = {}


def get_kernel(orb_product="gaussian", symmetric=False, norm="l2"):
    kernel_def_tuple = (orb_product, norm)
    if symmetric not in precompiled_kernels:
        precompiled_kernels[symmetric] = {}
    if kernel_def_tuple not in precompiled_kernels[symmetric]:
        precompiled_kernels[symmetric][kernel_def_tuple] = construct_kernel(
            orb_product=orb_product, symmetric=symmetric, norm=norm
        )
    return precompiled_kernels[symmetric][kernel_def_tuple]


# For preparing temporary arrays for kernel calculation.
def construct_weight_normalization(dist_func):
    orb_linear_kernel_function = construct_orb_linear_kernel_function(dist_func)

    @jit_
    def normalize_orb_scalar_rep_weights(
        scalar_reps,
        rep_weights,
    ):
        sq_norm = orb_linear_kernel_function(
            scalar_reps,
            scalar_reps,
            rep_weights,
            rep_weights,
        )
        norm_mult = sqrt_(sq_norm) ** (-1)
        rep_weights *= norm_mult

    @jit_
    def normalize_mol_scalar_rep_weights(scalar_reps, rep_weights, orb_rep_ubounds):
        norbs = orb_rep_ubounds.shape[0]
        for i_orb in range(norbs):
            lbound_orb, ubound_orb = get_lubound_from_ubounds(i_orb, orb_rep_ubounds)
            normalize_orb_scalar_rep_weights(
                scalar_reps[lbound_orb:ubound_orb],
                rep_weights[lbound_orb:ubound_orb],
            )

    @jit_(numba_parallel=True)
    def normalize_all_mol_scalar_rep_weights(
        mol_scalar_reps,
        rep_weights,
        mol_orb_ubounds,
        mol_rep_ubounds,
        orb_rep_ubounds,
    ):
        nmols = mol_rep_ubounds.shape[0]
        for i_mol in prange_(nmols):
            lbound_orb, ubound_orb = get_lubound_from_ubounds(i_mol, mol_orb_ubounds)
            lbound_rep, ubound_rep = get_lubound_from_ubounds(i_mol, mol_rep_ubounds)
            normalize_mol_scalar_rep_weights(
                mol_scalar_reps[lbound_rep:ubound_rep],
                rep_weights[lbound_rep:ubound_rep],
                orb_rep_ubounds[lbound_orb:ubound_orb],
            )

    return normalize_all_mol_scalar_rep_weights


weight_normalizations = {}


def get_weight_normalization(norm):
    global weight_normalizations
    if norm not in weight_normalizations:
        dist_func = dist_function_dict[norm]
        weight_normalizations[norm] = construct_weight_normalization(dist_func)
    return weight_normalizations[norm]


@jit_(numba_parallel=True)
def rescale_scalar_reps(mol_scalar_reps, sigmas, add_multiplier: float):
    # K.Karandashev: wrote this assuming * is faster than /, but not %100 sure how it's with current NumBa.
    inv_sigmas = add_multiplier / sigmas
    for rep_id in prange_(mol_scalar_reps.shape[0]):
        mol_scalar_reps[rep_id] *= inv_sigmas


def is_pair_rep(comp):
    return hasattr(comp, "comps")


def iterated_orb_reps(oml_comp, pair_rep=None, single_orb_list=False):
    if pair_rep is None:
        pair_rep = is_pair_rep(oml_comp)
    if pair_rep:
        return list(itertools.chain(oml_comp.comps[0].orb_reps, oml_comp.comps[1].orb_reps))
    else:
        if single_orb_list:
            return [oml_comp]
        else:
            return oml_comp.orb_reps


class OML_KernelInput:
    def __init__(self, oml_compound_list):
        self.pair_rep = is_pair_rep(oml_compound_list[0])
        self.temp_arrs_initialized_for = None

        self.scalar_reps = []
        self.orb_weights = []
        self.arep_weights = []
        self.mol_orb_ubounds = []
        self.mol_rep_ubounds = []
        self.orb_rep_ubounds = []

        self.num_mols = len(oml_compound_list)
        self.mol_orb_ubounds = empty_(self.num_mols, dtype=int_)
        mol_orb_ubound = 0
        mol_rep_ubound = 0

        for comp_id, comp in enumerate(oml_compound_list):
            mol_orbs = iterated_orb_reps(comp, pair_rep=self.pair_rep)
            mol_orb_ubound += len(mol_orbs)
            self.mol_orb_ubounds[comp_id] = mol_orb_ubound

            orb_rep_ubound = 0
            num_mol_areps = 0

            for orb in mol_orbs:
                self.orb_weights.append(orb.rho)
                areps = orb.orb_atom_reps
                nareps = len(areps)
                orb_rep_ubound += nareps
                num_mol_areps += nareps
                self.orb_rep_ubounds.append(orb_rep_ubound)
                for arep in orb.orb_atom_reps:
                    self.scalar_reps.append(copy_detached_(arep.scalar_reps))
                    self.arep_weights.append(arep.rho)

            mol_rep_ubound += num_mol_areps
            self.mol_rep_ubounds.append(mol_rep_ubound)

        self.scalar_reps = array_(self.scalar_reps)
        self.orb_weights = array_(self.orb_weights)
        self.arep_weights = array_(self.arep_weights)
        self.mol_orb_ubounds = array_(self.mol_orb_ubounds)
        self.mol_rep_ubounds = array_(self.mol_rep_ubounds)
        self.orb_rep_ubounds = array_(self.orb_rep_ubounds)

    def init_temp_arrs(self, sigmas, norm="l2"):
        if (self.temp_arrs_initialized_for is not None) and (
            self.temp_arrs_initialized_for == norm
        ):
            return
        assert all(sigmas > 0.0)
        self.temp_arrs_initialized_for = norm
        if norm == "l2":
            add_multiplier = 0.5
        else:
            add_multiplier = 1.0
        rescale_scalar_reps(self.scalar_reps, sigmas, add_multiplier)
        get_weight_normalization(norm)(
            self.scalar_reps,
            self.arep_weights,
            self.mol_orb_ubounds,
            self.mol_rep_ubounds,
            self.orb_rep_ubounds,
        )


def kernel_from_processed_input(
    A_input: OML_KernelInput,
    B_input: OML_KernelInput | None,
    sigmas: ndarray,
    global_sigma: float = None,
    norm: str = "l2",
):
    A_input.init_temp_arrs(sigmas, norm=norm)
    symmetric = B_input is None
    if not symmetric:
        B_input.init_temp_arrs(sigmas, norm=norm)
    if global_sigma is None:
        orb_product = "linear"
        kernel_args = ()
    else:
        orb_product = "gaussian"
        inv_sq_sigma = global_sigma ** (-2)
        kernel_args = (inv_sq_sigma,)
    kernel_func = get_kernel(orb_product=orb_product, symmetric=symmetric, norm=norm)
    if symmetric:
        return kernel_func(
            A_input.scalar_reps,
            A_input.orb_weights,
            A_input.arep_weights,
            A_input.mol_orb_ubounds,
            A_input.mol_rep_ubounds,
            A_input.orb_rep_ubounds,
            *kernel_args,
        )
    else:
        return kernel_func(
            A_input.scalar_reps,
            B_input.scalar_reps,
            A_input.orb_weights,
            B_input.orb_weights,
            A_input.arep_weights,
            B_input.arep_weights,
            A_input.mol_orb_ubounds,
            B_input.mol_orb_ubounds,
            A_input.mol_rep_ubounds,
            B_input.mol_rep_ubounds,
            A_input.orb_rep_ubounds,
            B_input.orb_rep_ubounds,
            *kernel_args,
        )


# Convenient interfaces.
def gaussian_kernel(A, B, sigmas, global_sigma, norm="l2"):
    A_input = OML_KernelInput(A)
    B_input = OML_KernelInput(B)
    return kernel_from_processed_input(
        A_input, B_input, sigmas, global_sigma=global_sigma, norm=norm
    )


def gaussian_kernel_symmetric(A, sigmas, global_sigma, norm="l2"):
    A_input = OML_KernelInput(A)
    return kernel_from_processed_input(A_input, None, sigmas, global_sigma=global_sigma, norm=norm)


def linear_kernel(A, B, sigmas, norm="l2"):
    A_input = OML_KernelInput(A)
    B_input = OML_KernelInput(B)
    return kernel_from_processed_input(A_input, B_input, sigmas, norm=norm)


def linear_kernel_symmetric(A, sigmas, norm="l2"):
    A_input = OML_KernelInput(A)
    return kernel_from_processed_input(A_input, None, sigmas, norm=norm)


# For the hyperparameter initial guess.
@jit_
def find_mol_vec_moments_wnorm(
    mol_moments_arr,
    scalar_reps,
    orb_weights,
    rep_weights,
    orb_rep_ubounds,
    moments,
    moment_ubounds,
):
    mol_moments_arr[:] = 0.0
    num_moments = moments.shape[0]
    norm = 0.0
    norbs = orb_rep_ubounds.shape[0]
    for i_orb in range(norbs):
        orb_weight = orb_weights[i_orb]
        lbound_orb, ubound_orb = get_lubound_from_ubounds(i_orb, orb_rep_ubounds)
        for i_rep in range(lbound_orb, ubound_orb):
            weight = abs_(orb_weight * rep_weights[i_rep])
            norm += weight
            for moment_id in range(num_moments):
                moment_lbound, moment_ubound = get_lubound_from_ubounds(moment_id, moment_ubounds)
                mol_moments_arr[moment_lbound:moment_ubound] += weight * (
                    scalar_reps[i_rep] ** moments[moment_id]
                )
    return norm


@jit_(numba_parallel=True)
def find_vec_moments(
    scalar_reps,
    orb_weights,
    rep_weights,
    mol_orb_ubounds,
    mol_rep_ubounds,
    orb_rep_ubounds,
    moments,
    dint_: dtype_ = dint_,
):
    nmols = mol_rep_ubounds.shape[0]
    nmoments = moments.shape[0]
    repsize = scalar_reps.shape[-1]
    mol_vec_moments = empty_((nmols, nmoments * repsize))
    mol_moment_norm = empty_((nmols,))

    moment_ubounds = empty_((nmoments,), dtype=dint_)
    for mom_id in range(nmoments):
        moment_ubounds[mom_id] = (mom_id + 1) * repsize

    for i_mol in prange_(nmols):
        mol_orb_lbound, mol_orb_ubound = get_lubound_from_ubounds(i_mol, mol_orb_ubounds)
        mol_rep_lbound, mol_rep_ubound = get_lubound_from_ubounds(i_mol, mol_rep_ubounds)
        mol_moment_norm[i_mol] = find_mol_vec_moments_wnorm(
            mol_vec_moments[i_mol],
            scalar_reps[mol_rep_lbound:mol_rep_ubound],
            orb_weights[mol_orb_lbound:mol_orb_ubound],
            rep_weights[mol_rep_lbound:mol_rep_ubound],
            orb_rep_ubounds[mol_orb_lbound:mol_orb_ubound],
            moments,
            moment_ubounds,
        )

    tot_moment_norm = sum_(mol_moment_norm)
    unnormed_moments = sum_(mol_vec_moments, axis=0)
    return unnormed_moments / tot_moment_norm


@jit_
def sign_checked_sqrts(vec):
    n = vec.shape[0]
    out_vec = zeros_((n,))
    for i in range(n):
        if vec[i] > 0.0:
            out_vec[i] = sqrt_(vec[i])
    return out_vec


def rep_stddevs(oml_compound_list):
    kernel_input = OML_KernelInput(oml_compound_list)
    avs_av2s = find_vec_moments(
        kernel_input.scalar_reps,
        kernel_input.orb_weights,
        kernel_input.arep_weights,
        kernel_input.mol_orb_ubounds,
        kernel_input.mol_rep_ubounds,
        kernel_input.orb_rep_ubounds,
        array_([1, 2]),
    )
    separator_id = kernel_input.scalar_reps.shape[-1]
    sigmas = sign_checked_sqrts(avs_av2s[separator_id:] - avs_av2s[:separator_id] ** 2)
    return sigmas


def renormed_smoothened_sigmas(sigmas, relatively_small_val=1.0e-3, norm="l2"):
    """
    Find sigma parameters that are too small and replace them with the average.
    Used if some components were zero in representations put into rep_stddevs.
    """
    new_sigmas = copy_(sigmas)
    nsigmas = new_sigmas.shape[0]
    match norm:
        case "l1":
            new_sigmas *= nsigmas
        case "l2":
            new_sigmas *= sqrt_(float(nsigmas))
    total_mean = mean_(sigmas)
    lb = relatively_small_val * total_mean
    small_ids = where_(sigmas <= lb)[0]
    nsmall_ids = small_ids.shape[0]
    if nsmall_ids == 0:
        return new_sigmas
    print("WARNING: had to smoothen", nsmall_ids, "sigma values.")
    large_ids = where_(sigmas > lb)[0]
    large_sigma_mean = mean_(sigmas[large_ids])
    new_sigmas[small_ids] = large_sigma_mean
    return new_sigmas
