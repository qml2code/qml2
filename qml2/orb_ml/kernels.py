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
    max_,
    mean_,
    prange_,
    sqrt_,
    sum_,
    where_,
    zeros_,
)
from ..kernels.kernels import l1_norm_dist, l2_norm_sq_dist
from .representations import OML_rep_params, component_id_mat_map

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
        kernel_matrix,
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

    return mol_mol_kernel_asymmetric


def construct_mol_mol_kernel_symmetric(orb_kernel_function):
    mol_prod = construct_mol_mol_kernel_function(orb_kernel_function)

    @jit_(numba_parallel=True)
    def mol_mol_kernel_symmetric(
        kernel_matrix,
        mol_scalar_reps,
        mol_orb_weights,
        rep_weights,
        mol_orb_ubounds,
        mol_rep_ubounds,
        orb_rep_ubounds,
        *orb_kern_args,
    ):
        nmols = mol_rep_ubounds.shape[0]
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

            for orb_id, orb in enumerate(mol_orbs):
                self.orb_weights.append(orb.rho)
                if self.pair_rep:
                    if orb_id < len(comp.comps[0].orb_reps):
                        self.orb_weights[-1] *= -1
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


class ConcatenatedPairKernelInput:
    """A list of Slater pairs as TWO single-determinant inputs rather than one merged one.

    `OML_KernelInput` puts both determinants of a pair into a single weighted orbital list
    and negates the first determinant's weights, so the kernel it produces is

        K(P, Q) = K00 - K01 - K10 + K11,

    Kab being the kernel between determinant a of P and determinant b of Q. That is the
    exact counterpart of `FJKRepresentationCalc(concatenate_components=False)`, whose
    feature vector is likewise a difference of the two determinants'.

    With `concatenate_components=True` the feature vector is instead the two determinants'
    vectors laid end to end, and the inner product of two concatenations is the sum of the
    per-block inner products, with no cross terms:

        K(P, Q) = K00 + K11.

    Hence this class: one ordinary `OML_KernelInput` holding determinant 0 of every pair,
    another holding determinant 1, and their kernels added.

    Splitting the pair changes nothing below it. Sigma rescaling is per feature and weight
    normalization is per ORBITAL (`normalize_orb_scalar_rep_weights`), so neither ever
    looked across determinants; and `OML_KernelInput` copies the scalar representations it
    is given, so the two halves rescale independently rather than twice over.

    The ORDER of the determinants matters exactly as it does in the representation: a
    training set assembled as (gas, solvated) cannot be used against pairs assembled the
    other way round.
    """

    def __init__(self, oml_slater_pair_list):
        for pair in oml_slater_pair_list:
            assert is_pair_rep(pair), (
                "ConcatenatedPairKernelInput needs Slater pairs to have determinants to "
                f"concatenate; got {type(pair).__name__}"
            )
            assert len(pair.comps) == 2, (
                "only two-determinant pairs are supported, "
                f"got one with {len(pair.comps)} determinants"
            )
        self.num_determinants = 2
        self.determinant_inputs = [
            OML_KernelInput([pair.comps[determinant_id] for pair in oml_slater_pair_list])
            for determinant_id in range(self.num_determinants)
        ]
        self.num_mols = len(oml_slater_pair_list)

    def init_temp_arrs(self, sigmas, norm="l2"):
        for determinant_input in self.determinant_inputs:
            determinant_input.init_temp_arrs(sigmas, norm=norm)


def get_kernel_input(compound_list, concatenate_components=False):
    if concatenate_components:
        return ConcatenatedPairKernelInput(compound_list)
    else:
        return OML_KernelInput(compound_list)


def kernel_from_processed_input(
    A_input: OML_KernelInput | ConcatenatedPairKernelInput,
    B_input: OML_KernelInput | ConcatenatedPairKernelInput | None,
    sigmas: ndarray,
    global_sigma: float = None,
    norm: str = "l2",
    out=None,
):
    """Kernel matrix, dispatching on how a Slater pair's determinants were combined.

    Kept as one entry point so that callers (`FJKModel` above all) need not know which of
    the two treatments they are holding.
    """
    if isinstance(A_input, ConcatenatedPairKernelInput):
        assert B_input is None or isinstance(B_input, ConcatenatedPairKernelInput), (
            "both sides of a concatenated-pair kernel have to be concatenated pairs; "
            f"got {type(B_input).__name__} on the right"
        )
        kernel_matrix = None
        for determinant_id in range(A_input.num_determinants):
            if B_input is None:
                B_determinant_input = None
            else:
                B_determinant_input = B_input.determinant_inputs[determinant_id]
            determinant_kernel = single_determinant_kernel_from_processed_input(
                A_input.determinant_inputs[determinant_id],
                B_determinant_input,
                sigmas,
                global_sigma=global_sigma,
                norm=norm,
                # `out` is filled by the first determinant and added to by the rest, so it
                # must not be handed to more than one of them.
                out=out if kernel_matrix is None else None,
            )
            if kernel_matrix is None:
                kernel_matrix = determinant_kernel
            else:
                kernel_matrix += determinant_kernel
        return kernel_matrix
    return single_determinant_kernel_from_processed_input(
        A_input, B_input, sigmas, global_sigma=global_sigma, norm=norm, out=out
    )


def single_determinant_kernel_from_processed_input(
    A_input: OML_KernelInput,
    B_input: OML_KernelInput | None,
    sigmas: ndarray,
    global_sigma: float = None,
    norm: str = "l2",
    out=None,
):
    A_input.init_temp_arrs(sigmas, norm=norm)
    symmetric = B_input is None
    if symmetric:
        kernel_matrix_shape = (A_input.num_mols, A_input.num_mols)
    else:
        B_input.init_temp_arrs(sigmas, norm=norm)
        kernel_matrix_shape = (A_input.num_mols, B_input.num_mols)
    if global_sigma is None:
        orb_product = "linear"
        kernel_args = ()
    else:
        orb_product = "gaussian"
        inv_sq_sigma = global_sigma ** (-2)
        kernel_args = (inv_sq_sigma,)
    kernel_func = get_kernel(orb_product=orb_product, symmetric=symmetric, norm=norm)
    if out is None:
        kernel_matrix = empty_(kernel_matrix_shape)
    else:
        kernel_matrix = out
        assert out.shape == kernel_matrix_shape
    if symmetric:
        kernel_func(
            kernel_matrix,
            A_input.scalar_reps,
            A_input.orb_weights,
            A_input.arep_weights,
            A_input.mol_orb_ubounds,
            A_input.mol_rep_ubounds,
            A_input.orb_rep_ubounds,
            *kernel_args,
        )
    else:
        kernel_func(
            kernel_matrix,
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
    return kernel_matrix


# Convenient interfaces.
def gaussian_kernel(A, B, sigmas, global_sigma, norm="l2", concatenate_components=False, out=None):
    A_input = get_kernel_input(A, concatenate_components=concatenate_components)
    B_input = get_kernel_input(B, concatenate_components=concatenate_components)
    return kernel_from_processed_input(
        A_input, B_input, sigmas, global_sigma=global_sigma, norm=norm, out=out
    )


def gaussian_kernel_symmetric(
    A, sigmas, global_sigma, norm="l2", concatenate_components=False, out=None
):
    A_input = get_kernel_input(A, concatenate_components=concatenate_components)
    return kernel_from_processed_input(
        A_input, None, sigmas, global_sigma=global_sigma, norm=norm, out=out
    )


def linear_kernel(A, B, sigmas, norm="l2", concatenate_components=False, out=None):
    A_input = get_kernel_input(A, concatenate_components=concatenate_components)
    B_input = get_kernel_input(B, concatenate_components=concatenate_components)
    return kernel_from_processed_input(A_input, B_input, sigmas, norm=norm, out=out)


def linear_kernel_symmetric(A, sigmas, norm="l2", concatenate_components=False, out=None):
    A_input = get_kernel_input(A, concatenate_components=concatenate_components)
    return kernel_from_processed_input(A_input, None, sigmas, norm=norm, out=out)


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


def rep_stddevs_from_kernel_input(compound_kernel_input):
    if isinstance(compound_kernel_input, ConcatenatedPairKernelInput):
        # Concatenation does not double the number of sigmas -- one set describes a single
        # determinant either way (see FJKRepresentationCalc.component_bounds) -- so the
        # spread is estimated on each determinant and the two averaged. Averaging standard
        # deviations is not the pooled standard deviation, but this is the INITIAL GUESS
        # that optimize_hyperparameters then scales by a fitted multiplier, so what matters
        # is the relative size of the components rather than the absolute value.
        determinant_stddevs = [
            rep_stddevs_from_kernel_input(determinant_input)
            for determinant_input in compound_kernel_input.determinant_inputs
        ]
        return sum(determinant_stddevs[1:], start=determinant_stddevs[0]) / len(
            determinant_stddevs
        )
    avs_av2s = find_vec_moments(
        compound_kernel_input.scalar_reps,
        compound_kernel_input.orb_weights,
        compound_kernel_input.arep_weights,
        compound_kernel_input.mol_orb_ubounds,
        compound_kernel_input.mol_rep_ubounds,
        compound_kernel_input.orb_rep_ubounds,
        array_([1, 2]),
    )
    separator_id = compound_kernel_input.scalar_reps.shape[-1]
    sigmas = sign_checked_sqrts(avs_av2s[separator_id:] - avs_av2s[:separator_id] ** 2)
    return sigmas


def rep_stddevs(oml_compound_list=None, compound_kernel_input=None, concatenate_components=False):
    """Spread of each representation component, for the initial sigma guess.

    `concatenate_components` has to match the kernel the sigmas are destined for. On a
    Slater pair list the default builds the DIFFERENCE input, whose orbital weights are
    negated for the first determinant, so the moments it reports are those of a quantity the
    concatenated kernel never forms.
    """
    if compound_kernel_input is None:
        compound_kernel_input = get_kernel_input(
            oml_compound_list, concatenate_components=concatenate_components
        )
    return rep_stddevs_from_kernel_input(compound_kernel_input)


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


def sigmas_redivied_by_mats(
    oml_rep_params: OML_rep_params,
    oml_compound_list=None,
    compound_kernel_input=None,
    stddevs=None,
    norm="l2",
    concatenate_components=False,
):
    if stddevs is None:
        stddevs = rep_stddevs(
            oml_compound_list=oml_compound_list,
            compound_kernel_input=compound_kernel_input,
            concatenate_components=concatenate_components,
        )
    comp_mat_map = component_id_mat_map(oml_rep_params)
    num_comps = max(max_(comp_ids) for comp_ids in comp_mat_map) + 1
    sigmas = empty_(num_comps)
    nmats = len(comp_mat_map)
    for comp_ids in comp_mat_map:
        mat_sigmas = stddevs[comp_ids]
        if norm == "l2":
            final_mat_sigma = sqrt_(sum_(mat_sigmas**2) * nmats)
        else:
            final_mat_sigma = sum_(mat_sigmas) * nmats
        sigmas[comp_ids] = final_mat_sigma
    return sigmas
