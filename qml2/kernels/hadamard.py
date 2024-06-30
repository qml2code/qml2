# NOTE: KK: At some point I considered making SORF diagonals bool_ class
# since they effectively correspond to changing or keeping the sign.
# Decided against that in case we start playing with SORF values beyond -1 and +1.
from typing import Tuple

from ..dimensionality_reduction import (
    project_scale_local_representations,
    project_scale_representations,
)
from ..jit_interfaces import (
    array_jittable_,
    bool_,
    cos_,
    dim0float_array_,
    empty_,
    int_,
    jit_,
    max_,
    ndarray_,
    optional_ndarray_,
    pi_,
    prange_,
    random_array_from_rng_,
    save_,
    sign_,
    sqrt_,
)
from ..utils import get_atom_environment_ranges, get_element_ids_from_sorted


# For creating SORF features.
@jit_
def create_SORF_matrices(
    nfeature_stacks: int, ntransforms: int, npcas: int, pi_: dim0float_array_ = pi_, rng=None
) -> Tuple[ndarray_, ndarray_]:
    biases = random_array_from_rng_((nfeature_stacks, npcas), rng=rng) * 2 * pi_
    sorf_diags = sign_(
        random_array_from_rng_((nfeature_stacks, ntransforms, npcas), rng=rng) - 0.5
    )
    return biases, sorf_diags


@jit_
def create_SORF_matrices_diff_species(
    nfeature_stacks: int,
    nspecies: int,
    ntransforms: int,
    npcas: int,
    pi_: dim0float_array_ = pi_,
    rng=None,
) -> Tuple[ndarray_, ndarray_]:
    biases = random_array_from_rng_((nfeature_stacks, nspecies, npcas), rng=rng) * 2 * pi_
    sorf_diags = sign_(
        random_array_from_rng_((nfeature_stacks, nspecies, ntransforms, npcas), rng=rng) - 0.5
    )
    return biases, sorf_diags


# Appears as: (1) fast Hadamard transform normalization constant.
#             (2) SORF matrix multiplier in Eq.~(20) of the QML-Lightning arXiv.
@jit_
def hadamard_norm_const(npcas: int_) -> ndarray_:
    return 1.0 / sqrt_(array_jittable_(float(npcas)))


# Normalization of the Random Fourier Feature vector as appearing in Eq.~(14) of the QML-Lightning arXiv.
@jit_
def rff_vec_norm_const(nfeatures: int_) -> ndarray_:
    return sqrt_(2.0 / array_jittable_(nfeatures))


# Taken from https://en.wikipedia.org/wiki/Fast_Walsh%E2%80%93Hadamard_transform.
@jit_
def fast_walsh_hadamard(
    array: ndarray_, norm_const: dim0float_array_, forward: bool_ = True
) -> None:
    """
    Unnormalized fast Walsh-Hadamard transform. For normalization divide by sqrt(a.shape[0])
    """
    n = array.shape[0]
    if forward:
        h = 1
    else:
        h = n // 2
    while True:
        for i in range(0, n, h * 2):
            for j in range(i, i + h):
                x = array[j]
                y = array[j + h]
                array[j] = x + y
                array[j + h] = x - y
        if forward:
            h *= 2
            if h >= n:
                break
        else:
            if h == 1:
                break
            h = h // 2
    array[:] *= norm_const


@jit_
def hadamard_rff_kernel_stack_phases(
    red_scaled_reps, temp_red_reps, sorf_diags, biases, norm_const
) -> None:
    Ntransforms = sorf_diags.shape[0]
    if red_scaled_reps.shape[0] == temp_red_reps[:].shape[0]:
        temp_red_reps[:] = red_scaled_reps[:]
    else:
        size = red_scaled_reps.shape[0]
        temp_red_reps[:size] = red_scaled_reps[:]
        temp_red_reps[size:] = 0.0
    for transform in range(Ntransforms):
        temp_red_reps *= sorf_diags[transform]
        fast_walsh_hadamard(temp_red_reps, norm_const)
    # appears from Eq. (20) in QML-Lightning arxiv.
    temp_red_reps /= norm_const
    # add biases
    temp_red_reps += biases


@jit_
def hadamard_rff_kernel_stack(
    red_scaled_reps, temp_red_reps, sorf_diags, biases, norm_const
) -> ndarray_:
    hadamard_rff_kernel_stack_phases(
        red_scaled_reps, temp_red_reps, sorf_diags, biases, norm_const
    )
    return cos_(save_(temp_red_reps))


@jit_
def local_hadamard_kernel_func(
    red_scaled_reps,
    temp_red_reps,
    element_ids,
    sorf_diags,
    biases,
    norm_const,
    natoms,
    kernel_component,
):
    kernel_component[:] = 0.0
    for i_atom in range(int(natoms)):
        el_id = int(element_ids[i_atom])
        kernel_component[:] += hadamard_rff_kernel_stack(
            red_scaled_reps[i_atom],
            temp_red_reps,
            sorf_diags[el_id],
            biases[el_id],
            norm_const,
        )


@jit_(numba_parallel=True)
def hadamard_kernel_processed_input(
    reduced_scaled_representations,
    sorf_diags,
    biases,
    kernel,
    nfeature_stacks: int,
    npcas: int,
):
    norm_const = hadamard_norm_const(npcas)

    nmols = reduced_scaled_representations.shape[0]
    nfeatures = kernel.shape[1]
    assert nfeature_stacks * npcas == nfeatures

    for feature_stack in prange_(nfeature_stacks):
        temp_red_reps = empty_((npcas,))
        lb_rff = npcas * feature_stack
        ub_rff = lb_rff + npcas
        for mol_id in range(nmols):
            kernel[mol_id, lb_rff:ub_rff] = hadamard_rff_kernel_stack(
                reduced_scaled_representations[mol_id],
                temp_red_reps,
                sorf_diags[feature_stack],
                biases[feature_stack],
                norm_const,
            )
    kernel[:, :] *= rff_vec_norm_const(nfeatures)  # normalization


@jit_(numba_parallel=True)
def local_hadamard_kernel_processed_input(
    reduced_scaled_representations,
    element_ids,
    all_sorf_diags,
    all_biases,
    kernel,
    ubound_arr,
    nfeature_stacks: int,
    npcas: int,
):
    assert reduced_scaled_representations.shape[0] == element_ids.shape[0]
    assert all_sorf_diags.shape[0] == all_biases.shape[0]
    # KK: not %100 sure it's necessary
    assert max_(element_ids) <= all_sorf_diags.shape[0]

    norm_const = hadamard_norm_const(npcas)

    nmols = ubound_arr.shape[0] - 1
    assert nmols == kernel.shape[0]
    nfeatures = kernel.shape[1]
    assert nfeature_stacks * npcas == nfeatures

    for feature_stack in prange_(nfeature_stacks):
        temp_red_reps = empty_((npcas,))
        lb_rff = npcas * feature_stack
        ub_rff = lb_rff + npcas
        for mol_id in range(nmols):
            lb_mol = ubound_arr[mol_id]
            ub_mol = ubound_arr[mol_id + 1]
            natoms = ub_mol - lb_mol
            local_hadamard_kernel_func(
                reduced_scaled_representations[lb_mol:ub_mol],
                temp_red_reps,
                element_ids[lb_mol:ub_mol],
                all_sorf_diags[feature_stack],
                all_biases[feature_stack],
                norm_const,
                natoms,
                kernel[mol_id, lb_rff:ub_rff],
            )
    kernel[:, :] *= rff_vec_norm_const(nfeatures)  # normalization


@jit_
def hadamard_kernel(
    representations,
    sorf_diags,
    biases,
    kernel,
    sigma: dim0float_array_,
    nfeature_stacks: int,
    npcas: int,
    reductor: optional_ndarray_ = None,
):
    if reductor is None:
        assert representations.shape[1] <= npcas
        reduced_scaled_representations = representations / sigma
    else:
        reduced_scaled_representations = project_scale_representations(
            representations, reductor, sigma
        )
    hadamard_kernel_processed_input(
        reduced_scaled_representations,
        sorf_diags,
        biases,
        kernel,
        nfeature_stacks,
        npcas,
    )


@jit_
def local_hadamard_kernel(
    representations,
    ncharges,
    na,
    sorted_elements,
    all_sorf_diags,
    all_biases,
    kernel,
    sigma: dim0float_array_,
    nfeature_stacks: int,
    npcas: int,
    reductors: optional_ndarray_ = None,
):
    ubound_arr = get_atom_environment_ranges(na)
    all_element_ids = get_element_ids_from_sorted(ncharges, sorted_elements)
    if reductors is None:
        assert representations.shape[1] <= npcas
        reduced_scaled_representations = representations / sigma
    else:
        reduced_scaled_representations = project_scale_local_representations(
            representations, all_element_ids, reductors, sigma
        )
    local_hadamard_kernel_processed_input(
        reduced_scaled_representations,
        all_element_ids,
        all_sorf_diags,
        all_biases,
        kernel,
        ubound_arr,
        nfeature_stacks,
        npcas,
    )
