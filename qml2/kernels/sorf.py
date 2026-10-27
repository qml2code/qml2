# NOTE: KK: At some point I considered making SORF diagonals bool_ class
# since they effectively correspond to changing or keeping the sign.
# Decided against that in case we start playing with SORF values beyond -1 and +1.
from typing import Tuple, Union

from ..basic_utils import checked_logical_environ_val
from ..dimensionality_reduction import (
    project_scale_local_representations,
    project_scale_representations,
)
from ..jit_interfaces import (
    OptionalGenerator_,
    abs_,
    array_jittable_,
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
from ..utils import check_allocation, get_atom_environment_ranges, get_element_ids_from_sorted
from .walsh_hadamard import unnormalized_fast_walsh_hadamard

avoid_sorf_numba_parallelization_env_var_name = "QML2_AVOID_SORF_NUMBA_PARALLELIZATION"


def allow_sorf_numba_parallelization():
    """
    This function is used as the `numba_parallel` value in `jit_` instances for routines where `prange` loops over feature vectors whose evaluation potentially includes calculating dot product with a reductor matrix.
    """
    return not checked_logical_environ_val(
        avoid_sorf_numba_parallelization_env_var_name, default_answer=False
    )


# For creating SORF.
@jit_
def rand_sign_polynom(r):
    """
    Function that looks like sign(r-0.5) for r in [0,1], but is continuous. Used instead of sign(r-0.5) in test mode to make tests more stable w.r.t. RNG input.

    Arguments:
        r (float) - typically a randomly chosen value in [0, 1].
    """
    d = 0.5 - r
    return (16 * (0.5 - abs_(d)) ** 4 - 1) * sign_(d)


@jit_
def gen_rand_signs(output_shape, rng=None, test_mode=False):
    rand_values = random_array_from_rng_(output_shape, rng=rng)
    if test_mode:
        return rand_sign_polynom(rand_values)
    else:
        return sign_(rand_values - 0.5)


@jit_
def create_sorf_matrices(
    nfeature_stacks: int_,
    ntransforms: int_,
    init_size: int_,
    pi_: dim0float_array_ = pi_,
    rng: OptionalGenerator_ = None,
    test_mode: bool = False,
) -> Tuple[ndarray_, ndarray_]:
    """
    Create biases and diagonal elements appearing in the SORF expression.

    nfeature_stacks - number of "stacks" of features (each stack generated from a single copy of initial representations).
    ntranforms - number of Hadamard transform - SORF diagonal multiplication stages.
    init_size - initial size of vectors to which SORF transform is applied.
    rng (numpy.random.Generator or None) - if not None is used to create output arrays; otherwise numpy.random RNG is used.
    """
    biases = random_array_from_rng_((nfeature_stacks, init_size), rng=rng) * 2 * pi_
    sorf_diags = gen_rand_signs(
        (nfeature_stacks, ntransforms, init_size), rng=rng, test_mode=test_mode
    )
    return biases, sorf_diags


@jit_
def create_sorf_matrices_diff_species(
    nfeature_stacks: int,
    nspecies: int,
    ntransforms: int,
    init_size: int,
    pi_: dim0float_array_ = pi_,
    rng: OptionalGenerator_ = None,
    test_mode: bool = False,
) -> Tuple[ndarray_, ndarray_]:
    """
    Create biases and diagonal elements appearing in the SORF expression.

    nfeature_stacks - number of "stacks" of features (each stack generated from a single copy of initial representations).
    nspecies - number of distinct atomic species encountered in molecules of interest.
    ntranforms - number of Hadamard transform - SORF diagonal multiplication stages.
    init_size - initial size of vectors to which SORF transform is applied.
    rng (numpy.random.Generator or None) - if not None is used to create output arrays; otherwise numpy.random RNG is used.
    """

    biases = random_array_from_rng_((nfeature_stacks, nspecies, init_size), rng=rng) * 2 * pi_
    sorf_diags = gen_rand_signs(
        (nfeature_stacks, nspecies, ntransforms, init_size), rng=rng, test_mode=test_mode
    )
    return biases, sorf_diags


# Appears as: (1) fast Hadamard transform normalization constant.
#             (2) SORF matrix multiplier in Eq.~(20) of the QML-Lightning arXiv.
@jit_
def hadamard_norm_const(init_size: int_) -> ndarray_:
    return 1.0 / sqrt_(array_jittable_(float(init_size)))


# Normalization of the Random Fourier Feature vector as appearing in Eq.~(14) of the QML-Lightning arXiv.
@jit_
def rff_vec_norm_const(nfeatures: int_) -> ndarray_:
    return sqrt_(2.0 / array_jittable_(float(nfeatures)))


@jit_
def generate_sorf_stack_unbiased_phases(
    red_scaled_reps, temp_red_reps, sorf_diags, norm_const
) -> None:
    Ntransforms = sorf_diags.shape[0]
    if red_scaled_reps.shape[0] == temp_red_reps[:].shape[0]:
        temp_red_reps[:] = red_scaled_reps[:]
    else:
        size = red_scaled_reps.shape[0]
        temp_red_reps[:size] = red_scaled_reps[:]
        temp_red_reps[size:] = 0.0
    # Each transform contributes one factor of norm_const and Eq. (20) of the QML-Lightning
    # arxiv divides by one, so the Ntransforms-1 net factors are applied once here instead of
    # being folded into every pass. The transform below is therefore the UNNORMALIZED one.
    temp_red_reps *= norm_const ** (Ntransforms - 1)
    for transform in range(Ntransforms):
        temp_red_reps *= sorf_diags[transform]
        unnormalized_fast_walsh_hadamard(temp_red_reps)


@jit_
def generate_sorf_stack_phases(
    red_scaled_reps, temp_red_reps, sorf_diags, biases, norm_const
) -> None:
    generate_sorf_stack_unbiased_phases(red_scaled_reps, temp_red_reps, sorf_diags, norm_const)
    # add biases
    temp_red_reps += biases


@jit_
def generate_sorf_stack(
    red_scaled_reps, temp_red_reps, sorf_diags, biases, norm_const
) -> ndarray_:
    generate_sorf_stack_phases(red_scaled_reps, temp_red_reps, sorf_diags, biases, norm_const)
    return cos_(save_(temp_red_reps))


@jit_
def local_sorf_func(
    red_scaled_reps,
    temp_red_reps,
    element_ids,
    sorf_diags,
    biases,
    norm_const,
    natoms,
    kernel_component,
) -> None:
    kernel_component[:] = 0.0
    for i_atom in range(int(natoms)):
        el_id = int(element_ids[i_atom])
        kernel_component[:] += generate_sorf_stack(
            red_scaled_reps[i_atom],
            temp_red_reps,
            sorf_diags[el_id],
            biases[el_id],
            norm_const,
        )


@jit_(numba_parallel=allow_sorf_numba_parallelization())
def generate_sorf_processed_input(
    reduced_scaled_representations,
    sorf_diags,
    biases,
    kernel,
    nfeature_stacks: int,
    init_size: int,
) -> None:
    norm_const = hadamard_norm_const(init_size)

    nmols = reduced_scaled_representations.shape[0]
    nfeatures = kernel.shape[1]
    assert nfeature_stacks * init_size == nfeatures
    assert kernel.shape[0] == nmols

    for feature_stack in prange_(nfeature_stacks):
        temp_red_reps = empty_((init_size,))
        lb_rff = init_size * feature_stack
        ub_rff = lb_rff + init_size
        for mol_id in range(nmols):
            kernel[mol_id, lb_rff:ub_rff] = generate_sorf_stack(
                reduced_scaled_representations[mol_id],
                temp_red_reps,
                sorf_diags[feature_stack],
                biases[feature_stack],
                norm_const,
            )
    kernel[:, :] *= rff_vec_norm_const(nfeatures)  # normalization


@jit_(numba_parallel=allow_sorf_numba_parallelization())
def generate_local_sorf_processed_input(
    reduced_scaled_representations,
    element_ids,
    all_sorf_diags,
    all_biases,
    kernel,
    ubound_arr,
    nfeature_stacks: int,
    init_size: int,
) -> None:
    assert reduced_scaled_representations.shape[0] == element_ids.shape[0]
    assert all_sorf_diags.shape[:2] == all_biases.shape[:2]
    assert ubound_arr[-1] == reduced_scaled_representations.shape[0]
    # KK: not %100 sure it's necessary
    assert max_(element_ids) < all_sorf_diags.shape[1]

    norm_const = hadamard_norm_const(init_size)

    nmols = ubound_arr.shape[0] - 1
    assert nmols == kernel.shape[0]
    nfeatures = kernel.shape[1]
    assert nfeature_stacks * init_size == nfeatures

    for feature_stack in prange_(nfeature_stacks):
        temp_red_reps = empty_((init_size,))
        lb_rff = init_size * feature_stack
        ub_rff = lb_rff + init_size
        for mol_id in range(nmols):
            lb_mol = ubound_arr[mol_id]
            ub_mol = ubound_arr[mol_id + 1]
            natoms = ub_mol - lb_mol
            local_sorf_func(
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
def generate_sorf(
    representations,
    sorf_diags,
    biases,
    sigma: dim0float_array_,
    nfeature_stacks: int,
    init_size: int,
    out: Union[ndarray_, None] = None,
    reductor: optional_ndarray_ = None,
) -> ndarray_:
    """
    Generate molecules' SORF from (global) representations.

    Arguments:
    representations - {number of molecules}x{representation size} array of global representations of molecules.
    sorf_diags, biases - 2D array of SORF diagonals and array of SORF phase biases. Can be generated with `create_sorf_matrices`.
    sigma - value of sigma.
    nfeature_stacks - number of "stacks" of features (each stack generated from a single copy of initial representation).
    init_size - initial size of vectors to which SORF transform is applied.
    out (optional) - if not None serves as the output array.
    reductor (optional) - if present is used to project representation vectors onto `init_size` values that are then transformed
    into SORF. If absent representation vectors are just copied into vectors of `init_size` size with zero padding before transforming into SORF.

    Output:
        out : {number of molecules} x (init_size * nfeature_stacks) array of molecular SORF.
    """
    if reductor is None:
        assert representations.shape[1] <= init_size
    reduced_scaled_representations = project_scale_representations(
        representations, reductor, sigma
    )
    out_shape = (representations.shape[0], init_size * nfeature_stacks)
    out = check_allocation(out_shape, out)
    generate_sorf_processed_input(
        reduced_scaled_representations,
        sorf_diags,
        biases,
        out,
        nfeature_stacks,
        init_size,
    )
    return out


@jit_
def generate_local_sorf(
    representations,
    ncharges,
    natoms,
    sorted_elements,
    all_sorf_diags,
    all_biases,
    sigma: dim0float_array_,
    nfeature_stacks: int,
    init_size: int,
    out: Union[ndarray_, None] = None,
    reductors: optional_ndarray_ = None,
) -> ndarray_:
    """
    Generate the molecules' SORF from local representations.

    Arguments:
    representations - {total number of atoms} x {representation size} array of local representations of molecules concatenated (e.g. via numpy.concatenate) together.
    ncharges - nuclear charges arrays for all molecules concatenated into a 1D array.
    natoms - numbers of atoms in each molecule.
    sorted_elements - sorted list of nuclear charges of elements that are encountered in molecules of interest.
    all_sorf_diags, all_biases - 3D array of SORF diagonals for each element and 2D array of SORF phase biases for each element.
    Can be generated with `create_sorf_matrices_diff_species`.
    sigma - value of sigma.
    nfeature_stacks - number of "stacks" of features (each stack generated from a single copy of initial representations).
    init_size - initial size of vectors to which SORF transform is applied.
    out (optional) - if not None serves as the output array.
    reductors (optional) - if present is used to project representation vectors onto `init_size` principle components that are then transformed
    into SORF. If absent representation vectors are just copied into vectors of `init_size` with zero padding before transforming into SORF.

    Output:
        out : {number of molecules} x (init_size * nfeature_stacks) array of molecular SORF.
    """
    ubound_arr = get_atom_environment_ranges(natoms)
    all_element_ids = get_element_ids_from_sorted(ncharges, sorted_elements)
    if reductors is None:
        assert representations.shape[1] <= init_size
    reduced_scaled_representations = project_scale_local_representations(
        representations, all_element_ids, reductors, sigma
    )
    out_shape = (natoms.shape[0], init_size * nfeature_stacks)

    out = check_allocation(out_shape, out)
    generate_local_sorf_processed_input(
        reduced_scaled_representations,
        all_element_ids,
        all_sorf_diags,
        all_biases,
        out,
        ubound_arr,
        nfeature_stacks,
        init_size,
    )
    return out
