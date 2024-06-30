from ..jit_interfaces import (
    abs_,
    arccos_,
    array_,
    cos_,
    dim0float_array_,
    dint_,
    dtype_,
    empty_,
    exp_,
    float_,
    int_,
    jit_,
    linspace_,
    log_,
    ndarray_,
    pi_,
    prange_,
    sign_,
    sin_,
    sqrt_,
    zeros_,
)
from .basic_utilities import (
    calculate_distances_wrcut,
    extend_for_pbc,
    get_from_sparse_matrix,
    get_neighbor_id,
)


@jit_(numba_parallel=True)
def decay(
    r: ndarray_,
    invrc: float_,
    natoms: int_,
    relevant_distance_ids: ndarray_,
    relevant_distance_nums: ndarray_,
    max_num_rel_distances: int_,
    pi_: float_ = pi_,
):
    f = empty_((natoms, max_num_rel_distances))
    for i in prange_(natoms):
        for j_id in range(int(relevant_distance_nums[i])):
            j = relevant_distance_ids[i, j_id]
            if j < i:
                continue
            f[i, j_id] = 0.5 * (cos_(pi_ * r[i, j_id] * invrc) + 1.0)
            if j < natoms:
                i_id = get_neighbor_id(j, i, relevant_distance_ids, relevant_distance_nums)
                f[j, i_id] = f[i, j_id]
    return f


@jit_
def calc_cos_angle_unclipped(r_ab, r_ac, r_bc):
    cos_angle = (r_ab**2 + r_ac**2 - r_bc**2) / r_ab / r_ac / 2
    return cos_angle


@jit_
def calc_cos_with_angle(r_ab, r_ac, r_bc):
    cos_angle = calc_cos_angle_unclipped(r_ab, r_ac, r_bc)
    if abs_(cos_angle) > 1.0:
        cos_angle = sign_(cos_angle)
    angle = arccos_(cos_angle)
    return cos_angle, angle


@jit_
def get_2body_component(
    rij: dim0float_array_,
    rdecay_val: dim0float_array_,
    eta2: float_,
    two_body_decay: float_,
    Rs2: ndarray_,
    sqrt2pi: float_,
    nbasis2: int_,
    radial: ndarray_,
):
    mu = log_(rij / sqrt_(1.0 + eta2 / rij**2))
    sigma = sqrt_(log_(1.0 + eta2 / rij**2))

    for k in range(nbasis2):
        radial[k] = (
            1.0
            / (sigma * sqrt2pi * Rs2[k])
            * rdecay_val
            * exp_(-((log_(Rs2[k]) - mu) ** 2) / (2.0 * sigma**2))
            / rij**two_body_decay
        )


@jit_
def get_3body_component(
    rij: dim0float_array_,
    rik: dim0float_array_,
    rjk: dim0float_array_,
    rdecayij: dim0float_array_,
    rdecayik: dim0float_array_,
    eta3: float_,
    three_body_decay: float_,
    three_body_weight: float_,
    zeta: float_,
    nabasis: int_,
    nbasis3: int_,
    Rs3: ndarray_,
    start_add: int_,
    rep: ndarray_,
):
    cos_1, angle = calc_cos_with_angle(rij, rik, rjk)
    cos_2 = calc_cos_angle_unclipped(rjk, rik, rij)
    cos_3 = calc_cos_angle_unclipped(rij, rjk, rik)

    radial = exp_(-eta3 * (0.5 * (rij + rik) - Rs3) ** 2) * rdecayij * rdecayik
    ksi3 = (
        (1.0 + 3.0 * cos_1 * cos_2 * cos_3)
        / (rik * rij * rjk) ** three_body_decay
        * three_body_weight
    )
    angular = zeros_((nabasis,))
    for l in range(nabasis // 2):
        o = l * 2 + 1
        angular[2 * l] += 2 * cos_(o * angle) * exp_(-((zeta * o) ** 2) / 2)
        angular[2 * l + 1] += 2 * sin_(o * angle) * exp_(-((zeta * o) ** 2) / 2)
    for l in range(nbasis3):
        z = start_add + l * nabasis
        rep[z : z + nabasis] += angular * radial[l] * ksi3


@jit_(numba_parallel=True)
def generate_fchl19(
    nuclear_charges: ndarray_,
    coordinates: ndarray_,
    elements: ndarray_ = array_([1, 6, 7, 8, 16], dtype=int_),
    nRs2: int_ = 24,
    nRs3: int_ = 20,
    nFourier: int_ = 1,
    eta2: float_ = 0.32,
    eta3: float_ = 2.7,
    zeta: float_ = pi_,
    rcut: float_ = 8.0,
    acut: float_ = 8.0,
    two_body_decay: float_ = 1.8,
    three_body_decay: float_ = 0.57,
    three_body_weight: float_ = 13.4,
    cell: ndarray_ = None,
    pi_: float_ = pi_,
    dint_: dtype_ = dint_,
):
    Rs2 = linspace_(0, rcut, 1 + nRs2)[1:]
    Rs3 = linspace_(0, acut, 1 + nRs3)[1:]

    Ts = linspace_(0, pi_, 2 * nFourier)

    nelements = elements.shape[0]
    natoms = coordinates.shape[0]

    sqrt2pi = sqrt_(2.0 * pi_)

    rep_size = nelements * nRs2 + (nelements * (nelements + 1)) * nRs3 * nFourier

    three_body_weight = sqrt_(eta3 / pi_) * three_body_weight

    nbasis2 = Rs2.shape[0]
    nbasis3 = Rs3.shape[0]
    nabasis = Ts.shape[0]

    # Initialize representation array
    rep = zeros_((natoms, rep_size))

    # Determine unique elements in the system
    element_types = zeros_((natoms,), dtype=dint_)
    for i in prange_(natoms):
        for j in range(nelements):
            if nuclear_charges[i] == elements[j]:
                element_types[i] = j
                break

    if cell is None:
        natoms_tot = natoms
    else:
        coordinates, element_types, natoms_tot, _ = extend_for_pbc(
            coordinates, element_types, natoms, max(rcut, acut), cell
        )

    # Calculate distances.
    max_rel_dist = max(rcut, 2 * acut)
    distance_matrix, relevant_distance_ids, relevant_distance_nums = calculate_distances_wrcut(
        coordinates, natoms_tot, max_rel_dist
    )
    max_num_rel_distances = max(relevant_distance_nums)
    # Two-body decay
    invcut = 1.0 / rcut
    rdecay = decay(
        distance_matrix,
        invcut,
        natoms,
        relevant_distance_ids,
        relevant_distance_nums,
        max_num_rel_distances,
    )
    add_rep = zeros_((natoms, max_num_rel_distances, nbasis2))
    # Calculate two-body terms (i,j) and add them to pairs with j < i.
    for i in prange_(natoms):
        radial = zeros_((nbasis2,))
        for j_id in range(int(relevant_distance_nums[i])):
            j = relevant_distance_ids[i, j_id]
            if j < i:
                continue
            rij = distance_matrix[i, j_id]
            if rij > rcut:
                continue
            get_2body_component(
                rij, rdecay[i, j_id], eta2, two_body_decay, Rs2, sqrt2pi, nbasis2, radial
            )
            n = element_types[j]

            start_index_n = n * nbasis2
            rep[i, start_index_n : start_index_n + nbasis2] += radial
            if j < natoms:
                i_id = get_neighbor_id(j, i, relevant_distance_ids, relevant_distance_nums)
                add_rep[j, i_id, :] = radial[:]
    # Add two-body terms for pairs with j < i.
    for j in prange_(natoms):
        for i_id in range(int(relevant_distance_nums[j])):
            i = relevant_distance_ids[j, i_id]
            if i > j:
                break
            if i >= natoms:
                break
            m = element_types[i]
            start_index_m = m * nbasis2
            rep[j, start_index_m : start_index_m + nbasis2] += add_rep[j, i_id, :]
    # Add three-body terms.
    for i in prange_(natoms):
        for j_id in range(int(relevant_distance_nums[i] - 1)):
            rij = distance_matrix[i, j_id]
            if rij > acut:
                continue
            j = relevant_distance_ids[i, j_id]
            n = element_types[j]

            for k_id in range(j_id + 1, int(relevant_distance_nums[i])):
                rik = distance_matrix[i, k_id]
                if rik > acut:
                    continue
                k = relevant_distance_ids[i, k_id]
                m = element_types[k]
                rjk = get_from_sparse_matrix(
                    distance_matrix,
                    j,
                    k,
                    relevant_distance_ids,
                    relevant_distance_nums,
                    default_val=max_rel_dist,
                )
                p = min(n, m)
                q = max(n, m)
                start_add = nelements * nbasis2 + nbasis3 * nabasis * (
                    -((p * (p + 1)) // 2) + q + nelements * p
                )

                get_3body_component(
                    rij,
                    rik,
                    rjk,
                    rdecay[i, j_id],
                    rdecay[i, k_id],
                    eta3,
                    three_body_decay,
                    three_body_weight,
                    zeta,
                    nabasis,
                    nbasis3,
                    Rs3,
                    start_add,
                    rep[i],
                )

    return rep
