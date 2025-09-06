# K.Karan. TODO: PBC for FCHL19 + gradients is incomplete
from ..data import nCartDim
from ..jit_interfaces import (
    abs_,
    any_,
    arccos_,
    array_,
    cos_,
    dim0float_array_,
    dint_,
    dot_,
    dtype_,
    empty_,
    exp_,
    float_,
    int_,
    jit_,
    linspace_,
    log_,
    max_,
    ndarray_,
    optional_ndarray_,
    pi_,
    prange_,
    sign_,
    sin_,
    sqrt_,
    tiny_,
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
def get_element_types(
    nuclear_charges, elements, natoms: int, nelements: int, dint_: dtype_ = dint_
):
    element_types = empty_(natoms, dtype=dint_)
    element_types[:] = -1

    for i in prange_(natoms):
        for j in range(nelements):
            if nuclear_charges[i] == elements[j]:
                element_types[i] = j
                break

    if any_(element_types == -1):
        raise Exception("Not all nuclear charges found in 'elements' keyword argument!")

    return element_types


@jit_(numba_parallel=True)
def generate_fchl19_no_gradients(
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
    cell: optional_ndarray_ = None,
    pi_: float_ = pi_,
):
    Rs2 = linspace_(0, rcut, 1 + nRs2)[1:]
    Rs3 = linspace_(0, acut, 1 + nRs3)[1:]

    Ts = linspace_(0, pi_, 2 * nFourier)

    nelements = elements.shape[0]
    natoms = coordinates.shape[0]

    sqrt2pi = sqrt_(2.0 * pi_)

    two_body_size = nelements * nRs2
    rep_size = nelements * nRs2 + (nelements * (nelements + 1)) * nRs3 * nFourier

    three_body_weight = sqrt_(eta3 / pi_) * three_body_weight

    nabasis = Ts.shape[0]

    # Initialize representation array
    rep = zeros_((natoms, rep_size))

    # Determine unique elements in the system
    element_types = get_element_types(nuclear_charges, elements, natoms, nelements)

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
    max_num_rel_distances = int(max_(relevant_distance_nums))
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
    add_rep = zeros_((natoms, max_num_rel_distances, nRs2))
    # Calculate two-body terms (i,j) and add them to pairs with j < i.
    for i in prange_(natoms):
        radial = zeros_((nRs2,))
        for j_id in range(int(relevant_distance_nums[i])):
            j = relevant_distance_ids[i, j_id]
            if j < i:
                continue
            rij = distance_matrix[i, j_id]
            if rij > rcut:
                continue
            get_2body_component(
                rij, rdecay[i, j_id], eta2, two_body_decay, Rs2, sqrt2pi, nRs2, radial
            )
            n = element_types[j]

            start_index_n = n * nRs2
            rep[i, start_index_n : start_index_n + nRs2] += radial
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
            start_index_m = m * nRs2
            rep[j, start_index_m : start_index_m + nRs2] += add_rep[j, i_id, :]
    # Add three-body terms.
    invcut = 1.0 / acut
    rdecay = decay(
        distance_matrix,
        invcut,
        natoms,
        relevant_distance_ids,
        relevant_distance_nums,
        max_num_rel_distances,
    )

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
                start_add = two_body_size + nRs3 * nabasis * (
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
                    nRs3,
                    Rs3,
                    start_add,
                    rep[i],
                )

    return rep


@jit_
def get_constants(rij, eta2: float_, invsqrij, two_body_decay: float_, log_Rs2):
    mu = log_(rij / sqrt_(1.0 + eta2 * invsqrij))
    sigma = sqrt_(log_(1.0 + eta2 * invsqrij))
    exp_s2 = exp_(sigma**2)
    exp_ln = exp_(-((log_Rs2 - mu) ** 2) / sigma**2 * 0.5) * sqrt_(2.0)
    scaling = 1.0 / rij**two_body_decay

    return mu, sigma, exp_s2, exp_ln, scaling


@jit_
def get_radial(rij, two_body_decay: float_, sigma, rdecay, Rs2, mu, nbasis2: int, sqrt2pi: float_):
    radial = zeros_(nbasis2)
    for k in range(nbasis2):
        radial[k] = (
            1.0
            / (sigma * sqrt2pi * Rs2[k])
            * rdecay
            * exp_(-((log_(Rs2[k]) - mu) ** 2) / (2.0 * sigma**2))
            / rij**two_body_decay
        )

    return radial


@jit_
def get_part(
    i: int_,
    j: int_,
    k: int_,
    coordinates,
    rij,
    log_Rs2,
    Rs2,
    mu,
    exp_ln,
    exp_s2,
    eta2: float_,
    sigma,
    two_body_decay: float_,
    invcut: float_,
    invrij,
    rdecay,
    scaling,
    radial_base,
    pi_=pi_,
):
    dx = -(coordinates[i, k] - coordinates[j, k])

    part = (
        (log_Rs2 - mu)
        * (-dx * (rij**2 * exp_s2 + eta2) / (rij * sqrt_(exp_s2)) ** 3)
        * sqrt_(exp_s2)
        / (sigma**2 * rij)
        + (log_Rs2 - mu) ** 2 * eta2 * dx / (sigma**4 * rij**4 * exp_s2)
    ) * exp_ln / (Rs2 * sigma * sqrt_(pi_) * 2) - exp_ln * eta2 * dx / (
        Rs2 * sigma**3 * sqrt_(pi_) * rij**4 * exp_s2 * 2.0
    )

    dscal = two_body_decay * dx / rij ** (two_body_decay + 2.0)
    ddecay = dx * 0.5 * pi_ * sin_(pi_ * rij * invcut) * invcut * invrij
    part = part * scaling * rdecay + radial_base * dscal * rdecay + radial_base * scaling * ddecay

    return part


@jit_
def get_3body_component_with_gradient(
    i: int_,
    j: int_,
    k: int_,
    n: int_,
    m: int_,
    coordinates,
    rij,
    rik,
    rjk,
    eta3: float_,
    Rs3,
    zeta: float_,
    invcut: float_,
    three_body_decay: float_,
    three_body_weight: float_,
    nabasis: int_,
    nbasis3: int_,
    nelements: int_,
    rdecay_ij,
    rdecay_ik,
    i_self_id: int_,
    j_id: int_,
    k_id: int_,
    rep,
    grad,
    tiny_: float_ = tiny_,
    pi_: float_ = pi_,
):
    rij2 = rij**2
    invrij = 1 / rij
    invrij2 = invrij**2

    rik2 = rik**2
    invrik = 1 / rik
    invrjk = 1 / rjk
    invrik2 = invrik**2

    a = coordinates[j, :]
    b = coordinates[i, :]
    c = coordinates[k, :]

    cos_i, angle = calc_cos_with_angle(rij, rik, rjk)
    cos_j = calc_cos_angle_unclipped(rij, rjk, rik)
    cos_k = calc_cos_angle_unclipped(rjk, rik, rij)

    radial = exp_(-eta3 * (0.5 * (rij + rik) - Rs3) ** 2)
    p = min(n, m)
    q = max(n, m)

    dot = dot_(a - b, c - b)

    angular = zeros_(nabasis)
    d_angular = zeros_(nabasis)

    angular[0] = exp_(-(zeta**2) / 2) * 2.0 * cos_(angle)
    angular[1] = exp_(-(zeta**2) / 2) * 2.0 * sin_(angle)

    det = rij2 * rik2 - dot**2
    if det > tiny_:
        d_angular[0] = exp_(-(zeta**2) / 2) * 2.0 * sin_(angle) / sqrt_(det)
        d_angular[1] = -exp_(-(zeta**2) / 2) * 2.0 * cos_(angle) / sqrt_(det)
    else:
        d_angular[:] = 0.0
    # Part of the derivative of the angular basis functions wrt atom j (dim(3))
    d_angular_d_j = c - b + dot * ((b - a) * invrij2)
    # Part of the derivative of the angular basis functions wrt atom k (dim(3))
    d_angular_d_k = a - b + dot * ((b - c) * invrik2)
    # Part of the derivative of the angular basis functions wrt atom i (dim(3))
    d_angular_d_i = -(d_angular_d_j + d_angular_d_k)

    # Part of the derivative of the radial basis functions wrt coordinates (dim(nbasis3))
    # including decay
    d_radial = radial * eta3 * (0.5 * (rij + rik) - Rs3)  # * rdecay(i,j) * rdecay(i,k)
    # Part of the derivative of the radial basis functions wrt atom j (dim(3))
    d_radial_d_j = (b - a) * invrij
    # Part of the derivative of the radial basis functions wrt atom k (dim(3))
    d_radial_d_k = (b - c) * invrik
    # Part of the derivative of the radial basis functions wrt atom i (dim(3))
    d_radial_d_i = -(d_radial_d_j + d_radial_d_k)

    # Part of the derivative of the i,j decay functions wrt coordinates (dim(3))
    d_ijdecay = -pi_ * (b - a) * sin_(pi_ * rij * invcut) * 0.5 * invrij * invcut
    # Part of the derivative of the i,k decay functions wrt coordinates (dim(3))
    d_ikdecay = -pi_ * (b - c) * sin_(pi_ * rik * invcut) * 0.5 * invrik * invcut

    invr_atm = (invrij * invrjk * invrik) ** three_body_decay

    # Axilrod-Teller-Muto term
    atm = (1.0 + 3.0 * cos_i * cos_j * cos_k) * invr_atm * three_body_weight

    atm_i = (3.0 * cos_j * cos_k) * invr_atm * invrij * invrik
    atm_j = (3.0 * cos_k * cos_i) * invr_atm * invrij * invrjk
    atm_k = (3.0 * cos_i * cos_j) * invr_atm * invrjk * invrik

    vi = dot_(a - b, c - b)
    vj = dot_(c - a, b - a)
    vk = dot_(b - c, a - c)

    d_atm_ii = 2 * b - a - c - vi * ((b - a) * invrij**2 + (b - c) * invrik**2)
    d_atm_ij = c - a - vj * (b - a) * invrij**2
    d_atm_ik = a - c - vk * (b - c) * invrik**2

    d_atm_ji = c - b - vi * (a - b) * invrij**2
    d_atm_jj = 2 * a - b - c - vj * ((a - b) * invrij**2 + (a - c) * invrjk**2)
    d_atm_jk = b - c - vk * (a - c) * invrjk**2

    d_atm_ki = a - b - vi * (c - b) * invrik**2
    d_atm_kj = b - a - vj * (c - a) * invrjk**2
    d_atm_kk = 2 * c - a - b - vk * ((c - a) * invrjk**2 + (c - b) * invrik**2)

    d_atm_extra_i = (
        ((a - b) * invrij**2 + (c - b) * invrik**2)
        * atm
        * three_body_decay
        / three_body_weight
    )
    d_atm_extra_j = (
        ((b - a) * invrij**2 + (c - a) * invrjk**2)
        * atm
        * three_body_decay
        / three_body_weight
    )
    d_atm_extra_k = (
        ((a - c) * invrjk**2 + (b - c) * invrik**2)
        * atm
        * three_body_decay
        / three_body_weight
    )

    # Get index of where the contributions of atoms i,j,k should be added
    s = nbasis3 * nabasis * (-(p * (p + 1)) // 2 + q + nelements * p)

    for l in range(nbasis3):
        z = s + l * nabasis

        rep[z : z + nabasis] += angular * radial[l] * atm * rdecay_ij * rdecay_ik

        for t in range(3):
            # Add up all gradient contributions wrt atom i
            grad[z : z + nabasis, i_self_id, t] += (
                d_angular * d_angular_d_i[t] * radial[l] * atm * rdecay_ij * rdecay_ik
                + angular * d_radial[l] * d_radial_d_i[t] * atm * rdecay_ij * rdecay_ik
                + angular
                * radial[l]
                * (
                    atm_i * d_atm_ii[t]
                    + atm_j * d_atm_ij[t]
                    + atm_k * d_atm_ik[t]
                    + d_atm_extra_i[t]
                )
                * three_body_weight
                * rdecay_ij
                * rdecay_ik
                + angular * radial[l] * (d_ijdecay[t] * rdecay_ik + rdecay_ij * d_ikdecay[t]) * atm
            )

            # Add up all gradient contributions wrt atom j
            grad[z : z + nabasis, j_id, t] += (
                d_angular * d_angular_d_j[t] * radial[l] * atm * rdecay_ij * rdecay_ik
                + angular * d_radial[l] * d_radial_d_j[t] * atm * rdecay_ij * rdecay_ik
                + angular
                * radial[l]
                * (
                    atm_i * d_atm_ji[t]
                    + atm_j * d_atm_jj[t]
                    + atm_k * d_atm_jk[t]
                    + d_atm_extra_j[t]
                )
                * three_body_weight
                * rdecay_ij
                * rdecay_ik
                - angular * radial[l] * d_ijdecay[t] * rdecay_ik * atm
            )

            # Add up all gradient contributions wrt atom k
            grad[z : z + nabasis, k_id, t] += (
                d_angular * d_angular_d_k[t] * radial[l] * atm * rdecay_ij * rdecay_ik
                + angular * d_radial[l] * d_radial_d_k[t] * atm * rdecay_ij * rdecay_ik
                + angular
                * radial[l]
                * (
                    atm_i * d_atm_ki[t]
                    + atm_j * d_atm_kj[t]
                    + atm_k * d_atm_kk[t]
                    + d_atm_extra_k[t]
                )
                * three_body_weight
                * rdecay_ij
                * rdecay_ik
                - angular * radial[l] * rdecay_ij * d_ikdecay[t] * atm
            )


@jit_(numba_parallel=True)
def generate_fchl_with_gradients(
    nuclear_charges,
    coordinates,
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
    cell: optional_ndarray_ = None,
    pi_: float_ = pi_,
    dint_: dtype_ = dint_,
    nCartDim: int_ = nCartDim,
):
    nelements = elements.shape[0]
    Rs2 = linspace_(0, rcut, 1 + nRs2)[1:]
    Rs3 = linspace_(0, acut, 1 + nRs3)[1:]

    Ts = linspace_(0, pi_, 2 * nFourier)
    sqrt2pi = sqrt_(2.0 * pi_)

    two_body_size = nelements * nRs2
    rep_size = two_body_size + (nelements * (nelements + 1)) * nRs3 * nFourier

    natoms = coordinates.shape[0]
    element_types = get_element_types(nuclear_charges, elements, natoms, nelements)
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
    max_num_rel_distances = int(max_(relevant_distance_nums))
    # ids of atoms affecting representation of an atom.
    # Differs from relevant_distance_ids by including atom_id itself.
    # TODO: number of atoms included can probably be decreased.
    max_num_rel_atoms = max_num_rel_distances + 1
    relevant_atom_ids = empty_((natoms_tot, max_num_rel_atoms), dtype=dint_)
    relevant_atom_nums = empty_((natoms_tot,), dtype=dint_)
    for i in prange_(natoms_tot):
        rel_dist_num = int(relevant_distance_nums[i])
        relevant_atom_ids[i, :rel_dist_num] = relevant_distance_ids[i, :rel_dist_num]
        relevant_atom_ids[i, rel_dist_num] = i
        relevant_atom_nums[i] = rel_dist_num + 1

    three_body_weight = sqrt_(eta3 / pi_) * three_body_weight

    invcut = 1.0 / rcut
    rdecay = decay(
        distance_matrix,
        invcut,
        natoms,
        relevant_distance_ids,
        relevant_distance_nums,
        max_num_rel_distances,
    )

    radial = zeros_(nRs2)
    log_Rs2 = log_(Rs2)
    rep = zeros_((natoms, rep_size))
    grad = zeros_((natoms, rep_size, max_num_rel_atoms, nCartDim))
    add_rep = zeros_((natoms, max_num_rel_distances, nRs2))
    add_grad = zeros_((natoms, max_num_rel_distances, nRs2, nCartDim))

    for i in prange_(natoms):
        relevant_distance_num = int(relevant_distance_nums[i])
        for j_id in range(relevant_distance_num):
            j = relevant_distance_ids[i, j_id]
            if j < i:
                continue
            rij = distance_matrix[i, j_id]
            if rij > rcut:
                continue

            invrij = 1 / rij
            invsqrij = invrij**2
            mu, sigma, exp_s2, exp_ln, scaling = get_constants(
                rij, eta2, invsqrij, two_body_decay, log_Rs2
            )

            radial = get_radial(
                rij, two_body_decay, sigma, rdecay[i, j_id], Rs2, mu, nRs2, sqrt2pi
            )

            n = element_types[j]
            start_index_n = int(n * nRs2)

            rep[i, start_index_n : start_index_n + nRs2] += radial
            if j < natoms:
                i_id = get_neighbor_id(j, i, relevant_distance_ids, relevant_distance_nums)
                add_rep[j, i_id, :] = radial[:]
            else:
                # to avoid compiler confusion
                i_id = 0

            radial_base = (
                1.0 / (sigma * sqrt2pi * Rs2) * exp_(-((log_Rs2 - mu) ** 2) / (2.0 * sigma**2))
            )

            for k in range(nCartDim):
                part = get_part(
                    i,
                    j,
                    k,
                    coordinates,
                    rij,
                    log_Rs2,
                    Rs2,
                    mu,
                    exp_ln,
                    exp_s2,
                    eta2,
                    sigma,
                    two_body_decay,
                    invcut,
                    invrij,
                    rdecay[i, j_id],
                    scaling,
                    radial_base,
                )

                grad[i, start_index_n : start_index_n + nRs2, relevant_distance_num, k] += part
                grad[i, start_index_n : start_index_n + nRs2, j_id, k] -= part

                if j < natoms:
                    add_grad[j, i_id, :, k] = part

    # add previously saved two-body components
    for j in prange_(natoms):
        relevant_distance_num = int(relevant_distance_nums[j])
        for i_id in range(relevant_distance_num):
            i = relevant_distance_ids[j, i_id]
            if j < i:
                continue
            rij = distance_matrix[j, i_id]
            if rij > rcut:
                continue
            m = element_types[i]

            start_index_m = int(m * nRs2)
            rep[j, start_index_m : start_index_m + nRs2] += add_rep[j, i_id, :]
            grad[j, start_index_m : start_index_m + nRs2, relevant_distance_num, :] -= add_grad[
                j, i_id, :, :
            ]
            grad[j, start_index_m : start_index_m + nRs2, i_id, :] += add_grad[j, i_id, :, :]

    nbasis3 = Rs3.shape[0]
    nabasis = Ts.shape[0]

    invcut = 1.0 / acut
    rdecay = decay(
        distance_matrix,
        invcut,
        natoms,
        relevant_distance_ids,
        relevant_distance_nums,
        max_num_rel_distances,
    )

    for i in prange_(natoms):
        num_rel_distances = int(relevant_distance_nums[i])
        for j_id in range(num_rel_distances - 1):
            rij = distance_matrix[i, j_id]
            if rij > acut:
                continue
            j = relevant_distance_ids[i, j_id]

            n = element_types[j]

            for k_id in range(j_id + 1, num_rel_distances):
                rik = distance_matrix[i, k_id]
                if rik > acut:
                    continue
                k = relevant_distance_ids[i, k_id]

                rjk = get_from_sparse_matrix(
                    distance_matrix,
                    j,
                    k,
                    relevant_distance_ids,
                    relevant_distance_nums,
                    default_val=max_rel_dist,
                )

                m = element_types[k]

                get_3body_component_with_gradient(
                    i,
                    j,
                    k,
                    n,
                    m,
                    coordinates,
                    rij,
                    rik,
                    rjk,
                    eta3,
                    Rs3,
                    zeta,
                    invcut,
                    three_body_decay,
                    three_body_weight,
                    nabasis,
                    nbasis3,
                    nelements,
                    rdecay[i, j_id],
                    rdecay[i, k_id],
                    num_rel_distances,
                    j_id,
                    k_id,
                    rep[i, two_body_size:],
                    grad[i, two_body_size:, :, :],
                )

    return rep, grad, relevant_atom_ids, relevant_atom_nums


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
    cell: optional_ndarray_ = None,
    gradients=False,
):
    if gradients:
        assert nFourier == 1
        return generate_fchl_with_gradients(
            nuclear_charges,
            coordinates,
            elements=elements,
            nRs2=nRs2,
            nRs3=nRs3,
            nFourier=nFourier,
            eta2=eta2,
            eta3=eta3,
            zeta=zeta,
            rcut=rcut,
            acut=acut,
            two_body_decay=two_body_decay,
            three_body_decay=three_body_decay,
            three_body_weight=three_body_weight,
            cell=cell,
        )
    else:
        return generate_fchl19_no_gradients(
            nuclear_charges,
            coordinates,
            elements=elements,
            nRs2=nRs2,
            nRs3=nRs3,
            nFourier=nFourier,
            eta2=eta2,
            eta3=eta3,
            zeta=zeta,
            rcut=rcut,
            acut=acut,
            two_body_decay=two_body_decay,
            three_body_decay=three_body_decay,
            three_body_weight=three_body_weight,
            cell=cell,
        )
