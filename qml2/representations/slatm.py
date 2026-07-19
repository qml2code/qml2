import itertools as itl
from typing import List

from ..jit_interfaces import (
    array_,
    bool_,
    cos_,
    empty_,
    exp_,
    float_,
    int_,
    jit_,
    linspace_,
    ndarray_,
    pi_,
    prange_,
    sqrt_,
    sum_,
    zeros_,
)
from ..utils import get_possible_nuclear_charges
from .basic_utilities import (
    calc_cos_angle_unclipped,
    calc_cos_with_angle,
    calculate_distances_wrcut,
    extend_for_pbc,
    get_from_sparse_matrix,
    representation_numba_parallelization,
)

# some global variables. TODO: perhaps make them changeable?
r0 = 0.1

mbtype_stop_int = -1


@jit_
def outside_interval(lbound: float_, val, ubound: float_):
    return (val < lbound) or (val >= ubound)


@jit_
def get_sbop_length(rcut: float_, dgrid, r0: float_ = r0):
    return int((float(rcut) - float(r0)) / float(dgrid)) + 1


@jit_
def get_sbot_length(dgrid, pi_: float_ = pi_):
    d2r = pi_ / 180  # degree to rad
    a0 = -20.0 * d2r
    a1 = pi_ + 20.0 * d2r
    return int((a1 - a0) / dgrid) + 1


@jit_
def add_sbop(
    representation_vector,
    mbtype,
    nuclear_charges,
    distances,
    neighbor_ids,
    num_neighbors,
    xs,
    ia: int_,
    sigma: float_ = 0.05,
    dgrid: float_ = 0.03,
    rcut: float_ = 4.8,
    rpower: int_ = 6,
    normalize: bool_ = True,
    pi_: float_ = pi_,
):
    """
    two-body terms

    :param obj: molecule object, consisting of two parts: [ zs, coords ]
    :type obj: list
    """
    z1 = int(mbtype[0])
    z2 = int(mbtype[1])

    if nuclear_charges[ia] == z1:
        z_other = z2
    elif nuclear_charges[ia] == z2:
        z_other = z1
    else:
        return

    # bop potential distribution
    coeff = 1 / sqrt_(2 * sigma**2 * pi_) if normalize else 1.0
    c0 = (z1 % 1000) * (z2 % 1000) * coeff
    inv_sigma = -0.5 / sigma**2
    xs0 = c0 / (xs**rpower) * dgrid

    for i_atom2_id in range(int(num_neighbors[ia])):
        ia2 = neighbor_ids[ia, i_atom2_id]
        if nuclear_charges[ia2] != z_other:
            continue
        r = distances[ia, i_atom2_id]
        if r > rcut:
            continue
        representation_vector += xs0 * exp_(inv_sigma * (xs - r) ** 2) / 2


@jit_
def add_sbot(
    representation_vector,
    mbtype,
    nuclear_charges,
    distances,
    neighbor_ids,
    num_neighbors,
    xs,
    ia: int_,
    sigma: float_ = 0.05,
    dgrid: float_ = 0.0262,
    rcut: float_ = 4.8,
    normalize: bool_ = True,
    pi_: float_ = pi_,
):
    """
    sigma -- standard deviation of gaussian distribution centered on a specific angle
            defaults to 0.05 (rad), approximately 3 degree
    dgrid    -- step of angle grid
            defaults to 0.0262 (rad), approximately 1.5 degree
    """
    z1 = int(mbtype[0])
    z2 = int(mbtype[1])
    z3 = int(mbtype[2])

    if nuclear_charges[ia] != z2:
        return

    # for a normalized gaussian distribution, u should multiply this coeff
    coeff = 1 / sqrt_(2 * sigma**2 * pi_) if normalize else 1.0

    eps = 2.2e-16

    prefactor = 1.0 / 3.0
    c0 = prefactor * (z1 % 1000) * (z2 % 1000) * (z3 % 1000) * coeff * dgrid
    cos_xs = cos_(xs) * c0
    inv_sigma = -1.0 / (2 * sigma**2)

    for ia1_id in range(int(num_neighbors[ia])):
        ia1 = neighbor_ids[ia, ia1_id]
        r_ij = distances[ia, ia1_id]
        if nuclear_charges[ia1] != z1:
            continue
        if outside_interval(eps, r_ij, rcut):
            continue
        if z1 == z3:
            starting_ia3_id = ia1_id + 1
        else:
            starting_ia3_id = 0
        for ia3_id in range(int(starting_ia3_id), int(num_neighbors[ia])):
            ia3 = neighbor_ids[ia, ia3_id]
            if nuclear_charges[ia3] != z3:
                continue
            r_jk = distances[ia, ia3_id]
            if outside_interval(eps, r_jk, rcut):
                continue

            r_ik = get_from_sparse_matrix(distances, ia1, ia3, neighbor_ids, num_neighbors)

            if outside_interval(eps, r_ik, rcut) or outside_interval(eps, r_jk, rcut):
                continue
            _, ang = calc_cos_with_angle(r_ij, r_jk, r_ik)
            cak = calc_cos_angle_unclipped(r_ik, r_ij, r_jk)
            cai = calc_cos_angle_unclipped(r_ik, r_jk, r_ij)
            r = r_ij * r_ik * r_jk
            representation_vector += (
                (c0 + cos_xs * cak * cai) / (r**3) * exp_((xs - ang) ** 2 * inv_sigma)
            )


@jit_
def generate_slatm_local_single_atom(
    representation_vector,
    ia: int_,
    nuclear_charges,
    distances,
    neighbor_ids,
    num_neighbors,
    mbtypes,
    sigmas,
    dgrids,
    rcut: float_,
    rpower: int_,
    alchemy: bool_ = False,
    normalize: bool_ = True,
    mbtype_stop_int: int = mbtype_stop_int,
    r0: float_ = r0,
    pi_: float_ = pi_,
):
    last_mbtype_length = 1
    current_lbound = 0
    representation_vector[:] = 0.0
    central_nuclear_charge = nuclear_charges[ia]

    sbop_size = get_sbop_length(rcut, dgrids[0])
    sbot_size = get_sbot_length(dgrids[1])

    # grid for 2-body terms
    sbop_xs = linspace_(r0, rcut, sbop_size)

    # grid for 3-body terms
    a0 = -20 * pi_ / 180
    a1 = pi_ + 20 * pi_ / 180
    sbot_xs = linspace_(a0, a1, sbot_size)

    for mbtype_id in range(mbtypes.shape[0]):
        mbtype = mbtypes[mbtype_id]
        if mbtype[1] == mbtype_stop_int:
            if central_nuclear_charge == mbtype[0]:
                representation_vector[current_lbound] += central_nuclear_charge
            if not alchemy:
                current_lbound += 1
        elif mbtype[2] == mbtype_stop_int:
            if last_mbtype_length == 1:
                last_mbtype_length += 1
                if alchemy:
                    current_lbound += 1
            add_sbop(
                representation_vector[current_lbound : current_lbound + sbop_size],
                mbtype,
                nuclear_charges,
                distances,
                neighbor_ids,
                num_neighbors,
                sbop_xs,
                ia,
                sigmas[0],
                dgrids[0],
                rcut,
                rpower,
                normalize,
            )
            if not alchemy:
                current_lbound += sbop_size
        else:  # len(mbtype) == 3:
            if last_mbtype_length == 2:
                last_mbtype_length += 1
                if alchemy:
                    current_lbound += sbop_size
            add_sbot(
                representation_vector[current_lbound : current_lbound + sbot_size],
                mbtype,
                nuclear_charges,
                distances,
                neighbor_ids,
                num_neighbors,
                sbot_xs,
                ia,
                sigmas[1],
                dgrids[1],
                rcut,
                normalize,
            )
            if not alchemy:
                current_lbound += sbot_size


@jit_
def get_total_rep_size(
    mbtypes, rcut: float_, dgrids, alchemy: bool_ = False, mbtype_stop_int: int_ = -1
):
    sbop_length = int(get_sbop_length(rcut, dgrids[0]))
    sbot_length = int(get_sbot_length(dgrids[1]))
    if alchemy:
        return 1 + sbop_length + sbot_length
    output = 0
    for mbtype_id in range(mbtypes.shape[0]):
        mbtype = mbtypes[mbtype_id]
        if mbtype[1] == mbtype_stop_int:
            output += 1
        elif mbtype[2] == mbtype_stop_int:
            output += sbop_length
        else:
            output += sbot_length
    return output


@jit_(numba_parallel=representation_numba_parallelization(), numba_cache=True)
def generate_slatm_local(
    nuclear_charges,
    distances,
    neighbor_ids,
    num_neighbors,
    mbtypes,
    natoms: int_,
    sigmas=array_([0.05, 0.05]),
    dgrids=array_([0.03, 0.03]),
    rcut: float_ = 4.8,
    rpower: int_ = 6,
    alchemy: bool_ = False,
    normalize: bool_ = True,
):
    """
    Calculate local SLATM representations in parallel.
    """
    rep_size = get_total_rep_size(mbtypes, rcut, dgrids, alchemy)
    local_reps = empty_((natoms, rep_size))
    for ia in prange_(natoms):
        generate_slatm_local_single_atom(
            local_reps[ia],
            ia,
            nuclear_charges,
            distances,
            neighbor_ids,
            num_neighbors,
            mbtypes,
            sigmas,
            dgrids,
            rcut,
            rpower,
            alchemy,
            normalize,
        )
    return local_reps


@jit_(numba_parallel=representation_numba_parallelization(), numba_cache=True)
def generate_slatm_global(
    nuclear_charges,
    distances,
    neighbor_ids,
    num_neighbors,
    mbtypes,
    natoms: int_,
    sigmas=array_([0.05, 0.05]),
    dgrids=array_([0.03, 0.03]),
    rcut: float_ = 4.8,
    rpower: int_ = 6,
    alchemy: bool_ = False,
    normalize: bool_ = True,
):
    """
    Calculate local SLATM representations and then calculate their sum in-place.
    """
    rep_size = get_total_rep_size(mbtypes, rcut, dgrids, alchemy)
    global_rep = zeros_(rep_size)
    for ia in prange_(natoms):
        buffer = empty_(rep_size)
        generate_slatm_local_single_atom(
            buffer,
            ia,
            nuclear_charges,
            distances,
            neighbor_ids,
            num_neighbors,
            mbtypes,
            sigmas,
            dgrids,
            rcut,
            rpower,
            alchemy,
            normalize,
        )
        global_rep += buffer
    return global_rep


def generate_slatm_basic(
    nuclear_charges,
    coordinates,
    mbtypes,
    true_natoms,
    local=False,
    sigmas=array_([0.05, 0.05]),
    dgrids=array_([0.03, 0.03]),
    rcut=4.8,
    rpower=6,
    alchemy=False,
    normalize=True,
):
    """
    Generate SLATM representation for/from only the first `true_natoms` atoms in `nuclear_charges` and `coordinates`.
    """
    distances, neighbor_ids, num_neighbors = calculate_distances_wrcut(
        coordinates, len(coordinates), rcut * 2
    )

    if local:
        return generate_slatm_local(
            nuclear_charges,
            distances,
            neighbor_ids,
            num_neighbors,
            mbtypes,
            true_natoms,
            sigmas=sigmas,
            dgrids=dgrids,
            rcut=rcut,
            rpower=rpower,
            alchemy=alchemy,
            normalize=normalize,
        )
    else:
        return generate_slatm_global(
            nuclear_charges,
            distances,
            neighbor_ids,
            num_neighbors,
            mbtypes,
            true_natoms,
            sigmas=sigmas,
            dgrids=dgrids,
            rcut=rcut,
            rpower=rpower,
            alchemy=alchemy,
            normalize=normalize,
        )


def generate_slatm(
    nuclear_charges,
    coordinates,
    mbtypes,
    local=False,
    sigmas=array_([0.05, 0.05]),
    dgrids=array_([0.03, 0.03]),
    rcut=4.8,
    rpower=6,
    alchemy=False,
    normalize=True,
    cell=None,
):
    """
    Generate Spectrum of London and Axillrod-Teller-Muto potential (SLATM) representation.
    Both global (``local=False``) and local (``local=True``) SLATM are available.

    A version that works for periodic boundary conditions will be released soon.

    NOTE: You will need to run the ``get_slatm_mbtypes()`` function to get the ``mbtypes`` input (or generate it manually).

    :param coordinates: Input coordinates
    :type coordinates: numpy array
    :param nuclear_charges: List of nuclear charges.
    :type nuclear_charges: numpy array
    :param mbtypes: Many-body types for the whole dataset, including 1-, 2- and 3-body types. Could be obtained by calling ``get_slatm_mbtypes()``.
    :type mbtypes: list
    :param local: Generate a local representation. Defaulted to False (i.e., global representation); otherwise, atomic version.
    :type local: bool
    :param sigmas: Controlling the width of Gaussian smearing function for 2- and 3-body parts, defaulted to [0.05,0.05], usually these do not need to be adjusted.
    :type sigmas: list
    :param dgrids: The interval between two sampled internuclear distances and angles, defaulted to [0.03,0.03], no need for change, compromised for speed and accuracy.
    :type dgrids: list
    :param rcut: Cut-off radius, defaulted to 4.8 Angstrom.
    :type rcut: float
    :param rpower: The power of R in 2-body potential, defaulted to London potential (=6).
    :type rpower: float
    :param alchemy: Switch to use the alchemy version of SLATM. (default=False)
    :type alchemy: bool
    :param normalize: Normalize some SLATM vector components. (default=True)
    :type normalize: bool
    :param cell: if not None defines periodic boundary conditions cell;
    :type cell: numpy array
    :return: 1D SLATM representation if local=False, 2D array of atomic SLATM representations if local=True.
    :rtype: numpy array
    """
    assert cell is None, "Not implemented yet; if needed please create a github issue."
    # NOTE: for now I leave some untested code as placeholders for PBC implementation
    natoms = len(nuclear_charges)
    if cell is not None:
        coordinates, nuclear_charges, _, _ = extend_for_pbc(
            coordinates, nuclear_charges, natoms, rcut, cell
        )
    return generate_slatm_basic(
        nuclear_charges,
        coordinates,
        mbtypes,
        natoms,
        local=local,
        sigmas=sigmas,
        dgrids=dgrids,
        rcut=rcut,
        rpower=rpower,
        alchemy=alchemy,
        normalize=normalize,
    )


def get_nuclear_charges_max_encounters(all_nuclear_charges):
    encountered_nuclear_charges = get_possible_nuclear_charges(all_nuclear_charges)
    max_nuclear_charge_encounters = zeros_(encountered_nuclear_charges.shape, dtype=int)
    for nuclear_charges in all_nuclear_charges:
        for nc_id, nc in enumerate(encountered_nuclear_charges):
            max_nuclear_charge_encounters[nc_id] = max(
                max_nuclear_charge_encounters[nc_id], sum_(nc == nuclear_charges)
            )
    return encountered_nuclear_charges, max_nuclear_charge_encounters


def get_slatm_mbtypes(nuclear_charges: List[ndarray_], pbc: str = "000") -> List[List[int_]]:
    """
    Get the list of minimal types of many-body terms in a dataset. This resulting list
    is necessary as input in the ``generate_slatm_representation()`` function.

    :param nuclear_charges: A list of the nuclear charges for each compound in the dataset.
    :type nuclear_charges: list of numpy arrays
    :param pbc: periodic boundary condition along x,y,z direction, defaulted to '000', i.e., molecule
    :type pbc: string
    :return: A list containing the types of many-body terms.
    :rtype: list
    """

    zsmax, nzmax = get_nuclear_charges_max_encounters(nuclear_charges)

    boas = [[zi, mbtype_stop_int, mbtype_stop_int] for zi in zsmax]

    bops = [[zi, zi, mbtype_stop_int] for zi in zsmax] + [
        list(x) + [mbtype_stop_int] for x in itl.combinations(zsmax, 2)
    ]

    bots = []
    for i in zsmax:
        for bop in bops:
            j, k = bop[:2]
            tas = [[i, j, k], [i, k, j], [j, i, k]]
            for tasi in tas:
                if (tasi in bots) or (tasi[::-1] in bots):
                    continue
                valid = True
                for zj, max_nzj in zip(zsmax, nzmax):
                    nzj = sum((zj == t) for t in tasi)
                    if nzj > max_nzj:
                        valid = False
                        break
                if valid:
                    bots.append(tasi)
    mbtypes = boas + bops + bots

    mbtypes = array_(mbtypes)

    return mbtypes


def generate_slatm_for_single_atom(
    nuclear_charges,
    coordinates,
    mbtypes,
    atom_id: int,
    local=False,
    sigmas=array_([0.05, 0.05]),
    dgrids=array_([0.03, 0.03]),
    rcut=4.8,
    rpower=6,
    alchemy=False,
    normalize=True,
    cell=None,
):
    """
    Generate SLATM representation from/for only atoms with id `atom_id`. Otherwise both arguments and keyword arguments are the same as in `generate_slatm`.
    """
    all_atom_ids = array_(list(range(len(nuclear_charges))))
    all_atom_ids[atom_id] = 0
    all_atom_ids[0] = atom_id
    shifted_coordinates = coordinates[all_atom_ids]
    shifted_nuclear_charges = nuclear_charges[all_atom_ids]
    if cell is not None:
        shifted_coordinates, shifted_nuclear_charges, _, _ = extend_for_pbc(
            shifted_coordinates, shifted_nuclear_charges, len(nuclear_charges), rcut, cell
        )
    return generate_slatm_basic(
        shifted_nuclear_charges,
        shifted_coordinates,
        mbtypes,
        1,
        local=local,
        sigmas=sigmas,
        dgrids=dgrids,
        rcut=rcut,
        rpower=rpower,
        alchemy=alchemy,
        normalize=normalize,
    )[0]
