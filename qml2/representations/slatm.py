import itertools as itl
from typing import List

import numpy as np
from numba import jit, njit, prange
from numpy import exp, int64, ndarray, pi, sqrt
from scipy.spatial import distance as ssd


@jit(nopython=True)
def linspace(x0, x1, nx):
    step = (x1 - x0) / (nx - 1)
    xs = np.empty(nx)
    for i in range(nx):
        xs[i] = x0 + i * step
    return xs


@jit(nopython=True)
def calc_angle(a, b, c):
    v1 = a - b
    v2 = c - b
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    cos_angle = np.dot(v1, v2)
    # Clipping to ensure numerical stability
    cos_angle = min(1.0, max(cos_angle, -1.0))
    angle = np.arccos(cos_angle)
    return angle


@jit(nopython=True)
def calc_cos_angle(a, b, c):
    v1 = a - b
    v2 = c - b
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    cos_angle = np.dot(v1, v2)
    return cos_angle


@jit(nopython=True, parallel=True)
def get_sbot_global(coordinates, nuclear_charges, z1, z2, z3, rcut, nx, dgrid, sigma, coeff):
    natoms = coordinates.shape[0]
    pi = np.pi
    d2r = pi / 180.0
    a0 = -20.0 * d2r
    a1 = pi + 20.0 * d2r
    xs = linspace(a0, a1, nx)
    prefactor = 1.0 / 3.0
    c0 = prefactor * (z1 % 1000) * (z2 % 1000) * (z3 % 1000) * coeff * dgrid
    ys = np.zeros(nx)
    inv_sigma = -1.0 / (2 * sigma**2)

    # Compute distance matrix
    distance_matrix = np.zeros((natoms, natoms))
    for i in prange(natoms):
        for j in range(i + 1, natoms):
            norm = np.sqrt(np.sum((coordinates[j, :] - coordinates[i, :]) ** 2))
            distance_matrix[i, j] = norm
            distance_matrix[j, i] = norm

    # Finding indices of atoms with specified nuclear charges
    ias1 = np.where(nuclear_charges == z1)[0]
    ias2 = np.where(nuclear_charges == z2)[0]
    ias3 = np.where(nuclear_charges == z3)[0]

    cos_xs = np.cos(xs) * c0

    # Calculation loop
    for ia1 in prange(len(ias1)):
        for ia2 in range(len(ias2)):
            for ia3 in range(len(ias3)):
                if z1 == z3 and ia1 >= ia3:
                    continue
                i, j, k = ias1[ia1], ias2[ia2], ias3[ia3]
                if not (
                    0 < distance_matrix[i, j] <= rcut
                    and 0 < distance_matrix[i, k] <= rcut
                    and 0 < distance_matrix[j, k] <= rcut
                ):
                    continue

                ang = calc_angle(coordinates[i], coordinates[j], coordinates[k])
                cak = calc_cos_angle(coordinates[i], coordinates[k], coordinates[j])
                cai = calc_cos_angle(coordinates[k], coordinates[i], coordinates[j])

                r = distance_matrix[i, j] * distance_matrix[i, k] * distance_matrix[j, k]
                ys += (c0 + cos_xs * cak * cai) / (r**3) * np.exp((xs - ang) ** 2 * inv_sigma)

    return ys


@jit(nopython=True, parallel=True)
def get_sbop_global(coordinates, nuclear_charges, z1, z2, rcut, nx, dgrid, sigma, coeff, rpower):
    #    natoms = coordinates.shape[0]
    ys = np.zeros(nx)
    c0 = (z1 % 1000) * (z2 % 1000) * coeff
    inv_sigma = -0.5 / sigma**2
    xs = linspace(0.1, rcut, nx)
    xs0 = c0 / (xs**rpower) * dgrid
    rcut2 = rcut**2

    # Allocate arrays to hold indices of atoms with specific nuclear charges
    ias1 = np.nonzero(nuclear_charges == z1)[0]
    ias2 = np.nonzero(nuclear_charges == z2)[0]

    # Main loop to calculate ys
    for ia1 in prange(len(ias1)):
        for ia2 in range(len(ias2)):
            # When z1 == z2, avoid double counting pairs by ensuring ia2 > ia1
            if z1 == z2 and ia1 >= ia2:
                continue
            r2 = np.sum((coordinates[ias1[ia1]] - coordinates[ias2[ia2]]) ** 2)
            if r2 < rcut2:
                r = np.sqrt(r2)
                ys += xs0 * np.exp(inv_sigma * (xs - r) ** 2)

    return ys


@njit(parallel=True)
def get_sbot_local(
    coordinates, nuclear_charges, ia_python, z1, z2, z3, rcut, nx, dgrid, sigma, coeff
):
    eps = 2.2e-16
    natoms = coordinates.shape[0]
    if coordinates.shape[0] != nuclear_charges.size:
        raise ValueError(
            f"ERROR: Coulomb matrix generation, {coordinates.shape[0]} coordinates, but {nuclear_charges.size} atom_types!"
        )

    ias1, ias2, ias3 = [], [], []
    for i in range(natoms):
        if int(nuclear_charges[i]) == z1:
            ias1.append(i)
        if int(nuclear_charges[i]) == z2:
            ias2.append(i)
        if int(nuclear_charges[i]) == z3:
            ias3.append(i)

    if ia_python not in ias2:
        return

    distance_matrix = np.zeros((natoms, natoms))
    for i in prange(natoms):
        for j in range(i + 1, natoms):
            norm = sqrt(np.sum((coordinates[j] - coordinates[i]) ** 2))
            distance_matrix[i, j] = norm
            distance_matrix[j, i] = norm

    a0 = -20 * pi / 180
    a1 = pi + 20 * pi / 180
    xs = np.linspace(a0, a1, nx)
    cos_xs = np.cos(xs) * coeff * dgrid

    prefactor = 1.0 / 3.0
    c0 = prefactor * (z1 % 1000) * (z2 % 1000) * (z3 % 1000) * coeff * dgrid
    inv_sigma = -1.0 / (2 * sigma**2)

    ys = np.zeros(nx)

    for ia1 in prange(len(ias1)):
        for ia3 in prange(len(ias3)) if z1 == z3 else range(len(ias3)):
            i, j, k = ias1[ia1], ia_python, ias3[ia3]
            if not (
                eps < distance_matrix[i, j] <= rcut
                and eps < distance_matrix[i, k] <= rcut
                and eps < distance_matrix[j, k] <= rcut
            ):
                continue
            ang = calc_angle(coordinates[i], coordinates[j], coordinates[k])
            cak = calc_cos_angle(coordinates[i], coordinates[k], coordinates[j])
            cai = calc_cos_angle(coordinates[k], coordinates[i], coordinates[j])
            r = distance_matrix[i, j] * distance_matrix[i, k] * distance_matrix[j, k]
            ys += (c0 + cos_xs * cak * cai) / (r**3) * exp((xs - ang) ** 2 * inv_sigma)

    return ys


@njit(parallel=True)
def get_sbop_local(
    coordinates, nuclear_charges, ia_python, z1, z2, rcut, nx, dgrid, sigma, coeff, rpower
):
    natoms = coordinates.shape[0]
    if coordinates.shape[0] != nuclear_charges.size:
        raise ValueError(
            f"ERROR: Coulomb matrix generation, {coordinates.shape[0]} coordinates, but {nuclear_charges.size} atom_types!"
        )

    ias1 = []
    ias2 = []

    for i in range(natoms):
        if int(nuclear_charges[i]) == z1:
            ias1.append(i)
        if int(nuclear_charges[i]) == z2:
            ias2.append(i)

    r0 = 0.1
    xs = np.linspace(r0, rcut, nx)
    ys = np.zeros(nx)

    c0 = (z1 % 1000) * (z2 % 1000) * coeff
    inv_sigma = -0.5 / sigma**2
    xs0 = c0 / (xs**rpower) * dgrid

    rcut2 = rcut**2
    ia = ia_python + 1

    if z1 == z2:
        for ia1 in prange(len(ias1)):
            for ia2 in range(ia1 + 1, len(ias2)):
                if (ias1[ia1] != ia) and (ias2[ia2] != ia):
                    r = np.sum((coordinates[ias1[ia1], :] - coordinates[ias2[ia2], :]) ** 2)
                    if r < rcut2:
                        ys += xs0 * exp(inv_sigma * (xs - sqrt(r)) ** 2)
    else:
        for ia1 in prange(len(ias1)):
            for ia2 in range(len(ias2)):
                if (ias1[ia1] != ia) and (ias2[ia2] != ia):
                    r = np.sum((coordinates[ias1[ia1], :] - coordinates[ias2[ia2], :]) ** 2)
                    if r < rcut2:
                        ys += xs0 * exp(inv_sigma * (xs - sqrt(r)) ** 2)

    return ys


def update_m(obj, ia, rcut=9.0, pbc=None):
    """
    retrieve local structure around atom `ia
    for periodic systems (or very large system)
    """
    zs, coords, c = obj
    v1, v2, v3 = c
    vs = ssd.norm(c, axis=0)

    nns = []
    for i, vi in enumerate(vs):
        n1_doulbe = rcut / vi
        n1 = int(n1_doulbe)
        if n1 - n1_doulbe == 0:
            n1s = (
                range(-n1, n1 + 1)
                if pbc[i]
                else [
                    0,
                ]
            )
        elif n1 == 0:
            n1s = (
                [-1, 0, 1]
                if pbc[i]
                else [
                    0,
                ]
            )
        else:
            n1s = (
                range(-n1 - 1, n1 + 2)
                if pbc[i]
                else [
                    0,
                ]
            )

        nns.append(n1s)

    n1s, n2s, n3s = nns

    n123s_ = np.array(list(itl.product(n1s, n2s, n3s)))
    n123s = []
    for n123 in n123s_:
        n123u = list(n123)
        if n123u != [0, 0, 0]:
            n123s.append(n123u)

    nau = len(n123s)
    n123s = np.array(n123s, np.float)

    na = len(zs)
    cia = coords[ia]

    if na == 1:
        ds = np.array([[0.0]])
    else:
        ds = ssd.squareform(ssd.pdist(coords))

    zs_u = []
    coords_u = []
    zs_u.append(zs[ia])
    coords_u.append(coords[ia])
    for i in range(na):
        di = ds[i, ia]
        if (di > 0) and (di <= rcut):
            zs_u.append(zs[i])
            coords_u.append(coords[ia])

            ts = np.zeros((nau, 3))
            for iau in range(nau):
                ts[iau] = np.dot(n123s[iau], c)

            coords_iu = coords[i] + ts
            dsi = ssd.norm(coords_iu - cia, axis=1)
            filt = np.logical_and(dsi > 0, dsi <= rcut)
            nx = filt.sum()
            zs_u += [
                zs[i],
            ] * nx
            coords_u += [
                list(coords_iu[filt, :]),
            ]

    obj_u = [zs_u, coords_u]

    return obj_u


def get_boa(z1, zs_):
    return z1 * np.array(
        [
            (zs_ == z1).sum(),
        ]
    )


def get_sbop(
    mbtype,
    obj,
    iloc=False,
    ia=None,
    normalize=True,
    sigma=0.05,
    rcut=4.8,
    dgrid=0.03,
    pbc="000",
    rpower=6,
):
    """
    two-body terms

    :param obj: molecule object, consisting of two parts: [ zs, coords ]
    :type obj: list
    """

    z1, z2 = mbtype
    zs, coords, c = obj

    if iloc:
        assert ia is not None, "#ERROR: plz specify `za and `ia "

    if pbc != "000":
        if rcut < 9.0:
            raise "#ERROR: rcut too small for systems with pbc"
        assert iloc, "#ERROR: for periodic system, plz use atomic rpst"
        zs, coords = update_m(obj, ia, rcut=rcut, pbc=pbc)

        # after update of `m, the query atom `ia will become the first atom
        ia = 0

    # bop potential distribution
    r0 = 0.1
    nx = int((rcut - r0) / dgrid) + 1

    coeff = 1 / np.sqrt(2 * sigma**2 * np.pi) if normalize else 1.0

    if iloc:
        ys = get_sbop_local(coords, zs, ia, z1, z2, rcut, nx, dgrid, sigma, coeff, rpower)
    else:
        ys = get_sbop_global(coords, zs, z1, z2, rcut, nx, dgrid, sigma, coeff, rpower)

    return ys


def get_sbot(
    mbtype, obj, iloc=False, ia=None, normalize=True, sigma=0.05, rcut=4.8, dgrid=0.0262, pbc="000"
):
    """
    sigma -- standard deviation of gaussian distribution centered on a specific angle
            defaults to 0.05 (rad), approximately 3 degree
    dgrid    -- step of angle grid
            defaults to 0.0262 (rad), approximately 1.5 degree
    """

    z1, z2, z3 = mbtype
    zs, coords, c = obj

    if iloc:
        assert ia is not None, "#ERROR: plz specify `za and `ia "

    if pbc != "000":
        assert iloc, "#ERROR: for periodic system, plz use atomic rpst"
        zs, coords = update_m(obj, ia, rcut=rcut, pbc=pbc)

        # after update of `m, the query atom `ia will become the first atom
        ia = 0

    # for a normalized gaussian distribution, u should multiply this coeff
    coeff = 1 / np.sqrt(2 * sigma**2 * np.pi) if normalize else 1.0

    # Setup grid in Python
    d2r = np.pi / 180  # degree to rad
    a0 = -20.0 * d2r
    a1 = np.pi + 20.0 * d2r
    nx = int((a1 - a0) / dgrid) + 1

    if iloc:
        ys = get_sbot_local(coords, zs, ia, z1, z2, z3, rcut, nx, dgrid, sigma, coeff)
    else:
        # eps_float64 = np.finfo(np.float64).eps
        ys = get_sbot_global(coords, zs, z1, z2, z3, rcut, nx, dgrid, sigma, coeff)

    return ys


def generate_slatm(
    nuclear_charges,
    coordinates,
    mbtypes,
    unit_cell=None,
    local=False,
    sigmas=[0.05, 0.05],
    dgrids=[0.03, 0.03],
    rcut=4.8,
    alchemy=False,
    pbc="000",
    rpower=6,
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
    :param alchemy: Swith to use the alchemy version of SLATM. (default=False)
    :type alchemy: bool
    :param pbc: defaulted to '000', meaning it's a molecule; the three digits in the string corresponds to x,y,z direction
    :type pbc: string
    :param rpower: The power of R in 2-body potential, defaulted to London potential (=6).
    :type rpower: float
    :return: 1D SLATM representation
    :rtype: numpy array
    """

    c = unit_cell
    #    iprt = False
    if c is None:
        c = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    if pbc != "000":
        # print(' -- handling systems with periodic boundary condition')
        assert c is not None, "ERROR: Please specify unit cell for SLATM"
        # =======================================================================
        # PBC may introduce new many-body terms, so at the stage of get statistics
        # info from db, we've already considered this point by letting maximal number
        # of nuclear charges being 3.
        # =======================================================================

    zs = nuclear_charges
    na = len(zs)
    coords = coordinates
    obj = [zs, coords, c]

    iloc = local

    if iloc:
        mbs = []
        X2Ns = []
        for ia in range(na):
            # if iprt: print '               -- ia = ', ia + 1
            n1 = 0
            n2 = 0
            n3 = 0
            mbs_ia = np.zeros(0)
            #            icount = 0
            for mbtype in mbtypes:
                if len(mbtype) == 1:
                    mbsi = get_boa(
                        mbtype[0],
                        np.array(
                            [
                                zs[ia],
                            ]
                        ),
                    )
                    # print ' -- mbsi = ', mbsi
                    if alchemy:
                        n1 = 1
                        n1_0 = mbs_ia.shape[0]
                        if n1_0 == 0:
                            mbs_ia = np.concatenate((mbs_ia, mbsi), axis=0)
                        elif n1_0 == 1:
                            mbs_ia += mbsi
                        else:
                            raise "#ERROR"
                    else:
                        n1 += len(mbsi)
                        mbs_ia = np.concatenate((mbs_ia, mbsi), axis=0)
                elif len(mbtype) == 2:
                    # print ' 001, pbc = ', pbc
                    mbsi = get_sbop(
                        mbtype,
                        obj,
                        iloc=iloc,
                        ia=ia,
                        sigma=sigmas[0],
                        dgrid=dgrids[0],
                        rcut=rcut,
                        pbc=pbc,
                        rpower=rpower,
                    )
                    mbsi *= 0.5  # only for the two-body parts, local rpst
                    # print ' 002'
                    if alchemy:
                        n2 = len(mbsi)
                        n2_0 = mbs_ia.shape[0]
                        if n2_0 == n1:
                            mbs_ia = np.concatenate((mbs_ia, mbsi), axis=0)
                        elif n2_0 == n1 + n2:
                            t = mbs_ia[n1 : n1 + n2] + mbsi
                            mbs_ia[n1 : n1 + n2] = t
                        else:
                            raise "#ERROR"
                    else:
                        n2 += len(mbsi)
                        mbs_ia = np.concatenate((mbs_ia, mbsi), axis=0)
                else:  # len(mbtype) == 3:
                    mbsi = get_sbot(
                        mbtype,
                        obj,
                        iloc=iloc,
                        ia=ia,
                        sigma=sigmas[1],
                        dgrid=dgrids[1],
                        rcut=rcut,
                        pbc=pbc,
                    )

                    # print(mbsi)
                    if alchemy:
                        n3 = len(mbsi)
                        n3_0 = mbs_ia.shape[0]
                        if n3_0 == n1 + n2:
                            mbs_ia = np.concatenate((mbs_ia, mbsi), axis=0)
                        elif n3_0 == n1 + n2 + n3:
                            t = mbs_ia[n1 + n2 : n1 + n2 + n3] + mbsi
                            mbs_ia[n1 + n2 : n1 + n2 + n3] = t
                        else:
                            raise "#ERROR"
                    else:
                        n3 += len(mbsi)
                        mbs_ia = np.concatenate((mbs_ia, mbsi), axis=0)

            mbs.append(mbs_ia)
            X2N = [n1, n2, n3]
            if X2N not in X2Ns:
                X2Ns.append(X2N)
        assert len(X2Ns) == 1, "#ERROR: multiple `X2N ???"
    else:
        n1 = 0
        n2 = 0
        n3 = 0
        mbs = np.zeros(0)
        for mbtype in mbtypes:
            if len(mbtype) == 1:
                mbsi = get_boa(mbtype[0], zs)
                if alchemy:
                    n1 = 1
                    n1_0 = mbs.shape[0]
                    if n1_0 == 0:
                        mbs = np.concatenate((mbs, [sum(mbsi)]), axis=0)
                    elif n1_0 == 1:
                        mbs += sum(mbsi)
                    else:
                        raise "#ERROR"
                else:
                    n1 += len(mbsi)
                    mbs = np.concatenate((mbs, mbsi), axis=0)
            elif len(mbtype) == 2:
                mbsi = get_sbop(
                    mbtype, obj, sigma=sigmas[0], dgrid=dgrids[0], rcut=rcut, rpower=rpower
                )

                if alchemy:
                    n2 = len(mbsi)
                    n2_0 = mbs.shape[0]
                    if n2_0 == n1:
                        mbs = np.concatenate((mbs, mbsi), axis=0)
                    elif n2_0 == n1 + n2:
                        t = mbs[n1 : n1 + n2] + mbsi
                        mbs[n1 : n1 + n2] = t
                    else:
                        raise "#ERROR"
                else:
                    n2 += len(mbsi)
                    mbs = np.concatenate((mbs, mbsi), axis=0)
            else:  # len(mbtype) == 3:
                mbsi = get_sbot(mbtype, obj, sigma=sigmas[1], dgrid=dgrids[1], rcut=rcut)

                if alchemy:
                    n3 = len(mbsi)
                    n3_0 = mbs.shape[0]
                    if n3_0 == n1 + n2:
                        mbs = np.concatenate((mbs, mbsi), axis=0)
                    elif n3_0 == n1 + n2 + n3:
                        t = mbs[n1 + n2 : n1 + n2 + n3] + mbsi
                        mbs[n1 + n2 : n1 + n2 + n3] = t
                    else:
                        raise "#ERROR"
                else:
                    n3 += len(mbsi)
                    mbs = np.concatenate((mbs, mbsi), axis=0)

    return mbs


def get_slatm_mbtypes(nuclear_charges: List[ndarray], pbc: str = "000") -> List[List[int64]]:
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

    zs = nuclear_charges

    nm = len(zs)
    zsmax = set()
    nas = []
    zs_ravel = []
    for zsi in zs:
        na = len(zsi)
        nas.append(na)
        zsil = list(zsi)
        zs_ravel += zsil
        zsmax.update(zsil)

    zsmax = np.array(list(zsmax))
    nass = []
    for i in range(nm):
        zsi = np.array(zs[i], np.int32)
        nass.append([(zi == zsi).sum() for zi in zsmax])

    nzmax = np.max(np.array(nass), axis=0)
    nzmax_u = []
    if pbc != "000":
        # the PBC will introduce new many-body terms, so set
        # nzmax to 3 if it's less than 3
        for nzi in nzmax:
            if nzi <= 2:
                nzi = 3
            nzmax_u.append(nzi)
        nzmax = nzmax_u

    boas = [
        [
            zi,
        ]
        for zi in zsmax
    ]

    bops = [[zi, zi] for zi in zsmax] + [list(x) for x in itl.combinations(zsmax, 2)]

    bots = []
    for i in zsmax:
        for bop in bops:
            j, k = bop
            tas = [[i, j, k], [i, k, j], [j, i, k]]
            for tasi in tas:
                if (tasi not in bots) and (tasi[::-1] not in bots):
                    nzsi = [(zj == tasi).sum() for zj in zsmax]
                    if np.all(nzsi <= nzmax):
                        bots.append(tasi)
    mbtypes = boas + bops + bots

    return mbtypes  # , np.array(zs_ravel), np.array(nas)
