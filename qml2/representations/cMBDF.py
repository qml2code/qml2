# TODO: Find a way to replace np.gradient with jit_interfaces.gradient_.
import numpy as np
from scipy.fft import next_fast_len
from scipy.signal import fftconvolve

from ..jit_interfaces import (
    abs_,
    arange_,
    arccos_,
    array_,
    cbrt_,
    concatenate_,
    cos_,
    dot_,
    einsum_,
    exp_,
    float_,
    int_,
    jit_,
    l2_norm_,
    max_,
    min_,
    pi_,
    prange_,
    repeat_,
    sign_,
    sin_,
    sort_,
    sqrt_,
    tiny_,
    unique_,
    zeros_,
)
from ..test_utils.toy_representation import all_atoms_relevant
from .basic_utilities import extend_for_pbc, representation_numba_parallelization


@jit_
def hermite_polynomial(x, degree: int_, a: float_ = 1):
    if degree == 1:
        return -2 * a * x
    elif degree == 2:
        x1 = (a * x) ** 2
        return 4 * x1 - 2 * a
    elif degree == 3:
        x1 = (a * x) ** 3
        return -8 * x1 + 12 * a * x
    elif degree == 4:
        x1 = (a * x) ** 4
        x2 = (a * x) ** 2
        return 16 * x1 - 48 * x2 + 12 * a**2
    elif degree == 5:
        x1 = (a * x) ** 5
        x2 = (a * x) ** 3
        return -32 * x1 + 160 * x2 - 120 * (a * x)


@jit_
def fcut(Rij, rcut, grad=False):
    if grad:
        arg = (pi_ * Rij) / rcut
        return 0.5 * (cos_(arg) + 1), (-pi_ * sin_(arg)) / (2 * rcut)
    else:
        return 0.5 * (cos_((pi_ * Rij) / rcut) + 1)


@jit_
def cut_to_abs_unity(val):
    if abs_(val) > 1.0:
        return sign_(val)
    else:
        return val


@jit_(numba_parallel=representation_numba_parallelization(), numba_cache=True)
def generate_data_with_gradients(
    size, charges, coords, rconvs_arr, aconvs_arr, cutoff_r: float_ = 12.0, n_atm: float_ = 2.0
):
    rconvs, drconvs = rconvs_arr[0], rconvs_arr[1]
    aconvs, daconvs = aconvs_arr[0], aconvs_arr[1]
    m1, n1 = rconvs.shape[0], rconvs.shape[1]
    m2, n2 = aconvs.shape[0], aconvs.shape[1]
    nrs = m1 * n1
    nAs = m2 * n2
    rstep = cutoff_r / rconvs.shape[-1]
    astep = pi_ / (aconvs.shape[-1])
    twob = zeros_((size, size, nrs))
    threeb = zeros_((size, size, size, nAs))
    twob_temps = zeros_((size, size, 8))
    threeb_temps = zeros_((size, size, size, 9))

    for i in prange_(size):
        z, atom = charges[i], coords[i]

        for j in range(i + 1, size):
            rij = atom - coords[j]
            rij_norm = l2_norm_(rij)

            if rij_norm != 0 and rij_norm < cutoff_r:
                grad_dist = rij / rij_norm
                z2 = charges[j]
                arg = (pi_ * rij_norm) / cutoff_r
                fcutij, gfcut = 0.5 * (cos_(arg) + 1), (-pi_ * sin_(arg)) / (2 * cutoff_r)
                ind = rij_norm / rstep
                ac = sqrt_(z * z2)
                pref = ac * fcutij
                twob_temps[i][j][:5] = ac, fcutij, gfcut, rij_norm, ind
                twob_temps[j][i][:5] = ac, fcutij, gfcut, rij_norm, ind
                twob_temps[i][j][5:] = grad_dist
                twob_temps[j][i][5:] = -grad_dist
                ind = int(ind)
                id2 = 0
                for i1 in range(m1):
                    for i2 in range(n1):
                        conv = pref * rconvs[i1][i2][ind]
                        twob[i][j][id2] = conv
                        twob[j][i][id2] = conv
                        id2 += 1

                for k in range(j + 1, size):
                    rik = atom - coords[k]
                    rik_norm = l2_norm_(rik)

                    if rik_norm != 0 and rik_norm < cutoff_r:
                        z3 = charges[k]

                        rkj = coords[k] - coords[j]

                        rkj_norm = l2_norm_(rkj)

                        fcutik, fcutjk = fcut(rik_norm, cutoff_r), fcut(rkj_norm, cutoff_r)
                        fcut_tot = fcutij * fcutik * fcutjk

                        cos1 = min_(1.0, max_(dot_(rij, rik) / (rij_norm * rik_norm), -1.0))
                        cos2 = min_(1.0, max_(dot_(-rij, -rkj) / (rij_norm * rkj_norm), -1.0))
                        cos3 = min_(1.0, max_(dot_(rkj, -rik) / (rkj_norm * rik_norm), -1.0))

                        ang1 = arccos_(cos1)
                        ang2 = arccos_(cos2)
                        ang3 = arccos_(cos3)
                        sin1, sin2, sin3 = (
                            abs_(sin_(ang1)),
                            abs_(sin_(ang2)),
                            abs_(sin_(ang3)),
                        )

                        ind1 = ang1 / astep
                        ind2 = ang2 / astep
                        ind3 = ang3 / astep

                        atm_temp = rij_norm * rik_norm * rkj_norm
                        atm = atm_temp**n_atm
                        atm1 = atm * atm_temp
                        atm = exp_(n_atm * (rij_norm + rik_norm + rkj_norm))

                        charge = np.cbrt(z * z2 * z3)

                        pref = charge * fcut_tot

                        # atm=1.0
                        threeb_temps[i][j][k] = (
                            charge,
                            pref,
                            atm,
                            atm1,
                            ind1,
                            sin1,
                            cos1,
                            cos2,
                            cos3,
                        )
                        threeb_temps[i][k][j] = (
                            charge,
                            pref,
                            atm,
                            atm1,
                            ind1,
                            sin1,
                            cos1,
                            cos3,
                            cos2,
                        )
                        threeb_temps[j][i][k] = (
                            charge,
                            pref,
                            atm,
                            atm1,
                            ind2,
                            sin2,
                            cos2,
                            cos1,
                            cos3,
                        )
                        threeb_temps[j][k][i] = (
                            charge,
                            pref,
                            atm,
                            atm1,
                            ind2,
                            sin2,
                            cos2,
                            cos3,
                            cos1,
                        )
                        threeb_temps[k][i][j] = (
                            charge,
                            pref,
                            atm,
                            atm1,
                            ind3,
                            sin3,
                            cos3,
                            cos1,
                            cos2,
                        )
                        threeb_temps[k][j][i] = (
                            charge,
                            pref,
                            atm,
                            atm1,
                            ind3,
                            sin3,
                            cos3,
                            cos2,
                            cos1,
                        )

                        ind1, ind2, ind3 = int(ind1), int(ind2), int(ind3)

                        id2 = 0
                        for i1 in range(m2):
                            for i2 in range(n2):
                                conv1 = (pref * aconvs[i1][i2][ind1]) / atm
                                conv2 = (pref * aconvs[i1][i2][ind2]) / atm
                                conv3 = (pref * aconvs[i1][i2][ind3]) / atm

                                threeb[i][j][k][id2] = conv1
                                threeb[i][k][j][id2] = conv1

                                threeb[j][i][k][id2] = conv2
                                threeb[j][k][i][id2] = conv2

                                threeb[k][j][i][id2] = conv3
                                threeb[k][i][j][id2] = conv3

                                id2 += 1

    twob_grad = zeros_((size, nrs, size, 3))
    threeb_grad = zeros_((size, nAs, size, 3))

    for i in range(size):
        grad_temp = zeros_((nrs, 3))
        agrad_temp = zeros_((nAs, 3))

        for j in range(size):
            rij = coords[i] - coords[j]
            agrad_temp2 = zeros_((nAs, 3))

            if j != i:
                ac, fcutij, gfcutij, rij_norm, ind = twob_temps[i][j][:5]
                grad_dist = -twob_temps[i][j][5:]
                if ac != 0:
                    ind = int(ind)

                    gradfcut = gfcutij * grad_dist

                    pref = ac * fcutij
                    id2 = 0

                    for i1 in range(m1):
                        for i2 in range(n1):
                            grad1 = pref * drconvs[i1][i2][ind] * grad_dist
                            grad2 = (twob[i][j][id2] * gradfcut) / fcutij
                            twob_grad[i][id2][j] = grad1 + grad2

                            grad_temp[id2] += -(grad1 + grad2)
                            id2 += 1

                for k in range(size):
                    if k != j and k != i:
                        rkj = coords[k] - coords[j]
                        rik = coords[i] - coords[k]
                        fcutik, gfcutik, rik_norm = twob_temps[i][k][1:4]
                        fcutjk, gfcutjk, rjk_norm = twob_temps[j][k][
                            1:4
                        ]  # K.Karan: is rjk_norm needed anywhere?
                        ac, pref, atm, atm1, ind1, sin1, cos1, cos2, cos3 = threeb_temps[i][j][k]
                        if ac != 0:
                            grad_distjk, grad_distik = twob_temps[j][k][5:], twob_temps[i][k][5:]
                            ind1 = int(ind1)

                            temp = gradfcut * fcutik * fcutjk
                            gradfcuti = -temp + (gfcutik * grad_distik * fcutij * fcutjk)
                            gradfcutj = temp + (gfcutjk * grad_distjk * fcutik * fcutij)

                            atm_gradi = -n_atm * (grad_distik - grad_dist) / atm
                            atm_gradj = -n_atm * (grad_distjk + grad_dist) / atm

                            denom = sin1 * rij_norm * rik_norm
                            if denom < tiny_:
                                denom += 2 * tiny_
                            gang1i = (
                                -(
                                    (cos1 * (-(rij_norm * grad_distik) + (rik_norm * grad_dist)))
                                    + ((rij + rik))
                                )
                                / denom
                            )
                            gang1j = -(-rik - (grad_dist * cos1 * rik_norm)) / denom

                            id2 = 0
                            for i1 in range(m2):
                                for i2 in range(n2):
                                    af, daf = aconvs[i1][i2][ind1], daconvs[i1][i2][ind1]
                                    grad1 = pref * daf
                                    grad2 = ac * af
                                    grad3 = pref * af
                                    agrad_temp2[id2] += (
                                        (grad1 * gang1j) + (grad2 * gradfcutj)
                                    ) / atm + (grad3 * atm_gradj)
                                    agrad_temp[id2] += (
                                        (grad1 * gang1i) + (grad2 * gradfcuti)
                                    ) / atm + (grad3 * atm_gradi)
                                    id2 += 1

            threeb_grad[i, :, j, :] = agrad_temp2

        twob_grad[i, :, i, :] = grad_temp

        threeb_grad[i, :, i, :] = agrad_temp

    return twob, twob_grad, threeb, threeb_grad


@jit_(numba_parallel=representation_numba_parallelization(), numba_cache=True)
def generate_data(
    size: int_,
    charges,
    coords,
    rconvs,
    aconvs,
    cutoff_r: float_ = 10.0,
    n_atm: float_ = 2.0,
    pi_: float_ = pi_,
):
    rconvs, aconvs = rconvs[0], aconvs[0]
    m1, n1 = rconvs.shape[0], rconvs.shape[1]
    m2, n2 = aconvs.shape[0], aconvs.shape[1]
    nrs = m1 * n1
    nAs = m2 * n2
    rstep = cutoff_r / rconvs.shape[-1]
    astep = pi_ / aconvs.shape[-1]
    twob = zeros_((size, size, nrs))
    threeb = zeros_((size, size, size, nAs))

    for i in prange_(size):
        z, atom = charges[i], coords[i]

        for j in range(i + 1, size):
            rij = atom - coords[j]
            rij_norm = l2_norm_(rij)

            if rij_norm != 0 and rij_norm < cutoff_r:
                z2 = charges[j]
                ind = int(rij_norm / rstep)
                pref = sqrt_(z * z2)

                id2 = 0
                for i1 in range(m1):
                    for i2 in range(n1):
                        conv = pref * rconvs[i1][i2][ind]
                        twob[i][j][id2] = conv
                        twob[j][i][id2] = conv
                        id2 += 1

                for k in range(j + 1, size):
                    rik = atom - coords[k]
                    rik_norm = l2_norm_(rik)

                    if rik_norm != 0 and rik_norm < cutoff_r:
                        z3 = charges[k]

                        rkj = coords[k] - coords[j]

                        rkj_norm = l2_norm_(rkj)

                        cos1 = cut_to_abs_unity(dot_(rij, rik) / (rij_norm * rik_norm))
                        cos2 = cut_to_abs_unity(dot_(rij, rkj) / (rij_norm * rkj_norm))
                        cos3 = cut_to_abs_unity(dot_(-rkj, rik) / (rkj_norm * rik_norm))
                        ang1 = arccos_(cos1)
                        ang2 = arccos_(cos2)
                        ang3 = arccos_(cos3)

                        ind1 = int(ang1 / astep)
                        ind2 = int(ang2 / astep)
                        ind3 = int(ang3 / astep)

                        atm = (rij_norm * rik_norm * rkj_norm) ** n_atm

                        charge = cbrt_(z * z2 * z3)

                        pref = charge

                        id2 = 0
                        for i1 in range(m2):
                            for i2 in range(n2):
                                if i2 == 0:
                                    conv1 = (pref * aconvs[i1][i2][ind1] * cos2 * cos3) / atm
                                    conv2 = (pref * aconvs[i1][i2][ind2] * cos1 * cos3) / atm
                                    conv3 = (pref * aconvs[i1][i2][ind3] * cos2 * cos1) / atm
                                else:
                                    conv1 = (pref * aconvs[i1][i2][ind1]) / atm
                                    conv2 = (pref * aconvs[i1][i2][ind2]) / atm
                                    conv3 = (pref * aconvs[i1][i2][ind3]) / atm

                                threeb[i][j][k][id2] = conv1
                                threeb[i][k][j][id2] = conv1

                                threeb[j][i][k][id2] = conv2
                                threeb[j][k][i][id2] = conv2

                                threeb[k][j][i][id2] = conv3
                                threeb[k][i][j][id2] = conv3

                                id2 += 1

    return twob, threeb


def get_convolutions(
    rstep=0.0008,
    rcut=10.0,
    alpha_list=[1.5, 5.0],
    n_list=[3.0, 5.0],
    order=4,
    a1=2.0,
    a2=2.0,
    astep=0.0002,
    nAs=4,
):
    """
    returns convolutions required for computing cMBDF functionals evaluated via Fast Fourier Transforms on a discretized grid
    """
    step_r = rcut / next_fast_len(int(rcut / rstep))
    astep = pi_ / next_fast_len(int(pi_ / astep))
    rgrid = arange_(0.0, rcut, step_r)
    rgrid2 = arange_(-rcut, rcut, step_r)
    agrid = arange_(0.0, pi_, astep)
    agrid2 = arange_(-pi_, pi_, astep)

    size = len(rgrid)
    gaussian = exp_(-a1 * (rgrid2**2)) * sqrt_(a1 / pi_)

    m = order + 1

    temp1, temp2 = [], []
    dtemp1, dtemp2 = [], []

    fms = [
        gaussian,
        *[
            gaussian
            * hermite_polynomial(rgrid2, i, sqrt_(array_(a1)))
            * ((-sqrt_(array_(a1))) ** m)
            for i in range(1, m + 1)
        ],
    ]

    for i in range(m):
        fm = fms[i]

        temp, dtemp = [], []
        for alpha in alpha_list:
            gn = exp_(-alpha * rgrid)
            arr = array_(fftconvolve(gn, fm, mode="same")) * step_r
            arr = arr / max_(abs_(arr))
            temp.append(arr)
            darr = np.gradient(arr, step_r)
            dtemp.append(darr)
        temp1.append(array_(temp))
        dtemp1.append(array_(dtemp))

        temp, dtemp = [], []
        for n in n_list:
            gn = (rgrid + 1) ** n
            arr = array_(fftconvolve(1 / gn, fm, mode="same")[:size]) * step_r
            arr = arr / max_(abs_(arr))
            temp.append(arr)
            darr = np.gradient(arr, step_r)
            dtemp.append(darr)
        temp2.append(array_(temp))
        dtemp2.append(array_(dtemp))

    rconvs = concatenate_((array_(temp1), array_(temp2)), axis=1)
    drconvs = concatenate_((array_(dtemp1), array_(dtemp2)), axis=1)

    size = len(agrid)
    gaussian = exp_(-a2 * (agrid2**2)) * sqrt_(a2 / pi_)

    m = order + 1

    temp1, dtemp1 = [], []

    fms = [
        gaussian,
        *[
            gaussian
            * hermite_polynomial(agrid2, i, sqrt_(array_(a2)))
            * ((-sqrt_(array_(a2))) ** m)
            for i in range(1, m + 1)
        ],
    ]

    for i in range(m):
        fm = fms[i]

        temp, dtemp = [], []
        for n in range(1, nAs + 1):
            gn = cos_(n * agrid)
            arr = array_(fftconvolve(gn, fm, mode="same")) * astep
            arr = arr / max_(abs_(arr))
            temp.append(arr)
            darr = np.gradient(arr, astep)
            dtemp.append(darr)
        temp1.append(array_(temp))
        dtemp1.append(array_(dtemp))

    aconvs, daconvs = array_(temp1), array_(dtemp1)

    return array_([rconvs, drconvs]), array_([aconvs, daconvs])


def get_asize(charges):
    keys = unique_(concatenate_(charges))
    asize = {int(key): max([(mol == key).sum() for mol in charges]) for key in keys}
    return asize


def generate_cmbdf(
    charges,
    coords,
    convs,
    local=True,
    pad=None,
    asize=None,
    rcut=10.0,
    n_atm=2.0,
    cell=None,
    gradients=False,
):
    """ "
    Generate the convolutional Many Body Distribution Functionals (cMBDF) representation.
    Both global (``local=False``) and local (``local=True``) forms are available.

    NOTE: You will need to run the ``get_convolutions()`` function to get the ``convs`` input. It can be run once and the output array saved locally.

    :param charges: List of nuclear charges
    :type charges: numpy array
    :param coords: Input coordinates.
    :type coords: numpy array
    :param convs: Array of arrays containing convolutions evaluated on a discretized grid required for computing the cMBDF functionals
    :type convs: numpy array
    :param local: Generate a local (atomic) representation. Defaulted to True; otherwise, generates a flattened, bagged feature vector for the molecule
    :type local: bool
    :param pad: Integer denoting the largest number of atoms expected to be encountered within a molecule; if None (default) equals size of the molecule. Molecules with a smaller number of atoms are padded by zeros
    :type pad: int
    :param asize: Dictionary containing largest number of each unique chemical element within a single molecule from the entire dataset. Required for generating the global representatio, i.e. when ``local=False``.
                  Can be obtained by passing the list of nuclear charges for all molecules from your dataset to the ``get_asize`` function.
    :type asize: dict
    :param rcut: Radial cut-off radius around each atom, defaulted to 10 Angstrom. Note that raising this cutoff has no effect on representation size.
                Long range effects can be captured by raising this cut-off and using a slower decaying radial functional, e.g. pass ``n_list=[1.0,2.0]`` to the ``get_convolutions()`` function to capture Coulomb-type interactions.
    :type rcut: float
    :param n_atm: Power term dictating decay of Axilrod-Teller-Muto type 3-body interactions. Defaulted to 2.0.
    :type n_atm: float
    :param cell: if not None defines the cell for periodic boundary conditions. NOTE: for now only works for local representation without gradients.
    :type cell: numpy array or None
    :gradients: Whether to return gradients of the representation with respect to atomic positions as well. Defaulted to false which returns a single array containing the representation.
                NOTE: gradients have only been implemented for the local version hence ``local`` parameter must be set to True.
    :type gradients: bool

    :return: 2D cMBDF representation if ``local==True``, otherwise 1D cMBDF representation. If ``gradients`` is set to True then two arrays are returned containing the local cMBDF representation and its gradient respectively
    :rtype: numpy array(s)
    """
    rconvs, aconvs = convs
    size = len(charges)
    if cell is not None:
        assert (not gradients) and local and (pad is None)
        true_size = size
        coords, charges, size, _ = extend_for_pbc(coords, charges, size, rcut * 2, cell)

    m1, n1 = rconvs[0].shape[0], rconvs[0].shape[1]
    m2, n2 = aconvs[0].shape[0], aconvs[0].shape[1]
    nr = m1 * n1
    na = m2 * n2
    desc_size = nr + na

    if pad is None:
        pad = size

    if local:
        mat = zeros_((pad, desc_size))

        if gradients:
            dmat = zeros_((pad, desc_size, pad, 3))
            twob, twob_grad, threeb, threeb_grad = generate_data_with_gradients(
                size, charges, coords, rconvs, aconvs, rcut, n_atm
            )
            mat[:size, :nr] = einsum_("ij... -> i...", twob)
            mat[:size, nr:] = einsum_("ijk... -> i...", threeb)
            dmat[:size, :nr, :size, :] = twob_grad
            dmat[:size, nr:, :size, :] = threeb_grad

            # TODO: K.Karan: replace the dirty fix for relevant atoms.
            return mat, dmat, all_atoms_relevant(size, size), repeat_(size, size)

        else:
            twob, threeb = generate_data(size, charges, coords, rconvs, aconvs, rcut, n_atm)
            mat[:size, :nr] = einsum_("ij... -> i...", twob)
            mat[:size, nr:] = einsum_("ijk... -> i...", threeb)
            if cell is not None:
                mat = mat[:true_size]

            return mat

    keys = list(asize.keys())
    rep_size = sum(asize.values())
    elements = {k: [] for k in keys}
    for i in range(size):
        elements[int(charges[i])].append(i)

    mat, ind = zeros_((rep_size, desc_size)), 0

    twob, threeb = generate_data(size, charges, coords, rconvs, aconvs, rcut, n_atm)

    for key in keys:
        num = len(elements[key])

        bags = zeros_((num, desc_size))

        if num != 0:
            bags[:, :nr] = einsum_("ij... -> i...", twob[elements[key]])
            bags[:, nr:] = einsum_("ijk... -> i...", threeb[elements[key]])

            mat[ind : ind + num] = -sort_(-bags, axis=0)

        ind += asize[key]

    return mat.ravel()
