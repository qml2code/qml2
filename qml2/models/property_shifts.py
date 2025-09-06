from ..jit_interfaces import empty_, jit_, prange_, sum_
from ..math import lstsq_solve
from ..utils import get_numba_list


@jit_
def element_counts(possible_nuclear_charges, nuclear_charges, output):
    for i in range(possible_nuclear_charges.shape[0]):
        output[i] = sum_(nuclear_charges == possible_nuclear_charges[i])


@jit_
def get_shift_coeffs(
    npoints,
    intensive_shift=True,
    extensive_shift=False,
    possible_nuclear_charges=None,
    all_nuclear_charges=None,
):
    nshifts = 0
    if intensive_shift:
        nshifts += 1
    if extensive_shift:
        assert all_nuclear_charges is not None
        assert possible_nuclear_charges is not None
        nelements = possible_nuclear_charges.shape[0]
        nshifts += nelements
    shift_coeffs = empty_((npoints, nshifts))
    if intensive_shift:
        shift_coeffs[:, -1] = 1.0
    # NOTE: "(all_nuclear_charges is not None) and (possible_nuclear_charges is not None)" here is redundant in normal Python, but is required to prevent
    # Numba's JIT compiler from thinking that "all_nuclear_charges[i]" can refer to "None[i]".
    if (
        extensive_shift
        and (all_nuclear_charges is not None)
        and (possible_nuclear_charges is not None)
    ):
        for i in prange_(npoints):
            element_counts(possible_nuclear_charges, all_nuclear_charges[i], shift_coeffs[i])
    return shift_coeffs


# K.Karan: Perhaps should be replaced with a standard library reference, I originally wrote the formulas explicitly
# in order to try something with variances of individual shifts, only to realize that I can just use BFGS for shifts.
def get_optimal_shift_guesses(
    quantity_vals,
    intensive_shift=True,
    extensive_shift=False,
    possible_nuclear_charges=None,
    all_nuclear_charges=None,
    shift_coeffs=None,
):
    npoints = len(quantity_vals)
    if shift_coeffs is None:
        if all_nuclear_charges is not None:
            all_nuclear_charges = get_numba_list(all_nuclear_charges)
        shift_coeffs = get_shift_coeffs(
            npoints,
            intensive_shift=intensive_shift,
            extensive_shift=extensive_shift,
            possible_nuclear_charges=possible_nuclear_charges,
            all_nuclear_charges=all_nuclear_charges,
        )
    optimal_shifts = lstsq_solve(shift_coeffs, quantity_vals)
    return optimal_shifts
