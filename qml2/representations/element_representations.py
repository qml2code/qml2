from ..jit_interfaces import zeros_

# K. Karan: Here I represent each element with:
# 0: period
# 1: number of valence s electrons
# 2: number of valence p electrons
# 3: number of valence d electrons
# Perhaps parts should be moved to qml2.data

PERIOD_VALENCE_REP = {
    1: [1, 1],
    2: [1, 2],
    3: [2, 1],
    4: [2, 2],
    5: [2, 2, 1],
    6: [2, 2, 2],
    7: [2, 2, 3],
    8: [2, 2, 4],
    9: [2, 2, 5],
    10: [2, 2, 6],
    11: [3, 1],
    12: [3, 2],
    13: [3, 2, 1],
    14: [3, 2, 2],
    15: [3, 2, 3],
    16: [3, 2, 4],
    17: [3, 2, 5],
    18: [3, 2, 6],
}


def period_valence_representation(nuclear_charges, rep_length=4):
    output = zeros_((len(nuclear_charges), rep_length))
    for i, nc in enumerate(nuclear_charges):
        nc_rep = PERIOD_VALENCE_REP[nc]
        lnc_rep = min(len(nc_rep), rep_length)
        output[i, :lnc_rep] = nc_rep[:lnc_rep]
    return output
