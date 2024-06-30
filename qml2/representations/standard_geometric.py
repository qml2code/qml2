import numpy as np
from numba import njit, prange

from ..jit_interfaces import argsort_, flip_, int_, jit_, l2_norm_, ndarray_, zeros_


@jit_
def extract_lower_triangle_diagonal(matrix: ndarray_, size: int_):
    result = zeros_((size * (size + 1) // 2,))
    idx = 0
    size = matrix.shape[0]
    for i in range(size):
        for j in range(i + 1):
            result[idx] = matrix[i, j]
            idx += 1
    return result


# TODO KK: Implement other sortings?
@jit_
def generate_coulomb_matrix(atomic_charges: ndarray_, coordinates: ndarray_, size: int_ = 29):
    natoms = atomic_charges.shape[0]
    cm = zeros_((natoms, natoms))

    summation = zeros_((natoms,))

    for i in range(natoms):
        for j in range(i, natoms):
            if i == j:
                cm[i, j] = 0.5 * (atomic_charges[i] ** 2.4)
            else:
                dist = l2_norm_(coordinates[i, :] - coordinates[j, :], ord=2)
                cm[i, j] = (atomic_charges[i] * atomic_charges[j]) / dist
                cm[j, i] = cm[i, j]
                summation[j] += cm[i, j] ** 2
            summation[i] += cm[i, j] ** 2

    # Sort the matrix by row norms
    sorted_indices = flip_(argsort_(summation))
    sorted_mat = cm[sorted_indices][:, sorted_indices]

    result = extract_lower_triangle_diagonal(sorted_mat, size)

    return result


@njit
def calculate_pair_norm(coords_i, coords_j, charge_i, charge_j):
    return charge_i * charge_j / np.linalg.norm(coords_i - coords_j)


@njit
def get_indices(nuclear_charges, type1):
    n = 0
    type1_indices = np.zeros(nuclear_charges.size, dtype=np.int32)
    for j in range(nuclear_charges.shape[0]):
        if nuclear_charges[j] == type1:
            type1_indices[n] = j
            n += 1
    return type1_indices[:n]


@njit(parallel=True)
def generate_bob(nuclear_charges, coordinates, bags, ncm=435, id=np.array([1, 6, 7, 8, 9])):
    natoms = nuclear_charges.shape[0]
    #    bags = np.array(list(bags.values()))
    nid = id.shape[0]
    #    ncm = compute_ncm(bags)
    pair_distance_matrix = np.zeros((natoms, natoms), dtype=np.float64)

    for i in prange(natoms):
        for j in range(i + 1, natoms):
            pair_distance_matrix[i, j] = calculate_pair_norm(
                coordinates[i], coordinates[j], nuclear_charges[i], nuclear_charges[j]
            )
            pair_distance_matrix[j, i] = pair_distance_matrix[i, j]

    cm = np.zeros(ncm, dtype=np.float64)
    start_indices = np.zeros(nid, dtype=np.int32)

    start_indices[0] = 0
    for i in range(1, nid):
        start_indices[i] = start_indices[i - 1] + (bags[i - 1] * (bags[i - 1] + 1)) // 2
        for j in range(i, nid):
            start_indices[i] += bags[j] * bags[i - 1]

    for i in prange(nid):
        type1 = id[i]
        type1_indices = get_indices(nuclear_charges, type1)
        natoms1 = len(type1_indices)

        bag = np.zeros(bags[i] * (bags[i] - 1) // 2, dtype=np.float64)

        for j in range(natoms1):
            idx1 = type1_indices[j]
            cm[start_indices[i] + j] = 0.5 * nuclear_charges[idx1] ** 2.4
            k = (j * j - j) // 2
            for l in range(j):
                idx2 = type1_indices[l]
                bag[k + l] = pair_distance_matrix[idx1, idx2]

        start_indices[i] += bags[i]

        nbag = (bags[i] * bags[i] - bags[i]) // 2
        cm[start_indices[i] : start_indices[i] + nbag] = -np.sort(-bag[:nbag])
        start_indices[i] += nbag

        for j in prange(i + 1, nid):
            type2 = id[j]
            type2_indices = get_indices(nuclear_charges, type2)
            natoms2 = len(type2_indices)

            bag = np.zeros(bags[i] * bags[j], dtype=np.float64)
            for k in range(natoms1):
                idx1 = type1_indices[k]
                for l in range(natoms2):
                    idx2 = type2_indices[l]
                    bag[natoms2 * k + l] = pair_distance_matrix[idx1, idx2]

            nbag = bags[i] * bags[j]
            cm[start_indices[i] : start_indices[i] + nbag] = -np.sort(-bag[:nbag])
            start_indices[i] += nbag

    return cm
