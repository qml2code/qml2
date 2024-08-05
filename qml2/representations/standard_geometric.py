from ..jit_interfaces import (
    argsort_,
    array_,
    dint_,
    dtype_,
    flip_,
    int_,
    jit_,
    l2_norm_,
    ndarray_,
    prange_,
    sort_,
    where_,
    zeros_,
)


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


@jit_
def calculate_pair_norm(coords_i, coords_j, charge_i, charge_j):
    return charge_i * charge_j / l2_norm_(coords_i - coords_j)


@jit_
def get_indices(nuclear_charges, type1: int, dint_: dtype_ = dint_):
    n = 0
    type1_indices = zeros_(nuclear_charges.shape, dtype=dint_)
    for j in range(nuclear_charges.shape[0]):
        if nuclear_charges[j] == type1:
            type1_indices[n] = j
            n += 1
    return type1_indices[:n]


def get_bob_bags(nuclear_charges_list, elements=array_([1, 6, 7, 8, 16])):
    bags = zeros_(elements.shape, dtype=dint_)
    for nuclear_charges in nuclear_charges_list:
        for element_id, element in enumerate(elements):
            element_count = where_(nuclear_charges == element)[0].shape[0]
            bags[element_id] = max(element_count, bags[element_id])
    return bags


# def compute_ncm(bags):
#    ncm = 0
#    keys = list(bags.keys())
#    for i, key in enumerate(keys):
#        ncm += bags[key] * (1 + bags[key])
#        for j in range(i):
#            ncm += 2 * bags[key] * bags[keys[j]]
#    return ncm // 2
def compute_ncm(bags):
    ncm = 0
    for i in range(len(bags)):
        ncm += bags[i] * (1 + bags[i])
        for j in range(i):
            ncm += 2 * bags[i] * bags[j]
    return ncm // 2


@jit_(numba_parallel=True)
def generate_bob(
    nuclear_charges,
    coordinates,
    bags,
    ncm: int = 435,
    elements=array_([1, 6, 7, 8, 16]),
    dint_: dtype_ = dint_,
):
    natoms = nuclear_charges.shape[0]
    #    bags = np.array(list(bags.values()))
    nelements = elements.shape[0]
    #    ncm = compute_ncm(bags)
    pair_distance_matrix = zeros_((natoms, natoms))

    for i in prange_(natoms):
        for j in range(i + 1, natoms):
            pair_distance_matrix[i, j] = calculate_pair_norm(
                coordinates[i], coordinates[j], nuclear_charges[i], nuclear_charges[j]
            )
            pair_distance_matrix[j, i] = pair_distance_matrix[i, j]

    cm = zeros_(ncm)
    start_indices = zeros_(nelements, dtype=dint_)

    start_indices[0] = 0
    for i in range(1, nelements):
        start_indices[i] = start_indices[i - 1] + (bags[i - 1] * (bags[i - 1] + 1)) // 2
        for j in range(i, nelements):
            start_indices[i] += bags[j] * bags[i - 1]

    for i in range(nelements):
        type1 = elements[i]
        type1_indices = get_indices(nuclear_charges, type1)
        natoms1 = len(type1_indices)

        bag = zeros_(bags[i] * (bags[i] - 1) // 2)

        for j in prange_(natoms1):
            idx1 = type1_indices[j]
            cm[start_indices[i] + j] = 0.5 * nuclear_charges[idx1] ** 2.4
            k = (j * j - j) // 2
            for l in range(j):
                idx2 = type1_indices[l]
                bag[k + l] = pair_distance_matrix[idx1, idx2]

        start_indices[i] += bags[i]

        nbag = (bags[i] * bags[i] - bags[i]) // 2
        cm[start_indices[i] : start_indices[i] + nbag] = -sort_(-bag[:nbag])
        start_indices[i] += nbag

        for j in range(i + 1, nelements):
            type2 = elements[j]
            type2_indices = get_indices(nuclear_charges, type2)
            natoms2 = len(type2_indices)

            bag = zeros_(bags[i] * bags[j])
            for k in prange_(natoms1):
                idx1 = type1_indices[k]
                for l in range(natoms2):
                    idx2 = type2_indices[l]
                    bag[natoms2 * k + l] = pair_distance_matrix[idx1, idx2]

            nbag = bags[i] * bags[j]
            cm[start_indices[i] : start_indices[i] + nbag] = -sort_(-bag[:nbag])
            start_indices[i] += nbag

    return cm
