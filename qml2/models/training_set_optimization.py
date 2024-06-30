# Contains procedures for dealing with problematic (i.e. non-invertible or close to non-invertible)
# kernel matrices.
from ..jit_interfaces import bool_, dint_, eye_, jit_, ones_, prange_, random_, sqrt_, zeros_
from ..kernels import l2_norm_sq_dist


class KernelUnstable(Exception):
    pass


@jit_
def all_indices_except(to_include, dint_=dint_):
    num_left = 0
    for el in to_include:
        if not el:
            num_left += 1
    output = zeros_((num_left,), dtype=dint_)
    arr_pos = 0
    for el_id, el in enumerate(to_include):
        if not el:
            output[arr_pos] = el_id
            arr_pos += 1
    return output[:arr_pos]


@jit_(numba_parallel=True)
def linear_dependent_entries(train_kernel, residue_tol_coeff):
    """
    Check entries in train_kernel whose RKHS vectors are close to linear, causing non-invertability of the kernel.
    """
    num_elements = train_kernel.shape[0]

    sqnorm_residue = zeros_(num_elements)
    residue_tolerance = zeros_(num_elements)

    for i in prange_(num_elements):
        sqnorm = train_kernel[i, i]
        sqnorm_residue[i] = sqnorm
        residue_tolerance[i] = sqnorm * residue_tol_coeff

    cur_orth_id = 0

    to_include = ones_(num_elements, dtype=bool_)

    orthonormalized_vectors = eye_(num_elements)

    for cur_orth_id in range(num_elements):
        if not to_include[cur_orth_id]:
            continue
        # Normalize the vector.
        cur_norm = sqrt_(sqnorm_residue[cur_orth_id])
        for i in prange_(cur_orth_id + 1):
            orthonormalized_vectors[cur_orth_id, i] /= cur_norm
        # Subtract projections of the normalized vector from all currently not orthonormalized vectors.
        # Also check that their residue is above the corresponding threshold.
        for i in prange_(cur_orth_id + 1, num_elements):
            if not to_include[i]:
                continue
            cur_product = 0.0
            for j in range(cur_orth_id + 1):
                if to_include[j]:
                    cur_product += train_kernel[i, j] * orthonormalized_vectors[cur_orth_id, j]
            sqnorm_residue[i] -= cur_product**2
            if sqnorm_residue[i] < residue_tolerance[i]:
                to_include[i] = False
            else:
                for j in range(cur_orth_id + 1):
                    orthonormalized_vectors[i, j] -= (
                        cur_product * orthonormalized_vectors[cur_orth_id, j]
                    )
        cur_orth_id += 1
    return all_indices_except(to_include)


@jit_(numba_parallel=True)
def kernel2sqdist(train_kernel):
    """
    Transform train_kernel to matrix of distances in RKHS. Helps identify closest RKHS vectors to be excluded.
    """
    num_train = train_kernel.shape[0]
    sqdist_mat = zeros_((num_train, num_train))
    for i in prange_(num_train):
        for j in range(i):
            sqdist_mat[i, j] = train_kernel[i, i] + train_kernel[j, j] - 2 * train_kernel[i, j]
            sqdist_mat[j, i] = sqdist_mat[i, j]
    return sqdist_mat


@jit_
def min_id_sqdist(sqdist_row, to_include, entry_id):
    """
    Find entry closest to entry_id in terms of RKHS distance.
    """
    cur_min_sqdist = 0.0
    cur_min_sqdist_id = 0
    minimal_sqdist_init = False
    num_train = sqdist_row.shape[0]

    for j in range(num_train):
        if entry_id != j:
            cur_sqdist = sqdist_row[j]
            if ((cur_sqdist < cur_min_sqdist) or (not minimal_sqdist_init)) and to_include[j]:
                minimal_sqdist_init = True
                cur_min_sqdist = cur_sqdist
                cur_min_sqdist_id = j
    return cur_min_sqdist_id, cur_min_sqdist


@jit_(numba_parallel=True)
def rep_sqdist_mat(rep_arr):
    """
    Square distance from representations.
    """
    num_vecs = rep_arr.shape[0]
    sqdist_mat = zeros_((num_vecs, num_vecs))
    for i in prange_(num_vecs):
        for j in range(i):
            sqdist_mat[i, j] = l2_norm_sq_dist(rep_arr[i], rep_arr[j])
            sqdist_mat[j, i] = sqdist_mat[i, j]
    return sqdist_mat


@jit_(numba_parallel=True)
def sqdist_exclude_nearest(sqdist_mat, min_sqdist, num_cut_closest_entries, dint_=dint_):
    """
    Exclude training set entries to increase minimal square distance in a training set.
    """
    num_train = sqdist_mat.shape[0]

    minimal_distance_ids = zeros_(num_train, dtype=dint_)
    minimal_distances = zeros_(num_train)
    to_include = ones_(num_train, dtype=bool_)

    for i in prange_(num_train):
        minimal_distance_ids[i], minimal_distances[i] = min_id_sqdist(sqdist_mat[i], to_include, i)

    num_ignored = 0

    while True:
        cur_min_id, cur_min_sqdist = min_id_sqdist(minimal_distances, to_include, -1)
        if (cur_min_sqdist > min_sqdist) and (min_sqdist > 0.0):
            break
        if random_() > 0.5:
            new_ignored = cur_min_id
        else:
            new_ignored = minimal_distance_ids[cur_min_id]

        to_include[new_ignored] = False
        num_ignored += 1
        if num_ignored == 1:
            print("Smallest ignored distance:", cur_min_sqdist)
        if num_ignored == num_cut_closest_entries:
            print("Largest ignored distance:", cur_min_sqdist)
            break
        for i in prange_(num_train):
            if to_include[i]:
                if minimal_distance_ids[i] == new_ignored:
                    minimal_distance_ids[i], minimal_distances[i] = min_id_sqdist(
                        sqdist_mat[i], to_include, i
                    )

    return all_indices_except(to_include)


@jit_
def kernel_exclude_nearest(train_kernel, min_sqdist, num_cut_closest_entries):
    """
    Exclude training set entries to increase minimum RKHS square distance in the training set.
    """
    sqdist_mat = kernel2sqdist(train_kernel)
    return sqdist_exclude_nearest(sqdist_mat, min_sqdist, num_cut_closest_entries)
