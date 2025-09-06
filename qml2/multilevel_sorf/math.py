# Miscellaneous routines for models.py.
# TODO K.Karan.: if importance multipliers are revisited this should be revised.


from ..jit_interfaces import abs_, array_, dot_, empty_, jit_, median_, prange_, sqrt_, sum_
from ..math import svd_aligned
from ..models.sorf_hyperparameter_optimization import (
    leaveoneout_eigenvalue_multipliers,
    leaveoneout_errors_from_precalc,
    leaveoneout_errors_quantity_ders,
)
from ..utils import l2_sq_norm


def find_least_error_1D(A, b, mult=None):
    """
    Find `x` minimizing L1 norm of `Ax-b`; for now only works for 1D case. If mult is not None solve `Ax-b*mult`.
    """
    assert A.shape[1] == 1
    assert A.shape[0] == b.shape[0]
    renorm_b = b / A[:, 0]
    if mult is not None:
        renorm_b *= mult
    if renorm_b.shape[0] == 1:
        return renorm_b[0]
    abs_A = abs_(A)
    abs_A_renorm_b_list = [(rb, aA) for rb, aA in zip(renorm_b, abs_A)]
    abs_A_renorm_b_list.sort()
    tot_der = -sum_(abs_A)
    for rb, aA in abs_A_renorm_b_list:
        tot_der += 2 * aA
        if tot_der > 0:
            return array_([rb])
    # If we don't have an answer at this point all aA are close to zero.
    print("WARNING: find_least_error_1D called for small A vector, magnitude:", sum_(abs_A))
    # we'll just pick a point in the middle of possible minima.
    return array_([median_(renorm_b)])


def error_matrices_for_shifts(
    weighted_shift_coeffs,
    weighted_quantities,
    Z_U,
    eigenvalue_multipliers,
    stat_factors,
    transformed_inv_K_Z,
):
    shift_A_mat = leaveoneout_errors_quantity_ders(
        weighted_shift_coeffs, Z_U, eigenvalue_multipliers, stat_factors, None
    )
    # The b vector is basically leave-one-out errors times -1.
    transformed_alphas_rhs = dot_(weighted_quantities, Z_U)

    reproduced_weighted_quantities = dot_(transformed_inv_K_Z, transformed_alphas_rhs)

    shift_b = leaveoneout_errors_from_precalc(
        reproduced_weighted_quantities, weighted_quantities, stat_factors
    )

    shift_b *= -1
    return shift_A_mat, shift_b


def get_inv_K_mult_Z_from_eigenvectors(Z_matrix, lambda_val, K_eigenvalues, K_eigenvectors):
    inv_K_eigenvalues = 1.0 / (K_eigenvalues + lambda_val)
    transformed_Z_matrix = dot_(Z_matrix, K_eigenvectors)
    transformed_Z_matrix *= inv_K_eigenvalues
    return dot_(transformed_Z_matrix, K_eigenvectors.T)


def get_training_plane_sq_distance_components(Z_matrix, l2reg, Z_singular_values=None, Z_Vh=None):
    if Z_Vh is None:
        _, Z_singular_values, Z_Vh = svd_aligned(Z_matrix)
    train_kernel_eigenvalue_multipliers = sqrt_(
        leaveoneout_eigenvalue_multipliers(Z_singular_values, l2reg)
    )
    return train_kernel_eigenvalue_multipliers, Z_Vh


@jit_
def get_training_plane_sq_distance(
    feature_vector, train_kernel_eigenvalue_multipliers, train_kernel_eigenvectors
):
    return l2_sq_norm(feature_vector) - l2_sq_norm(
        train_kernel_eigenvalue_multipliers * dot_(train_kernel_eigenvectors, feature_vector)
    )


@jit_
def get_training_plane_sq_distances(
    feature_vectors, train_kernel_eigenvalue_multipliers, train_kernel_eigenvectors
):
    ncompounds = feature_vectors.shape[0]
    output = empty_(feature_vectors.shape[0])
    for i in prange_(ncompounds):
        output[i] = get_training_plane_sq_distance(
            feature_vectors[i], train_kernel_eigenvalue_multipliers, train_kernel_eigenvectors
        )
    return output


@jit_
def get_model_uncertainty_fitted_ratios(
    Z_matrix,
    alphas,
    quantities,
    train_kernel_eigenvalue_multipliers,
    train_kernel_eigenvectors,
):
    reproduced_errors = abs_(dot_(Z_matrix, alphas) - quantities)
    ntrain = reproduced_errors.shape[0]
    sq_distances = empty_(ntrain)
    for i in prange_(ntrain):
        sq_distances[i] = get_training_plane_sq_distance(
            Z_matrix[i],
            train_kernel_eigenvalue_multipliers,
            train_kernel_eigenvectors,
        )

    ratios = reproduced_errors / sq_distances
    return ratios, sq_distances


@jit_(numba_parallel=True)
def inplace_add_dot_product(matrix, added_vectors):
    """
    Perform the following operation without allocating additional arrays:
    matrix+=dot_(added_vectors.T, added_vectors)
    """
    npoints = added_vectors.shape[0]
    nfeatures = added_vectors.shape[1]
    assert nfeatures == matrix.shape[0]
    assert nfeatures == matrix.shape[1]
    for point_id in range(npoints):
        added_features = added_vectors[point_id, :]
        for feature_id in prange_(nfeatures):
            matrix[feature_id] += added_features * added_features[feature_id]
