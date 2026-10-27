"""Unit tests for qml2.models.training_set_optimization.

These routines exist to make an ill-conditioned kernel usable by throwing training entries
away, so the properties worth pinning down are the ones a caller relies on: that the
entries named are really the redundant ones, that dropping them leaves an invertible
kernel, and that the exclusion loops always terminate.

No stored checksums -- every assertion here is a property that must hold for any input,
checked against an independent calculation rather than against a recorded number.
"""

import numpy as np
from conftest import int2rng

from qml2.models.training_set_optimization import (
    all_indices_except,
    kernel2sqdist,
    kernel_exclude_nearest,
    linear_dependent_entries,
    rep_sqdist_mat,
    sqdist_exclude_nearest,
)

NPOINTS = 10
NDIMENSIONS = 4


def gaussian_kernel(first, second, sigma=1.0):
    square_distances = ((first[:, None, :] - second[None, :, :]) ** 2).sum(-1)
    return np.exp(-square_distances / (2 * sigma**2))


def representations(rng, npoints=NPOINTS):
    return rng.normal(size=(npoints, NDIMENSIONS))


def test_all_indices_except():
    """Returns the positions that are False, in order."""
    mask = np.array([True, False, True, False, False])
    assert np.array_equal(all_indices_except(mask), np.array([1, 3, 4]))
    assert all_indices_except(np.ones(4, dtype=bool)).shape[0] == 0
    assert np.array_equal(all_indices_except(np.zeros(3, dtype=bool)), np.arange(3))


def test_rep_sqdist_mat_matches_explicit_distances():
    reps = representations(int2rng(1))
    expected = ((reps[:, None, :] - reps[None, :, :]) ** 2).sum(-1)
    produced = rep_sqdist_mat(reps)
    assert np.allclose(produced, expected)
    assert np.allclose(produced, produced.T)
    assert np.allclose(np.diag(produced), 0.0)


def test_kernel2sqdist_matches_rkhs_formula():
    """||f_i - f_j||^2 = K_ii + K_jj - 2 K_ij, and for a Gaussian kernel of unit diagonal
    that is 2(1 - K_ij)."""
    reps = representations(int2rng(2))
    kernel = gaussian_kernel(reps, reps)
    produced = kernel2sqdist(kernel)
    expected = 2.0 * (1.0 - kernel)
    assert np.allclose(produced, expected)
    assert np.allclose(np.diag(produced), 0.0)


def test_linear_dependent_entries_finds_duplicates():
    """Duplicated points share an RKHS vector exactly, so they are what must be named."""
    rng = int2rng(3)
    reps = representations(rng)
    reps[7] = reps[2]
    reps[9] = reps[3]
    kernel = gaussian_kernel(reps, reps)
    assert np.linalg.matrix_rank(kernel) == NPOINTS - 2

    excluded = linear_dependent_entries(kernel, 1.0e-9)
    assert set(excluded.tolist()) == {7, 9}

    # The point of naming them: what is left must be invertible.
    kept = np.setdiff1d(np.arange(NPOINTS), excluded)
    assert np.linalg.matrix_rank(kernel[np.ix_(kept, kept)]) == kept.shape[0]


def test_linear_dependent_entries_keeps_well_conditioned_kernel():
    reps = representations(int2rng(4))
    kernel = gaussian_kernel(reps, reps)
    assert linear_dependent_entries(kernel, 1.0e-9).shape[0] == 0


def test_sqdist_exclude_nearest_respects_count():
    reps = representations(int2rng(5))
    square_distances = rep_sqdist_mat(reps)
    for num_cut in (1, 3, 5):
        excluded = sqdist_exclude_nearest(square_distances, 0.0, num_cut)
        assert excluded.shape[0] == num_cut
        assert excluded.shape[0] == np.unique(excluded).shape[0]


def test_sqdist_exclude_nearest_raises_minimum_distance():
    """With a distance threshold, every surviving pair must clear it."""
    reps = representations(int2rng(6))
    square_distances = rep_sqdist_mat(reps)
    off_diagonal = square_distances[~np.eye(NPOINTS, dtype=bool)]
    threshold = float(np.median(off_diagonal))

    excluded = sqdist_exclude_nearest(square_distances, threshold, 0)
    kept = np.setdiff1d(np.arange(NPOINTS), excluded)
    assert kept.shape[0] >= 2
    kept_distances = square_distances[np.ix_(kept, kept)]
    assert kept_distances[~np.eye(kept.shape[0], dtype=bool)].min() > threshold


def test_sqdist_exclude_nearest_terminates_on_degenerate_arguments():
    """Regression: neither stopping criterion fires for these, and the loop used to hang.

    min_sqdist <= 0 disables the distance test and a non-positive count never matched the
    counter, so the call never returned. It has to come back with an empty exclusion
    instead.
    """
    reps = representations(int2rng(7), npoints=8)
    square_distances = rep_sqdist_mat(reps)

    assert sqdist_exclude_nearest(square_distances, 0.0, 0).shape[0] == 0
    assert sqdist_exclude_nearest(square_distances, -1.0, -5).shape[0] == 0
    # Asking for more exclusions than there are entries must stop, not run off the end.
    excluded = sqdist_exclude_nearest(square_distances, 0.0, 99)
    assert excluded.shape[0] == 7


def test_kernel_exclude_nearest_agrees_with_the_two_step_route():
    """kernel_exclude_nearest is kernel2sqdist followed by sqdist_exclude_nearest."""
    reps = representations(int2rng(8))
    kernel = gaussian_kernel(reps, reps)
    threshold = 0.5

    from_kernel = kernel_exclude_nearest(kernel, threshold, 0)
    from_distances = sqdist_exclude_nearest(kernel2sqdist(kernel), threshold, 0)
    assert np.array_equal(from_kernel, from_distances)


if __name__ == "__main__":
    test_all_indices_except()
    test_rep_sqdist_mat_matches_explicit_distances()
    test_kernel2sqdist_matches_rkhs_formula()
    test_linear_dependent_entries_finds_duplicates()
    test_linear_dependent_entries_keeps_well_conditioned_kernel()
    test_sqdist_exclude_nearest_respects_count()
    test_sqdist_exclude_nearest_raises_minimum_distance()
    test_sqdist_exclude_nearest_terminates_on_degenerate_arguments()
    test_kernel_exclude_nearest_agrees_with_the_two_step_route()
