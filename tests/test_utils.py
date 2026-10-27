"""Unit tests for numeric helpers in qml2.utils.

Currently: ``nearest_neighbour_distances``, a vectorized rewrite of an obvious loop. The
identity ``||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b`` is exact on paper but loses cancellation
in floating point, so it is checked against the loop it replaced rather than against a
recorded number.
"""

import numpy as np
from conftest import int2rng

from qml2.utils import nearest_neighbour_distances


def brute_force_nearest_distances(representations, train_indices, test_indices):
    """The loop the vectorized version replaced. Unambiguous, and slow enough to matter."""
    train_block = representations[train_indices]
    return np.array(
        [
            np.sqrt(np.min(np.sum((train_block - representations[index]) ** 2, axis=1)))
            for index in test_indices
        ]
    )


def test_nearest_neighbour_distances_match_the_loop():
    """The vectorization must be exact, not merely close."""
    rng = int2rng(1)
    representations = rng.normal(size=(30, 12))
    train_indices = np.arange(0, 20)
    test_indices = np.arange(20, 30)

    produced = nearest_neighbour_distances(representations, train_indices, test_indices)
    expected = brute_force_nearest_distances(representations, train_indices, test_indices)
    assert np.allclose(produced, expected, rtol=0.0, atol=1e-10)


def test_nearest_neighbour_distances_handle_a_coincident_point():
    """A test point identical to a training point gives 0, not nan.

    This is what the clip guards: the identity leaves a tiny negative square distance from
    cancellation, and sqrt of that is nan -- which would silently poison a median.
    """
    rng = int2rng(2)
    representations = rng.normal(size=(10, 2000)) * 1000.0
    representations[7] = representations[3]

    distances = nearest_neighbour_distances(representations, np.arange(0, 5), np.array([7, 8, 9]))
    assert np.all(np.isfinite(distances))
    assert np.all(distances >= 0.0)
    assert np.isclose(distances[0], 0.0, atol=1e-6)


def test_nearest_neighbour_distances_shape_and_order():
    rng = int2rng(3)
    representations = rng.normal(size=(12, 5))
    test_indices = np.array([9, 10, 11])
    distances = nearest_neighbour_distances(representations, np.arange(9), test_indices)
    assert distances.shape == test_indices.shape
    reordered = nearest_neighbour_distances(representations, np.arange(9), test_indices[::-1])
    assert np.allclose(reordered, distances[::-1])


if __name__ == "__main__":
    test_nearest_neighbour_distances_match_the_loop()
    test_nearest_neighbour_distances_handle_a_coincident_point()
    test_nearest_neighbour_distances_shape_and_order()
