"""Unit tests for qml2.models.learning_curves.

A learning curve is a plot somebody reads and believes, so the failures that matter are
the quiet ones: a point that comes back nan, a curve that silently depends on global
random state, a "scaffold split" that leaks a group across the train/test boundary. Each
of those gets a test here, alongside the obvious property that more training data helps.
"""

import numpy as np
import pytest
from conftest import int2rng

from qml2.models.learning_curves import (
    group_train_test_split,
    learning_curve_from_predictions,
    learning_curve_from_regularized_kernel,
    learning_curve_predictions_from_regularized_kernel,
)
from qml2.models.loss_functions import MAE, RMSE

NTRAIN = 60
NTEST = 20
NDIMENSIONS = 3
L2REG = 1.0e-8


def regression_problem(rng, ntrain=NTRAIN):
    """A smooth, learnable target and its RAW kernels.

    Raw: learning_curve_*_from_regularized_kernel applies l2reg itself, per training
    subset, so a kernel that already carried it would be regularized twice.
    """
    train_reps = rng.normal(size=(ntrain, NDIMENSIONS))
    test_reps = rng.normal(size=(NTEST, NDIMENSIONS))

    def target(reps):
        return np.sin(reps[:, 0]) + 0.5 * reps[:, 1]

    def gaussian(first, second):
        square_distances = ((first[:, None, :] - second[None, :, :]) ** 2).sum(-1)
        return np.exp(-square_distances / 2.0)

    train_kernel = gaussian(train_reps, train_reps)
    return train_kernel, target(train_reps), gaussian(test_reps, train_reps), target(test_reps)


def test_learning_curve_from_predictions_means_and_stds():
    test_quantities = np.zeros(4)
    predictions = [
        [np.full(4, 0.1), np.full(4, 0.3)],  # MAEs 0.1 and 0.3
        [np.full(4, 0.05)],  # MAE 0.05
    ]
    means, stds = learning_curve_from_predictions(predictions, test_quantities)
    assert np.allclose(means, [0.2, 0.05])
    assert np.allclose(stds, [0.1, 0.0])


def test_learning_curve_from_predictions_honours_loss_and_importance():
    test_quantities = np.zeros(4)
    predictions = [[np.array([1.0, -1.0, 1.0, -1.0])]]

    mean_MAE, _ = learning_curve_from_predictions(
        predictions, test_quantities, error_loss_function=MAE()
    )
    mean_RMSE, _ = learning_curve_from_predictions(
        predictions, test_quantities, error_loss_function=RMSE()
    )
    assert np.isclose(mean_MAE[0], 1.0)
    assert np.isclose(mean_RMSE[0], 1.0)

    # Halving every error must halve the MAE.
    weighted, _ = learning_curve_from_predictions(
        predictions,
        test_quantities,
        error_loss_function=MAE(),
        test_importance_multipliers=np.full(4, 0.5),
    )
    assert np.isclose(weighted[0], 0.5)


def test_learning_curve_improves_with_training_set_size():
    problem = regression_problem(int2rng(1))
    sizes = [10, 20, 40]
    curves, _ = learning_curve_from_regularized_kernel(
        *problem, sizes, l2reg=L2REG, rng=int2rng(2)
    )
    for loss_name, (means, stds) in curves.items():
        assert len(means) == len(sizes), loss_name
        assert all(np.isfinite(means)), loss_name
        assert means[-1] < means[0], loss_name


def test_learning_curve_reports_every_requested_loss_from_one_pass():
    """The reason predictions and scoring are separate: one expensive pass, many curves."""
    problem = regression_problem(int2rng(3))
    sizes = [10, 20]
    default_curves, _ = learning_curve_from_regularized_kernel(
        *problem, sizes, l2reg=L2REG, rng=int2rng(4)
    )
    assert sorted(default_curves) == ["MAE", "RMSE"]

    chosen, _ = learning_curve_from_regularized_kernel(
        *problem, sizes, l2reg=L2REG, rng=int2rng(4), error_loss_functions={"mae": MAE()}
    )
    assert sorted(chosen) == ["mae"]
    # Same predictions underneath, so the shared loss must agree exactly.
    assert np.allclose(chosen["mae"][0], default_curves["MAE"][0])
    # RMSE >= MAE on the same errors, so the two curves are genuinely different.
    assert np.all(np.array(default_curves["RMSE"][0]) >= np.array(default_curves["MAE"][0]))


def test_learning_curve_regularization_modes():
    """l2reg, l2reg_diag_ratio and opt_l2reg are three ways to say the same thing."""
    train_kernel, train_quantities, test_kernel, _ = regression_problem(int2rng(5))
    sizes = [10, 20]

    # The Gaussian kernel has a unit diagonal, so the absolute and relative forms coincide.
    absolute, ratios_absolute = learning_curve_predictions_from_regularized_kernel(
        train_kernel, train_quantities, test_kernel, sizes, l2reg=1e-6, rng=int2rng(6)
    )
    relative, ratios_relative = learning_curve_predictions_from_regularized_kernel(
        train_kernel,
        train_quantities,
        test_kernel,
        sizes,
        l2reg_diag_ratio=1e-6,
        rng=int2rng(6),
    )
    assert np.allclose(
        np.concatenate([np.concatenate(p) for p in absolute]),
        np.concatenate([np.concatenate(p) for p in relative]),
    )
    assert np.allclose(np.concatenate(ratios_absolute), np.concatenate(ratios_relative))


def test_opt_l2reg_chooses_per_subset():
    """The point of opt_l2reg: each subset gets the regularization its own size warrants."""
    train_kernel, train_quantities, test_kernel, _ = regression_problem(int2rng(7))
    _, ratios = learning_curve_predictions_from_regularized_kernel(
        train_kernel,
        train_quantities,
        test_kernel,
        [10, 20],
        opt_l2reg=True,
        rng=int2rng(8),
        l2reg_total_iterpts=16,
    )
    flat = np.concatenate(ratios)
    assert np.all(np.isfinite(flat)) and np.all(flat > 0.0)
    assert len(set(flat.tolist())) > 1, "every subset got the same l2reg, so nothing was optimized"


def test_learning_curve_refuses_impossible_sizes():
    """Regression: an oversized request used to yield nan instead of an error."""
    problem = regression_problem(int2rng(9))
    with pytest.raises(ValueError, match="exceeds"):
        learning_curve_from_regularized_kernel(
            *problem, [NTRAIN + 1], l2reg=L2REG, rng=int2rng(10)
        )
    with pytest.raises(ValueError, match="positive"):
        learning_curve_from_regularized_kernel(*problem, [0], l2reg=L2REG, rng=int2rng(10))


def test_learning_curve_regularization_guards():
    """Exactly one way of setting l2reg, and kernels of the right shape."""
    train_kernel, train_quantities, test_kernel, _ = regression_problem(int2rng(11))
    with pytest.raises(ValueError, match="exactly one"):
        learning_curve_predictions_from_regularized_kernel(
            train_kernel, train_quantities, test_kernel, [10]
        )
    with pytest.raises(ValueError, match="exactly one"):
        learning_curve_predictions_from_regularized_kernel(
            train_kernel, train_quantities, test_kernel, [10], l2reg=1e-6, opt_l2reg=True
        )
    with pytest.raises(ValueError, match="test_kernel is"):
        learning_curve_predictions_from_regularized_kernel(
            train_kernel, train_quantities, test_kernel.T, [10], l2reg=1e-6
        )


def test_learning_curve_is_reproducible_with_an_rng():
    problem = regression_problem(int2rng(12))
    first, _ = learning_curve_from_regularized_kernel(
        *problem, [10, 20], l2reg=L2REG, rng=int2rng(13)
    )
    second, _ = learning_curve_from_regularized_kernel(
        *problem, [10, 20], l2reg=L2REG, rng=int2rng(13)
    )
    different, _ = learning_curve_from_regularized_kernel(
        *problem, [10, 20], l2reg=L2REG, rng=int2rng(14)
    )
    assert np.allclose(first["MAE"][0], second["MAE"][0])
    assert not np.allclose(first["MAE"][0], different["MAE"][0])


def test_group_train_test_split_never_shares_a_group():
    rng = int2rng(10)
    group_ids = rng.integers(0, 12, size=NTRAIN)
    train_indices, test_indices = group_train_test_split(group_ids, 40, seed=11)

    assert not set(group_ids[train_indices]) & set(group_ids[test_indices])
    assert train_indices.shape[0] + test_indices.shape[0] == NTRAIN
    assert not set(train_indices.tolist()) & set(test_indices.tolist())
    # Groups are indivisible, so the pool only approaches the target from below.
    assert 0.9 * 40 <= train_indices.shape[0] <= 40


def test_group_train_test_split_argument_guards():
    group_ids = np.arange(NTRAIN) % 10
    with pytest.raises(ValueError, match="rng or seed"):
        group_train_test_split(group_ids, 30, seed=1, rng=int2rng(1))
    with pytest.raises(ValueError, match="exceeds"):
        group_train_test_split(group_ids, NTRAIN + 5, seed=1)


def test_group_train_test_split_refuses_an_unfillable_target():
    """One dominant group cannot be split, and saying so beats returning a short pool.

    This is the FreeSolv acyclic='pool' situation in miniature: a single group holds more
    than the requested training set, so no combination of whole groups comes close.
    """
    group_ids = np.zeros(NTRAIN, dtype=int)
    group_ids[:50] = 0  # one group of 50
    group_ids[50:] = np.arange(1, NTRAIN - 49)
    with pytest.raises(ValueError, match="whole groups"):
        group_train_test_split(group_ids, 40, seed=1)


if __name__ == "__main__":
    test_learning_curve_from_predictions_means_and_stds()
    test_learning_curve_from_predictions_honours_loss_and_importance()
    test_learning_curve_improves_with_training_set_size()
    test_learning_curve_reports_every_requested_loss_from_one_pass()
    test_learning_curve_regularization_modes()
    test_opt_l2reg_chooses_per_subset()
    test_learning_curve_refuses_impossible_sizes()
    test_learning_curve_regularization_guards()
    test_learning_curve_is_reproducible_with_an_rng()
    test_group_train_test_split_never_shares_a_group()
    test_group_train_test_split_argument_guards()
    test_group_train_test_split_refuses_an_unfillable_target()
