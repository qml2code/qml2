# Standardized learning curve building.
from typing import List

import numpy as np
from numpy import ndarray

from ..jit_interfaces import LinAlgError_, mean_, std_
from ..math import cho_solve_from_matrix
from ..optimizers import grid_optimize_1D
from .hyperparameter_optimization import KRRLeaveOneOutL2regLoss
from .kfolds import random_group_subset
from .loss_functions import MAE, RMSE


def subset_indices_per_size(ntrain, training_set_size, max_subset_num, rng):
    """Disjoint training subsets of one size, drawn from a single shuffle.

    Disjoint because subsets of a given size are meant to be independent draws whose spread
    becomes the error bar; overlapping ones would understate it.
    """
    if training_set_size <= 0:
        raise ValueError(f"training_set_size must be positive, got {training_set_size}")
    nsubsets = min(max_subset_num, ntrain // training_set_size)
    if nsubsets == 0:
        # Without this the subset loop below simply does not run and the point comes back
        # as nan from the mean of an empty list -- a learning curve with a silent hole in
        # it rather than a failure anyone would notice.
        raise ValueError(
            f"training_set_size {training_set_size} exceeds the {ntrain} available "
            "training points, so no subset of that size can be drawn"
        )
    shuffled = rng.permutation(ntrain)
    return [
        shuffled[index * training_set_size : (index + 1) * training_set_size]
        for index in range(nsubsets)
    ]


def optimized_l2reg_diag_ratio(
    subset_kernel,
    subset_quantities,
    loss_function=MAE(),
    ln_l2reg_diag_ratio_bounds=(-25.0, 0.0),
    total_iterpts=32,
):
    """Grid-search the l2reg that minimizes the LEAVE-ONE-OUT loss on this subset.

    Leave-one-out rather than the test set: choosing the regularization by test performance
    would tune the model on the data the learning curve then reports, and the curve would
    flatter itself. The criterion costs one eigendecomposition of the subset kernel, reused
    across every grid point.

    The grid runs over ln(l2reg_diag_ratio) -- l2reg divided by the mean diagonal element --
    which is the scale-free form the model classes optimize in, so the same bounds work for
    kernels of any magnitude.
    """
    l2reg_loss = KRRLeaveOneOutL2regLoss(
        subset_quantities, train_kernel=subset_kernel, loss_function=loss_function
    )

    def objective(ln_l2reg_diag_ratio):
        try:
            return l2reg_loss.calculate_for_ln_l2reg_rel_diag(ln_l2reg_diag_ratio)
        except LinAlgError_:
            # Too little regularization for this subset; grid_optimize_1D skips None.
            return None

    return np.exp(
        grid_optimize_1D(objective, list(ln_l2reg_diag_ratio_bounds), total_iterpts=total_iterpts)
    )


def learning_curve_predictions_from_regularized_kernel(
    train_kernel: ndarray,
    train_quantities: ndarray,
    test_kernel: ndarray,
    training_set_sizes: List[int],
    l2reg=None,
    l2reg_diag_ratio=None,
    opt_l2reg=False,
    max_subset_num: int = 8,
    rng=None,
    l2reg_opt_loss_function=MAE(),
    ln_l2reg_diag_ratio_bounds=(-25.0, 0.0),
    l2reg_total_iterpts: int = 32,
):
    """Predictions for every training subset of a learning curve, from a kernel.

    For a kernel somebody else computed -- FJK, SOAP, anything expensive enough that
    recomputing it per learning-curve point is the wrong shape. Mirrors the model classes'
    ``learning_curve_predictions``: it returns predictions, and turning those into loss
    numbers is a separate step, so one expensive pass can be scored under several losses.

    NOTE ON THE KERNEL: pass the RAW kernel. Regularization is applied here, per subset, so
    a kernel that already carries l2reg on its diagonal would be regularized twice. (The
    function this replaces required the opposite, which was a standing foot-gun.)

    Exactly one of these must be given:

        l2reg                 absolute ridge term.
        l2reg_diag_ratio      ridge term relative to the mean diagonal element -- the
                              scale-free form, and the convention the model classes use.
        opt_l2reg=True        choose it per subset by minimizing the LEAVE-ONE-OUT loss on
                              that subset, over a grid in ln(l2reg_diag_ratio). This is the
                              "optimize l2reg individually for each point" behaviour; each
                              subset gets the regularization its own size warrants, rather
                              than one value chosen for the largest.

    Returns ``(all_predictions, all_l2reg_diag_ratios)``. The second is worth looking at
    even when you only wanted the curve: a subset whose ratio collapses towards machine
    epsilon has an ill-conditioned solve, and that shows up as a single wild point in an
    otherwise smooth curve.
    """
    ntrain = train_quantities.shape[0]
    ntest = test_kernel.shape[0]
    if train_kernel.shape != (ntrain, ntrain):
        raise ValueError(f"train_kernel is {train_kernel.shape}, expected {(ntrain, ntrain)}")
    if test_kernel.shape != (ntest, ntrain):
        raise ValueError(f"test_kernel is {test_kernel.shape}, expected {(ntest, ntrain)}")

    nspecified = sum([l2reg is not None, l2reg_diag_ratio is not None, bool(opt_l2reg)])
    if nspecified != 1:
        raise ValueError(
            "specify exactly one of l2reg, l2reg_diag_ratio, or opt_l2reg=True; "
            f"got {nspecified}"
        )
    if rng is None:
        rng = np.random.default_rng()

    all_predictions = []
    all_l2reg_diag_ratios = []
    for training_set_size in training_set_sizes:
        subset_predictions = []
        subset_ratios = []
        for subset in subset_indices_per_size(ntrain, training_set_size, max_subset_num, rng):
            subset_kernel = train_kernel[subset][:, subset]
            subset_quantities = train_quantities[subset]

            if opt_l2reg:
                used_ratio = optimized_l2reg_diag_ratio(
                    subset_kernel,
                    subset_quantities,
                    loss_function=l2reg_opt_loss_function,
                    ln_l2reg_diag_ratio_bounds=ln_l2reg_diag_ratio_bounds,
                    total_iterpts=l2reg_total_iterpts,
                )
            elif l2reg_diag_ratio is not None:
                used_ratio = l2reg_diag_ratio
            else:
                used_ratio = l2reg / np.mean(np.diagonal(subset_kernel))

            try:
                alphas = cho_solve_from_matrix(
                    subset_kernel, subset_quantities, l2reg_diag_ratio=used_ratio
                )
            except LinAlgError_ as error:
                raise LinAlgError_(
                    f"non-invertible kernel for a training subset of size "
                    f"{training_set_size} at l2reg_diag_ratio={used_ratio:.3e}"
                ) from error
            subset_predictions.append(np.dot(test_kernel[:, subset], alphas))
            subset_ratios.append(used_ratio)
        all_predictions.append(subset_predictions)
        all_l2reg_diag_ratios.append(subset_ratios)
    return all_predictions, all_l2reg_diag_ratios


def learning_curve_from_regularized_kernel(
    train_kernel: ndarray,
    train_quantities: ndarray,
    test_kernel: ndarray,
    test_quantities: ndarray,
    training_set_sizes: List[int],
    error_loss_functions=None,
    test_importance_multipliers=None,
    **prediction_kwargs,
):
    """Learning curve from a kernel somebody else computed, under several losses at once.

    Mirrors the model classes' ``learning_curve``: builds the predictions, then scores them.
    ``error_loss_functions`` is a name -> loss mapping, defaulting to MAE and RMSE, so one
    pass over the expensive part produces every curve the paper prints rather than one per
    invocation.

    See :func:`learning_curve_predictions_from_regularized_kernel` for the keyword
    arguments controlling regularization -- in particular that the kernel must be RAW.

    Returns ``({name: (means, stds)}, all_l2reg_diag_ratios)``.
    """
    if error_loss_functions is None:
        error_loss_functions = {"MAE": MAE(), "RMSE": RMSE()}

    all_predictions, all_l2reg_diag_ratios = learning_curve_predictions_from_regularized_kernel(
        train_kernel,
        train_quantities,
        test_kernel,
        training_set_sizes,
        **prediction_kwargs,
    )
    results = {
        name: learning_curve_from_predictions(
            all_predictions,
            test_quantities,
            error_loss_function=loss_function,
            test_importance_multipliers=test_importance_multipliers,
        )
        for name, loss_function in error_loss_functions.items()
    }
    return results, all_l2reg_diag_ratios


def learning_curve_from_predictions(
    all_predictions, test_quantities, error_loss_function=MAE(), test_importance_multipliers=None
):
    """
    Learning curves from predictions (used in different model classes). Includes calculating both the mean loss and the standard deviation (e.g. over several k-folds).
    """
    loss_means = []
    loss_stds = []
    for subset_predictions in all_predictions:
        subset_losses = []
        for predictions in subset_predictions:
            errors = predictions - test_quantities
            if test_importance_multipliers is not None:
                errors *= test_importance_multipliers
            subset_losses.append(error_loss_function(errors))
        MAE_mean = mean_(subset_losses)
        MAE_std = std_(subset_losses)
        loss_means.append(MAE_mean)
        loss_stds.append(MAE_std)
    return loss_means, loss_stds


def group_train_test_split(
    group_ids,
    training_set_size,
    rng=None,
    seed=None,
    min_fill_fraction=0.9,
):
    """Train/test split for a learning curve that never puts a group on both sides.

    This is the ONE step of a group-based learning curve that has to be group-aware. The
    successive training subsets drawn from the returned pool do not: the pool already
    shares no group with the test set, so every subset of it is group-disjoint from test
    too. That is what lets a scaffold-split learning curve still hit exact training-set
    sizes -- 50, 100, 200 -- which a curve built by accumulating whole scaffolds could
    not, since scaffold sizes do not add up to round numbers.

    So the usual shape is unchanged apart from where the split comes from::

        group_ids = scaffold_group_ids(all_SMILES, acyclic="skeleton")
        train_idx, test_idx = group_train_test_split(group_ids, 474, seed=seed)
        ...
        model.learning_curve_predictions(..., training_set_sizes, ...)

    Args:
        group_ids: per-molecule group label; negative labels are dropped entirely.
        training_set_size: TARGET pool size. Groups are indivisible, so the realized pool
            is generally a little smaller -- use ``len(train_idx)`` as the last point of
            ``training_set_sizes`` rather than the number requested here.
        rng, seed: supply one. ``seed`` is the convenient form for paper scripts, which
            index everything by seed.
        min_fill_fraction: fail rather than return a pool far below the target; see
            :func:`qml2.models.kfolds.random_group_subset`.

    Returns:
        ``(training_indices, test_indices)``, both sorted, sharing no group.
    """
    if rng is None:
        rng = np.random.default_rng(seed)
    elif seed is not None:
        raise ValueError("pass rng or seed, not both")
    return random_group_subset(
        group_ids, training_set_size, rng=rng, min_fill_fraction=min_fill_fraction
    )
