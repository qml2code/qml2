"""Unit tests for grouped_cv_loss -- aggregating leave-one-group-out CV residuals.

The layer is small, but the two ways it can be silently wrong are not obvious: support
weights that multiply inside the first-level loss rather than outside it, and macro-mode
arguments quietly ignored in micro mode. Both get a test, as does the documented
micro/macro relationship, which doubles as a correctness check on the weighting.
"""

import warnings

import numpy as np
import pytest
from conftest import int2rng

from qml2.models.hyperparameter_optimization import KRRKfoldL2regLoss, KRRKfoldL2regOpt
from qml2.models.kfolds import (
    GroupKFolds,
    KFolds,
    build_kfolds,
    group_index_lists,
    leave_one_group_out_folds,
)
from qml2.models.loss_functions import (
    MAE,
    MSE,
    RMSE,
    SelfConsistentHuber,
    group_residual_indices,
    grouped_cv_loss,
)
from qml2.models.sorf_hyperparameter_optimization import SORFKfoldL2regLoss, SORFKfoldL2regOpt

NPOINTS = 40
NGROUPS = 7
COMPROMISE_COEFFICIENT = 0.5


def residuals_and_groups(seed=0, ngroups=NGROUPS, npoints=NPOINTS):
    rng = int2rng(seed)
    return rng.normal(size=npoints), rng.integers(0, ngroups, size=npoints)


def test_micro_reduces_to_the_pooled_loss():
    """Micro mode must be exactly the loss applied to every residual at once."""
    residuals, group_ids = residuals_and_groups()
    for point_loss in (MAE(), MSE(), RMSE(), SelfConsistentHuber(COMPROMISE_COEFFICIENT)):
        assert np.isclose(
            grouped_cv_loss(residuals, group_ids, point_loss, mode="micro"),
            point_loss(residuals),
            rtol=0.0,
            atol=1e-14,
        )


def test_macro_equal_weight_is_the_mean_of_per_group_losses():
    residuals, group_ids = residuals_and_groups(seed=1)
    members = group_residual_indices(group_ids)
    for point_loss in (MAE(), SelfConsistentHuber(COMPROMISE_COEFFICIENT)):
        expected = np.mean([point_loss(residuals[indices]) for indices in members])
        assert np.isclose(
            grouped_cv_loss(residuals, group_ids, point_loss, mode="macro"),
            expected,
            rtol=0.0,
            atol=1e-14,
        )


def test_support_weighting_bridges_macro_and_micro():
    """The documented relationship, used as a correctness check on the weighting.

    Exact for a loss that is a plain mean of a per-point function; RMSE is a nonlinear
    function of such a mean, so it is deliberately asserted NOT to coincide -- otherwise
    the test would pass for a weighting scheme that simply ignored group sizes.
    """
    residuals, group_ids = residuals_and_groups(seed=2)
    for point_loss in (MAE(), MSE()):
        micro = grouped_cv_loss(residuals, group_ids, point_loss, mode="micro")
        macro = grouped_cv_loss(
            residuals, group_ids, point_loss, mode="macro", aggregation_weights="support"
        )
        assert np.isclose(micro, macro, rtol=0.0, atol=1e-12)

    micro_rmse = grouped_cv_loss(residuals, group_ids, RMSE(), mode="micro")
    macro_rmse = grouped_cv_loss(
        residuals, group_ids, RMSE(), mode="macro", aggregation_weights="support"
    )
    assert not np.isclose(micro_rmse, macro_rmse, rtol=1e-6)


def test_weights_multiply_outside_the_first_level_loss():
    """w * L(r_g), never L(w * r_g).

    MSE separates the two: it is quadratic, so L(w*r) = w^2 L(r) while the correct form is
    w * L(r). A weighting applied inside would therefore square the support weights.
    """
    residuals, group_ids = residuals_and_groups(seed=3, ngroups=4, npoints=20)
    members = group_residual_indices(group_ids)
    sizes = np.array([float(len(indices)) for indices in members])
    point_loss = MSE()

    produced = grouped_cv_loss(
        residuals, group_ids, point_loss, mode="macro", aggregation_weights="support"
    )
    normalized = sizes / sizes.mean()
    outside = np.mean([w * point_loss(residuals[m]) for w, m in zip(normalized, members)])
    inside = np.mean([point_loss(w * residuals[m]) for w, m in zip(normalized, members)])

    assert np.isclose(produced, outside, rtol=0.0, atol=1e-12)
    # The two must actually differ here, or the assertion above proves nothing.
    assert not np.isclose(outside, inside, rtol=1e-6)


def test_sqrt_support_sits_between_uniform_and_support():
    residuals, group_ids = residuals_and_groups(seed=4)
    values = [
        grouped_cv_loss(residuals, group_ids, MAE(), mode="macro", aggregation_weights=w)
        for w in (None, "sqrt_support", "support")
    ]
    assert min(values[0], values[2]) <= values[1] <= max(values[0], values[2])


def test_explicit_aggregation_weights():
    residuals, group_ids = residuals_and_groups(seed=5, ngroups=4, npoints=20)
    members = group_residual_indices(group_ids)
    weights = np.array([1.0, 2.0, 3.0, 4.0])
    produced = grouped_cv_loss(
        residuals, group_ids, MAE(), mode="macro", aggregation_weights=weights
    )
    normalized = weights / weights.mean()
    expected = np.mean([w * MAE()(residuals[m]) for w, m in zip(normalized, members)])
    assert np.isclose(produced, expected, rtol=0.0, atol=1e-12)

    with pytest.raises(ValueError, match="entries for"):
        grouped_cv_loss(residuals, group_ids, MAE(), mode="macro", aggregation_weights=np.ones(3))


def test_singleton_groups_warn_but_stay_finite():
    """Self-consistent losses ARE defined on one residual, so no fallback -- but the
    objective then rests partly on pure noise, and that is worth saying out loud."""
    residuals = np.array([0.5, -0.2, 1.1, -0.7, 0.3])
    group_ids = np.array([0, 0, 1, 2, 3])  # three singleton groups
    point_loss = SelfConsistentHuber(COMPROMISE_COEFFICIENT)

    with pytest.warns(UserWarning, match="single point"):
        value = grouped_cv_loss(residuals, group_ids, point_loss, mode="macro")
    assert np.isfinite(value) and value > 0.0

    # The documented property that makes the no-fallback decision safe.
    single = np.array([0.7])
    assert np.isfinite(point_loss(single))
    assert np.isclose(point_loss(3.0 * single), 3.0 * point_loss(single), rtol=1e-12)

    # No singletons -> no warning at all, so the warning stays informative.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        grouped_cv_loss(residuals, np.array([0, 0, 1, 1, 1]), point_loss, mode="macro")


def test_micro_refuses_macro_only_arguments():
    """Silently ignoring them would read as if macro behaviour had been requested."""
    residuals, group_ids = residuals_and_groups(seed=6)
    with pytest.raises(ValueError, match="macro"):
        grouped_cv_loss(residuals, group_ids, MAE(), mode="micro", across_group_loss=MAE())
    with pytest.raises(ValueError, match="macro"):
        grouped_cv_loss(residuals, group_ids, MAE(), mode="micro", aggregation_weights="support")


def test_across_group_loss_is_applied():
    residuals, group_ids = residuals_and_groups(seed=7)
    members = group_residual_indices(group_ids)
    group_values = np.array([MAE()(residuals[m]) for m in members])
    produced = grouped_cv_loss(residuals, group_ids, MAE(), mode="macro", across_group_loss=RMSE())
    assert np.isclose(produced, RMSE()(group_values), rtol=0.0, atol=1e-14)


def test_guards():
    residuals, group_ids = residuals_and_groups(seed=8)
    with pytest.raises(ValueError, match="mode must be"):
        grouped_cv_loss(residuals, group_ids, MAE(), mode="nonsense")
    with pytest.raises(ValueError, match="group labels"):
        grouped_cv_loss(residuals, group_ids[:-1], MAE())
    # A negative label means the caller lost the residual-to-fold alignment.
    bad = group_ids.copy()
    bad[3] = -1
    with pytest.raises(ValueError, match="belong to a fold"):
        grouped_cv_loss(residuals, bad, MAE())
    with pytest.raises(ValueError, match="aggregation_weights must be"):
        grouped_cv_loss(residuals, group_ids, MAE(), mode="macro", aggregation_weights="nope")


# ---------------------------------------------------------------------------
# The k-fold l2reg loss classes now express their aggregation through grouped_cv_loss.
# ---------------------------------------------------------------------------


def kfold_problem(seed=3, npoints=50, ndimensions=12):
    rng = int2rng(seed)
    representations = rng.normal(size=(npoints, ndimensions))
    quantities = rng.normal(size=npoints)
    square_distances = ((representations[:, None, :] - representations[None, :, :]) ** 2).sum(-1)
    kernel = np.exp(-square_distances / 4.0)
    feature_matrix = rng.normal(size=(npoints, 24))
    group_ids = rng.integers(0, 8, size=npoints)
    return kernel, feature_matrix, quantities, group_ids


def shared_folds(npoints, group_ids):
    """Both fold kinds, seeded so the comparisons below are against a fixed split."""
    return KFolds(npoints, 4, rng=int2rng(0)), GroupKFolds(group_ids, 4, rng=int2rng(1))


L2REG = 1.0e-2


@pytest.mark.parametrize("fold_kind", ["plain", "group"])
@pytest.mark.parametrize("loss_name", ["MAE", "RMSE"])
def test_kfold_loss_modes_match_the_explicit_formulas(fold_kind, loss_name):
    """micro pools every residual; macro averages the per-fold losses.

    These are the two behaviours the old ``average_over_kfolds`` flag selected --
    False and True respectively -- so this pins the replacement as exact.
    """
    kernel, feature_matrix, quantities, group_ids = kfold_problem()
    plain, grouped = shared_folds(quantities.shape[0], group_ids)
    folds = plain if fold_kind == "plain" else grouped
    point_loss = MAE() if loss_name == "MAE" else RMSE()

    for build in (
        lambda **kw: KRRKfoldL2regLoss(quantities, kfolds=folds, train_kernel=kernel.copy(), **kw),
        lambda **kw: SORFKfoldL2regLoss(
            quantities, kfolds=folds, train_Z=feature_matrix.copy(), **kw
        ),
    ):
        fold_errors = build(loss_function=point_loss).kfold_errors(L2REG)
        micro = build(loss_function=point_loss, mode="micro").calculate_for_l2reg(L2REG)
        macro = build(loss_function=point_loss, mode="macro").calculate_for_l2reg(L2REG)
        assert np.isclose(micro, point_loss(np.concatenate(fold_errors)), rtol=0.0, atol=1e-13)
        assert np.isclose(
            macro, np.mean([point_loss(e) for e in fold_errors]), rtol=0.0, atol=1e-13
        )


def test_kfold_loss_defaults_to_micro_with_MAE():
    kernel, _, quantities, group_ids = kfold_problem()
    plain, _ = shared_folds(quantities.shape[0], group_ids)
    loss = KRRKfoldL2regLoss(quantities, kfolds=plain, train_kernel=kernel.copy())
    fold_errors = loss.kfold_errors(L2REG)
    assert loss.grouped_cv_mode == "micro"
    assert np.isclose(
        loss.calculate_for_l2reg(L2REG), MAE()(np.concatenate(fold_errors)), rtol=0.0, atol=1e-13
    )


def test_point_loss_overrides_loss_function():
    kernel, _, quantities, group_ids = kfold_problem()
    plain, _ = shared_folds(quantities.shape[0], group_ids)
    fold_errors = KRRKfoldL2regLoss(
        quantities, kfolds=plain, train_kernel=kernel.copy()
    ).kfold_errors(L2REG)
    overridden = KRRKfoldL2regLoss(
        quantities,
        kfolds=plain,
        train_kernel=kernel.copy(),
        loss_function=MAE(),
        point_loss=RMSE(),
    )
    assert np.isclose(
        overridden.calculate_for_l2reg(L2REG),
        RMSE()(np.concatenate(fold_errors)),
        rtol=0.0,
        atol=1e-13,
    )


def test_point_loss_defaults_to_the_passed_loss_function():
    """point_loss=None must pick up loss_function, not fall straight through to MAE.

    loss_function is set by the MRO SIBLING -- the concrete classes are
    (KfoldL2regLossMixin, <...>LeaveOneOutL2regLoss) -- so this resolves through a base the
    mixin itself does not inherit from, which is easy to mistake for dead code.
    """
    kernel, feature_matrix, quantities, group_ids = kfold_problem()
    plain, _ = shared_folds(quantities.shape[0], group_ids)
    fold_errors = KRRKfoldL2regLoss(
        quantities, kfolds=plain, train_kernel=kernel.copy()
    ).kfold_errors(L2REG)

    for build in (
        lambda **kw: KRRKfoldL2regLoss(quantities, kfolds=plain, train_kernel=kernel.copy(), **kw),
        lambda **kw: SORFKfoldL2regLoss(
            quantities, kfolds=plain, train_Z=feature_matrix.copy(), **kw
        ),
    ):
        inherited = build(loss_function=RMSE())
        assert isinstance(inherited.point_loss, RMSE)
        assert isinstance(build().point_loss, MAE)

    krr = KRRKfoldL2regLoss(
        quantities, kfolds=plain, train_kernel=kernel.copy(), loss_function=RMSE()
    )
    assert np.isclose(
        krr.calculate_for_l2reg(L2REG), RMSE()(np.concatenate(fold_errors)), rtol=0.0, atol=1e-13
    )

    # ... and through the Opt wrapper, which stores point_loss raw and lets the Loss resolve it.
    optimizer = KRRKfoldL2regOpt(
        lambda parameters: kernel.copy(), quantities, nkfolds=4, loss_function=RMSE()
    )
    assert isinstance(optimizer.get_l2reg_optimizer_from_parameters(None).point_loss, RMSE)


def test_kfold_loss_support_weighting_recovers_micro():
    kernel, _, quantities, group_ids = kfold_problem()
    _, grouped = shared_folds(quantities.shape[0], group_ids)
    micro = KRRKfoldL2regLoss(
        quantities, kfolds=grouped, train_kernel=kernel.copy(), mode="micro"
    ).calculate_for_l2reg(L2REG)
    macro_support = KRRKfoldL2regLoss(
        quantities,
        kfolds=grouped,
        train_kernel=kernel.copy(),
        mode="macro",
        aggregation_weights="support",
    ).calculate_for_l2reg(L2REG)
    assert np.isclose(micro, macro_support, rtol=0.0, atol=1e-12)


def test_kfold_loss_micro_refuses_macro_only_arguments():
    kernel, _, quantities, group_ids = kfold_problem()
    plain, _ = shared_folds(quantities.shape[0], group_ids)
    loss = KRRKfoldL2regLoss(
        quantities,
        kfolds=plain,
        train_kernel=kernel.copy(),
        mode="micro",
        aggregation_weights="support",
    )
    with pytest.raises(ValueError, match="macro"):
        loss.calculate_for_l2reg(L2REG)


def test_kfold_opt_forwards_the_aggregation_settings():
    kernel, _, quantities, group_ids = kfold_problem()
    optimizer = KRRKfoldL2regOpt(
        lambda parameters: kernel.copy(),
        quantities,
        nkfolds=4,
        mode="macro",
        aggregation_weights="sqrt_support",
    )
    inner = optimizer.get_l2reg_optimizer_from_parameters(None)
    assert inner.grouped_cv_mode == "macro"
    assert inner.aggregation_weights == "sqrt_support"
    assert np.isfinite(inner.calculate_for_l2reg(L2REG))


def test_kfold_rng_makes_the_split_reproducible():
    """Without it, KFolds draws from the GLOBAL numpy state.

    Two loss objects built from identical arguments then hold DIFFERENT folds, which makes
    any comparison between them meaningless and the run irreproducible. The classes let the
    caller set nkfolds and the training ratio, so not letting them set the generator was a
    gap rather than a design choice.
    """
    kernel, feature_matrix, quantities, group_ids = kfold_problem()

    def krr(seed):
        return KRRKfoldL2regLoss(
            quantities, nkfolds=4, kfold_rng=int2rng(seed), train_kernel=kernel.copy()
        )

    def sorf(seed):
        return SORFKfoldL2regLoss(
            quantities, nkfolds=4, kfold_rng=int2rng(seed), train_Z=feature_matrix.copy()
        )

    for build in (krr, sorf):
        assert np.array_equal(
            build(11).kfolds.all_shuffled_indices, build(11).kfolds.all_shuffled_indices
        )
        assert not np.array_equal(
            build(11).kfolds.all_shuffled_indices, build(12).kfolds.all_shuffled_indices
        )
        assert np.isclose(
            build(11).calculate_for_l2reg(L2REG), build(11).calculate_for_l2reg(L2REG)
        )

    # Group folds honour it too.
    grouped = [
        KRRKfoldL2regLoss(
            quantities,
            nkfolds=3,
            group_ids=group_ids,
            kfold_rng=int2rng(seed),
            train_kernel=kernel.copy(),
        ).kfolds.all_shuffled_indices
        for seed in (4, 4, 9)
    ]
    assert np.array_equal(grouped[0], grouped[1])
    assert not np.array_equal(grouped[0], grouped[2])


def test_kfold_rng_omitted_keeps_the_global_rng_behaviour():
    """None must not silently change what the classes always did."""
    kernel, _, quantities, _ = kfold_problem()
    splits = []
    for _ in range(2):
        np.random.seed(3)
        splits.append(
            KRRKfoldL2regLoss(
                quantities, nkfolds=4, train_kernel=kernel.copy()
            ).kfolds.all_shuffled_indices
        )
    assert np.array_equal(*splits)


def test_build_kfolds_forwards_rng_on_the_plain_path():
    """Regression: build_kfolds took an rng but passed it only to GroupKFolds, so on the
    plain path it silently did nothing."""
    first = build_kfolds(40, nkfolds=4, rng=int2rng(5)).all_shuffled_indices
    second = build_kfolds(40, nkfolds=4, rng=int2rng(5)).all_shuffled_indices
    assert np.array_equal(first, second)
    assert not np.array_equal(
        first, build_kfolds(40, nkfolds=4, rng=int2rng(6)).all_shuffled_indices
    )


def fold_blocks(folds):
    return sorted(
        tuple(sorted(folds.train_test_indices(i)[1].tolist())) for i in range(folds.nkfolds)
    )


def test_leave_one_group_out_flag_matches_the_constructor():
    """The flag must be exactly leave_one_group_out_folds, through every entry point."""
    kernel, feature_matrix, quantities, group_ids = kfold_problem()
    ngroups = len(group_index_lists(group_ids))
    reference = fold_blocks(leave_one_group_out_folds(group_ids))
    assert len(reference) == ngroups

    candidates = [
        build_kfolds(quantities.shape[0], group_ids=group_ids, leave_one_group_out=True),
        KRRKfoldL2regLoss(
            quantities, group_ids=group_ids, leave_one_group_out=True, train_kernel=kernel.copy()
        ).kfolds,
        KRRKfoldL2regOpt(
            lambda parameters: kernel.copy(),
            quantities,
            group_ids=group_ids,
            leave_one_group_out=True,
        ).kfolds,
        SORFKfoldL2regLoss(
            quantities,
            group_ids=group_ids,
            leave_one_group_out=True,
            train_Z=feature_matrix.copy(),
        ).kfolds,
        SORFKfoldL2regOpt(
            lambda parameters: feature_matrix.copy(),
            quantities,
            group_ids=group_ids,
            leave_one_group_out=True,
        ).kfolds,
    ]
    for folds in candidates:
        assert folds.nkfolds == ngroups
        assert fold_blocks(folds) == reference
        # Every validation block holds exactly one group; that is what LOGO means.
        for block in fold_blocks(folds):
            assert len(set(group_ids[list(block)])) == 1


def test_leave_one_group_out_ignores_nkfolds_and_rng():
    """The fold count IS the group count, and one group per fold leaves the rng nothing to
    do but permute fold order, which no aggregation over folds depends on."""
    kernel, _, quantities, group_ids = kfold_problem()
    ngroups = len(group_index_lists(group_ids))
    reference = fold_blocks(leave_one_group_out_folds(group_ids))

    fewer = KRRKfoldL2regLoss(
        quantities,
        group_ids=group_ids,
        leave_one_group_out=True,
        nkfolds=3,
        train_kernel=kernel.copy(),
    )
    assert fewer.kfolds.nkfolds == ngroups

    seeded = KRRKfoldL2regLoss(
        quantities,
        group_ids=group_ids,
        leave_one_group_out=True,
        kfold_rng=int2rng(9),
        train_kernel=kernel.copy(),
    )
    assert fold_blocks(seeded.kfolds) == reference


def test_leave_one_group_out_guards():
    kernel, _, quantities, group_ids = kfold_problem()
    with pytest.raises(ValueError, match="requires group_ids"):
        build_kfolds(quantities.shape[0], leave_one_group_out=True)
    with pytest.raises(ValueError, match="builds the folds itself"):
        build_kfolds(
            quantities.shape[0],
            group_ids=group_ids,
            leave_one_group_out=True,
            kfolds=leave_one_group_out_folds(group_ids),
        )
    with pytest.raises(ValueError, match="requires group_ids"):
        KRRKfoldL2regLoss(quantities, leave_one_group_out=True, train_kernel=kernel.copy())


def test_group_ids_alone_is_not_leave_one_group_out():
    """The reason the flag exists: group_ids at the default nkfolds is a DIFFERENT criterion,
    and with more groups than folds it is silently different rather than an error."""
    kernel, _, quantities, _ = kfold_problem()
    many_groups = int2rng(21).integers(0, 20, size=quantities.shape[0])
    ngroups = len(group_index_lists(many_groups))
    assert ngroups > 8

    default = KRRKfoldL2regLoss(quantities, group_ids=many_groups, train_kernel=kernel.copy())
    assert default.kfolds.nkfolds == 8 != ngroups

    flagged = KRRKfoldL2regLoss(
        quantities, group_ids=many_groups, leave_one_group_out=True, train_kernel=kernel.copy()
    )
    assert flagged.kfolds.nkfolds == ngroups


if __name__ == "__main__":
    test_micro_reduces_to_the_pooled_loss()
    test_macro_equal_weight_is_the_mean_of_per_group_losses()
    test_support_weighting_bridges_macro_and_micro()
    test_weights_multiply_outside_the_first_level_loss()
    test_sqrt_support_sits_between_uniform_and_support()
    test_explicit_aggregation_weights()
    test_singleton_groups_warn_but_stay_finite()
    test_micro_refuses_macro_only_arguments()
    test_across_group_loss_is_applied()
    test_guards()
    for fold_kind in ("plain", "group"):
        for loss_name in ("MAE", "RMSE"):
            test_kfold_loss_modes_match_the_explicit_formulas(fold_kind, loss_name)
    test_kfold_loss_defaults_to_micro_with_MAE()
    test_point_loss_overrides_loss_function()
    test_point_loss_defaults_to_the_passed_loss_function()
    test_kfold_loss_support_weighting_recovers_micro()
    test_kfold_loss_micro_refuses_macro_only_arguments()
    test_kfold_opt_forwards_the_aggregation_settings()
    test_kfold_rng_makes_the_split_reproducible()
    test_kfold_rng_omitted_keeps_the_global_rng_behaviour()
    test_build_kfolds_forwards_rng_on_the_plain_path()
    test_leave_one_group_out_flag_matches_the_constructor()
    test_leave_one_group_out_ignores_nkfolds_and_rng()
    test_leave_one_group_out_guards()
    test_group_ids_alone_is_not_leave_one_group_out()
