"""Unit tests for qml2.models.kfolds -- group-aware splitting.

The invariant everything else rests on is that a group is never split across a train/test
boundary. For force models the group is a conformer, whose energy and force components
must move together; for scaffold splits it is a chemotype. Both reduce to the same check.
"""

import numpy as np
import pytest
from conftest import get_qm7_smiles, int2rng

from qml2.cheminfo.scaffolds import scaffold_group_ids
from qml2.models.kfolds import (
    GroupKFolds,
    GroupKFoldsMultipleObservables,
    conformer_folds,
    frequency_ordered_group_split,
    group_index_lists,
    leave_one_group_out_folds,
    observable_group_ids,
    observable_ranges,
    random_group_subset,
)
from qml2.models.loss_functions import MAE
from qml2.models.sorf_hyperparameter_optimization import SORFKfoldL2regLoss

NPOINTS = 40


def test_group_index_lists_drops_negative_labels():
    labels = np.array([0, 1, 0, -1, 2, -1, 1])
    groups = group_index_lists(labels)
    assert [g.tolist() for g in groups] == [[0, 2], [1, 6], [4]]


def test_group_kfolds_partition_and_never_split_a_group():
    rng = int2rng(1)
    group_ids = rng.integers(0, 9, size=NPOINTS)
    folds = GroupKFolds(group_ids, 4, rng=int2rng(2))

    seen = []
    for fold_id in range(folds.nkfolds):
        train_indices, test_indices = folds.train_test_indices(fold_id)
        seen.append(test_indices)
        assert not set(group_ids[train_indices]) & set(group_ids[test_indices])
        assert not set(train_indices.tolist()) & set(test_indices.tolist())
    assert np.array_equal(np.sort(np.concatenate(seen)), np.arange(NPOINTS))
    assert folds.fold_sizes().sum() == NPOINTS


def test_group_kfolds_refuses_too_few_groups():
    with pytest.raises(ValueError, match="cannot fill"):
        GroupKFolds(np.arange(NPOINTS) % 3, 5)
    with pytest.raises(ValueError, match="at least 2"):
        GroupKFolds(np.arange(NPOINTS), 1)


def test_leave_one_group_out_gives_one_fold_per_group():
    group_ids = np.arange(NPOINTS) % 8
    folds = leave_one_group_out_folds(group_ids)
    assert folds.nkfolds == 8
    for fold_id in range(folds.nkfolds):
        _, test_indices = folds.train_test_indices(fold_id)
        assert np.unique(group_ids[test_indices]).shape[0] == 1


def test_random_group_subset_complement_shares_no_group():
    rng = int2rng(3)
    group_ids = rng.integers(0, 10, size=NPOINTS)
    subset, complement = random_group_subset(group_ids, 25, rng=int2rng(4))
    assert not set(group_ids[subset]) & set(group_ids[complement])
    assert subset.shape[0] + complement.shape[0] == NPOINTS
    assert subset.shape[0] <= 25


def test_random_group_subset_reports_an_unfillable_target():
    group_ids = np.zeros(NPOINTS, dtype=int)
    group_ids[30:] = np.arange(1, NPOINTS - 29)
    with pytest.raises(ValueError, match="whole groups"):
        random_group_subset(group_ids, 20, rng=int2rng(5))
    # With the floor removed the shortfall is accepted instead.
    subset, _ = random_group_subset(group_ids, 20, rng=int2rng(5), min_fill_fraction=0.0)
    assert subset.shape[0] < 20


def test_observable_ranges_and_group_ids():
    counts = [4, 1, 3]
    assert np.array_equal(observable_ranges(counts), np.array([0, 4, 5, 8]))
    # Default: one group per item, so an item's observables share a label.
    assert np.array_equal(observable_group_ids(counts), np.array([0, 0, 0, 0, 1, 2, 2, 2]))
    # Items can be grouped further, e.g. by scaffold.
    assert np.array_equal(
        observable_group_ids(counts, item_group_ids=[7, 7, 9]),
        np.array([7, 7, 7, 7, 7, 9, 9, 9]),
    )


def test_conformer_folds_keep_energy_and_forces_together():
    """A conformer owns 3*natoms + 1 observables; none of them may cross the boundary."""
    natoms = np.array([3, 5, 4, 6, 3, 5, 4, 7])
    observables_per_conformer = natoms * 3 + 1
    conformer_ids = observable_group_ids(observables_per_conformer)

    folds = conformer_folds(natoms.shape[0], 4, rng=int2rng(6))
    expanded = GroupKFoldsMultipleObservables(observables_per_conformer, folds)

    total = int(observables_per_conformer.sum())
    seen = []
    for fold_id in range(expanded.nkfolds):
        train_indices, test_indices = expanded.train_test_indices(fold_id)
        seen.append(test_indices)
        assert not set(conformer_ids[train_indices]) & set(conformer_ids[test_indices])
    assert np.array_equal(np.sort(np.concatenate(seen)), np.arange(total))


def test_two_expansions_of_one_split_agree_on_conformers():
    """OQML expands the same split onto observables (rows) and atoms (columns).

    Built independently the two would drift apart, because the greedy fold fill balances on
    block size and the blocks differ (3*natoms + 1 against natoms). Sharing one item-level
    folds object is what keeps them consistent; this is that guarantee.
    """
    natoms = np.array([3, 5, 4, 6, 3, 5, 4, 7])
    folds = conformer_folds(natoms.shape[0], 4, rng=int2rng(7))
    observable_folds = GroupKFoldsMultipleObservables(natoms * 3 + 1, folds)
    atom_folds = GroupKFoldsMultipleObservables(natoms, folds)

    observable_ids = observable_group_ids(natoms * 3 + 1)
    atom_ids = observable_group_ids(natoms)
    for fold_id in range(folds.nkfolds):
        _, test_conformers = folds.train_test_indices(fold_id)
        _, test_observables = observable_folds.train_test_indices(fold_id)
        _, test_atoms = atom_folds.train_test_indices(fold_id)
        expected = set(test_conformers.tolist())
        assert set(observable_ids[test_observables]) == expected
        assert set(atom_ids[test_atoms]) == expected


def test_conformer_folds_accept_scaffold_level_grouping():
    """Grouping the conformers themselves needs nothing new: pass their group labels."""
    natoms = np.array([3, 5, 4, 6, 3, 5])
    # Conformers 0-2 belong to one molecule, 3-5 to another.
    molecule_ids = np.array([0, 0, 0, 1, 1, 1])
    folds = conformer_folds(natoms.shape[0], 2, conformer_group_ids=molecule_ids)
    observable_ids = observable_group_ids(natoms * 3 + 1, item_group_ids=molecule_ids)
    expanded = GroupKFoldsMultipleObservables(natoms * 3 + 1, folds)
    for fold_id in range(expanded.nkfolds):
        train_indices, test_indices = expanded.train_test_indices(fold_id)
        assert not set(observable_ids[train_indices]) & set(observable_ids[test_indices])


# ---------------------------------------------------------------------------
# Frequency-ordered splits.
# ---------------------------------------------------------------------------

# The reference dataset is the QM7 SMILES shipped in test_data, NOT FreeSolv or the
# electrochemical-stability set: those live under $DATA and are absent on some machines, so
# a test keyed to them would fail for the wrong reason.
REFERENCE_NMOLECULES = 1000
REFERENCE_TRAIN_FRACTION = 0.8

# Committed properties of the deterministic split on that prefix, with
# scaffold_group_ids(acyclic="skeleton", include_chirality=False). A change in the grouping
# or in the tie-break shows up here rather than as a quietly different number downstream.
# The group counts are the sensitive part; the split sizes come out exact because QM7 has
# plenty of small groups for the greedy fill to land on.
REFERENCE_NGROUPS = 132
REFERENCE_LARGEST_GROUP = 126
REFERENCE_SPLIT_SIZES = (800, 200)

ENSEMBLE_SEEDS = (274658, 20786, 109790, 478390)


def uneven_group_ids():
    """Groups of sizes 3, 2, 2, 1, 1 -- two equal-size blocks to exercise tie-breaking."""
    return np.array([0, 0, 0, 1, 1, 2, 2, 3, 4])


def reference_group_ids():
    """Scaffold labels for the committed QM7 prefix, at the paper's grouping."""
    all_SMILES = get_qm7_smiles(REFERENCE_NMOLECULES)
    return all_SMILES, scaffold_group_ids(all_SMILES, acyclic="skeleton", include_chirality=False)


def test_frequency_split_matches_hand_computed_deepchem_order():
    """Pins the tie-break without needing DeepChem installed.

    Groups and DeepChem's key (size, smallest member index), both DESCENDING:

        group 0 = [0,1,2]  (3, 0)   -> first
        group 2 = [5,6]    (2, 5)   -> before group 1, because 5 > 3
        group 1 = [3,4]    (2, 3)
        group 4 = [8]      (1, 8)   -> before group 3, because 8 > 7
        group 3 = [7]      (1, 7)

    With train_size=5 the fill takes groups 0 and 2 and stops; the rest is test. Note that
    group 2 preceding group 1 is the whole point -- a stable sort on size alone, or a
    tie-break on first appearance, would give [0,1,2,3,4] and a different split.
    """
    train_indices, test_indices = frequency_ordered_group_split(uneven_group_ids(), 5)
    assert train_indices.tolist() == [0, 1, 2, 5, 6]
    assert test_indices.tolist() == [3, 4, 7, 8]


def test_frequency_split_deterministic_and_ignores_rng():
    group_ids = uneven_group_ids()
    first = frequency_ordered_group_split(group_ids, 5)
    second = frequency_ordered_group_split(group_ids, 5, rng=int2rng(1))
    third = frequency_ordered_group_split(group_ids, 5, rng=int2rng(2))
    for other in (second, third):
        assert np.array_equal(first[0], other[0])
        assert np.array_equal(first[1], other[1])


def test_frequency_split_covers_and_separates(shuffle_ties=False):
    rng = int2rng(3)
    group_ids = rng.integers(0, 12, size=NPOINTS)
    train_indices, test_indices = frequency_ordered_group_split(
        group_ids, 30, shuffle_ties=shuffle_ties, rng=int2rng(4)
    )
    assert not set(train_indices.tolist()) & set(test_indices.tolist())
    assert np.array_equal(
        np.sort(np.concatenate([train_indices, test_indices])), np.arange(NPOINTS)
    )
    assert not set(group_ids[train_indices]) & set(group_ids[test_indices])
    assert np.array_equal(train_indices, np.sort(train_indices))
    assert np.array_equal(test_indices, np.sort(test_indices))


def test_frequency_split_covers_and_separates_with_shuffled_ties():
    test_frequency_split_covers_and_separates(shuffle_ties=True)


def test_frequency_split_keeps_the_dominant_group_in_train():
    """The defining property: the largest group never lands in test, in either mode."""
    rng = int2rng(5)
    # Enough groups that a block of equal-size ones straddles the boundary, so the
    # shuffle_ties=True branch below is exercised rather than warned about.
    group_ids = rng.integers(0, 16, size=NPOINTS)
    sizes = np.bincount(group_ids)
    largest_label = int(np.argmax(sizes))
    largest_members = set(np.flatnonzero(group_ids == largest_label).tolist())

    for shuffle_ties in (False, True):
        train_indices, test_indices = frequency_ordered_group_split(
            group_ids, 30, shuffle_ties=shuffle_ties, rng=int2rng(6)
        )
        assert largest_members <= set(train_indices.tolist())
        # ... and nothing as large as the dominant group is ever held out. Stated against
        # the largest PRESENT group rather than sizes.min(), which is 0 whenever a label in
        # the range happens to draw no members.
        test_group_sizes = {int(sizes[label]) for label in group_ids[test_indices]}
        assert max(test_group_sizes) < int(sizes.max())


def test_frequency_split_random_ties_move_only_boundary_groups():
    """Different seeds must move membership but not sizes, and only at the boundary."""
    rng = int2rng(7)
    group_ids = rng.integers(0, 14, size=NPOINTS)
    sizes = np.bincount(group_ids)
    train_size = 28

    draws = [
        frequency_ordered_group_split(group_ids, train_size, shuffle_ties=True, rng=int2rng(seed))
        for seed in range(4)
    ]
    shapes = {(train.shape[0], test.shape[0]) for train, test in draws}
    assert len(shapes) == 1, "realized sizes depend only on group sizes, not on tie order"

    memberships = [set(test.tolist()) for _, test in draws]
    assert len(set(map(frozenset, memberships))) > 1, "seeds must actually differ"

    moved = set()
    for first in memberships:
        for second in memberships:
            moved |= first ^ second
    moved_sizes = {int(sizes[group_ids[index]]) for index in moved}
    # Only groups whose size straddles the boundary may move; the big ones are pinned.
    assert max(moved_sizes) < int(sizes.max())

    # Same seed, same answer.
    repeat = frequency_ordered_group_split(
        group_ids, train_size, shuffle_ties=True, rng=int2rng(0)
    )
    assert np.array_equal(repeat[1], draws[0][1])


def test_frequency_split_drops_negative_labels():
    group_ids = np.array([0, 0, -1, 1, 1, -1, 2, 3])
    train_indices, test_indices = frequency_ordered_group_split(group_ids, 4)
    combined = set(train_indices.tolist()) | set(test_indices.tolist())
    assert 2 not in combined and 5 not in combined
    assert combined == {0, 1, 3, 4, 6, 7}


def test_frequency_split_warns_when_ties_cannot_matter():
    """Sizes 4,3,2,1 with train_size=7: the boundary falls BETWEEN size levels.

    Every seed then gives the same split, and an "ensemble" would be N identical draws
    with error bars that look real but are zero. The warning is the only thing standing
    between that and a published number.
    """
    group_ids = np.array([0, 0, 0, 0, 1, 1, 1, 2, 2, 3])
    with pytest.warns(UserWarning, match="no effect"):
        first = frequency_ordered_group_split(group_ids, 7, shuffle_ties=True, rng=int2rng(1))
    with pytest.warns(UserWarning, match="no effect"):
        second = frequency_ordered_group_split(group_ids, 7, shuffle_ties=True, rng=int2rng(2))
    assert np.array_equal(first[1], second[1])


def test_frequency_split_min_fill_fraction_is_tunable():
    """The floor is a keyword argument, not a module constant.

    Same name and default as random_group_subset, which applies the identical floor for the
    identical reason -- a dominant group making the target unreachable.
    """
    dominated = np.zeros(NPOINTS, dtype=int)
    dominated[30:] = np.arange(1, NPOINTS - 29)

    with pytest.raises(ValueError, match="whole groups"):
        frequency_ordered_group_split(dominated, 20)
    # Lowering it accepts the shortfall instead of refusing.
    train_indices, _ = frequency_ordered_group_split(dominated, 20, min_fill_fraction=0.0)
    assert train_indices.shape[0] < 20
    # And raising it makes an otherwise-fine split fail, so the argument is really consulted.
    group_ids = int2rng(31).integers(0, 9, size=NPOINTS)
    frequency_ordered_group_split(group_ids, 30, min_fill_fraction=0.9)
    with pytest.raises(ValueError, match="whole groups"):
        frequency_ordered_group_split(group_ids, 30, min_fill_fraction=1.01)


def test_frequency_split_guards():
    group_ids = np.arange(NPOINTS) % 10
    with pytest.raises(ValueError, match="exceeds"):
        frequency_ordered_group_split(group_ids, NPOINTS + 1)

    dominated = np.zeros(NPOINTS, dtype=int)
    dominated[30:] = np.arange(1, NPOINTS - 29)
    with pytest.raises(ValueError, match="whole groups"):
        frequency_ordered_group_split(dominated, 20)


def test_frequency_split_reference_sizes():
    """Regression against committed properties of the 80/20 split on real molecules."""
    _, group_ids = reference_group_ids()
    assert np.unique(group_ids).shape[0] == REFERENCE_NGROUPS
    assert int(np.bincount(group_ids).max()) == REFERENCE_LARGEST_GROUP

    train_size = int(round(REFERENCE_TRAIN_FRACTION * group_ids.shape[0]))
    train_indices, test_indices = frequency_ordered_group_split(group_ids, train_size)
    assert (train_indices.shape[0], test_indices.shape[0]) == REFERENCE_SPLIT_SIZES

    # The paper's ensemble: same sizes from every seed, differing membership.
    memberships = set()
    for seed in ENSEMBLE_SEEDS:
        shuffled = frequency_ordered_group_split(
            group_ids, train_size, shuffle_ties=True, rng=np.random.default_rng(seed)
        )
        assert (shuffled[0].shape[0], shuffled[1].shape[0]) == REFERENCE_SPLIT_SIZES
        memberships.add(frozenset(shuffled[1].tolist()))
    assert len(memberships) == len(ENSEMBLE_SEEDS), "seeds must give distinct splits here"


def test_frequency_split_feeds_a_group_aware_hyperparameter_loss():
    """The intended downstream use: split, then optimize hyperparameters on the train side.

    Group labels carry over by plain indexing -- ``group_ids[train_indices]`` -- because
    every group consumer keys on the label rather than assuming a contiguous range. So the
    folds built inside the loss respect the same scaffolds the split did.
    """
    rng = int2rng(11)
    npoints = 60
    group_ids = rng.integers(0, 15, size=npoints)
    train_indices, test_indices = frequency_ordered_group_split(group_ids, 48)

    train_group_ids = group_ids[train_indices]
    feature_matrix = rng.normal(size=(npoints, 20))[train_indices]
    quantities = rng.normal(size=npoints)[train_indices]

    loss = SORFKfoldL2regLoss(
        quantities,
        group_ids=train_group_ids,
        nkfolds=4,
        train_Z=feature_matrix.copy(),
        loss_function=MAE(),
    )
    value = loss.calculate_for_l2reg(1.0e-2)
    assert np.isfinite(value) and value > 0.0

    # The loss's own folds must not split a scaffold either.
    for fold_id in range(loss.kfolds.nkfolds):
        fold_train, fold_test = loss.kfolds.train_test_indices(fold_id)
        assert not set(train_group_ids[fold_train]) & set(train_group_ids[fold_test])


def test_frequency_split_agrees_with_deepchem():
    """The real "we match the standard tool" check; skipped unless DeepChem is installed."""
    deepchem = pytest.importorskip("deepchem")

    all_SMILES = get_qm7_smiles(REFERENCE_NMOLECULES)
    # acyclic="pool" deliberately: DeepChem's ScaffoldSplitter puts every ring-free molecule
    # in a single group, so reproducing it needs that convention rather than the default.
    group_ids = scaffold_group_ids(all_SMILES, acyclic="pool", include_chirality=False)
    frac_train = REFERENCE_TRAIN_FRACTION
    train_size = int(round(frac_train * group_ids.shape[0]))
    ours_train, ours_test = frequency_ordered_group_split(group_ids, train_size)

    dataset = deepchem.data.NumpyDataset(
        X=np.zeros((len(all_SMILES), 1)),
        y=np.zeros(len(all_SMILES)),
        ids=np.array(all_SMILES),
    )
    theirs_train, _, theirs_test = deepchem.splits.ScaffoldSplitter().split(
        dataset, frac_train=frac_train, frac_valid=0.0, frac_test=1.0 - frac_train
    )
    assert set(ours_train.tolist()) == set(int(i) for i in theirs_train)
    assert set(ours_test.tolist()) == set(int(i) for i in theirs_test)


if __name__ == "__main__":
    test_group_index_lists_drops_negative_labels()
    test_group_kfolds_partition_and_never_split_a_group()
    test_group_kfolds_refuses_too_few_groups()
    test_leave_one_group_out_gives_one_fold_per_group()
    test_random_group_subset_complement_shares_no_group()
    test_random_group_subset_reports_an_unfillable_target()
    test_observable_ranges_and_group_ids()
    test_conformer_folds_keep_energy_and_forces_together()
    test_two_expansions_of_one_split_agree_on_conformers()
    test_conformer_folds_accept_scaffold_level_grouping()
    test_frequency_split_matches_hand_computed_deepchem_order()
    test_frequency_split_deterministic_and_ignores_rng()
    test_frequency_split_covers_and_separates()
    test_frequency_split_covers_and_separates_with_shuffled_ties()
    test_frequency_split_keeps_the_dominant_group_in_train()
    test_frequency_split_random_ties_move_only_boundary_groups()
    test_frequency_split_drops_negative_labels()
    test_frequency_split_warns_when_ties_cannot_matter()
    test_frequency_split_min_fill_fraction_is_tunable()
    test_frequency_split_guards()
    test_frequency_split_feeds_a_group_aware_hyperparameter_loss()
    test_frequency_split_reference_sizes()
    # Skips when DeepChem is absent, and pytest.skip RAISES rather than returns -- from
    # BaseException, not Exception -- so a bare call would abort the whole run.
    try:
        test_frequency_split_agrees_with_deepchem()
    except pytest.skip.Exception as reason:
        print(f"skipped test_frequency_split_agrees_with_deepchem: {reason}")
