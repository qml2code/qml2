"""Splitting a dataset into folds, with or without respecting groups.

Splits ONLY -- the losses that consume them live in ``hyperparameter_optimization`` and
``sorf_hyperparameter_optimization``. An earlier generation of k-fold loss classes (the
``callable_ninv_MAE`` family, which evaluated a hyperparameter by refitting the model once
per fold) used to live here too; it was deleted once the leave-one-out and k-fold losses
next door superseded it, since those get the same numbers from a single eigendecomposition
instead of ``nkfolds`` refits.

``KFolds`` is the plain random split. GROUP-aware splitting is the reason this module
exists as its own file: a random split lets a molecule land in test while a near-twin of it
sits in training, whereas holding out whole GROUPS -- scaffolds, in practice -- measures
generalization to chemotypes the model has not seen.

Nothing here knows what a group means. A group is an integer label per data point, and
``qml2.cheminfo.scaffolds`` is what turns molecules into those labels; any other
grouping works identically.

On learning curves, note which step is group-aware and which is not. The train/test SPLIT
must be group-aware, and :func:`random_group_subset` does that by moving whole groups.
Subsets drawn from the resulting training pool for the curve's successive points do NOT
need to be: the pool already shares no group with test, so every subset of it is
group-disjoint from test too. That distinction is what lets a group-split learning curve
hit exact training-set sizes -- ``50, 100, 200, ...`` -- which a curve built by
accumulating whole groups cannot, since group sizes do not add up to round numbers.
"""

import warnings

import numpy as np


# Separating sets into training and test.
def training_test_mols(total_set_size, training_set_size, nkfolds, rng=None):
    """Independent shuffles of all indices, one per fold, split at a fixed separator.

    ``rng=None`` draws from the GLOBAL numpy random state, which is what this always did
    and what ``np.random.seed`` therefore still controls. Pass a Generator to decouple the
    folds from global state: without one, two KFolds built from the same arguments are
    different splits, so anything comparing two loss objects has to build the folds once
    and share them.
    """
    temp_id_arr = np.array(range(total_set_size), dtype=int)
    shuffled_mols = np.empty((nkfolds, total_set_size), dtype=int)
    set_separators = np.repeat(training_set_size, nkfolds)
    shuffle = np.random.shuffle if rng is None else rng.shuffle
    for kfold_id in range(nkfolds):
        shuffle(temp_id_arr)
        shuffled_mols[kfold_id, :] = temp_id_arr[:]
    return shuffled_mols, set_separators


# Creating k-folds.
# KK asked here whether a standard k-fold generator could keep a molecule's atoms together
# in one set; GroupKFoldsMultipleObservables below does exactly that, driven by group labels
# rather than by a bespoke separator.
class KFolds:
    def __init__(
        self,
        total_set_size,
        nkfolds,
        training_set_ratio=0.5,
        training_set_size=None,
        rng=None,
    ):
        """
        Stores indices of KFolds for hyperparameter optimization.

        ``rng``: pass a numpy Generator to make the split reproducible. None keeps the
        historical behaviour of drawing from the global numpy random state.
        """
        self.total_set_size = total_set_size
        self.nkfolds = nkfolds
        self.training_set_size = training_set_size
        if (self.training_set_size is None) and (training_set_ratio is not None):
            self.training_set_size = int(training_set_ratio * total_set_size)
        self.all_shuffled_indices = None
        self.all_training_test_set_separators = None
        self.all_shuffled_indices, self.all_training_test_set_separators = training_test_mols(
            self.total_set_size, self.training_set_size, nkfolds, rng=rng
        )

    def train_test_indices(self, kfold_id):
        shuffled_indices = self.all_shuffled_indices[kfold_id]
        separator = self.all_training_test_set_separators[kfold_id]
        return shuffled_indices[:separator], shuffled_indices[separator:]


# ---------------------------------------------------------------------------
# Group-aware splitting.
#
# A "group" is an integer label per data point; points sharing a label are never split
# across a train/test boundary. See the module docstring.
# ---------------------------------------------------------------------------


def group_index_lists(group_ids):
    """The member indices of each group, as a list of index arrays.

    Points labelled negative are treated as UNGROUPED and dropped -- that is the
    convention ``scaffold_group_ids(..., on_unparsable="skip")`` writes. Groups come back
    in order of first appearance, so the output depends on the input order alone.
    """
    group_ids = np.asarray(group_ids)
    if group_ids.ndim != 1:
        raise ValueError("group_ids must be a 1D array of labels, one per data point")
    order = {}
    members = []
    for index, label in enumerate(group_ids):
        if label < 0:
            continue
        label = int(label)
        if label not in order:
            order[label] = len(members)
            members.append([])
        members[order[label]].append(index)
    return [np.array(m, dtype=int) for m in members]


def random_group_subset(group_ids, subset_size, rng=None, min_fill_fraction=0.9):
    """Indices of a random subset assembled from WHOLE groups.

    The complement of the returned subset shares no group with it, which is the property
    the whole exercise rests on. Because groups are indivisible the realized size only
    approaches ``subset_size`` from below; groups are tried in random order and each is
    taken if it still fits, so a large group early in the order does not block the small
    ones behind it.

    Returns ``(subset_indices, complement_indices)``, both sorted.

    Raises if the realized size falls below ``min_fill_fraction * subset_size``, which
    happens when one group is so large that it cannot fit -- on FreeSolv with
    ``acyclic="pool"`` a single group holds half the dataset, and silently returning a
    much smaller training set than asked for would show up later as an unexplained kink
    in a learning curve. Pass ``min_fill_fraction=0`` to accept whatever fits.
    """
    if rng is None:
        rng = np.random.default_rng()
    groups = group_index_lists(group_ids)
    total = sum(len(g) for g in groups)
    if subset_size > total:
        raise ValueError(f"subset_size {subset_size} exceeds the {total} grouped points")

    chosen = []
    running = 0
    for position in rng.permutation(len(groups)):
        group = groups[position]
        if running + len(group) <= subset_size:
            chosen.append(group)
            running += len(group)
        if running == subset_size:
            break

    if running < min_fill_fraction * subset_size:
        largest = max((len(g) for g in groups), default=0)
        raise ValueError(
            f"could only fill {running} of the requested {subset_size} points with whole "
            f"groups (largest group holds {largest}). Use a finer grouping -- for "
            "scaffolds, acyclic='skeleton' instead of 'pool' -- or lower "
            "min_fill_fraction to accept the shortfall."
        )

    subset = np.sort(np.concatenate(chosen)) if chosen else np.zeros(0, dtype=int)
    mask = np.ones(np.asarray(group_ids).shape[0], dtype=bool)
    mask[subset] = False
    # Ungrouped points belong to neither side; drop them from the complement too.
    ungrouped = np.asarray(group_ids) < 0
    mask[ungrouped] = False
    return subset, np.flatnonzero(mask)


def _greedy_group_fill(ordered_groups, train_size):
    """Walk groups in order, taking each into train if it still fits, else into test.

    Scanning continues past a group that does not fit, so a smaller group behind a large
    one can still land in train. The decision depends only on group SIZES, which is why
    every tie ordering realizes the same train/test sizes.
    """
    train_groups = []
    test_groups = []
    running = 0
    for group in ordered_groups:
        if running + len(group) <= train_size:
            train_groups.append(group)
            running += len(group)
        else:
            test_groups.append(group)
    return train_groups, test_groups, running


def frequency_ordered_group_split(
    group_ids, train_size, shuffle_ties=False, rng=None, min_fill_fraction=0.9
):
    """Split indices into (train, test), filling whole groups largest-first.

    Groups are ordered by descending size, so the dominant scaffold is ALWAYS in train and
    the churn is confined to the small groups straddling the train/test boundary. Within a
    block of equal-size groups the order is set by ``shuffle_ties``:

      shuffle_ties=False (default)
          Deterministic: ties broken by the group's smallest member index, largest-first,
          matching DeepChem's / MoleculeNet's ``ScaffoldSplitter``. This is the reference
          split, for comparability with the scaffold-split literature. ``rng`` is ignored.
          Note the tie-break is a ROW-ORDER key, not a chemical one, so the split is
          reproducible only for a fixed input order.

      shuffle_ties=True
          Same-size groups are shuffled with ``rng`` first; groups of DIFFERENT sizes keep
          their strict largest-first order. This averages over the boundary order-artifact
          that the deterministic split inherits, while keeping the dominant group in train.
          Use several seeds for an ensemble with error bars.

    Both are distinct from the MoleculeNet/SCAGE "random scaffold split", which shuffles
    ALL groups and so lets the dominant group flip between train and test. Fixing the
    frequency ordering and randomizing only ties is a deliberate variance-reduction choice:
    the resulting error bars reflect boundary ambiguity -- roughly 20 molecules on FreeSolv
    -- rather than the coin-flip placement of the largest group.

    Args:
        group_ids: 1D int array, one label per data point. Negative labels are UNGROUPED
            and dropped from both outputs, as in :func:`group_index_lists`.
        train_size: TARGET number of training points. Groups being indivisible, the
            realized train size only approaches this from below.
        shuffle_ties: see above.
        rng: used only when ``shuffle_ties=True``. None creates a non-reproducible
            default_rng; for an ensemble pass ``np.random.default_rng(seed)``, one seed per
            draw.
        min_fill_fraction: the realized train size must reach this fraction of
            ``train_size`` or the split is refused. One dominant group can make the target
            unreachable, and a silently short training set surfaces later as an unexplained
            result rather than an error. Pass 0 to accept whatever fits. Named to match
            :func:`random_group_subset`, which applies the same floor for the same reason.

    Returns:
        ``(train_indices, test_indices)``, both sorted ascending, disjoint, together
        covering every grouped point once, sharing no group.
    """
    groups = group_index_lists(group_ids)
    n_grouped = sum(len(group) for group in groups)
    if train_size > n_grouped:
        raise ValueError(
            f"train_size {train_size} exceeds the {n_grouped} grouped points available"
        )

    # DeepChem's key is (group_size, smallest_member_index), both descending. Pinned by a
    # regression test rather than trusted, because the key has varied across versions.
    deterministic_order = sorted(groups, key=lambda g: (len(g), int(g.min())), reverse=True)

    if shuffle_ties:
        if rng is None:
            rng = np.random.default_rng()
        shuffled = [groups[position] for position in rng.permutation(len(groups))]
        # Stable sort on size alone: different sizes take strict descending order and the
        # shuffled order survives only WITHIN each equal-size block. Sorting by member
        # index here would undo the shuffle, which is the whole tie-break in this branch.
        ordered = sorted(shuffled, key=lambda g: -len(g))

        # Randomizing ties only does something if the boundary falls INSIDE an equal-size
        # block; otherwise every seed collapses to the deterministic split and an
        # "ensemble" would be N identical draws with zero error bars that look real.
        boundary_train, boundary_test, _ = _greedy_group_fill(deterministic_order, train_size)
        train_block_sizes = {len(group) for group in boundary_train}
        test_block_sizes = {len(group) for group in boundary_test}
        if not (train_block_sizes & test_block_sizes):
            warnings.warn(
                "shuffle_ties=True has no effect for this (group_ids, train_size): no "
                "block of equal-size groups straddles the train/test boundary, so every "
                "seed gives the same split. Change train_size or use a finer grouping if "
                "an ensemble with error bars was intended.",
                stacklevel=2,
            )
    else:
        ordered = deterministic_order

    train_groups, test_groups, running = _greedy_group_fill(ordered, train_size)

    if running < min_fill_fraction * train_size:
        largest = max((len(group) for group in groups), default=0)
        raise ValueError(
            f"could only fill {running} of the requested {train_size} points with whole "
            f"groups (largest group holds {largest}). Use a finer grouping -- for "
            "scaffolds, acyclic='skeleton' instead of 'pool' -- or lower min_fill_fraction "
            "to accept the shortfall."
        )

    def assemble(group_list):
        if not group_list:
            return np.zeros(0, dtype=int)
        return np.sort(np.concatenate(group_list))

    return assemble(train_groups), assemble(test_groups)


def group_kfold_members(group_ids, nkfolds, rng=None):
    """Assign whole groups to ``nkfolds`` folds, largest group to the emptiest fold.

    Returns a list of ``nkfolds`` index arrays. The greedy largest-first fill keeps the
    folds as even as group sizes allow; with one dominant group they cannot be even at
    all, which is worth knowing before reading a cross-validation number built on them.
    """
    if nkfolds < 2:
        raise ValueError("nkfolds must be at least 2")
    groups = group_index_lists(group_ids)
    if len(groups) < nkfolds:
        raise ValueError(f"{len(groups)} groups cannot fill {nkfolds} folds; use a finer grouping")
    if rng is not None:
        groups = [groups[i] for i in rng.permutation(len(groups))]
    # Sort AFTER shuffling: size decides placement, and the shuffle only breaks ties, so
    # the fill is balanced while still differing between seeds.
    groups.sort(key=lambda g: -len(g))

    sizes = np.zeros(nkfolds, dtype=int)
    members = [[] for _ in range(nkfolds)]
    for group in groups:
        target = int(np.argmin(sizes))
        members[target].extend(group.tolist())
        sizes[target] += len(group)
    return [np.array(sorted(m), dtype=int) for m in members]


class GroupKFolds(KFolds):
    def __init__(self, group_ids, nkfolds, rng=None):
        """K-folds whose validation blocks are whole groups.

        Presents the same ``train_test_indices(kfold_id)`` interface as :class:`KFolds`,
        so it drops into the same loss classes. Unlike ``KFolds``, whose folds are
        independent shuffles at a FIXED separator, these folds PARTITION the set: every
        point is in exactly one validation block. That is what the smoother-matrix k-fold
        losses require, and it also means the separator differs between folds, since
        groups do not divide evenly.
        """
        fold_members = group_kfold_members(group_ids, nkfolds, rng=rng)
        self.nkfolds = nkfolds
        self.group_ids = np.asarray(group_ids)
        self.total_set_size = self.group_ids.shape[0]
        self.fold_members = fold_members

        grouped = np.concatenate(fold_members)
        self.all_shuffled_indices = np.empty((nkfolds, grouped.shape[0]), dtype=int)
        self.all_training_test_set_separators = np.empty(nkfolds, dtype=int)
        for kfold_id, test_indices in enumerate(fold_members):
            mask = np.ones(self.total_set_size, dtype=bool)
            mask[test_indices] = False
            mask[self.group_ids < 0] = False
            train_indices = np.flatnonzero(mask)
            self.all_shuffled_indices[kfold_id, : train_indices.shape[0]] = train_indices
            self.all_shuffled_indices[kfold_id, train_indices.shape[0] :] = test_indices
            self.all_training_test_set_separators[kfold_id] = train_indices.shape[0]
        self.training_set_size = None

    def fold_sizes(self):
        """Validation-block size per fold; uneven folds are the thing to check for."""
        return np.array([len(m) for m in self.fold_members], dtype=int)


def leave_one_group_out_folds(group_ids):
    """One fold per group: the group is the validation block, everything else trains.

    The group analogue of leave-one-out, and the criterion the SORF group losses
    implement. With every group a singleton this reduces exactly to leave-one-out.
    """
    return GroupKFolds(group_ids, len(group_index_lists(group_ids)))


def build_kfolds(
    total_set_size,
    kfolds=None,
    nkfolds=8,
    training_set_ratio=0.5,
    training_set_size=None,
    group_ids=None,
    rng=None,
    leave_one_group_out=False,
):
    """Pass a ready-made folds object through, or construct a plain :class:`KFolds`.

    Lets every loss class accept folds either as an object -- a :class:`GroupKFolds` when
    the split has to respect scaffolds -- or as the numbers one would build a plain
    ``KFolds`` from.

    ``rng`` reaches BOTH branches -- it used to be forwarded only to GroupKFolds, so
    passing it on the plain path silently did nothing.

    ``leave_one_group_out=True`` selects one fold per group via
    :func:`leave_one_group_out_folds`, which is otherwise only reachable by computing the
    group count yourself and passing it as ``nkfolds``. ``nkfolds`` and ``rng`` are then
    both immaterial: the fold count IS the group count, and with one group per fold the rng
    can only permute the order of folds, which no aggregation over them depends on.

    NOTE: the smoother-matrix k-fold losses require each fold's training and validation
    sets to PARTITION the full set. ``KFolds.train_test_indices`` gives that (it splits a
    single shuffle at a separator), and so does ``GroupKFolds``. A folds object that
    resampled with replacement would not, and would silently produce wrong errors.
    """
    if leave_one_group_out:
        if kfolds is not None:
            raise ValueError(
                "leave_one_group_out=True builds the folds itself; passing kfolds as well "
                "is ambiguous about which split was intended"
            )
        if group_ids is None:
            raise ValueError("leave_one_group_out=True requires group_ids")
        return leave_one_group_out_folds(group_ids)
    if kfolds is not None:
        return kfolds
    if group_ids is not None:
        # training_set_ratio/training_set_size are deliberately ignored here: group folds
        # partition the set into nkfolds validation blocks, so the training fraction is
        # 1 - 1/nkfolds by construction and cannot be dialled independently.
        return GroupKFolds(group_ids, nkfolds, rng=rng)
    return KFolds(
        total_set_size,
        nkfolds,
        training_set_ratio=training_set_ratio,
        training_set_size=training_set_size,
        rng=rng,
    )


# ---------------------------------------------------------------------------
# Force models: one conformer owns many observables.
#
# A force training set is not one row per molecule. Each conformer contributes an energy
# and three force components per atom, and those observables have to move between train
# and validation TOGETHER -- a model fitted to some of a conformer's forces and validated
# on the rest is not predicting anything out of sample, it is interpolating within a
# single geometry.
#
# So the conformer is the group, and these expand a conformer-level split onto whatever
# each conformer owns. Grouping the CONFORMERS themselves -- by scaffold, say, so every
# conformer of a molecule moves together -- needs nothing new here: pass the conformer
# group labels to the item-level GroupKFolds and expand that instead.
# ---------------------------------------------------------------------------


def observable_ranges(observables_per_item):
    """Start offsets of each item's consecutive block of observables, plus the total."""
    counts = np.asarray(observables_per_item, dtype=int)
    ranges = np.zeros(counts.shape[0] + 1, dtype=int)
    np.cumsum(counts, out=ranges[1:])
    return ranges


def observable_group_ids(observables_per_item, item_group_ids=None):
    """Group label per OBSERVABLE, so an item's observables never split across folds.

    ``item_group_ids`` groups the items themselves -- by scaffold, for instance -- and
    defaults to one group per item, which is the conformer-level grouping force models
    need.
    """
    ranges = observable_ranges(observables_per_item)
    if item_group_ids is None:
        item_group_ids = np.arange(ranges.shape[0] - 1)
    item_group_ids = np.asarray(item_group_ids)
    group_ids = np.empty(ranges[-1], dtype=int)
    for item_id in range(ranges.shape[0] - 1):
        group_ids[ranges[item_id] : ranges[item_id + 1]] = item_group_ids[item_id]
    return group_ids


def conformer_folds(nconformers, nkfolds, conformer_group_ids=None, rng=None):
    """Conformer-level folds: one group per conformer unless told otherwise."""
    if conformer_group_ids is None:
        conformer_group_ids = np.arange(nconformers)
    return GroupKFolds(conformer_group_ids, nkfolds, rng=rng)


class GroupKFoldsMultipleObservables(KFolds):
    def __init__(self, observables_per_item, item_folds):
        """Item-level folds expanded onto the observables each item owns.

        Takes an EXISTING item-level folds object rather than building its own, and that
        is the whole point. OQML needs two expansions of one split -- the observables index
        the rows of its matrix, the atoms index the columns -- and they have to agree on
        which conformer went into which fold. Built separately they would not: the greedy
        fill balances on block size, and a conformer's observable block (3*natoms + 1) and
        its atom block (natoms) are different sizes, so the two would drift apart and the
        model would be validated on rows it had partly trained on.
        """
        ranges = observable_ranges(observables_per_item)
        self.nkfolds = item_folds.nkfolds
        self.total_set_size = int(ranges[-1])
        self.item_folds = item_folds
        self.observable_ranges = ranges

        self.fold_members = []
        for fold_id in range(self.nkfolds):
            _, test_items = item_folds.train_test_indices(fold_id)
            members = [np.arange(ranges[item_id], ranges[item_id + 1]) for item_id in test_items]
            self.fold_members.append(
                np.sort(np.concatenate(members)) if members else np.zeros(0, dtype=int)
            )

        self.all_shuffled_indices = np.empty((self.nkfolds, self.total_set_size), dtype=int)
        self.all_training_test_set_separators = np.empty(self.nkfolds, dtype=int)
        for fold_id, test_indices in enumerate(self.fold_members):
            mask = np.ones(self.total_set_size, dtype=bool)
            mask[test_indices] = False
            train_indices = np.flatnonzero(mask)
            self.all_shuffled_indices[fold_id, : train_indices.shape[0]] = train_indices
            self.all_shuffled_indices[fold_id, train_indices.shape[0] :] = test_indices
            self.all_training_test_set_separators[fold_id] = train_indices.shape[0]
        self.training_set_size = None

    def fold_sizes(self):
        return np.array([len(m) for m in self.fold_members], dtype=int)
