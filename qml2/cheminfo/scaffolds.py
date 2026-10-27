"""Grouping molecules by Bemis-Murcko scaffold.

The point of a scaffold group is to be held out WHOLE. A random split lets a molecule
sit in the test set while a near-twin of it sits in training, so the learning curve
measures interpolation between neighbours; holding out entire scaffolds instead measures
generalization to chemotypes the model has never seen, which is what the FreeSolv
literature (MoleculeNet and successors) reports.

This module is the ONLY place in qml2 that knows what a scaffold is. Everything
downstream -- k-folds, leave-one-group-out losses, learning curves -- consumes the plain
integer array returned by :func:`scaffold_group_ids` and calls it a "group", so none of
it depends on RDKit or on chemistry at all. Any other grouping (by formula, by provenance,
by assay) can be fed to the same machinery.

One parameter controls how coarse the grouping is, plus RDKit's standard chirality flag:

``include_chirality``
    Passed straight through to RDKit's canonical SMILES, as everywhere else. ``False``
    (default) makes stereoisomers share a scaffold, which is the usual convention.

``acyclic``
    What to do with molecules that have no rings, whose Murcko scaffold is EMPTY. This is
    not an edge case: roughly a third of FreeSolv is acyclic, so the choice moves the
    split substantially.

    * ``"skeleton"`` (default) -- group by the generic carbon skeleton of the whole
      molecule. Keeps the grouping scaffold-like while leaving it splittable, which is
      what the datasets here need. It deviates from the benchmark convention, so say so
      in text.
    * ``"pool"`` -- one group for all of them. The MoleculeNet/DeepChem convention, and
      the right choice when reproducing those numbers, but it produces a single
      unsplittable group: on FreeSolv that group holds half the dataset, which both
      distorts the target split fractions and makes a randomized-tie split degenerate
      (every seed returns the same split).
    * ``"singleton"`` -- every acyclic molecule alone in its own group. Finest grain;
      makes the acyclic part behave like a random split.

RDKit is an optional dependency of qml2 (it arrives with the ``morfeus`` extra), so the
import is guarded the same way ``qml2.ensemble`` guards it: importing this module always
works, and the failure surfaces only when a scaffold is actually requested.
"""

from collections import defaultdict

import numpy as np

from ..basic_utils import ExceptionRaisingClass

try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Scaffolds import MurckoScaffold

    # RDKit is chatty about sanitization on the kind of SMILES these datasets carry.
    RDLogger.DisableLog("rdApp.*")
except ModuleNotFoundError as ex:
    Chem = ExceptionRaisingClass(ex, add_attrs=["MolFromSmiles", "MolToSmiles"])
    MurckoScaffold = ExceptionRaisingClass(
        ex, add_attrs=["GetScaffoldForMol", "MakeScaffoldGeneric"]
    )

# Prefix marking a key as belonging to an acyclic molecule, so an acyclic "skeleton" key
# can never collide with a real scaffold's SMILES.
ACYCLIC_PREFIX = "__acyclic__"

ACYCLIC_POLICIES = ("skeleton", "pool", "singleton")

UNPARSABLE_POLICIES = ("error", "singleton", "skip")


def _generic_smiles(mol, include_chirality=False):
    """Canonical SMILES of the generic framework: every atom carbon, every bond single.

    ``MakeScaffoldGeneric`` raises on molecules whose generic form fails RDKit's valence
    check (it happily makes a pentavalent carbon out of a pentavalent phosphorus). Falling
    back on the atom-typed SMILES keeps such a molecule in a group of its own rather than
    losing it, which is the conservative failure: it can only split a group too finely,
    never merge two things that should stay apart.
    """
    try:
        return Chem.MolToSmiles(
            MurckoScaffold.MakeScaffoldGeneric(mol), isomericSmiles=include_chirality
        )
    except Exception:
        return Chem.MolToSmiles(mol, isomericSmiles=include_chirality)


def scaffold_key(smiles, acyclic="skeleton", include_chirality=False):
    """The group key for one molecule, or ``None`` if the SMILES will not parse.

    Two molecules share a group exactly when this returns the same string for both, so the
    key is the whole definition of the grouping. See the module docstring for the two
    parameters.
    """
    if acyclic not in ACYCLIC_POLICIES:
        raise ValueError(f"acyclic must be one of {ACYCLIC_POLICIES}, not {acyclic!r}")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    scaffold_smiles = Chem.MolToSmiles(scaffold, isomericSmiles=include_chirality)

    if scaffold_smiles != "":
        return scaffold_smiles

    # No rings: the Murcko scaffold is empty and `acyclic` decides the grain.
    if acyclic == "pool":
        return ACYCLIC_PREFIX
    if acyclic == "singleton":
        # The molecule's own canonical SMILES, so every distinct acyclic molecule is alone.
        return ACYCLIC_PREFIX + "::" + Chem.MolToSmiles(mol, isomericSmiles=include_chirality)
    # "skeleton": the generic graph of the whole molecule -- every atom carbon, every bond
    # single -- so acyclic molecules differing only in heteroatom placement share a group.
    return ACYCLIC_PREFIX + "::" + _generic_smiles(mol, include_chirality=include_chirality)


def scaffold_keys(smiles_list, **kwargs):
    """``scaffold_key`` over a list, preserving order."""
    return [scaffold_key(smiles, **kwargs) for smiles in smiles_list]


def scaffold_groups(smiles_list, **kwargs):
    """Map each scaffold key to the array of molecule indices carrying it.

    Unparsable SMILES collect under the key ``None``. Returned as a plain dict; use
    :func:`scaffold_group_ids` for the form the rest of qml2 consumes.
    """
    groups = defaultdict(list)
    for index, key in enumerate(scaffold_keys(smiles_list, **kwargs)):
        groups[key].append(index)
    return {key: np.array(indices, dtype=int) for key, indices in groups.items()}


def scaffold_group_ids(smiles_list, on_unparsable="error", **kwargs):
    """Integer group label per molecule -- the handoff to the rest of qml2.

    Returns an array of length ``len(smiles_list)`` whose entries are group indices in
    ``0..ngroups-1``, and which is all any downstream grouping code needs.

    Labels are assigned in order of first appearance rather than by group size, so the
    numbering depends only on the input order and not on the parameters; a molecule's
    label is therefore comparable across two calls only if the SMILES list is the same.

    ``on_unparsable`` : what to do with SMILES RDKit rejects.
        * ``"error"`` (default) -- raise, naming the offending indices. Silently grouping
          molecules that failed to parse is how a broken dataset becomes a plausible
          learning curve.
        * ``"singleton"`` -- give each one its own group and keep it in the set.
        * ``"skip"`` -- label them ``-1``; callers must drop them.
    """
    if on_unparsable not in UNPARSABLE_POLICIES:
        raise ValueError(f"on_unparsable must be one of {UNPARSABLE_POLICIES}")

    keys = scaffold_keys(smiles_list, **kwargs)
    unparsable = [index for index, key in enumerate(keys) if key is None]
    if unparsable and on_unparsable == "error":
        raise ValueError(
            f"{len(unparsable)} of {len(keys)} SMILES could not be parsed (indices "
            f"{unparsable[:5]}{' ...' if len(unparsable) > 5 else ''}). Pass "
            "on_unparsable='singleton' to keep them as one-molecule groups, or 'skip' to "
            "label them -1 and drop them."
        )

    group_ids = np.empty(len(keys), dtype=int)
    next_id = 0
    seen = {}
    for index, key in enumerate(keys):
        if key is None:
            if on_unparsable == "skip":
                group_ids[index] = -1
                continue
            # "singleton": a key that cannot collide with any scaffold SMILES.
            key = f"__unparsable__::{index}"
        if key not in seen:
            seen[key] = next_id
            next_id += 1
        group_ids[index] = seen[key]
    return group_ids


def group_sizes(group_ids):
    """Size of each group, largest first. The histogram input."""
    valid = np.asarray(group_ids)
    valid = valid[valid >= 0]
    if valid.shape[0] == 0:
        return np.zeros(0, dtype=int)
    return np.sort(np.bincount(valid))[::-1]


def scaffold_distribution(smiles_list, on_unparsable="singleton", **kwargs):
    """Everything needed to describe one parameter choice, in one dict.

    ``on_unparsable`` defaults to ``"singleton"`` here rather than ``"error"``: this is the
    surveying entry point, and a survey that raises on a dataset it was asked to describe
    is useless. ``n_unparsable`` reports how many there were.
    """
    keys = scaffold_keys(smiles_list, **kwargs)
    n_unparsable = sum(key is None for key in keys)
    ids = scaffold_group_ids(smiles_list, on_unparsable=on_unparsable, **kwargs)
    sizes = group_sizes(ids)
    n_molecules = len(smiles_list)
    n_acyclic = sum(key is not None and key.startswith(ACYCLIC_PREFIX) for key in keys)
    return {
        "n_molecules": n_molecules,
        "n_groups": int(sizes.shape[0]),
        "n_unparsable": n_unparsable,
        "n_acyclic": n_acyclic,
        "sizes": sizes,
        "largest_group": int(sizes[0]) if sizes.shape[0] else 0,
        # The fraction in the biggest group is the number that decides whether a split is
        # feasible at all: it is a hard floor on the size of whichever side receives it.
        "largest_group_fraction": (float(sizes[0]) / n_molecules if sizes.shape[0] else 0.0),
        "n_singletons": int(np.sum(sizes == 1)),
        "mean_group_size": (float(np.mean(sizes)) if sizes.shape[0] else 0.0),
        "median_group_size": (float(np.median(sizes)) if sizes.shape[0] else 0.0),
    }
