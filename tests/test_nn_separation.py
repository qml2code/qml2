"""Unit tests for qml2.cheminfo.nn_separation.

These functions are the measurement behind the scaffold-split diagnostic, so what matters
is that the numbers are right, not that the code runs. The l2 distance
routine that pairs with these lives in qml2.utils, and is tested in test_utils.py.
"""

import numpy as np
import pytest
from conftest import add_checksum_to_dict, compare_or_create, get_qm7_smiles, int2rng

from qml2.cheminfo.nn_separation import (
    ECFP4_RADIUS,
    ecfp4_fingerprints,
    ecfp_fingerprints,
    nearest_neighbour_similarities,
    slatm_representations,
)

# Small, varied, and cheap to embed: two aromatics, two acyclics, one halide. Hand-picked
# so the similarity assertions below can be reasoned about chemically.
SMILES_SAMPLE = ["c1ccccc1", "Cc1ccccc1", "CCO", "CCCCO", "ClCCl"]

# Real molecules for the checksums, from test_data rather than $DATA so the test runs on
# every machine. Kept small because the SLATM path embeds a conformer per molecule.
CHECKSUM_NMOLECULES = 100
SLATM_CHECKSUM_NMOLECULES = 40


def test_ecfp4_fingerprints_basics():
    fingerprints = ecfp4_fingerprints(SMILES_SAMPLE)
    assert len(fingerprints) == len(SMILES_SAMPLE)
    assert fingerprints[0].GetNumBits() == 2048
    # Benzene against itself is 1.0; benzene against ethanol is far from it.
    similarities = nearest_neighbour_similarities(fingerprints, np.array([0]), np.array([0, 2]))
    assert np.isclose(similarities[0], 1.0)
    assert similarities[1] < 0.3


def test_ecfp4_is_ecfp_at_radius_two():
    """The named shorthand must be exactly the general call, and the radius must matter.

    ECFP names carry the DIAMETER, so ECFP4 is radius 2 -- the reason the general function
    takes a radius and the named one does not expose it at all.
    """
    assert ECFP4_RADIUS == 2

    named = ecfp4_fingerprints(SMILES_SAMPLE)
    general = ecfp_fingerprints(SMILES_SAMPLE, radius=2)
    assert all(a == b for a, b in zip(named, general))

    # Radius 3 is ECFP6 and must give different bits, or the argument does nothing.
    wider = ecfp_fingerprints(SMILES_SAMPLE, radius=3)
    assert any(a != b for a, b in zip(named, wider))


def test_ecfp4_fingerprints_refuse_unparsable_smiles():
    """An unparsable SMILES must raise, not yield an empty fingerprint that would read as
    'similar to nothing' and bias every number computed from it."""
    with pytest.raises(ValueError, match="could not parse"):
        ecfp4_fingerprints(["c1ccccc1", "this is not a molecule"])


def test_nearest_neighbour_similarities_ordering_and_range():
    fingerprints = ecfp4_fingerprints(SMILES_SAMPLE)
    train_indices = np.array([0, 2])
    test_indices = np.array([1, 3, 4])
    similarities = nearest_neighbour_similarities(fingerprints, train_indices, test_indices)

    assert similarities.shape == test_indices.shape
    assert np.all((similarities >= 0.0) & (similarities <= 1.0))
    # Toluene is nearer to benzene than dichloromethane is to anything in training.
    assert similarities[0] > similarities[2]
    # Reordering the test indices must permute the output the same way.
    reordered = nearest_neighbour_similarities(fingerprints, train_indices, test_indices[::-1])
    assert np.allclose(reordered, similarities[::-1])


def test_ecfp4_fingerprints_checksums():
    """Pin the actual BITS, not just properties of them.

    Safe to pin across machines: ECFP4 is a pure graph algorithm -- no geometry, no RNG, no
    BLAS -- so the same SMILES give the same bits on any build. Contrast
    test_slatm_representations_checksums, which cannot promise that.
    """
    all_SMILES = get_qm7_smiles(CHECKSUM_NMOLECULES)
    fingerprints = ecfp4_fingerprints(all_SMILES)
    bits = np.array([fingerprint.ToList() for fingerprint in fingerprints], dtype=float)
    assert bits.shape == (CHECKSUM_NMOLECULES, 2048)

    checksums = {}
    add_checksum_to_dict(checksums, "ecfp4_bits", bits, int2rng(1), nstack_checksums=4, stacks=8)
    compare_or_create(checksums, "ECFP4_fingerprints")


def test_slatm_representations_checksums():
    """Pin the SLATM vectors, geometry included.

    CAVEAT, and it is the reason this is a separate test from the ECFP one: these numbers
    depend on RDKit's ETKDG embedding as well as on generate_slatm. The embedding is seeded
    (embed_seed), and that makes it bit-reproducible across processes, thread counts and
    environments sharing an RDKit build -- verified -- but nothing guarantees it across
    RDKit VERSIONS. So an RDKit upgrade may require regenerating this benchmark with
    QML2_BENCHMARK_GENERATION_ON=1, and a failure here means "the geometry moved", not
    necessarily "the representation broke". Check the ECFP checksum first: if that still
    passes, the SMILES are fine and the difference is geometric.
    """
    all_SMILES = get_qm7_smiles(SLATM_CHECKSUM_NMOLECULES)
    representations = slatm_representations(all_SMILES)
    assert representations.shape[0] == SLATM_CHECKSUM_NMOLECULES

    checksums = {}
    add_checksum_to_dict(
        checksums, "slatm", representations, int2rng(2), nstack_checksums=4, stacks=8
    )
    compare_or_create(checksums, "SLATM_from_SMILES")


def test_slatm_representations_shape_and_reproducibility():
    """One vector per molecule, on a shared basis, reproducible for a fixed embed seed."""
    representations = slatm_representations(SMILES_SAMPLE)
    assert representations.shape[0] == len(SMILES_SAMPLE)
    assert representations.ndim == 2
    assert np.all(np.isfinite(representations))

    repeated = slatm_representations(SMILES_SAMPLE)
    assert np.allclose(representations, repeated)

    # Identical molecules must land on identical vectors; distinct ones must not.
    with_duplicate = slatm_representations(SMILES_SAMPLE + ["c1ccccc1"])
    assert np.allclose(with_duplicate[0], with_duplicate[-1])
    assert not np.allclose(with_duplicate[0], with_duplicate[1])


if __name__ == "__main__":
    test_ecfp4_fingerprints_basics()
    test_ecfp4_is_ecfp_at_radius_two()
    test_ecfp4_fingerprints_refuse_unparsable_smiles()
    test_nearest_neighbour_similarities_ordering_and_range()
    test_ecfp4_fingerprints_checksums()
    test_slatm_representations_checksums()
    test_slatm_representations_shape_and_reproducibility()
