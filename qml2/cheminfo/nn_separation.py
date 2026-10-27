"""Nearest-neighbour separation between two subsets of a dataset.

How close does a test set sit to the training set it was split from? Take each TEST
molecule's nearest neighbour in the TRAINING set and record that similarity (or distance);
the distribution of those values over the test set is the answer. A split that genuinely
separates shifts the distribution away from the training set; a split that merely sounds
principled leaves it where a random split puts it.

Nothing here knows about splits -- every function takes the two index sets and reports the
numbers. The splitters live in :mod:`qml2.models.kfolds`, and deciding what the numbers
mean is the caller's business.

Two feature spaces, because the answer should not depend on one choice of yardstick:

* **ECFP4 / Tanimoto** (:func:`ecfp4_fingerprints`, :func:`nearest_neighbour_similarities`)
  -- Morgan radius 2 (the 4 in ECFP4 is a diameter; see ECFP4_RADIUS), 2048 bits. The
  field-standard measure, and what the critique literature reports, so it is the
  comparable number. Conventions: >= 0.7 very similar (likely an analogue), >= 0.4
  similar, < 0.3 structurally distinct.
* **SLATM / l2 distance** (:func:`slatm_representations`, with
  :func:`qml2.utils.nearest_neighbour_distances` -- that one is not chemistry-specific, so
  it lives with the other l2 helpers) -- the representation the models themselves use,
  answering the "ECFP is not your model's space" objection. Reported as a raw distance:
  there is no universal threshold, and only the comparison between splits matters, so
  nothing has to be calibrated.

RDKit is an optional dependency of qml2, so its import is guarded exactly as in
:mod:`qml2.cheminfo.scaffolds`: importing this module always works and the failure
surfaces only when a fingerprint or a geometry is actually requested.
"""

import numpy as np

from ..basic_utils import ExceptionRaisingClass
from ..representations import generate_slatm, get_slatm_mbtypes

try:
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import AllChem, rdFingerprintGenerator

    RDLogger.DisableLog("rdApp.*")
except ModuleNotFoundError as ex:
    Chem = ExceptionRaisingClass(ex, add_attrs=["MolFromSmiles", "AddHs"])
    DataStructs = ExceptionRaisingClass(ex, add_attrs=["BulkTanimotoSimilarity"])
    AllChem = ExceptionRaisingClass(ex, add_attrs=["EmbedMolecule", "MMFFOptimizeMolecule"])
    rdFingerprintGenerator = ExceptionRaisingClass(ex, add_attrs=["GetMorganGenerator"])

# The number in an ECFP name is the DIAMETER of the largest atom environment, not the
# radius (Rogers & Hahn, JCIM 2010), while RDKit's generator takes a RADIUS. So ECFP4 is
# radius 2 and ECFP6 is radius 3. Checked rather than assumed: a radius-2 environment on a
# straight chain spans 4 bonds end to end, radius 3 spans 6.
ECFP4_RADIUS = 2
MORGAN_BITS = 2048

# Iteration cap for the conformer cleanup in slatm_representations. Enough to relax an
# ETKDG embedding; this is not a geometry optimization anybody should quote.
MMFF_MAX_ITERATIONS = 200


def ecfp_fingerprints(all_SMILES, radius, nbits=MORGAN_BITS):
    """Morgan/ECFP bit vectors, one per SMILES, in input order.

    ``radius`` is required and is a RADIUS, so the caller cannot accidentally read it as
    the diameter an ECFP name carries; :func:`ecfp4_fingerprints` is the named shorthand
    for the usual choice.

    Built through ``rdFingerprintGenerator``, which is the current API for what used to be
    ``GetMorganFingerprintAsBitVect``; RDKit 2025 deprecates the latter and the bits are
    identical. Raises on a SMILES RDKit cannot parse rather than returning a silently empty
    fingerprint, which would read as "similar to nothing" and quietly bias every number
    computed from it.
    """
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nbits)
    fingerprints = []
    for smiles in all_SMILES:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            raise ValueError(f"RDKit could not parse {smiles!r}")
        fingerprints.append(generator.GetFingerprint(molecule))
    return fingerprints


def ecfp4_fingerprints(all_SMILES, nbits=MORGAN_BITS):
    """ECFP4 bit vectors: :func:`ecfp_fingerprints` at radius 2.

    Exists so the call site can name the fingerprint the literature names, without a
    ``radius`` argument sitting next to a ``4`` that means something else.
    """
    return ecfp_fingerprints(all_SMILES, radius=ECFP4_RADIUS, nbits=nbits)


def slatm_representations(all_SMILES, embed_seed=1):
    """Global SLATM, one vector per SMILES, from a single embedded conformer each.

    NOTE ON GEOMETRY, because it decides what these numbers can be used for: the
    learning-curve pipeline uses morfeus conformer ENSEMBLES, and regenerating those costs
    hours. This embeds ONE ETKDG conformer per molecule and relaxes it with MMFF. That is a
    proxy for the model's geometry, adequate where the question is RELATIVE -- does one
    split put the test set further from training than another -- because every molecule and
    every split is measured on the same geometries. It is not adequate for anything that
    needs the pipeline's actual conformers.

    ``mbtypes`` are derived from the whole list, so the vectors share a basis and are
    comparable; the length therefore depends on how many elements the dataset contains.
    """
    all_charges = []
    all_coordinates = []
    for smiles in all_SMILES:
        molecule = Chem.AddHs(Chem.MolFromSmiles(smiles))
        if AllChem.EmbedMolecule(molecule, randomSeed=embed_seed) != 0:
            raise ValueError(f"RDKit could not embed a conformer for {smiles!r}")
        AllChem.MMFFOptimizeMolecule(molecule, maxIters=MMFF_MAX_ITERATIONS)
        all_charges.append(np.array([atom.GetAtomicNum() for atom in molecule.GetAtoms()]))
        all_coordinates.append(np.array(molecule.GetConformer().GetPositions()))

    mbtypes = get_slatm_mbtypes(all_charges)
    # Argument order is (nuclear_charges, coordinates, mbtypes) -- charges FIRST, as in
    # generate_coulomb_matrix. Swapping them does not raise until deep inside numba.
    return np.array(
        [
            generate_slatm(charges, coordinates, mbtypes, local=False)
            for charges, coordinates in zip(all_charges, all_coordinates)
        ]
    )


def nearest_neighbour_similarities(fingerprints, train_indices, test_indices):
    """Max Tanimoto similarity of each TEST molecule to the TRAINING set.

    Returned in the order of ``test_indices``. Higher means the test molecule has a close
    analogue in training, i.e. the split leaked.
    """
    train_fingerprints = [fingerprints[index] for index in train_indices]
    return np.array(
        [
            max(DataStructs.BulkTanimotoSimilarity(fingerprints[index], train_fingerprints))
            for index in test_indices
        ]
    )
