# TODO K.Karan.: Need to check that the test set includes molecules larger in furthest atom distance than
# cutoff distance for local representations. Use LJ random configurations?
import random

import numpy as np
from conftest import add_checksum_to_dict, compare_or_create, perturbed_xyz_examples

from qml2.representations import generate_slatm, get_slatm_mbtypes
from qml2.utils import read_xyz_file

ntest_mols = 2000


def test_SLATM():
    rng = random.Random(2)

    xyzs = perturbed_xyz_examples(rng, ntest_mols)
    all_nuclear_charges = []
    all_coordinates = []
    for xyz in xyzs:
        nuclear_charges, _, coordinates, _ = read_xyz_file(xyz)
        all_nuclear_charges.append(nuclear_charges)
        all_coordinates.append(coordinates)
    mbtypes = get_slatm_mbtypes(all_nuclear_charges)
    all_representations = []
    for nuclear_charges, coordinates in zip(all_nuclear_charges, all_coordinates):
        representation = generate_slatm(nuclear_charges, coordinates, mbtypes)
        all_representations.append(representation)
    merged_reps = np.array(all_representations)

    checksums = {}

    add_checksum_to_dict(checksums, "SLATM", merged_reps, rng, nstack_checksums=8, stacks=32)
    compare_or_create(checksums, "SLATM", max_rel_difference=1.0e-10)


if __name__ == "__main__":
    test_SLATM()
