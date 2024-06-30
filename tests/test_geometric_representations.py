# TODO K.Karan.: Need to check that the test set includes molecules larger in furthest atom distance than
# cutoff distance for local representations. Use LJ random configurations?
import importlib
import random

import numpy as np
from conftest import add_checksum_to_dict, compare_or_create, perturbed_xyz_examples

from qml2.utils import read_xyz_file

interface_submodule = importlib.import_module("qml2.representations")
imported_representations = interface_submodule.__dict__

ntest_mols = 2000


def run_single_representation_test(rep_func_name, rng, checksums_storage, local=False):
    rep_func = imported_representations[rep_func_name]
    xyzs = perturbed_xyz_examples(rng, ntest_mols)
    all_representations = []
    for xyz in xyzs:
        nuclear_charges, _, coordinates, _ = read_xyz_file(xyz)
        all_representations.append(rep_func(nuclear_charges, coordinates))
    if local:
        merged_reps = np.concatenate(all_representations)
    else:
        merged_reps = np.array(all_representations)

    add_checksum_to_dict(
        checksums_storage, rep_func_name, merged_reps, rng, nstack_checksums=8, stacks=32
    )


def test_representations():
    test_name = "representations"
    checksums_storage = {}
    checksums_rng = random.Random(1)
    global_rep_names = ["generate_coulomb_matrix"]
    local_rep_names = ["generate_fchl19"]
    for rep_name in global_rep_names:
        run_single_representation_test(rep_name, checksums_rng, checksums_storage, local=False)
    for rep_name in local_rep_names:
        run_single_representation_test(rep_name, checksums_rng, checksums_storage, local=True)
    compare_or_create(checksums_storage, test_name, max_rel_difference=1.0e-10)


if __name__ == "__main__":
    test_representations()
