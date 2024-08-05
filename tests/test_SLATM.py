# TODO K.Karan.: Need to check that the test set includes molecules larger in furthest atom distance than
# cutoff distance for local representations. Use LJ random configurations?
import random

from conftest import add_checksum_to_dict, compare_or_create, perturbed_xyz_examples

from qml2 import Compound, CompoundList
from qml2.jit_interfaces import array_
from qml2.representations import get_slatm_mbtypes

ntest_mols = 2000


def test_SLATM():
    rng = random.Random(2)

    xyzs = perturbed_xyz_examples(rng, ntest_mols)
    compounds = CompoundList([Compound(xyz=xyz) for xyz in xyzs])
    all_nuclear_charges = compounds.all_nuclear_charges()
    mbtypes = get_slatm_mbtypes(all_nuclear_charges)
    compounds.generate_slatm(mbtypes, test_mode=True)
    merged_reps = array_(compounds.all_representations())

    checksums = {}

    add_checksum_to_dict(checksums, "SLATM", merged_reps, rng, nstack_checksums=8, stacks=32)
    compare_or_create(checksums, "SLATM", max_rel_difference=1.0e-10)


if __name__ == "__main__":
    test_SLATM()
