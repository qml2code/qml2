# TODO K.Karan.: Need to check that the test set includes molecules larger in furthest atom distance than
# cutoff distance for local representations. Use LJ random configurations?

from conftest import add_checksum_to_dict, compare_or_create, int2rng, perturbed_xyz_examples

from qml2 import Compound, CompoundList
from qml2.jit_interfaces import concatenate_
from qml2.representations import get_asize, get_convolutions

ntest_mols = 2000


def test_cMBDF():
    rng = int2rng(2)

    xyzs = perturbed_xyz_examples(rng, ntest_mols)
    compounds = CompoundList([Compound(xyz=xyz) for xyz in xyzs])
    all_nuclear_charges = compounds.all_nuclear_charges()
    convolutions = get_convolutions()
    asize = get_asize(all_nuclear_charges)
    compounds.generate_cmbdf(convolutions, asize=asize, test_mode=True)
    merged_reps = concatenate_(compounds.all_representations())

    checksums = {}

    add_checksum_to_dict(checksums, "cMBDF", merged_reps, rng, nstack_checksums=8, stacks=32)
    compare_or_create(checksums, "cMBDF", max_rel_difference=1.0e-10)


if __name__ == "__main__":
    test_cMBDF()
