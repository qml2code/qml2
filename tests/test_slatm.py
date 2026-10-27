# TODO K.Karan.: Need to check that the test set includes molecules larger in furthest atom distance than
# cutoff distance for local representations. Use LJ random configurations?

from conftest import add_checksum_to_dict, compare_or_create, int2rng, perturbed_xyz_examples

from qml2 import Compound, CompoundList
from qml2.jit_interfaces import array_, concatenate_
from qml2.representations import get_slatm_mbtypes

ntest_mols = 2000


def run_slatm_test_for_compounds(compounds, local=False, calculators=None):
    if calculators is None:
        all_nuclear_charges = compounds.all_nuclear_charges()
        mbtypes = get_slatm_mbtypes(all_nuclear_charges)

    checksums = {}

    if local:
        rep_name = "aSLATM"
        rep_merger = concatenate_
    else:
        rep_name = "SLATM"
        rep_merger = array_

    for alchemy in [True, False]:
        checksum_rng = int2rng(4)
        if calculators is None:
            compounds.generate_slatm(mbtypes, local=local, test_mode=True, alchemy=alchemy)
        else:
            slatm_func = calculators[alchemy]
            compounds.generate_rep_from_func(slatm_func, test_mode=True)
        merged_reps = rep_merger(compounds.all_representations())
        add_checksum_to_dict(
            checksums,
            f"alchemy_{alchemy}",
            merged_reps,
            checksum_rng,
            nstack_checksums=8,
            stacks=32,
        )
    compare_or_create(checksums, rep_name, max_rel_difference=1.0e-10)


def run_slatm_test(local=False):
    rng = int2rng(2)
    xyzs = perturbed_xyz_examples(rng, ntest_mols)
    compounds = CompoundList([Compound(xyz=xyz) for xyz in xyzs])
    run_slatm_test_for_compounds(compounds, local=local)


def test_slatm():
    run_slatm_test()


def test_aslatm():
    run_slatm_test(local=True)


if __name__ == "__main__":
    test_slatm()
    test_aslatm()
