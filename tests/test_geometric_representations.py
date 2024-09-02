# TODO K.Karan.: Need to check that the test set includes molecules larger in furthest atom distance than
# cutoff distance for local representations. Use LJ random configurations?
from conftest import add_checksum_to_dict, compare_or_create, int2rng, perturbed_xyz_examples

from qml2 import Compound, CompoundList
from qml2.jit_interfaces import array_, concatenate_

ntest_mols = 2000


def checksums_single_representation(
    rep_func_name, rng, checksums_storage, local=False, rep_kwargs={}
):
    xyzs = perturbed_xyz_examples(rng, ntest_mols)
    compound_list = CompoundList([Compound(xyz=xyz) for xyz in xyzs])
    call = getattr(compound_list, rep_func_name)
    call(test_mode=True, **rep_kwargs)
    all_representations = compound_list.all_representations()
    if local:
        merged_reps = concatenate_(all_representations)
    else:
        merged_reps = array_(all_representations)

    add_checksum_to_dict(
        checksums_storage, rep_func_name, merged_reps, rng, nstack_checksums=8, stacks=32
    )


def checksums_bob(rng, checksums_storage, **kwargs):
    from qml2.representations import compute_ncm, get_bob_bags

    xyzs = perturbed_xyz_examples(rng, ntest_mols)
    compound_list = CompoundList([Compound(xyz=xyz) for xyz in xyzs])
    all_nuclear_charges = compound_list.all_nuclear_charges()
    bags = get_bob_bags(all_nuclear_charges)
    ncm = compute_ncm(bags)

    compound_list.generate_bob(bags, ncm=ncm, test_mode=True)

    merged_reps = array_(compound_list.all_representations())

    add_checksum_to_dict(checksums_storage, "bob", merged_reps, rng, nstack_checksums=8, stacks=32)


def run_single_representation_test(rep_name, **kwargs):
    rep_func_name = "generate_" + rep_name
    checksums_storage = {}
    checksums_rng = int2rng(1)
    if rep_name == "bob":
        checksums_bob(checksums_rng, checksums_storage, **kwargs)
    else:
        checksums_single_representation(rep_func_name, checksums_rng, checksums_storage, **kwargs)

    compare_or_create(checksums_storage, rep_name, max_rel_difference=1.0e-10)


def test_coulomb_matrix():
    run_single_representation_test("coulomb_matrix")


def test_fchl19():
    run_single_representation_test("fchl19", local=True, rep_kwargs={"rcut": 8.0, "acut": 12.0})


def test_bob():
    run_single_representation_test("bob")


if __name__ == "__main__":
    test_coulomb_matrix()
    test_fchl19()
    test_bob()
