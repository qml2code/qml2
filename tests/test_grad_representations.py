from conftest import (
    add_checksum_to_dict,
    compare_or_create,
    int2rng,
    nullify_uninitialized_grad,
    perturbed_xyz_examples,
)

from qml2 import Compound, CompoundList
from qml2.jit_interfaces import concatenate_

ntest_mols = 400


def checksums_grad_representation(rep_func_name, rng, rep_kwargs={}):
    xyzs = perturbed_xyz_examples(rng, ntest_mols)

    compound_list = CompoundList([Compound(xyz=xyz) for xyz in xyzs])
    call = getattr(compound_list, rep_func_name)
    call(test_mode=True, gradients=True, **rep_kwargs)
    all_representations = compound_list.all_representations()
    merged_reps = concatenate_(all_representations)

    checksums_representation = {}
    add_checksum_to_dict(
        checksums_representation, rep_func_name, merged_reps, rng, nstack_checksums=8, stacks=32
    )
    del merged_reps

    checksums_grad = {}
    all_grads = []
    all_relevant_atom_ids = []
    all_relevant_atom_nums = []
    for comp in compound_list:
        grads = comp.grad_representation
        relevant_atom_ids = comp.grad_relevant_atom_ids
        relevant_atom_nums = comp.grad_relevant_atom_nums
        nullify_uninitialized_grad(grads, relevant_atom_ids, relevant_atom_nums)
        all_grads.append(grads)
        all_relevant_atom_ids.append(relevant_atom_ids)
        all_relevant_atom_nums.append(relevant_atom_nums)

    add_checksum_to_dict(
        checksums_grad, "grad_" + rep_func_name, all_grads, rng, nstack_checksums=8, stacks=32
    )
    add_checksum_to_dict(
        checksums_grad,
        "relevant_atom_ids_" + rep_func_name,
        all_relevant_atom_ids,
        rng,
        nstack_checksums=8,
        stacks=32,
    )
    add_checksum_to_dict(
        checksums_grad,
        "relevant_atom_nums_" + rep_func_name,
        all_relevant_atom_nums,
        rng,
        nstack_checksums=8,
        stacks=32,
    )

    return checksums_representation, checksums_grad


def run_grad_representation_test(rep_name, **kwargs):
    rep_func_name = "generate_" + rep_name
    checksums_rng = int2rng(1)
    checksums_representation, checksums_grad = checksums_grad_representation(
        rep_func_name, checksums_rng, **kwargs
    )
    # for representation compare for the same checksums as generated for representation tests.
    compare_or_create(checksums_representation, "short_" + rep_name, max_rel_difference=1.0e-10)
    compare_or_create(checksums_grad, "grad_" + rep_name, max_difference=1.0e-8)


def test_grad_fchl19():
    run_grad_representation_test("fchl19", rep_kwargs={"rcut": 8.0, "acut": 12.0})


if __name__ == "__main__":
    test_grad_fchl19()
