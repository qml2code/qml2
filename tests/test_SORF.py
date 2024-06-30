# Test generation of Structured Orthogonal Random Features.
import random

from conftest import (
    add_checksum_to_dict,
    compare_or_create,
    fix_reductor_signs,
    fix_reductors_signs,
)

from qml2.dimensionality_reduction import get_reductor, get_reductors_diff_species
from qml2.jit_interfaces import default_rng_, randint_, random_, seed_, zeros_
from qml2.kernels.hadamard import (
    create_SORF_matrices,
    create_SORF_matrices_diff_species,
    hadamard_kernel,
    local_hadamard_kernel,
)
from qml2.models.hyperparameter_init_guesses import vector_std

repsize = 80
npcas = 64
nfeature_stacks = 8
ntransforms = 3

nfeatures = nfeature_stacks * npcas


def run_global_SORF(checksums_storage, checksums_rng):
    nvecs = 128
    rep_vecs = random_((nvecs, repsize))
    sigma = vector_std(rep_vecs)

    sorf_rng = default_rng_(8)
    reductor = get_reductor(rep_vecs, npcas)
    fix_reductor_signs(reductor)
    biases, sorf_diags = create_SORF_matrices(nfeature_stacks, ntransforms, npcas, rng=sorf_rng)

    kernel = zeros_((nvecs, nfeatures))

    hadamard_kernel(
        rep_vecs, sorf_diags, biases, kernel, sigma, nfeature_stacks, npcas, reductor=reductor
    )

    add_checksum_to_dict(
        checksums_storage, "global_SORF", kernel, checksums_rng, stacks=8, nstack_checksums=8
    )

    other_rep_vecs = random_((nvecs, repsize // 2))
    hadamard_kernel(other_rep_vecs, sorf_diags, biases, kernel, sigma, nfeature_stacks, npcas)

    add_checksum_to_dict(
        checksums_storage,
        "global_SORF_no_red",
        kernel,
        checksums_rng,
        stacks=8,
        nstack_checksums=8,
    )


def run_local_SORF(checksums_storage, checksums_rng):
    ncomps = 64

    atom_nums = randint_(1, 16, (ncomps,))
    tot_natoms = sum(atom_nums)

    ncharges = randint_(1, 3, (tot_natoms,))

    rep_vecs = random_((tot_natoms, repsize))
    sigma = vector_std(rep_vecs)

    sorf_rng = default_rng_(4)
    reductors, sorted_elements = get_reductors_diff_species(rep_vecs, ncharges, npcas)
    fix_reductors_signs(reductors)
    all_biases, all_sorf_diags = create_SORF_matrices_diff_species(
        nfeature_stacks, sorted_elements.shape[0], ntransforms, npcas, rng=sorf_rng
    )

    kernel = zeros_((ncomps, nfeatures))

    local_hadamard_kernel(
        rep_vecs,
        ncharges,
        atom_nums,
        sorted_elements,
        all_sorf_diags,
        all_biases,
        kernel,
        sigma,
        nfeature_stacks,
        npcas,
        reductors=reductors,
    )

    add_checksum_to_dict(
        checksums_storage, "local_SORF", kernel, checksums_rng, stacks=8, nstack_checksums=8
    )

    other_rep_vecs = random_((tot_natoms, repsize // 2))
    local_hadamard_kernel(
        other_rep_vecs,
        ncharges,
        atom_nums,
        sorted_elements,
        all_sorf_diags,
        all_biases,
        kernel,
        sigma,
        nfeature_stacks,
        npcas,
    )

    add_checksum_to_dict(
        checksums_storage, "local_SORF_no_red", kernel, checksums_rng, stacks=8, nstack_checksums=8
    )


def test_SORF():
    checksums_storage = {}
    checksums_rng = random.Random(8)
    seed_(8)
    run_global_SORF(checksums_storage, checksums_rng)
    run_local_SORF(checksums_storage, checksums_rng)

    compare_or_create(checksums_storage, "sorf", max_rel_difference=1.0e-6)


if __name__ == "__main__":
    test_SORF()
