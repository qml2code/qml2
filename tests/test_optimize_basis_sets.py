# Test QML2's implementation of basis set optimization procedures.

import random

import numpy as np
import pytest
from conftest import add_checksum_to_dict, compare_or_create, perturbed_xyz_nhatoms_interval

max_nhatoms = 4


def all_log_rescalings(oml_comp_list):
    output = []
    for comp in oml_comp_list:
        for l in comp.ao_rescalings:
            output += list(l)
    return np.log(np.array(output))


def test_optimize_basis_sets():
    _ = pytest.importorskip("pyscf")
    from qml2.orb_ml import OML_Compound, OML_Compound_list
    from qml2.orb_ml.oml_compound import OML_pyscf_calc_params

    example_xyzs = perturbed_xyz_nhatoms_interval(0, max_nhatoms)[:16]

    calc_params = OML_pyscf_calc_params(orb_grad_tol=1.0e-11, scf_conv_tol=1.0e-11)

    example_compounds_default_rescaling = []
    example_compounds_custom_rescaling = []

    custom_rescaling = {"H": [[0]], "S": [[1, 3], [2, 4]]}
    for el in ["C", "O", "N"]:
        custom_rescaling[el] = [[0], [1, 2]]

    for xyz in example_xyzs:
        kwargs = {"xyz": xyz, "pyscf_calc_params": calc_params, "optimize_ao_rescalings": True}
        example_compounds_default_rescaling.append(OML_Compound(**kwargs))
        example_compounds_custom_rescaling.append(
            OML_Compound(**kwargs, basis_rescaled_orbitals=custom_rescaling)
        )

    example_compounds_default_rescaling = OML_Compound_list(example_compounds_default_rescaling)
    example_compounds_custom_rescaling = OML_Compound_list(example_compounds_custom_rescaling)

    example_compounds_default_rescaling.run_calcs(fixed_num_threads=1)
    example_compounds_custom_rescaling.run_calcs(fixed_num_threads=1)

    default_log_rescalings = all_log_rescalings(example_compounds_default_rescaling)
    custom_log_rescalings = all_log_rescalings(example_compounds_custom_rescaling)

    checksums_storage = {}
    checksums_rng = random.Random(4)

    add_checksum_to_dict(
        checksums_storage,
        "default_basis_resc",
        default_log_rescalings,
        checksums_rng,
        nstack_checksums=8,
        stacks=8,
    )
    add_checksum_to_dict(
        checksums_storage,
        "custom_basis_resc",
        custom_log_rescalings,
        checksums_rng,
        nstack_checksums=8,
        stacks=8,
    )

    compare_or_create(checksums_storage, "optimize_basis_sets", max_difference=5.0e-2)


if __name__ == "__main__":
    test_optimize_basis_sets()
