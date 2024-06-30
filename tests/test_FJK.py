# K.Karan.: Limiting test system size and making convergence more strict to negate
# numerical noise associated with pySCF's SCF procedures. Also to decrease CPU time.
# NOTE: the numerical noise seems to manifest only when a test created on one machine is
# reproduced on another machine, when run on a single machine the test generates much more consistent values.
# This means that, in a way, FJK's predictive accuracy as reported in https://doi.org/10.1063/5.0083301 partially relied
# based on FJK models learning to account for insufficient convergence of pySCF calculations.
# Also, the test's errors between machine can be noticeably worse when smaller molecules (3 heavy atoms and less) are considered.

import random

import numpy as np
import pytest
from conftest import add_checksum_to_dict, compare_or_create, perturbed_xyz_nhatoms_interval

min_nhatoms = 4
max_nhatoms = 5


def run_single_FJK_pair_test(pair_name, pair_kwargs, checksums_storage, checksums_rng):
    _ = pytest.importorskip("pyscf")
    from qml2.orb_ml import OML_Compound, OML_Compound_list, OML_Slater_pair
    from qml2.orb_ml.kernels import gaussian_kernel, gaussian_kernel_symmetric, rep_stddevs
    from qml2.orb_ml.oml_compound import OML_pyscf_calc_params
    from qml2.orb_ml.representations import OML_rep_params

    example_xyzs = perturbed_xyz_nhatoms_interval(min_nhatoms, max_nhatoms)

    calc_params = OML_pyscf_calc_params(orb_grad_tol=1.0e-11, scf_conv_tol=1.0e-11)
    rep_params = OML_rep_params(max_angular_momentum=1)

    A_xyzs = example_xyzs[:32]
    B_xyzs = example_xyzs[32:48]

    comp_param_kwargs = {"pyscf_calc_params": calc_params}

    if pair_kwargs is None:

        def comp_func(xyz):
            return OML_Compound(xyz=xyz, **comp_param_kwargs)

    else:

        def comp_func(xyz):
            return OML_Slater_pair(
                xyz=xyz, **comp_param_kwargs, second_oml_comp_kwargs=pair_kwargs
            )

    A_compounds = OML_Compound_list([comp_func(xyz) for xyz in A_xyzs])
    B_compounds = OML_Compound_list([comp_func(xyz) for xyz in B_xyzs])

    rep_param_kwargs = {"rep_params": rep_params}
    A_compounds.generate_orb_reps(**rep_param_kwargs, fixed_num_threads=1)
    B_compounds.generate_orb_reps(**rep_param_kwargs, fixed_num_threads=1)

    # Calculate sigmas.
    sigmas = rep_stddevs(A_compounds)
    sigmas *= np.sqrt(float(len(sigmas)))
    global_sigma = 0.5

    for norm in ["l1", "l2"]:
        asym_kernel = gaussian_kernel(A_compounds, B_compounds, sigmas, global_sigma, norm=norm)
        sym_kernel = gaussian_kernel_symmetric(A_compounds, sigmas, global_sigma, norm=norm)
        add_checksum_to_dict(
            checksums_storage,
            pair_name + "_asym_kernel_" + norm,
            asym_kernel,
            checksums_rng,
            nstack_checksums=8,
            stacks=8,
        )
        add_checksum_to_dict(
            checksums_storage,
            pair_name + "_sym_kernel_" + norm,
            sym_kernel,
            checksums_rng,
            nstack_checksums=8,
            stacks=8,
        )


def test_FJK():
    test_name = "FJK"

    d = {
        "single": None,
        "HOMO": {"used_orb_type": "HOMO_removed", "calc_type": "UHF"},
        "LUMO": {"used_orb_type": "LUMO_added", "calc_type": "UHF"},
        "charge": {"charge": 1, "calc_type": "UHF"},
        "spin": {"spin": 2, "calc_type": "UHF"},
    }
    checksums_storage = {}
    checksums_rng = random.Random(1)
    for name, kwargs in d.items():
        run_single_FJK_pair_test(name, kwargs, checksums_storage, checksums_rng)
    compare_or_create(checksums_storage, test_name, max_rel_difference=5.0e-2)


if __name__ == "__main__":
    test_FJK()
