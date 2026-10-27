# K.Karan.: Limiting test system size and making convergence more strict to negate
# numerical noise associated with pySCF's SCF procedures. Also to decrease CPU time.
# NOTE: the numerical noise seems to manifest only when a test created on one machine is
# reproduced on another machine, when run on a single machine the test generates much more consistent values.
# This means that, in a way, FJK's predictive accuracy as reported in https://doi.org/10.1063/5.0083301 partially relied
# based on FJK models learning to account for insufficient convergence of pySCF calculations.
# Also, the test's errors between machine can be noticeably worse when smaller molecules (3 heavy atoms and less) are considered.

import pytest
from conftest import compare_or_create
from test_FJK import run_single_FJK_pair_test

tblite_calc_kwargs = {
    "localization_procedure": "Pipek-Mezey",
    "software": "tblite",
    "calc_type": "GFN2-xTB",
    "oml_rep_params_kwargs": {"max_angular_momentum": 1, "fjd_coup_mats": True},
}


def test_FJK_tblite():
    pytest.importorskip("tblite")
    pytest.importorskip("pyscf")
    test_name = "FJK_tblite"
    d = {
        "single": None,
    }
    # "solvation": {"smd_solvent": "water"},
    checksums_storage = {}
    for name, kwargs in d.items():
        run_single_FJK_pair_test(
            name,
            kwargs,
            checksums_storage,
            excluded_xyzs=["0023.xyz", "0025.xyz"],
            **tblite_calc_kwargs,
        )
    compare_or_create(checksums_storage, test_name, max_rel_difference=0.1)


def test_FJK_tblite_concatenated_pairs():
    pytest.importorskip("tblite")
    pytest.importorskip("pyscf")
    test_name = "FJK_tblite_concatenated_pairs"

    d = {
        "solvation": {"smd_solvent": "water"},
    }
    # K.Karan: uncommented spin test because it is too unstable
    # "spin": {"spin": 2, "calc_type": "UHF"},
    checksums_storage = {}
    for name, kwargs in d.items():
        run_single_FJK_pair_test(
            name,
            kwargs,
            checksums_storage,
            concatenate_components=True,
            excluded_xyzs=["0023.xyz", "0025.xyz"],
            **tblite_calc_kwargs,
        )
    compare_or_create(checksums_storage, test_name, max_rel_difference=0.1)


if __name__ == "__main__":
    test_FJK_tblite()
    test_FJK_tblite_concatenated_pairs()
