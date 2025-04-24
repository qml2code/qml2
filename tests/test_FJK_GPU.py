# K.Karan.: Same test as test_FJK.py, but performed using gpu4pyscf https://github.com/pyscf/gpu4pyscf
import pytest
from conftest import compare_or_create, int2rng
from test_FJK import run_single_FJK_pair_test


def test_FJK_GPU(use_Huckel=False):
    _ = pytest.importorskip("gpu4pyscf")
    if use_Huckel:
        test_name = "FJK_Huckel"
    else:
        test_name = "FJK"

    d = {
        "single": None,
        "HOMO": {"used_orb_type": "HOMO_removed", "calc_type": "UHF"},
        "LUMO": {"used_orb_type": "LUMO_added", "calc_type": "UHF"},
        "charge": {"charge": 1, "calc_type": "UHF"},
    }
    checksums_storage = {}
    checksums_rng = int2rng(1)
    for name, kwargs in d.items():
        run_single_FJK_pair_test(
            name, kwargs, checksums_storage, checksums_rng, use_gpu=True, use_Huckel=use_Huckel
        )
    compare_or_create(
        checksums_storage, test_name, max_rel_difference=0.1, partial_comparison=True
    )


def test_FJK_GPU_Huckel():
    test_FJK_GPU(use_Huckel=True)


if __name__ == "__main__":
    test_FJK_GPU()
    test_FJK_GPU_Huckel()
