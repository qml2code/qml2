# K.Karan.: Same test as test_FJK.py, but performed using gpu4pyscf https://github.com/pyscf/gpu4pyscf
import pytest
from test_FJK import run_all_FJK_pair_tests


def test_FJK_GPU(use_Huckel=False):
    _ = pytest.importorskip("gpu4pyscf")
    run_all_FJK_pair_tests(use_Huckel=use_Huckel, use_gpu=True)


def test_FJK_GPU_Huckel():
    test_FJK_GPU(use_Huckel=True)


if __name__ == "__main__":
    test_FJK_GPU()
    test_FJK_GPU_Huckel()
