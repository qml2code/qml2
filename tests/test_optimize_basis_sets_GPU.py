# Test QML2's implementation of basis set optimization procedures.

import pytest
from test_optimize_basis_sets import run_optimize_basis_sets_test


def test_optimize_basis_sets_GPU():
    _ = pytest.importorskip("gpu4pyscf")
    run_optimize_basis_sets_test(use_gpu=True)


if __name__ == "__main__":
    test_optimize_basis_sets_GPU()
