import pytest
from test_kernels import run_global_kernels_test, run_local_kernels_test


def test_global_kernels_gpu():
    _ = pytest.importorskip("cupy")
    run_global_kernels_test(
        "qml2.cupy.global_kernels",
    )


def test_raw_global_kernels():
    _ = pytest.importorskip("cupy")
    run_global_kernels_test("qml2.cupy.raw_kernels", partial_comparison=True)


def test_raw_local_kernels():
    _ = pytest.importorskip("cupy")
    run_local_kernels_test("qml2.cupy.raw_kernels", partial_comparison=True)


if __name__ == "__main__":
    test_global_kernels_gpu()
    test_raw_global_kernels()
    test_raw_local_kernels()
