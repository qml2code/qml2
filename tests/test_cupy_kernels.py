import pytest
from test_kernels import run_global_kernels_test, run_local_kernels_test

_ = pytest.importorskip("cupy")


def test_global_kernels_gpu():
    run_global_kernels_test(
        "qml2.cupy.global_kernels",
    )


def test_raw_global_kernels():
    run_global_kernels_test("qml2.cupy.raw_kernels", partial_comparison=True)


def test_raw_local_kernels():
    run_local_kernels_test("qml2.cupy.raw_kernels", partial_comparison=True)


if __name__ == "__main__":
    test_global_kernels_gpu()
    test_raw_global_kernels()
    test_raw_local_kernels()
