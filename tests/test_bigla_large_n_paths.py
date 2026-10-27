# The large-N branches of qml2.math: above SCIPY_MAX_N, scipy's LP64 LAPACK overflows, so
# gram_eigh and solve_w_reg_factor fall through to bigla (ILP64, memory-frugal) and then to
# numpy (ILP64, correct but ~2 N^2 eigh workspace and a one-shot LU).
#
# The branch only triggers past N ~ 45000, which is far too large to test, so SCIPY_MAX_N is
# monkeypatched down. What matters is that all three rungs agree NUMERICALLY -- most of all in
# eigenvector orientation, which is the failure a reconstruction test cannot see: Q diag(w) Q.T
# is symmetric in a transposition mistake, so a wrongly-oriented Q still reconstructs A.
import numpy as np
import pytest

from qml2 import math as qmath

bigla = pytest.importorskip("bigla", reason="bigla is an optional dependency")
if not bigla.is_ilp64():  # importable but no ILP64 backend -> the paths under test are skipped
    pytest.skip("bigla resolved no ILP64 backend", allow_module_level=True)

N = 64
TOL = 1e-9


def _sym(n=N, seed=0):
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((n, n))
    return M + M.T


def _spd(n=N, seed=1):
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((n, n))
    return M @ M.T + n * np.eye(n)


def _align_columns(Q, ref):
    """Eigenvector sign is arbitrary; fix it per column before comparing."""
    signs = np.sign(np.einsum("ij,ij->j", Q, ref))
    signs[signs == 0] = 1
    return Q * signs


def test_gram_eigh_bigla_path_matches_scipy_path(monkeypatch):
    """The rung swap must be invisible to callers -- including the orientation of Q."""
    K = _sym()
    monkeypatch.setattr(qmath, "SCIPY_MAX_N", 10**9)  # force the scipy branch
    w_scipy, Q_scipy = qmath.gram_eigh(K.copy())
    monkeypatch.setattr(qmath, "SCIPY_MAX_N", 1)  # force the large-N branch
    w_bigla, Q_bigla = qmath.gram_eigh(K.copy())

    np.testing.assert_allclose(w_bigla, w_scipy, rtol=1e-11)
    np.testing.assert_allclose(_align_columns(Q_bigla, Q_scipy), Q_scipy, atol=1e-8)


def test_gram_eigh_bigla_path_matches_numpy_fallback(monkeypatch):
    """And the third rung agrees with the second, so a machine without bigla is consistent."""
    K = _sym()
    monkeypatch.setattr(qmath, "SCIPY_MAX_N", 1)
    w_bigla, Q_bigla = qmath.gram_eigh(K.copy())
    monkeypatch.setattr(qmath, "_bigla_if_usable", lambda: None)  # simulate bigla absent
    w_numpy, Q_numpy = qmath.gram_eigh(K.copy())

    np.testing.assert_allclose(w_bigla, w_numpy, rtol=1e-11)
    np.testing.assert_allclose(_align_columns(Q_bigla, Q_numpy), Q_numpy, atol=1e-8)


def test_gram_eigh_bigla_path_reconstructs(monkeypatch):
    """Column convention, stated independently of any reference implementation."""
    K = _sym()
    monkeypatch.setattr(qmath, "SCIPY_MAX_N", 1)
    w, Q = qmath.gram_eigh(K.copy())
    np.testing.assert_allclose(Q @ (w[:, None] * Q.T), K, atol=1e-8)


def test_cho_solve_bigla_path_matches_scipy_path(monkeypatch):
    """solve_w_reg_factor: the bigla Cholesky rung must match the scipy one, regulariser
    included."""
    A, B = _spd(), np.random.default_rng(2).standard_normal((N, 3))
    monkeypatch.setattr(qmath, "SCIPY_MAX_N", 10**9)
    x_scipy = qmath.cho_solve_from_matrix(A.copy(), B.copy(), l2reg=1e-6)
    monkeypatch.setattr(qmath, "SCIPY_MAX_N", 1)
    with pytest.warns(UserWarning, match="bigla"):
        x_bigla = qmath.cho_solve_from_matrix(A.copy(), B.copy(), l2reg=1e-6)
    np.testing.assert_allclose(x_bigla, x_scipy, atol=1e-7)


def test_cho_solve_falls_back_to_numpy_without_bigla(monkeypatch):
    """With bigla unusable the numpy rung still runs, and the warning names it."""
    A, B = _spd(), np.random.default_rng(3).standard_normal((N, 2))
    monkeypatch.setattr(qmath, "SCIPY_MAX_N", 10**9)
    x_scipy = qmath.cho_solve_from_matrix(A.copy(), B.copy(), l2reg=1e-6)
    monkeypatch.setattr(qmath, "SCIPY_MAX_N", 1)
    monkeypatch.setattr(qmath, "_bigla_if_usable", lambda: None)
    with pytest.warns(UserWarning, match="np.linalg.solve"):
        x_numpy = qmath.cho_solve_from_matrix(A.copy(), B.copy(), l2reg=1e-6)
    np.testing.assert_allclose(x_numpy, x_scipy, atol=1e-7)


def test_input_not_destroyed_when_overwrite_not_requested(monkeypatch):
    """bigla defaults to overwrite_a=False like scipy; the large-N path must not quietly
    destroy a caller's matrix just because it changed backend."""
    K = _sym()
    K_ref = K.copy()
    monkeypatch.setattr(qmath, "SCIPY_MAX_N", 1)
    qmath.gram_eigh(K, overwrite_a=False)
    np.testing.assert_array_equal(K, K_ref)


# The branches below did not exist when the cases above were written. Each pins a behaviour that
# would otherwise regress silently, because in every case the WRONG answer is still a plausible one.


def _general(n=N, seed=4):
    """Non-symmetric, well-conditioned: has no Cholesky factor at all."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, n)) + n * np.eye(n)


def test_lu_request_is_not_served_by_a_cholesky(monkeypatch):
    """The large-N branch must honour the factorization it was ASKED for.

    Routing an lu_solve_from_matrix call to bigla's Cholesky is a change of precondition, not of
    speed: Cholesky needs symmetric positive-definite input, and a caller reaching for LU generally
    has neither. On a good day potrf raises; on an indefinite matrix whose leading minors happen to
    be positive it returns a wrong answer quietly. Assert on the WARNING as well as the numbers --
    the numbers alone cannot distinguish "took the LU rung" from "took a Cholesky that happened to
    work".
    """
    A, B = _general(), np.random.default_rng(5).standard_normal((N, 2))
    monkeypatch.setattr(qmath, "SCIPY_MAX_N", 10**9)
    x_scipy = qmath.lu_solve_from_matrix(A.copy(), B.copy())

    monkeypatch.setattr(qmath, "SCIPY_MAX_N", 1)
    with pytest.warns(UserWarning) as record:
        x_large = qmath.lu_solve_from_matrix(A.copy(), B.copy())
    messages = " ".join(str(w.message) for w in record)
    assert "Cholesky" not in messages, f"an LU request was served by a Cholesky: {messages}"
    assert "LU" in messages, f"the warning should name the LU rung taken: {messages}"
    np.testing.assert_allclose(x_large, x_scipy, atol=1e-7)


def test_cholesky_request_still_says_cholesky(monkeypatch):
    """The other side of the gate: an SPD request must still reach bigla's Cholesky, or the gate has
    been drawn too tightly and the reusable factorisation is lost."""
    A, B = _spd(), np.random.default_rng(6).standard_normal((N, 2))
    monkeypatch.setattr(qmath, "SCIPY_MAX_N", 1)
    with pytest.warns(UserWarning, match="Cholesky"):
        qmath.cho_solve_from_matrix(A.copy(), B.copy(), l2reg=1e-6)


def test_svd_aligned_large_path_matches_scipy_path(monkeypatch):
    """svd_aligned had no large-N branch at all before -- it overflowed silently. The new branch must
    agree with the scipy one AND keep the C-contiguity its name promises."""
    rng = np.random.default_rng(7)
    A = rng.standard_normal((40, 12))
    monkeypatch.setattr(qmath, "SCIPY_MAX_ELEMENTS", 10**9)
    U_s, s_s, Vh_s = qmath.svd_aligned(A.copy())
    monkeypatch.setattr(qmath, "SCIPY_MAX_ELEMENTS", 10)
    with pytest.warns(UserWarning, match="svd_aligned"):
        U_b, s_b, Vh_b = qmath.svd_aligned(A.copy())

    np.testing.assert_allclose(s_b, s_s, rtol=1e-10)
    np.testing.assert_allclose((U_b * s_b) @ Vh_b, A, atol=1e-10)
    assert U_b.flags.c_contiguous and Vh_b.flags.c_contiguous, "contiguity contract broken"


def test_svd_aligned_guard_counts_ELEMENTS_not_one_dimension():
    """SCIPY_MAX_N is a square-matrix notion (46341 ~ sqrt(2**31)). For a rectangular matrix the
    binding quantity is the element count, so a tall-skinny array must not be refused merely for
    being tall -- hence svd_aligned guards on A.size against SCIPY_MAX_ELEMENTS."""
    assert qmath.SCIPY_MAX_ELEMENTS == 2**31 - 1
    # 100000 x 100 is 1e7 elements: far inside the budget although one dimension dwarfs SCIPY_MAX_N.
    assert 100_000 * 100 < qmath.SCIPY_MAX_ELEMENTS < 10**6 * 5_000


def test_bigla_probe_accepts_either_api(monkeypatch):
    """_bigla_if_usable must work with a bigla exposing is_ilp64() OR only backend_info().

    A bare `except Exception` around `bigla.is_ilp64()` silently disabled bigla ENTIRELY for any
    build lacking that accessor -- the integration looked live while never once running.
    """
    import sys
    import types

    def fake(has_is_ilp64, ilp64=True):
        m = types.ModuleType("bigla")
        if has_is_ilp64:
            m.is_ilp64 = lambda: ilp64
        m.backend_info = lambda: types.SimpleNamespace(ilp64=ilp64)
        return m

    for has_accessor in (True, False):
        monkeypatch.setitem(sys.modules, "bigla", fake(has_accessor))
        assert (
            qmath._bigla_if_usable() is not None
        ), f"probe failed with is_ilp64 {'present' if has_accessor else 'absent'}"
    # and an LP64 backend must still be declined through either route
    for has_accessor in (True, False):
        monkeypatch.setitem(sys.modules, "bigla", fake(has_accessor, ilp64=False))
        assert qmath._bigla_if_usable() is None
