"""
Everything related to getting KRR coefficients with different regressors.
"""
import warnings

import numpy as np
import scipy

from .jit_interfaces import (
    array_,
    cho_factor_,
    cho_solve_,
    copy_detached_,
    diag_indices_from_,
    dot_,
    flip_,
    float_,
    jit_,
    logical_not_,
    lstsq_,
    lu_factor_,
    lu_solve_,
    maximum_,
    mean_,
    reshape_,
    searchsorted_,
    sum_,
    svd_,
    zeros_,
)
from .utils import where2slice

# maximum N when it makes sense to switch from numpy to scipy
SCIPY_MAX_N = 45000
# The real LP64 ceiling is on the ELEMENT COUNT: LAPACK's 32-bit integers cannot address more than
# 2**31-1 entries, which is why the square-matrix limit lands at sqrt(2**31) ~ 46341 and why
# SCIPY_MAX_N is set just below it. Rectangular routines must compare against this instead, since a
# tall-skinny matrix overflows while min(m, n) is still small.
SCIPY_MAX_ELEMENTS = 2**31 - 1


# KK: Based on https://stackoverflow.com/questions/59292279/solving-linear-systems-of-equations-with-svd-decomposition
@jit_(numba_parallel=True)
def solve_from_svd(U_mat, singular_values, Vh_mat, rhs, rcond: float_):
    """Least-squares solve from an ALREADY-COMPUTED SVD, truncating at ``rcond * s[0]``.

    Takes the FACTORS, not the matrix -- the counterpart of :func:`svd_solve_from_matrix`, which
    computes the decomposition and then calls this. That split is the naming convention in this
    module: ``*_from_matrix`` takes the matrix, while ``solve_from_svd`` / :func:`solve_using_eig`
    take factors a caller already has, so the name says which without reading the signature (they
    differ in arity, so a wrong import is otherwise a silent change of meaning).

    ``rcond`` is what makes this the right route for a rank-deficient system, and the reason a
    QR-based ``gels`` cannot be substituted for it -- see :func:`lstsq_solve`.
    """
    cutoff_s = rcond * singular_values[0]
    cutoff_id = rhs.shape[0] - searchsorted_(flip_(singular_values), cutoff_s)
    rescaled_U_mat_vec_prod = dot_(U_mat.T[:cutoff_id], rhs)
    rescaled_U_mat_vec_prod /= singular_values[:cutoff_id]
    # TODO KK: can this be optimized w.r.t. memory access?
    return dot_(rescaled_U_mat_vec_prod, Vh_mat[:cutoff_id])


def svd_solve_from_matrix(mat, rhs, rcond=0.0):
    """
    Solve "mat*x=rhs" using SVD decomposition.

    TODO (bigla / large-N): no size guard here either, and this is the caller most likely to meet a
    genuinely large matrix -- its only consumer is ``models/krr_forces.py`` (GPRForceModel), where
    the kernel carries one energy plus ``3*natoms`` force rows per conformer. The fix is the same
    one
    described in :func:`lstsq_solve`: route the decomposition through :func:`svd_aligned` (already
    bigla-backed and size-aware) instead of the bare ``svd_``, keeping :func:`solve_from_svd` and
    so
    ``rcond`` semantics intact. Deferred with the force-model work.
    """
    U_mat, singular_values, Vh_mat = svd_(mat)
    return solve_from_svd(U_mat, singular_values, Vh_mat, rhs, rcond)


@jit_(numba_parallel=True)
def solve_using_eig(eigenvalues, eigenvectors, rhs):
    rhs_eigenvectors = dot_(eigenvectors.T, rhs)
    rhs_eigenvectors /= eigenvalues
    return dot_(rhs_eigenvectors, eigenvectors.T)


def get_factorization_w_l2reg(
    A, factorization, l2reg=None, l2reg_diag_ratio=None, overwrite_a=False
):
    regularize = (l2reg is not None) or (l2reg_diag_ratio is not None)
    if regularize:
        diag_indices = diag_indices_from_(A)
        # Detached copy of the original diagonal, reused twice: to derive ``l2reg`` from
        # ``l2reg_diag_ratio`` and (when A is not overwritten) to restore A afterwards.
        # ``copy_detached_`` (not a bare index, which in the torch backend stays attached to
        # the autograd graph) keeps the restore and the regularization out of A's gradient,
        # matching the original where l2reg was an external constant.
        A_diag = copy_detached_(A[diag_indices])
        if l2reg is None:
            l2reg = mean_(A_diag) * l2reg_diag_ratio
        A[diag_indices] += l2reg
    if overwrite_a and A.flags.c_contiguous:
        # TODO: K.Karan.: as far as I understand using "reshape_" instead of "np.asfortranarray" capitalizes on A being Hermitian and thus
        # not requiring transposition while creating a FORTRAN array; this results in np.reshape being faster in this context.
        A = reshape_(A, A.shape, order="F")
    fact = factorization(A, overwrite_a=overwrite_a)
    if regularize and not overwrite_a:
        A[diag_indices] = A_diag[:]
    return fact


def solve_w_reg_factor(
    A,
    B,
    factorization,
    solver,
    l2reg=None,
    l2reg_diag_ratio=None,
    overwrite_a=False,
    overwrite_b=False,
    solver_name="scipy solver",
):
    """Solve ``A x = B`` (optionally ridge-regularised) by ``factorization`` + ``solver``
    (e.g. cho_factor/cho_solve_from_matrix, lu_factor/lu_solve_from_matrix), auto-switching by size
    on ``SCIPY_MAX_N``:
    both scipy Cholesky and LU share the LP64 LAPACK that overflows for ``A`` larger than
    ~46341x46341, so above the threshold we fall back (with a warning naming ``solver_name`` and
    the path taken) to whichever ILP64 backend is available:

    * ``bigla`` when installed AND it resolved a genuine ILP64 LAPACK (:func:`_bigla_if_usable`):
      Cholesky, so the factorisation stays reusable and peak memory stays low;
    * otherwise numpy's ILP64 ``np.linalg.solve`` -- a one-shot general LU, which drops the
      reusable factorization and ``overwrite_b``.

    All three paths keep the ridge regularisation identical."""
    if A.shape[0] > SCIPY_MAX_N:
        A = add_l2reg_to_diagonal(
            A, l2reg=l2reg, l2reg_diag_ratio=l2reg_diag_ratio, overwrite_a=overwrite_a
        )
        bigla = _bigla_if_usable()
        # The requested factorization MUST be honoured here. Routing an lu_factor_ request to a
        # Cholesky is not a performance substitution but a change of precondition: Cholesky needs
        # symmetric positive-definite input, and a caller who asked for LU generally has a matrix
        # that is neither. It would raise on a good day and, on an indefinite matrix whose leading
        # minors happen to be positive, return a wrong answer quietly.
        # add_l2reg_to_diagonal already made the copy if the caller asked for one, so overwriting
        # below is safe regardless of overwrite_a.
        if bigla is not None and factorization is cho_factor_:
            # ILP64 Cholesky. Preferred over np.linalg.solve for two reasons: it keeps the
            # factorisation reusable (np.linalg.solve is a one-shot general LU, so a lambda-grid or
            # k-fold loop refactorises every time), and its peak memory is far lower exactly where
            # that matters.
            _warn_too_large_for_scipy(
                A.shape[0], solver_name, "bigla's ILP64 Cholesky", stacklevel=4
            )
            factor = bigla.cho_factor(A, lower=True, overwrite_a=True)
            return bigla.cho_solve(factor, B, overwrite_b=False)
        if bigla is not None and factorization is lu_factor_ and hasattr(bigla, "lu_factor"):
            # ILP64 LU -- same reusability argument as above, without assuming definiteness.
            # hasattr guards against a bigla predating the LU routines.
            _warn_too_large_for_scipy(A.shape[0], solver_name, "bigla's ILP64 LU", stacklevel=4)
            factor = bigla.lu_factor(A, overwrite_a=True)
            return bigla.lu_solve(factor, B, overwrite_b=False)
        _warn_too_large_for_scipy(
            A.shape[0],
            solver_name,
            "numpy's ILP64 np.linalg.solve " "(general LU, no reusable factorization)",
            stacklevel=4,
        )
        return np.linalg.solve(A, B)
    if overwrite_b:
        assert B.flags.f_contiguous, "unlike A, B must be FORTRAN-contiguous to allow overwriting"
    fact = get_factorization_w_l2reg(
        A, factorization, l2reg=l2reg, l2reg_diag_ratio=l2reg_diag_ratio, overwrite_a=overwrite_a
    )
    return solver(fact, B, overwrite_b=overwrite_b)


def add_l2reg_to_diagonal(A, l2reg=None, l2reg_diag_ratio=None, overwrite_a=False):
    """Return ``A + l2reg*I`` (deriving ``l2reg`` from ``l2reg_diag_ratio * mean(diag A)`` when only
    the ratio is given). The non-factorised counterpart of the regularisation in
    :func:`get_factorization_w_l2reg`, used by the numpy (ILP64) large-N solve path. Copies ``A``
    unless ``overwrite_a`` (and returns ``A`` unchanged when neither regulariser is given)."""
    if l2reg is None and l2reg_diag_ratio is None:
        return A
    if not overwrite_a:
        A = A.copy()
    diag_indices = diag_indices_from_(A)
    if l2reg is None:
        l2reg = mean_(copy_detached_(A[diag_indices])) * l2reg_diag_ratio
    A[diag_indices] += l2reg
    return A


def _bigla_if_usable():
    """Return the ``bigla`` module when it can actually handle a matrix past scipy's ceiling.

    bigla is an OPTIONAL dependency, imported here rather than at module scope so qml2 installs
    and runs without it. Two checks, in this order:

    1. Is it importable at all?
    2. Did it resolve a genuine ILP64 LAPACK? Importing successfully does NOT imply this --
       bigla falls back to an LP64 library with ``max_dim=46340`` rather than failing at import,
       so a caller that branches on the import alone would take the bigla path and then raise
       on the first large matrix. ``bigla.is_ilp64()`` is the question we actually mean.

    Returns None when either check fails, so callers fall through to the numpy ILP64 path.
    """
    try:
        import bigla
    except ImportError:
        return None
    # Ask through whichever API this bigla has. `is_ilp64()` is the convenience accessor; older
    # builds expose only `backend_info().ilp64`. Probing explicitly rather than catching an
    # AttributeError matters: a bare `except Exception` here silently disabled bigla ENTIRELY for
    # every build without `is_ilp64`, so the large-N path fell through to np.linalg.solve and the
    # whole integration was dead code that still looked live.
    if hasattr(bigla, "is_ilp64"):
        probe = bigla.is_ilp64
    elif hasattr(bigla, "backend_info"):

        def probe():
            return bigla.backend_info().ilp64

    else:
        return None
    try:
        if not probe():
            return None
    except Exception:  # backend discovery is best-effort; never let it break a solve
        return None
    return bigla


def _warn_too_large_for_scipy(n, solver_name, fallback, stacklevel=3):
    """Warn that ``A`` (N=``n``) is too large for scipy ``solver_name``, naming the ``fallback``
    actually taken so the log says which of the two ILP64 paths ran."""
    warnings.warn(
        f"matrix is too large (N={n} > SCIPY_MAX_N={SCIPY_MAX_N}) for {solver_name}: scipy's LP64 "
        f"LAPACK overflows past N~46341, so falling back to {fallback}.",
        stacklevel=stacklevel,
    )


def cho_solve_from_matrix(
    A, B, l2reg=None, l2reg_diag_ratio=None, overwrite_a=False, overwrite_b=False
):
    """
    Solve "A*x=B" (``A`` symmetric positive-definite) via Cholesky, optionally ridge-regularised.
    Auto-switches by size in :func:`solve_w_reg_factor`: scipy/torch ``cho_factor`` +
    ``cho_solve_from_matrix``
    for ``N <= SCIPY_MAX_N``, else numpy's ILP64 ``np.linalg.solve`` (scipy's LP64 LAPACK overflows
    for ``A`` larger than ~46341x46341).
    """
    return solve_w_reg_factor(
        A,
        B,
        cho_factor_,
        cho_solve_,
        l2reg=l2reg,
        l2reg_diag_ratio=l2reg_diag_ratio,
        overwrite_a=overwrite_a,
        overwrite_b=overwrite_b,
        solver_name="cho_solve_from_matrix",
    )


def lu_solve_from_matrix(
    A, B, l2reg=None, l2reg_diag_ratio=None, overwrite_a=False, overwrite_b=False
):
    """
    Solve "A*x=B" via LU decomposition. Auto-switches by size in :func:`solve_w_reg_factor` to
    numpy's ILP64 ``np.linalg.solve`` for ``N > SCIPY_MAX_N`` (scipy's LP64 LAPACK overflows past
    N~46341).
    """
    return solve_w_reg_factor(
        A,
        B,
        lu_factor_,
        lu_solve_,
        l2reg=l2reg,
        l2reg_diag_ratio=l2reg_diag_ratio,
        overwrite_a=overwrite_a,
        overwrite_b=overwrite_b,
        solver_name="lu_solve_from_matrix",
    )


def inplace_orthogonalize_vectors(A):
    """
    Overwrite matrix A (nvectors x vector_length) with an orthonormal basis of the space
    spanned by its rows.

    A QR factorization of the transpose, whose Q has orthonormal columns spanning the
    column space of A.T, which is the row space of A. This replaced an
    `scipy.linalg.svd(A, full_matrices=False)`, answering the note that used to stand here
    asking for a standard numpy/scipy routine: at the shape this is actually used with --
    514 x 78789, a SLATM training set -- the SVD took 7.4 s single-threaded against 3.2 s
    for the QR, for a result the callers cannot distinguish.

    "Cannot distinguish" is the part to be careful about. The rows come back in NO
    particular order, where the SVD returned them by decreasing singular value. That is
    immaterial to a caller that keeps the WHOLE basis: the span is the same, so the norm
    this preserves is the same, and so is every distance measured in the projected space --
    the two bases differ by a rotation, which a distance cannot see. A caller that kept only
    the first k rows would be selecting principal components, which this no longer returns,
    and must use an SVD instead.

    The sign convention below makes the result CANONICAL, which is a property the SVD this
    replaced did not have. A QR factorization is unique once the diagonal of R is required
    positive, so fixing it pins the basis exactly; an SVD basis is determined only up to a
    sign per row, and, where singular values repeat, only up to a rotation within the
    degenerate subspace. Callers that store results computed from this basis -- the
    benchmarked tests do -- were therefore exposed to a LAPACK build changing its mind,
    which is the "sign-flipping problem" the MSORF feature test's comments refer to.

    Rank deficiency is handled less gracefully than by an SVD; pass `pivoting=True` below if
    a caller ever needs to survive it. Note that the uniqueness argument also assumes full
    rank, so a rank-deficient input is doubly outside what this promises.
    """
    assert (
        A.shape[0] <= A.shape[1]
    ), f"cannot orthogonalize {A.shape[0]} vectors of length {A.shape[1]}"
    orthonormal_columns, upper_triangular = scipy.linalg.qr(A.T, mode="economic")
    # Column j of Q pairs with row j of R, so flipping both leaves the factorization intact
    # while forcing R's diagonal positive. A zero diagonal entry means rank deficiency, and
    # is left alone rather than given an arbitrary sign.
    diagonal_signs = np.sign(np.diag(upper_triangular))
    diagonal_signs[diagonal_signs == 0.0] = 1.0
    orthonormal_columns *= diagonal_signs
    A[:, :] = orthonormal_columns.T[:, :]


def lstsq_solve(A, B, rcond=None):
    """
    A convenient shorthand for just recovering the solution instead of the entire lstsq_ output.

    TODO (bigla / large-N): this has NO size guard, unlike :func:`cho_solve_from_matrix` and
    :func:`svd_aligned`. Past ~2**31 matrix elements (``SCIPY_MAX_ELEMENTS``) the LP64 LAPACK
    behind
    ``lstsq_`` overflows, silently -- no warning, no fallback.

    The reason it was not simply wired to bigla when the others were: **bigla's least-squares is
    not a
    drop-in for this signature.** ``bigla.lstsq`` goes through LAPACK ``gels`` (QR/LQ), which
    *requires full rank* and has no ``rcond`` at all -- whereas ``rcond`` here exists precisely to
    truncate small singular values, i.e. to handle the rank-deficient case. Substituting ``gels``
    would turn a minimum-norm truncated solution into a ``LinAlgError`` on exactly the inputs
    ``rcond`` was added for. That is the same class of silent-precondition swap as routing an LU
    request to a Cholesky, and it is not worth repeating.

    The faithful large-N route already exists in this module: ``bigla.svd`` (via
    :func:`svd_aligned`, which is now size-aware) feeding :func:`solve_from_svd`, which applies
    ``rcond`` itself. So the work is to add an SVD-backed branch here, not to call ``bigla.lstsq``.

    Not urgent, because the current callers are all small in the dimension that matters:

    * ``models/property_shifts.py`` ``optimal_shifts`` -- ``(npoints, nelements)``; nelements is a
      handful, so the element count stays far below the ceiling even at 10**6 points. NOTE this is
      the ENERGY path (every fit with quantity shifts), not a force-only one.
    * ``models/loss_functions.py`` (initial guess, and the ``fmle_rcond`` solve) -- likewise small.
    * The plausibly-large case is force fits, through :func:`svd_solve_from_matrix` rather
      than this function.

    Decide it together with the force-model work; see
    /home/konstantin/claude_misc/forces/README.md.
    """
    return lstsq_(A, B, rcond=rcond)[0]


def svd_aligned(A, overwrite_a=False):
    """
    Shorthand for C-contiguous output of scipy.linalg.svd(A, full_matrices=False).

    Size-aware like :func:`cho_solve_from_matrix`, and for the same reason: scipy's LP64 LAPACK
    overflows once the matrix holds more than ~2**31 elements. Note the criterion here is the
    ELEMENT COUNT, not a single dimension -- ``SCIPY_MAX_N`` (=sqrt(2**31)) is a square-matrix
    notion, and a tall-skinny Z matrix (say 10**6 x 5000) overflows at min(m, n) = 5000. Above the
    ceiling this uses ``bigla.svd`` when available; without bigla there is no ILP64 SVD to fall
    back
    to (numpy's ``linalg.svd`` links the same 64-bit LAPACK numpy ships, so it is tried), and the
    warning says so.
    """
    if A.size > SCIPY_MAX_ELEMENTS:
        bigla = _bigla_if_usable()
        if bigla is not None and hasattr(bigla, "svd"):
            _warn_too_large_for_scipy(
                max(A.shape),
                "svd_aligned",
                "bigla's ILP64 SVD (memory-aware driver)",
                stacklevel=3,
            )
            U, s, Vh = bigla.svd(A, full_matrices=False, overwrite_a=overwrite_a)
            return np.ascontiguousarray(U), s, np.ascontiguousarray(Vh)
        _warn_too_large_for_scipy(
            max(A.shape), "svd_aligned", "numpy's ILP64 np.linalg.svd", stacklevel=3
        )
        U, s, Vh = np.linalg.svd(A, full_matrices=False)
        return np.ascontiguousarray(U), s, np.ascontiguousarray(Vh)
    U, s, Vh = scipy.linalg.svd(A, full_matrices=False, overwrite_a=overwrite_a)
    return np.ascontiguousarray(U), s, np.ascontiguousarray(Vh)


def gram_eigh(K, overwrite_a=False):
    """Ascending ``(eigenvalues, eigenvectors)`` of a symmetric matrix ``K`` (e.g. a Gram matrix),
    auto-selecting the LAPACK backend by size (``SCIPY_MAX_N``):

    * ``N <= SCIPY_MAX_N``: ``scipy.linalg.eigh(driver='evr')`` -- ``dsyevr`` uses an O(N) workspace
      (vs numpy ``dsyevd``'s ~2 N^2) and, with ``overwrite_a``, factorises in place, so its peak
      memory is markedly smaller.
    * ``N > SCIPY_MAX_N``, ``bigla`` usable: ``bigla.eigh(driver="auto")`` -- ILP64, and its driver
      policy reads MemAvailable to choose between ``dsyevd`` (fast, ~2 N^2 workspace) and ``dsyev``
      (O(N) workspace), which is the difference between fitting on the node and not.
    * ``N > SCIPY_MAX_N``, no bigla: ``numpy.linalg.eigh`` -- numpy links the ILP64 (64-bit)
      OpenBLAS, whereas scipy's is LP64 (32-bit) and overflows once ``N^2 > 2^31`` (``N`` ~ 46341).
      Correct, but its ``dsyevd`` workspace is ~2 N^2 with no way to opt out.

    Both back-ends return eigenvalues ascending with matching orthonormal eigenvector columns, so the
    All three return eigenvalues ascending with matching orthonormal eigenvector COLUMNS, so the
    choice is transparent to callers (``regression_using_Z_gram_eigh`` etc.). ``overwrite_a`` is
    honoured by the scipy and bigla branches; numpy always copies."""
    if K.shape[0] <= SCIPY_MAX_N:
        return scipy.linalg.eigh(K, driver="evr", overwrite_a=overwrite_a, check_finite=False)
    bigla = _bigla_if_usable()
    if bigla is not None:
        # bigla mirrors scipy: eigenvalues ascending, eigenvectors as COLUMNS, so this is
        # interchangeable with both other branches. Its "auto" driver reads MemAvailable and
        # picks dsyevd or the O(N)-workspace dsyev accordingly -- the whole reason to prefer it
        # over np.linalg.eigh, whose dsyevd workspace is ~2 N^2 with no way to opt out.
        return bigla.eigh(K, lower=True, overwrite_a=overwrite_a, driver="auto")
    return np.linalg.eigh(K)


def gram_eigh_from_features(Z_matrix):
    """
    Eigendecomposition of the SMALLER Gram matrix of ``Z_matrix`` (shape
    ``(ntrain, nfeatures)``): ``eigh(Z Z^T)`` when ``ntrain <= nfeatures`` else
    ``eigh(Z^T Z)``. Returns ``(gram_eigenvalues, gram_eigenvectors)`` -- the squared
    singular values ``s^2`` together with, respectively, the left singular vectors ``U``
    (first case) or the right singular vectors ``V`` (second) of ``Z``.

    Reusable analogue of :func:`svd_aligned` for :func:`regression_using_Z_gram_eigh`:
    decompose once, then reuse the result across several ``l2reg`` values (e.g.
    leave-one-out ``l2reg`` optimisation), just as a cached ``svd_aligned`` result is
    reused via ``Z_U`` / ``Z_singular_values`` / ``Z_Vh``. Which singular-vector set is
    returned follows from the shape, so it stays consistent with the solve as long as the
    same ``Z_matrix`` is used.
    """
    ntrain, nfeatures = Z_matrix.shape
    if ntrain <= nfeatures:
        return gram_eigh(dot_(Z_matrix, Z_matrix.T), overwrite_a=True)
    return gram_eigh(dot_(Z_matrix.T, Z_matrix), overwrite_a=True)


def regression_using_Z_gram_eigh(
    Z_matrix, l2reg, quantities, gram_eigenvalues=None, gram_eigenvectors=None
):
    """
    ``regression_using_Z_SVD`` obtained from the eigendecomposition of the smaller Gram
    matrix rather than the thin SVD of ``Z_matrix``, never materialising the rectangular
    singular-vector block.

    Using ``Z = U diag(s) Vh``, the ridge solution
    ``alphas = Vh.T diag(s/(s^2+l2reg)) U.T y`` is rewritten with only the squared
    singular values ``s^2`` (the Gram eigenvalues) and one side's singular vectors:

    * ``ntrain <= nfeatures`` (the SORF regime): ``eigh(Z Z^T)`` -> ``U`` and ``s^2``;
      ``alphas = Z^T U diag(1/(s^2+l2reg)) U^T y``.
    * ``nfeatures < ntrain``: ``eigh(Z^T Z)`` -> ``V`` and ``s^2``;
      ``alphas = V diag(1/(s^2+l2reg)) V^T Z^T y``.

    Why this can beat the thin SVD: it never forms the
    ``O(min(ntrain, nfeatures) x max(ntrain, nfeatures))`` singular-vector block that
    dominates a thin SVD when the two dimensions are very unequal, so it does
    substantially fewer flops for ``ntrain << nfeatures`` regardless of the BLAS/LAPACK
    backend. Separately -- and backend-dependent -- the Gram build is all BLAS-3
    ``gemm``s, which parallelise well, whereas the thin-SVD bidiagonalisation barely
    multithreads on some backends (e.g. the OpenBLAS in the numpy/scipy wheels here) yet
    may scale better on others (e.g. MKL). That implementation dependence is why the
    choice is kept switchable via ``use_gram_svd``.

    Trade-off: forming a Gram matrix squares the condition number, so on badly conditioned
    ``Z`` prefer the explicit SVD (``use_gram_svd=False``). ``gram_eigenvalues`` /
    ``gram_eigenvectors`` (from :func:`gram_eigh_from_features`) may be supplied to reuse a cached
    decomposition across ``l2reg`` values.
    """
    ntrain, nfeatures = Z_matrix.shape
    if gram_eigenvalues is None:
        gram_eigenvalues, gram_eigenvectors = gram_eigh_from_features(Z_matrix)
    # 1 / (s^2 + l2reg); clip tiny negative round-off in the (PSD) Gram eigenvalues.
    inv_regularized = 1.0 / (maximum_(gram_eigenvalues, 0.0) + l2reg)
    if ntrain <= nfeatures:
        # gram_eigenvectors are the left singular vectors U of Z.
        transformed = dot_(gram_eigenvectors.T, quantities)
        transformed *= inv_regularized
        return dot_(Z_matrix.T, dot_(gram_eigenvectors, transformed))
    # gram_eigenvectors are the right singular vectors V of Z.
    transformed = dot_(gram_eigenvectors.T, dot_(Z_matrix.T, quantities))
    transformed *= inv_regularized
    return dot_(gram_eigenvectors, transformed)


def regression_using_Z_SVD(
    Z_matrix,
    l2reg,
    quantities,
    Z_U=None,
    Z_singular_values=None,
    Z_Vh=None,
    gram_eigenvalues=None,
    gram_eigenvectors=None,
    use_gram_svd=True,
):
    """
    Regression for SORF from the SVD spectrum of the feature matrix ``Z_matrix`` (shape
    ``(ntrain, nfeatures)``); returns feature-space ``alphas``.

    The spectrum is obtained one of two ways:

    * ``use_gram_svd=True`` (default): :func:`regression_using_Z_gram_eigh` -- eigenvalues
      of the smaller Gram matrix, skipping the rectangular singular-vector block. Fewer
      flops for ``ntrain << nfeatures`` and, on backends that parallelise ``gemm`` better
      than the thin SVD, faster on many cores as well (see that function for the caveats).
    * ``use_gram_svd=False``: the explicit thin SVD via :func:`svd_aligned` -- costlier but
      resolves tiny singular values on ill-conditioned ``Z`` more accurately. Kept because
      the relative speed of the two is backend-dependent (BLAS/LAPACK implementation).

    Decomposition reuse across ``l2reg`` values mirrors between the two modes: with
    ``use_gram_svd=True`` pass cached ``gram_eigenvalues`` / ``gram_eigenvectors`` (from
    :func:`gram_eigh_from_features`); with ``use_gram_svd=False`` pass cached ``Z_U`` /
    ``Z_singular_values`` / ``Z_Vh`` (from :func:`svd_aligned`). Supplied SVD factors take
    precedence and force the explicit-SVD solve.
    """
    if Z_U is None and use_gram_svd:
        assert Z_matrix is not None
        return regression_using_Z_gram_eigh(
            Z_matrix,
            l2reg,
            quantities,
            gram_eigenvalues=gram_eigenvalues,
            gram_eigenvectors=gram_eigenvectors,
        )
    if Z_U is None:
        assert Z_matrix is not None
        Z_U, Z_singular_values, Z_Vh = svd_aligned(Z_matrix)
    transformed_alphas_rhs = dot_(Z_U.T, quantities)
    transformed_alphas_rhs *= Z_singular_values / (Z_singular_values**2 + l2reg)
    return dot_(Z_Vh.T, transformed_alphas_rhs)


def regression_using_Z(
    Z_matrix,
    quantities,
    l2reg=None,
    l2reg_diag_ratio=None,
    kernel_factorization=None,
    return_kernel_factorization=False,
    use_primal=None,
):
    """
    Ridge regression for SORF returning feature-space ``alphas`` that minimise
    ``||Z alphas - quantities||^2 + l2reg ||alphas||^2``.

    The identical solution is reached from either the ``ntrain x ntrain`` dual kernel
    ``Z Z^T`` or the ``nfeatures x nfeatures`` primal Gram ``Z^T Z``; the cheaper one
    depends on the shape:

    * dual (default in the SORF regime ``nfeatures >= ntrain``): factor
      ``Z Z^T + l2reg I``, solve, then ``alphas = Z^T @ beta``;
    * primal (``nfeatures < ntrain``): factor ``Z^T Z + l2reg I`` and solve directly
      against ``Z^T @ quantities``.

    ``use_primal`` forces the branch; ``None`` picks it from the shape. Note the
    ``l2reg_diag_ratio`` reference is the mean diagonal of whichever matrix is factored,
    so it differs between branches -- for a shape-independent regulariser pass an absolute
    ``l2reg``. A supplied ``kernel_factorization`` must match the chosen branch.

    The factored matrix has size ``min(ntrain, nfeatures)``; when that exceeds ``SCIPY_MAX_N`` the
    scipy Cholesky is replaced (with a warning) by numpy's ILP64 ``np.linalg.solve`` -- see
    :func:`cho_solve_from_matrix` -- which returns no reusable factorization
    (``kernel_factorization`` is then
    ``None``).
    """
    ntrain, nfeatures = Z_matrix.shape
    if use_primal is None:
        use_primal = nfeatures < ntrain
    if use_primal:
        if nfeatures <= SCIPY_MAX_N:
            if kernel_factorization is None:
                gram = dot_(Z_matrix.T, Z_matrix)
                kernel_factorization = get_factorization_w_l2reg(
                    gram,
                    cho_factor_,
                    l2reg=l2reg,
                    l2reg_diag_ratio=l2reg_diag_ratio,
                    overwrite_a=True,
                )
            alphas = cho_solve_(kernel_factorization, dot_(Z_matrix.T, quantities))
        else:
            _warn_too_large_for_scipy(
                nfeatures,
                "regression_using_Z",
                "numpy's ILP64 np.linalg.solve (general LU, no reusable factorization)",
            )
            gram = add_l2reg_to_diagonal(
                dot_(Z_matrix.T, Z_matrix),
                l2reg=l2reg,
                l2reg_diag_ratio=l2reg_diag_ratio,
                overwrite_a=True,
            )
            alphas = np.linalg.solve(gram, dot_(Z_matrix.T, quantities))
            kernel_factorization = None
        if return_kernel_factorization:
            return alphas, kernel_factorization
        return alphas
    if ntrain <= SCIPY_MAX_N:
        if kernel_factorization is None:
            k = dot_(Z_matrix, Z_matrix.T)
            kernel_factorization = get_factorization_w_l2reg(
                k, cho_factor_, l2reg=l2reg, l2reg_diag_ratio=l2reg_diag_ratio, overwrite_a=True
            )
        alphas_untransformed = cho_solve_(kernel_factorization, quantities)
    else:
        _warn_too_large_for_scipy(
            ntrain,
            "regression_using_Z",
            "numpy's ILP64 np.linalg.solve (general LU, no reusable factorization)",
        )
        k = add_l2reg_to_diagonal(
            dot_(Z_matrix, Z_matrix.T),
            l2reg=l2reg,
            l2reg_diag_ratio=l2reg_diag_ratio,
            overwrite_a=True,
        )
        alphas_untransformed = np.linalg.solve(k, quantities)
        kernel_factorization = None
    alphas = dot_(Z_matrix.T, alphas_untransformed)
    if return_kernel_factorization:
        return alphas, kernel_factorization
    return alphas


# KK: Accelerates training if you train on several properties that can be sorted in such a way that each property is available for each molecule for which the
# next property is available (but not necessarily vice-versa). In this case re-indexing allows usage of a single Cholesky decomposition for all properties.
# Originally developed for doing G2S for molecules of different sizes. Might require heavy revision to be usable again.
def nullify_ignored(arr, indices_to_ignore):
    if indices_to_ignore is not None:
        for row_id, cur_ignore_indices in enumerate(indices_to_ignore):
            arr[row_id][where2slice(logical_not_(cur_ignore_indices))] = 0.0


class Cho_multi_factors:
    def __init__(self, train_kernel, indices_to_ignore=None, ignored_orderable=False):
        """
        For dealing with several Cholesky decompositions at once.
        """
        self.indices_to_ignore = indices_to_ignore
        self.ignored_orderable = ignored_orderable
        self.single_cho_decomp = indices_to_ignore is None
        if not self.single_cho_decomp:
            self.single_cho_decomp = not self.indices_to_ignore.any()
        if self.single_cho_decomp:
            self.cho_factors = [cho_factor_(train_kernel)]
        else:
            if self.ignored_orderable:
                ignored_nums = []
                for i, cur_ignored in enumerate(self.indices_to_ignore):
                    ignored_nums.append((i, sum_(cur_ignored)))
                ignored_nums.sort(key=lambda x: x[1])
                self.availability_order = array_([i[0] for i in ignored_nums])
                self.avail_quant_nums = [
                    self.indices_to_ignore.shape[0] - sum_(cur_ignored)
                    for cur_ignored in self.indices_to_ignore.T
                ]
                self.cho_factors = [
                    cho_factor_(
                        train_kernel[self.availability_order, :][:, self.availability_order]
                    )
                ]
            else:
                self.cho_factors = []
                for cur_ignore_ids in self.indices_to_ignore.T:
                    s = where2slice(cur_ignore_ids)
                    self.cho_factors.append(cho_factor_(train_kernel[s, :][:, s]))

    def solve_with(self, rhs):
        if len(rhs.shape) == 1:
            assert self.single_cho_decomp
            cycled_rhs = array_([rhs])
        else:
            if not self.single_cho_decomp:
                if self.ignored_orderable:
                    assert len(self.avail_quant_nums) == rhs.shape[1]
                else:
                    assert len(self.cho_factors) == rhs.shape[1]
            cycled_rhs = rhs.T
        output = zeros_(cycled_rhs.shape)
        for rhs_id, rhs_component in enumerate(cycled_rhs):
            if self.indices_to_ignore is None:
                included_indices = array_(range(len(rhs_component)))
            else:
                if self.ignored_orderable:
                    included_indices = self.availability_order[: self.avail_quant_nums[rhs_id]]
                else:
                    included_indices = where2slice(self.indices_to_ignore[:, rhs_id])
            if self.single_cho_decomp:
                cur_decomp = self.cho_factors[0]
            else:
                if self.ignored_orderable:
                    cur_decomp = (
                        self.cho_factors[0][0][: self.avail_quant_nums[rhs_id], :][
                            :, : self.avail_quant_nums[rhs_id]
                        ],
                        self.cho_factors[0][1],
                    )
                else:
                    cur_decomp = self.cho_factors[rhs_id]
            output[rhs_id, included_indices] = cho_solve_from_matrix(
                cur_decomp, rhs_component[included_indices]
            )
        if len(rhs.shape) == 1:
            return output[0]
        else:
            return output
