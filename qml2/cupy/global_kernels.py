"""
Implementation of global functions found in `qml2.kernels.kernels` capitalizing on CuPy's array operations (namely,
`gaussian_kernel`, `laplacian_kernel`, and `matern_kernel` with their symmetric versions).

Arguments and keyword arguments are completely the analogous to those for `qml2.kernels.kernels`, except the
`out` keyword is not supported.
"""

import cupy as cp

from ..kernels.kernels import sqrt3, sqrt5


def l1_dists_no_sym_check(X_train, X_test):
    return cp.sum(cp.abs(X_train[:, cp.newaxis, :] - X_test[cp.newaxis, :, :]), axis=2)


def l1_dists(X_train, X_test):
    X_train = cp.asarray(X_train)
    if X_test is None:
        # symmetric kernel
        return l1_dists_no_sym_check(X_train, X_train)
    else:
        X_test = cp.asarray(X_test)
        return l1_dists_no_sym_check(X_train, X_test)


def get_sq_norms(X):
    return cp.sum(X**2, axis=1)


def l2_sq_dists(X_train, X_test):
    X_train = cp.asarray(X_train)
    if X_test is None:
        # symmetric kernel
        sq_dists = cp.dot(X_train, X_train.T)
        sq_norms = cp.copy(cp.diagonal(sq_dists))
        sq_dists *= -2
        sq_dists += sq_norms.reshape(-1, 1) + sq_norms.reshape(1, -1)
    else:
        # asymmetric kernel
        X_test = cp.asarray(X_test)
        X_test_sq_norms = get_sq_norms(X_test)
        X_train_sq_norms = get_sq_norms(X_train)
        sq_dists = (
            X_train_sq_norms.reshape(-1, 1)
            + X_test_sq_norms.reshape(1, -1)
            - 2 * cp.dot(X_train, X_test.T)
        )
    return sq_dists


def l2_dists(X_train, X_test):
    sq_dists = l2_sq_dists(X_train, X_test)
    # Sometimes due to numerical errors some square distance values are negative with low magnitude.
    sq_dists[cp.where(sq_dists < 0.0)] = 0.0
    return cp.sqrt(sq_dists)


def gaussian_kernel(X_train, X_test, sigma):
    gamma = 1.0 / (2.0 * sigma**2)

    sq_dists = l2_sq_dists(X_train, X_test)

    K = cp.exp(-gamma * sq_dists)
    cp.cuda.Stream.null.synchronize()

    K = cp.asnumpy(K)
    return K


def gaussian_kernel_symmetric(X, sigma):
    return gaussian_kernel(X, None, sigma)


def matern_kernel(X_train, X_test, sigma, order=0, metric="l1"):
    match metric:
        case "l1":
            dists = l1_dists(X_train, X_test)
        case "l2":
            dists = l2_dists(X_train, X_test)
        case _:
            raise Exception("Unimplemented Matern metric.")
    dists /= sigma
    match order:
        case 0:
            K = cp.exp(-dists)
        case 1:
            K = cp.exp(-dists * sqrt3) * (1 + sqrt3 * dists)
        case 2:
            K = cp.exp(-dists * sqrt5) * (1 + sqrt5 * dists + dists**2 * 5.0 / 3.0)
    cp.cuda.Stream.null.synchronize()

    K = cp.asnumpy(K)
    return K


def matern_kernel_symmetric(X, sigma, **kwargs):
    return matern_kernel(X, None, sigma, **kwargs)


def laplacian_kernel(*args):
    return matern_kernel(*args, order=0, metric="l1")


def laplacian_kernel_symmetric(*args):
    return matern_kernel_symmetric(*args, order=0, metric="l1")
