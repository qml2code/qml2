#####
# For initial guesses of hyperparameters.
#####
from ..jit_interfaces import jit_, max_, mean_, prange_, sqrt_, sum_, zeros_
from ..utils import l1_norm_dist, l2_norm_sq_dist


# I think the oldest reference for this is a paper by Bing (if someone remembers DOI please add).
def max_dist(X_arr, dist_func):
    @jit_(numba_parallel=True)
    def func(X_arr):
        l = X_arr.shape[0]
        m = zeros_((2,))
        for i in prange_(l):
            for j in range(i):
                m[1] = dist_func(X_arr[i], X_arr[j])
                m[0] = max_(m)
        return m[0]

    return func(X_arr)


def max_Laplace_dist(X_arr):
    return max_dist(X_arr, l1_norm_dist)


def max_Gauss_dist(X_arr):
    return sqrt_(max_dist(X_arr, l2_norm_sq_dist))


# NOTE: the function is non-jittable at least as of Python 3.12.2
# due to axis keyword in mean_.
# @jit_
def vector_std(X_arr):
    X_mean = mean_(X_arr, axis=0)
    X_div2 = mean_((X_arr - X_mean) ** 2, axis=0)
    return sqrt_(sum_(X_div2))
