import multiprocessing as multiprocessing_
from typing import List, Optional, Tuple, Union

import numpy as np
from numba import get_num_threads, jit, prange
from scipy.linalg import cho_factor, cho_solve, lu_factor, lu_solve

from .jit_manager import defined_jit_, numba_flag

Pool_ = multiprocessing_.Pool

jit_ = defined_jit_(jit, numba_flag)

DimsSequenceType_ = Union[List[int], Tuple[int]]  # Union[List[int], Tuple[int, ...]]

LinAlgError_ = np.linalg.LinAlgError

concatenate_ = np.concatenate

array_jittable_ = np.array

get_num_threads_ = get_num_threads


# For compatibility when compiled with Torch.
class Module_:
    def __init__(self):
        pass

    def forward(self):
        pass

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def compile(self):
        pass


#        print("WARNING: .compile() was called in Numpy/Numba mode.")


def Parameter_(arg):
    return arg


# TODO: KK: TBH depends on some Numba quirk I don't understand, needs to be tested more thoroughly.
@jit_
def all_tuple_dim_smaller_(t1, t2):
    """
    If t1=arr1.shape and t2=arr2.shape determines whether arr2 is large enough to store arr1.
    """
    return t1 <= t2


# KK: I saw conflicting (outdated?) information on whether flip allocates a new copy
flip_ = np.flip

# To maintain compatibility with pyTorch, we distinguish between standalone integers, integers that are part of
# an array, and the corresponding types. In Numba those are largely the same thing, as reflected by numerous repetitions.
# datatypes
float_ = np.float64
int_ = np.int64
bool_ = np.bool_
dfloat_ = np.float64
dint_ = np.int64
dint32_ = np.int32
dbool_ = np.bool_
ndarray_ = np.ndarray
optional_ndarray_ = Union[ndarray_, None]
dtype_ = type
dim0float_array_ = dfloat_
dim0int_array_ = dint_
OptionalGenerator_ = Optional[np.random.Generator]
# constructors for standalone
constr_float_ = float_
constr_int_ = int_
constr_bool_ = bool_
# constructors for array components.
constr_dfloat_ = float_
constr_dint_ = int_
constr_dbool_ = bool_


def is_scalar_(val):
    if isinstance(val, ndarray_):
        return val.shape == ()
    else:
        return True


@jit_
def zero_scalar_():
    return 0.0


def hermitian_matrix_inverse_(mat):
    return np.linalg.pinv(mat, hermitian=True)


# special variables
inf_ = np.inf
isinf_ = np.isinf
isnan_ = np.isnan
tiny_ = np.finfo(float_).tiny
# array constructors
array_ = np.array
empty_ = np.empty
zeros_ = np.zeros
ones_ = np.ones
eye_ = np.eye
repeat_ = np.repeat
arange_ = np.arange

# random-related
random_ = np.random.random
default_rng_ = np.random.default_rng


@jit_
def random_array_from_rng_(size=(1,), rng=None):
    if rng is None:
        return random_(size)
    else:
        return rng.random(size)


@jit_
def random_from_rng_(rng=None):
    return random_array_from_rng_(rng=rng)[0]


standard_normal_ = np.random.standard_normal


@jit_
def standard_normal_array_from_rng_(size=(1,), rng=None):
    if rng is None:
        return standard_normal_(size)
    else:
        return rng.standard_normal(size)


@jit_
def standard_normal_from_rng_(rng=None):
    return standard_normal_array_from_rng_(rng=rng)[0]


randint_ = np.random.randint


@jit_
def randint_array_from_rng_(lbound, ubound, size=(1,), rng=None):
    if rng is None:
        return randint_(lbound, ubound, size=size)
    else:
        return rng.integers(lbound, ubound, size=size)


@jit_
def randint_from_rng_(lbound, ubound, rng=None):
    return randint_array_from_rng_(lbound, ubound, rng=rng)[0]


seed_ = np.random.seed
permutation_ = np.random.permutation
# copying
copy_ = np.copy
copy_detached_ = copy_  # only different in pyTorch
# array lookup and manipulation
where_ = np.where


@jit_
def elements_where_(val_arr, bool_arr):
    # because where_ in these situations causes problems with TorchScript.
    return val_arr[where_(bool_arr)]


argsort_ = np.argsort
diag_indices_from_ = np.diag_indices_from
trace_ = np.trace

append_ = np.append
delete_ = np.delete
searchsorted_ = np.searchsorted
max_ = np.max
min_ = np.min
argmin_ = np.argmin
argmax_ = np.argmax
sign_ = np.sign
floor_ = np.floor
mean_ = np.mean
median_ = np.median
std_ = np.std
sort_ = np.sort
# logical
logical_not_ = np.logical_not
logical_and_ = np.logical_and
any_ = np.any
all_ = np.all
# linear operations
dot_ = np.dot
cross_ = np.cross
matmul_ = np.matmul
# trigonometry
cos_ = np.cos
sin_ = np.sin
arccos_ = np.arccos
# analytic functions
sqrt_ = np.sqrt
exp_ = np.exp
cosh_ = np.cosh
sinh_ = np.sinh
tanh_ = np.tanh
log_ = np.log
# Generating grids.
linspace_ = np.linspace
# loops
prange_ = prange
# common functions
sum_ = np.sum
prod_ = np.prod
abs_ = np.abs
square_ = np.square
l2_norm_ = np.linalg.norm
# linear algebra
cho_factor_ = cho_factor
svd_ = np.linalg.svd
eigh_ = np.linalg.eigh
cho_solve_ = cho_solve
lu_factor_ = lu_factor
lu_solve_ = lu_solve
lstsq_ = np.linalg.lstsq
# important constants
pi_ = np.pi


# Only needed to keep torch differentiability.
@jit_
def save_(arr: ndarray_):
    return arr


@jit_
def permuted_range_(n: int_):
    output = empty_((n,), dtype=dint_)
    for i in range(n):
        output[i] = i
    return permutation_(output)


# other
reshape_ = np.reshape
