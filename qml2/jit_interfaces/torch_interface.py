# from torch._prims_common import DimsSequenceType

from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from torch import tensor

from ..basic_utils import checked_environ_val
from .jit_manager import defined_jit_, torch_flag

torch_default_device_env_name = "QML2_DEFAULT_TORCH_DEVICE"

# NOTE: KK: Introduced for consistency with NumPy. Perhaps will be managed properly in the future.
torch.set_default_dtype(torch.float64)

multiprocessing_ = torch.multiprocessing


# Pool_ = multiprocessing_.Pool
# TODO: As of 2024.07.11 the code has some weird problems with torch.multiprocessing.Pool.
# Check if it's still the case couple of Torch releases later.
class Pool_:
    def __init__(self, n):
        self.n = n

    def map(self, f, arr):
        return [f(el) for el in arr]

    def close(self):
        pass


DimsSequenceType_ = List[int]  # Union[List[int], Tuple[int]]  # Union[List[int], Tuple[int, ...]]


# For building Torch models.
class Module_(torch.nn.Module):
    def __init__(self):
        super().__init__()


Parameter_ = torch.nn.Parameter


# Which device everything is stored on.
# Taken from:https://stackoverflow.com/questions/53266350/how-to-tell-pytorch-to-not-use-the-gpu
def recommended_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# For conveniently checking where PyTorch puts everything.
def print_default_device():
    test_array = array_([1])
    print(test_array.device)


# JIT compilation.
jit_ = defined_jit_(torch.jit.script, torch_flag, possible_jit_failures=[TypeError, RuntimeError])

# datatypes
float_ = float
int_ = int
bool_ = bool
dfloat_ = torch.float64
dint_ = torch.int64
dint32_ = torch.int32
dbool_ = torch.bool
ndarray_ = torch.Tensor
dtype_ = torch.dtype
optional_dtype_ = Union[dtype_, None]
optional_ndarray_ = Optional[ndarray_]
dim0float_array_ = torch.Tensor
dim0int_array_ = torch.Tensor
OptionalGenerator_ = Optional[torch.Generator]
# constructors for standalone
constr_float_ = float


def is_scalar_(val):
    if isinstance(val, ndarray_):
        return len(val.shape) == 0  # check it's a 0-dim tensor
    else:
        return True


@jit_
def zero_scalar_():
    return torch.tensor(0.0)


@jit_
def constr_int_(val: ndarray_):
    return int(val)


constr_bool_ = bool


# constructors for 0-dim tensors.
@jit_
def constr_dfloat_(val: float, dfloat_: dtype_ = dfloat_):
    return torch.tensor(val, dtype=dfloat_)


@jit_
def constr_dint_(val: int, dint_: dtype_ = dfloat_):
    return torch.tensor(val, dtype=dint_)


@jit_
def constr_dbool_(val: bool, dbool_: dtype_ = dbool_):
    return torch.tensor(val, dtype=dbool_)


# Functions for which transition from Numpy/Numba to Torch is not straightforward
def array_(
    converted_array,
    torch_device: Optional[torch.device] = None,
    torch_requires_grad: Optional[bool_] = False,
    dtype: optional_dtype_ = None,
):
    try:
        return torch.tensor(
            converted_array, device=torch_device, requires_grad=torch_requires_grad, dtype=dtype
        )
    except (TypeError, ValueError):
        reformatted_list = [
            array_(
                el, torch_device=torch_device, torch_requires_grad=torch_requires_grad, dtype=dtype
            )
            for el in converted_array
        ]
        sample_el = reformatted_list[0]
        output = torch.empty((len(reformatted_list), *sample_el.shape))
        for i, el in enumerate(reformatted_list):
            assert el.shape == sample_el.shape
            output[i, :] = el[:]
        return output


array_jittable_ = torch.tensor


@jit_
def append_(tensor1: ndarray_, tensor2: ndarray_, axis: int = 0):
    shape1 = tensor1.shape
    shape2 = tensor2.shape
    ndims = len(shape1)
    assert ndims == len(shape2)
    shape_final: List[int] = [s for s in shape1]
    for d in range(ndims):
        if d == axis:
            shape_final[d] += shape2[d]
        else:
            assert shape_final[d] == shape2[d]
    output = torch.empty(shape_final, dtype=tensor1.dtype)
    torch.cat((tensor1, tensor2), dim=axis, out=output)
    return output


# TODO dirty, but I guess the function shouldn't be used anywhere anyway
# @jit_
def delete_(tensor, slice, axis=0):
    converted = np.array(tensor)
    return array_(np.delete(converted, slice, axis=axis))


empty_ = torch.empty
zeros_ = torch.zeros
ones_ = torch.ones


@jit_
def random_(size: DimsSequenceType_ = ()):
    return torch.rand(size)


def default_rng_(seed: Optional[int] = None):
    output = torch.Generator(device="cpu")
    if seed is not None:
        output.manual_seed(seed)
    return output


@jit_
def random_array_from_rng_(size: DimsSequenceType_ = (1,), rng: OptionalGenerator_ = None):
    return torch.rand(size, generator=rng)


randint_ = torch.randint


@jit_
def standard_normal_(size: DimsSequenceType_ = ()):
    return torch.normal(zeros_(size), ones_(size))


seed_ = torch.manual_seed
eye_ = torch.eye
# ensuring the device is maintained in all constructors (KK: is it needed?)
device_enforcement = True

# KK: Might be excessive?
# def enforce_creator_device():
#    available_creator_names = ["empty_", "zeros_", "ones_", "random_"] #, "eye_"]
#    global_vars = globals()
#    for creator_name in available_creator_names:
#        old_func = global_vars[creator_name]
#        @jit_
#        def new_func(
#            size: size_type,
#            dtype: dtype_ = dfloat_,
#            torch_device: Union[torch.device, None] = default_device,
#        ):
#            return old_func(size, dtype=dtype, device=torch_device)
#
#        globals()[creator_name] = new_func


@jit_
def repeat_(repeated_value: ndarray_, num_repetitions: int):
    # KK: I don't think there is an exact analogue of np.repeat in torch
    output = empty_((num_repetitions,), dtype=type(repeated_value))
    output[:] = repeated_value
    return output


permuted_range_ = torch.randperm


@jit_
def permutation_(input_tensor: ndarray_):
    permuted_indices = permuted_range_(input_tensor.shape[0])
    return input_tensor[permuted_indices]


@jit_
def cho_solve_(factorization, rhs):
    return torch.cholesky_solve(rhs[:, None], factorization)[:, 0]


LinAlgError_ = torch._C._LinAlgError


@jit_
def diag_indices_from_(mat):
    s = mat.shape[0]
    l = list(range(s))
    return (l, l)


# WARNING: Think more about copy_/copy_detached_ usage.
@jit_
def copy_(tensor):
    return tensor.clone()


@jit_
def copy_detached_(tensor: ndarray_):
    return tensor.detach().clone()


# for defining constancts
@jit_
def const_float_(val: float):
    return tensor(val, dtype=torch.float64)


@jit_
def const_int_(val: int):
    return tensor(val, dtype=torch.float64)


@jit_
def const_bool_(val: bool):
    return tensor(val, dtype=torch.bool)


# loops
# KK: there might be a way to do proper parallelization here, but I do not know it.
@jit_
def prange_(l: int_):
    return list(range(l))


# special variables
inf_ = torch.inf
isinf_ = torch.isinf
isnan_ = torch.isnan
tiny_ = torch.finfo(float_).tiny
# important constants
pi_ = tensor(torch.pi)
# array lookup and manipulation
where_ = torch.where


@jit_
def elements_where_(val_arr, bool_arr):
    # because where_ in these situations causes problems with TorchScript.
    ids = where_(bool_arr)[0]
    new_val_arr = copy_(val_arr[: int(ids[-1]) + 1])
    j = 0
    for i in ids:
        new_val_arr[j] = val_arr[int(i)]
        j += 1
    return new_val_arr


argsort_ = torch.argsort
searchsorted_ = torch.searchsorted


# TODO: KK might need to implement dtype later on.
@jit_
def concatenate_(arrays: List[ndarray_], axis: int_ = 0, out: Union[ndarray_, None] = None):
    if out is None:
        return torch.cat(arrays, dim=axis)
    else:
        return torch.cat(arrays, dim=axis, out=out)


max_ = torch.max
min_ = torch.min
sign_ = torch.sign
floor_ = torch.floor


@jit_
def mean_(t, axis: int = 0):
    return torch.mean(t, dim=axis)


median_ = torch.median

std_ = torch.std


@jit_
def flip_(array: ndarray_, axis: Tuple[int] = (0,)):
    return torch.flip(array, axis)


@jit_
def sort_(array: ndarray_):
    return torch.sort(array)[0]


# logical
logical_not_ = torch.logical_not
logical_and_ = torch.logical_and
any_ = torch.any
all_ = torch.all
# common linear operations
matmul_ = torch.matmul


@jit_
def dot_(v1, v2, out: Union[ndarray_, None] = None):
    if out is None:
        return matmul_(v1, v2)
    if v1.requires_grad or v2.requires_grad:
        temp = matmul_(v1, v2)
        out[:] = temp[:]
        return temp
    else:
        return matmul_(v1, v2, out=out)


cross_ = torch.cross

# trigonometry
cos_ = torch.cos
sin_ = torch.sin
arccos_ = torch.arccos
# analytic functions
sqrt_ = torch.sqrt
exp_ = torch.exp
cosh_ = torch.cosh
sinh_ = torch.sinh
tanh_ = torch.tanh
log_ = torch.log
# Generating grids.
linspace_ = torch.linspace


# common functions
@jit_
def sum_(t, axis: Optional[int_] = None):
    if axis is None:
        return torch.sum(t)
    else:
        return torch.sum(t, dim=int(axis))


def prod_(t):
    return torch.prod(array_(t))


abs_ = torch.abs
square_ = torch.square
l2_norm_ = torch.linalg.vector_norm

# linear algebra.
cho_factor_ = torch.linalg.cholesky
svd_ = torch.linalg.svd
eigh_ = torch.linalg.eigh
lu_factor_ = torch.linalg.lu_factor
lstsq_ = torch.linalg.lstsq


# TODO: KK: looks dirty, needs revision
@jit_
def lu_solve_(lu_output: Tuple[ndarray_, ndarray_], rhs: ndarray_):
    return torch.linalg.lu_solve(lu_output[0], lu_output[1], rhs[:, None])[:, 0]


def hermitian_matrix_inverse_(mat):
    cho_decomp = cho_factor_(mat)
    return torch.cholesky_inverse(cho_decomp)


@jit_
def save_(arr: ndarray_):
    if arr.requires_grad:
        return arr.clone()
    else:
        return arr


@jit_
def all_tuple_dim_smaller_(t1: DimsSequenceType_, t2: DimsSequenceType_):
    """
    If t1=arr1.shape and t2=arr2.shape determines whether arr2 is large enough to store arr1.
    """
    for i in range(len(t1)):
        if t1[i] > t2[i]:
            return False
    return True


# other
reshape_ = torch.reshape

if __name__ != "__main__":
    torch_default_device = checked_environ_val(torch_default_device_env_name, var_class=str)
    if torch_default_device:
        torch.set_default_device(torch_default_device)
    print("QML2 now uses Torch, device:")
    print_default_device()
