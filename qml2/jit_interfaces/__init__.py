# KK: As far as I understand, well written Numpy/Numba code is faster than well-written Torch code on CPUs.
# Therefore, I think for now it makes most sense to write implementation of everything primarily with Numpy/Numba in mind
# and treat Torch as a useful addition for benchmarking gradient calculation routines and doing hyperparameter
# optimization.
# KK: For now I am a bit messy about dealing with the fact that an int can be int64 or tensor with dtype int64 in torch
# vs and int being just and int64 in Numba. In the future, might be worthwhile to keep torch's distinction everywhere
# by using int_ and int64_/tint_? (tint_ for Tensor.dint64)
# KK: Writing interfaces for all functions can be annoying. However, apart from Numba-Torch intercompilability,
# I hope this would ensure that adjusting old code to potential Numpy/Numba or pyTorch syntax revision
# can be done as easily as changing the corresponding lines in numba_interface.py or torch_interface.py.
# WARNING: If the same functionality is supported in Numpy/Numba and Torch under different names,
# then it enters under the name given to it by Numpy/Numba.
import importlib

from ..basic_utils import checked_environ_val
from .jit_manager import available_jits, numba_flag

default_jit_env_var_name = "QML2_DEFAULT_JIT"


def set_defaults_from_interface(interface_name):
    interface_submodule = importlib.import_module(interface_name, package=__name__)
    imported_vars = interface_submodule.__dict__
    for imported_var_name, imported_var in imported_vars.items():
        if len(imported_var_name) < 2:
            continue
        if (imported_var_name[-1] != "_") or (imported_var_name[-2] == "_"):
            continue
        globals()[imported_var_name] = imported_var


def set_default_jit(new_jit_flag):
    assert new_jit_flag in available_jits
    global used_jit_name
    used_jit_name = new_jit_flag
    set_defaults_from_interface("." + used_jit_name + "_interface")


if __name__ != "__main__":
    # Check environment for default jit flag, if none pick numba
    default_flag = checked_environ_val(
        default_jit_env_var_name, default_answer=numba_flag, var_class=str
    )
    default_flag_internal = default_flag.lower()
    assert default_flag_internal in available_jits
    set_default_jit(default_flag_internal)
