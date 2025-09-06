# Introduced to ensure flags for different JIT compilers do not obstruct each other.
import inspect

from ..basic_utils import ExceptionRaisingFunc, checked_logical_environ_val

numba_flag = "numba"
torch_flag = "torch"
available_jits = [numba_flag, torch_flag]
debug = False
skip_jit = False
skip_jit_failures = False

debug_env_var_name = "QML2_DEBUG"
skip_jit_env_var_name = "QML2_SKIP_JIT"
skip_jit_failures_env_var_name = "QML2_SKIP_JIT_FAILURES"


def set_debug(new_debug_val):
    global debug
    debug = new_debug_val


def set_skip_jit(new_skip_jit_val):
    """
    For convenient debugging
    """
    global skip_jit
    skip_jit = new_skip_jit_val


def set_skip_jit_failures(skip_jit_failures_val):
    global skip_jit_failures
    skip_jit_failures = skip_jit_failures_val


var_subroutine_dict = {
    debug_env_var_name: set_debug,
    skip_jit_env_var_name: set_skip_jit,
    skip_jit_failures_env_var_name: set_skip_jit_failures,
}


# For ignoring jit failures (say for code which was only written for Numba and needs to be ignored when Torch is used)
class JITFailureException(Exception):
    pass


class JITExceptionRaisingFunc(ExceptionRaisingFunc):
    def __init__(self, func, ex):
        super().__init__(ex, returned_exception_type=JITFailureException)
        self.exception_text = (
            """
Failure in JIT compilation.
source origin:"""
            + inspect.getsourcefile(func)
            + """
function source:
"""
            + inspect.getsource(func)
            + """
traceback:
"""
            + self.exception_text
        )


# The class has to be created due to numba.jit and torch.jit.script having different flags.
class _jit_:
    def __init__(self, used_jit_=None, jit_keywords={}):
        self.used_jit_ = used_jit_
        self.jit_keywords = jit_keywords

    def __call__(self, signature_or_function):
        return self.used_jit_.exception_skipping_jit(signature_or_function, **self.jit_keywords)


def get_jit_keywords(
    used_jit_name=None, skip=False, numba_parallel=False, numba_fastmath=True, numba_nopython=True
):
    kws = {}
    for var_name, var in locals().items():
        var_name_spl = var_name.split("_")
        if var_name_spl[0] != used_jit_name:
            continue
        new_var_name = "_".join(var_name_spl[1:])
        kws[new_var_name] = var
    return kws


class defined_jit_:
    def __init__(self, used_jit_func, used_jit_name, possible_jit_failures=[]):
        self.used_jit_func = used_jit_func
        self.used_jit_name = used_jit_name
        self.possible_jit_failures = possible_jit_failures

    def exception_skipping_jit(self, signature_or_function, local_skip_jit=False, **kwargs):
        if skip_jit or local_skip_jit:
            return signature_or_function
        try:
            return self.used_jit_func(signature_or_function, **kwargs)
        except (*self.possible_jit_failures,) as ex:
            #            assert type(ex) in self.possible_jit_failures
            if skip_jit_failures:
                return JITExceptionRaisingFunc(signature_or_function, ex)
            else:
                raise JITFailureException

    def __call__(self, signature_or_function=None, **other_keywords):
        jit_keywords = get_jit_keywords(used_jit_name=self.used_jit_name, **other_keywords)
        if debug and self.used_jit_name == numba_flag:
            jit_keywords["debug"] = debug
        if "skip" in other_keywords:
            jit_keywords["local_skip_jit"] = other_keywords["skip"]
        if signature_or_function is None:
            return _jit_(used_jit_=self, jit_keywords=jit_keywords)
        else:
            return self.exception_skipping_jit(signature_or_function, **jit_keywords)


if __name__ != "__main__":
    for var, sub in var_subroutine_dict.items():
        val = checked_logical_environ_val(var, default_answer=False)
        sub(val)
