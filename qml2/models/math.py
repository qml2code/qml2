"""
Everything related to ensuring loss functions optimized during hyperparameter optimization are well-defined everywhere.
"""
from typing import Callable

from ..jit_interfaces import LinAlgError_, any_, inf_, isinf_, isnan_, repeat_, zeros_


class SpecialValue(Exception):
    pass


possible_numerical_exceptions = (LinAlgError_, ZeroDivisionError)


def check_special_value(func_output):
    gradient = isinstance(func_output, tuple)
    if gradient:
        val = func_output[0]
    else:
        val = func_output
    spec_val = isinf_(val) or isnan_(val)
    if spec_val or (not gradient):
        return spec_val
    grad = func_output[1]
    return any_(isinf_(grad)) or any_(isinf_(grad))


# Inverse of the function.
# Used in BOSS minimization.
def ninv_f_special_value(x, gradient=False):
    if isinstance(x, tuple):
        assert gradient
        x = x[1]
    if gradient:
        return 0.0, zeros_(x.shape)
    else:
        return 0.0


def ninv_f(func_output, gradient=True, **func_kwargs):
    if isinstance(func_output, Callable):

        def new_func(x):
            try:
                return ninv_f(func_output(x, **func_kwargs))
            except possible_numerical_exceptions:
                return ninv_f_special_value(x, gradient=gradient)

        return new_func

    gradient = isinstance(func_output, tuple)

    if check_special_value(func_output):
        return ninv_f_special_value(func_output, gradient=gradient)

    if gradient:
        val = func_output[0]
        grad = func_output[1]
    else:
        val = func_output
    ninv_val = -1.0 / val
    if not gradient:
        return ninv_val
    return ninv_val, ninv_val**2 * grad


def f_winf(func, gradient=False, **func_kwargs):
    def new_func(x):
        try:
            output = func(x, **func_kwargs)
            if check_special_value(output):
                raise SpecialValue
            return output
        except (*possible_numerical_exceptions, SpecialValue):
            pass
        val = inf_
        if not gradient:
            return val
        if x.shape == ():
            return val, inf_
        else:
            return val, repeat_(inf_, x.shape)

    return new_func
