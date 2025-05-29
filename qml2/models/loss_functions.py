# K.Karan: The reason for "Rescaled" in "RescaledHuber" and "RescaledLogCosh" is to underline
# that they differ from standard definitions of Huber and LogCosh functions to conform to limiting behavior
# discussed in the mSORF paper.

from scipy.optimize import minimize

from ..basic_utils import display_scipy_convergence
from ..jit_interfaces import (
    LinAlgError_,
    abs_,
    any_,
    dot_,
    empty_,
    exp_,
    isinf_,
    isnan_,
    jit_,
    log_,
    mean_,
    prange_,
    sign_,
    sort_,
    sqrt_,
    sum_,
    tanh_,
    tiny_,
    zeros_,
)
from ..math import lstsq_solve


# Calculate loss function with gradients as a function of errors.
class ErrorLoss:
    def __init__(self):
        pass

    def find_minimum_linear_errors(self, A, b, method="L-BFGS-B", tol=1.0e-9, initial_guess=None):
        """
        Use a gradient-based optimization method to find the minimum of
        self(Ax-b) w.r.t. x.
        """
        if initial_guess is None:
            # Use the minimum of |Ax-b|**2 as initial guess.
            initial_guess = lstsq_solve(A, b)

        def optimized_function(x):
            if any_(isnan_(x)) or any_(isinf_(x)):
                raise LinAlgError_
            errors = dot_(A, x) - b
            val, val_err_der = self.calc_wders(errors)
            return val, dot_(val_err_der, A)

        optimization_result = minimize(
            optimized_function,
            initial_guess,
            method=method,
            jac=True,
            options={"disp": display_scipy_convergence},
            tol=tol,
        )
        return optimization_result.x


@jit_
def calc_MAE(err_vals):
    return mean_(abs_(err_vals))


class MAE(ErrorLoss):
    def __call__(self, err_vals):
        return calc_MAE(err_vals)

    def calc_wders(self, err_vals):
        return self(err_vals), sign_(err_vals) / err_vals.shape[0]


class MSE(ErrorLoss):
    def __init__(self, fmle_rcond=None):
        self.fmle_rcond = fmle_rcond

    def __call__(self, err_vals):
        return mean_(err_vals**2)

    def calc_wders(self, err_vals):
        return self(err_vals), 2 * err_vals / err_vals.shape[0]

    def find_minimum_linear_errors(self, A, b, **kwargs):
        return lstsq_solve(A, b, rcond=self.fmle_rcond)


class RMSE(MSE):
    def __call__(self, err_vals):
        return sqrt_(super().__call__(err_vals))

    def calc_wders(self, err_vals):
        MSE_val, MSE_der = super().calc_wders(err_vals)
        RMSE_val = sqrt_(MSE_val)
        RMSE_der = MSE_der / 2.0 / RMSE_val
        return RMSE_val, RMSE_der


# Rescaled Huber.
@jit_
def rescaled_huber(err_vals, delta_value):
    output = 0.0
    for err_val in err_vals:
        if abs_(err_val) >= 2 * delta_value:
            output += abs_(err_val) - delta_value
        else:
            output += err_val**2 / 4 / delta_value
    return output / err_vals.shape[0]


@jit_
def rescaled_huber_ders(err_vals, delta_value):
    der = empty_(err_vals.shape)
    nerr_vals = err_vals.shape[0]
    for i in prange_(err_vals.shape[0]):
        err_val = err_vals[i]
        if abs_(err_val) >= 2 * delta_value:
            der[i] = sign_(err_val)
        else:
            der[i] = err_val / 2 / delta_value
    return der / nerr_vals


class RescaledHuber(ErrorLoss):
    def __init__(self, delta_value):
        self.delta_value = delta_value

    def __call__(self, err_vals):
        return rescaled_huber(err_vals, self.delta_value)

    def calc_wders(self, err_vals):
        return rescaled_huber(err_vals, self.delta_value), rescaled_huber_ders(
            err_vals, self.delta_value
        )


# Rescaled LogCosh
@jit_
def calc_rescaled_log_cosh(err_vals, ln2_delta_value):
    unnorm_func_vals = abs_(err_vals)
    if ln2_delta_value > tiny_:
        unnorm_func_vals += ln2_delta_value * log_(
            (1 + exp_(-2 * abs_(err_vals) / ln2_delta_value)) / 2
        )
    return mean_(unnorm_func_vals)


@jit_
def calc_rescaled_log_cosh_ders(err_vals, ln2_delta_value):
    return tanh_(err_vals / ln2_delta_value) / err_vals.shape[0]


class RescaledLogCosh(ErrorLoss):
    def __init__(self, delta_value):
        self.ln2_delta_value = delta_value / log_(2.0)

    def __call__(self, err_vals):
        return calc_rescaled_log_cosh(err_vals, self.ln2_delta_value)

    def calc_wders(self, err_vals):
        return calc_rescaled_log_cosh(err_vals, self.ln2_delta_value), calc_rescaled_log_cosh_ders(
            err_vals, self.ln2_delta_value
        )


# For calculating self-consistent error losses.
class SelfConsistentLoss(ErrorLoss):
    def __init__(self, compromise_coefficient, **kwargs):
        self.compromise_coefficient = compromise_coefficient
        self.init_calculator(**kwargs)

    def init_calculator(self):
        self.calculator = None
        self.calculator_args = None
        self.der_calculator = None

    def get_der_calculator_args(self, val):
        pass

    def __call__(self, err_vals):
        return self.calculator(err_vals, *self.calculator_args)

    def calc_wders(self, err_vals):
        loss = self(err_vals)
        if loss <= tiny_:
            return loss, zeros_(err_vals.shape)
        der_calculator_args = self.get_der_calculator_args(loss)
        ders = self.der_calculator(err_vals, *der_calculator_args)
        ders *= loss / dot_(err_vals, ders)
        return loss, ders


class SquaredLoss(ErrorLoss):
    def __init__(self, base_loss_function: ErrorLoss):
        self.base_loss_function = base_loss_function

    def __call__(self, err_vals):
        return self.base_loss_function.__call__(err_vals) ** 2

    def calc_wders(self, err_vals):
        loss, loss_grad = self.base_loss_function.calc_wders(err_vals)
        return loss**2, loss_grad * loss * 2


# SC Huber.
@jit_
def self_consistent_huber_rough_search_results(err_vals, compromise_coefficient):
    sorted_abs_errs = sort_(abs_(err_vals))
    geq_sum_abs_err = sum_(sorted_abs_errs)
    lt_sum_err2 = 0.0
    tot_num = err_vals.shape[0]
    geq_num = tot_num
    prev_eq_val = -geq_sum_abs_err

    prev_eq_val_positive = prev_eq_val > 0.0

    for abs_err in sorted_abs_errs:
        new_geq_num = geq_num - 1
        new_lt_sum_err2 = lt_sum_err2 + abs_err**2
        new_geq_sum_abs_err = geq_sum_abs_err - abs_err
        # division by 2 appears because the divider between sum_{<} and sum_{\geq} is 2*delta=2*compromise_coefficient*L
        corresponding_loss_val = abs_err / compromise_coefficient / 2
        new_eq_val = (
            corresponding_loss_val**2 * (tot_num + compromise_coefficient * new_geq_num)
            - corresponding_loss_val * new_geq_sum_abs_err
            - new_lt_sum_err2 / 4 / compromise_coefficient
        )
        new_eq_val_positive = new_eq_val > 0.0
        if new_eq_val_positive != prev_eq_val_positive:
            break
        geq_num = new_geq_num
        lt_sum_err2 = new_lt_sum_err2
        geq_sum_abs_err = new_geq_sum_abs_err

    determinant = sqrt_(
        geq_sum_abs_err**2 + lt_sum_err2 * (geq_num + tot_num / compromise_coefficient)
    )

    return geq_sum_abs_err, geq_num, determinant


@jit_
def self_consistent_huber_from_precalc(
    geq_sum_abs_err, determinant, compromise_coefficient, geq_num, tot_num
):
    return (geq_sum_abs_err + determinant) / 2 / (tot_num + compromise_coefficient * geq_num)


@jit_
def self_consistent_huber(err_vals, compromise_coefficient):
    geq_sum_abs_err, geq_num, determinant = self_consistent_huber_rough_search_results(
        err_vals, compromise_coefficient
    )
    tot_num = err_vals.shape[0]
    return self_consistent_huber_from_precalc(
        geq_sum_abs_err, determinant, compromise_coefficient, geq_num, tot_num
    )


class SelfConsistentHuber(SelfConsistentLoss):
    def init_calculator(self):
        self.calculator = self_consistent_huber
        self.der_calculator = rescaled_huber_ders
        self.calculator_args = (self.compromise_coefficient,)

    def get_der_calculator_args(self, val):
        delta_value = self.compromise_coefficient * val
        return (delta_value,)


class SquaredSelfConsistentHuber(SquaredLoss):
    def __init__(self, compromise_coefficient):
        super().__init__(SelfConsistentHuber(compromise_coefficient))


# SCLogCosh
@jit_
def self_consistent_log_cosh_newton_val_der(x, err_vals, ln2_compromise_coefficient):
    """
    Value and derivative of the RHS of the equation solved by Newton method in the SSC LogCosh.
    """
    assert x >= 0.0
    if x <= tiny_:
        # the ->0 limit is available analytically
        return calc_MAE(err_vals), -(1 + ln2_compromise_coefficient * log_(2.0))
    new_ln2_delta_value = ln2_compromise_coefficient * x
    val = calc_rescaled_log_cosh(err_vals, new_ln2_delta_value) - x
    err_der = calc_rescaled_log_cosh_ders(err_vals, new_ln2_delta_value)
    newton_der = (val - dot_(err_vals, err_der)) / x
    return val, newton_der


@jit_
def self_consistent_log_cosh_newton_calculation(
    err_vals, ln2_compromise_coefficient, newton_relative_convergence
):
    # start at l=0.
    solution = 0.0
    while True:
        val, der = self_consistent_log_cosh_newton_val_der(
            solution, err_vals, ln2_compromise_coefficient
        )
        new_solution = solution - val / der
        if new_solution <= tiny_ or new_solution <= solution:
            # in the first case err_vals are all almost zero
            # in the second case we must've reached machine precision
            return solution
        rel_err = abs_(new_solution - solution) / new_solution
        if rel_err < newton_relative_convergence:
            return new_solution
        solution = new_solution


class SelfConsistentLogCosh(SelfConsistentLoss):
    def init_calculator(self, newton_relative_convergence=tiny_):
        self.ln2_compromise_coefficient = self.compromise_coefficient / log_(2.0)
        self.newton_relative_convergence = newton_relative_convergence
        self.calculator = self_consistent_log_cosh_newton_calculation
        self.der_calculator = calc_rescaled_log_cosh_ders
        self.calculator_args = (self.ln2_compromise_coefficient, self.newton_relative_convergence)

    def get_der_calculator_args(self, val):
        return (val * self.ln2_compromise_coefficient,)


class SquaredSelfConsistentLogCosh(SquaredLoss):
    def __init__(self, compromise_coefficient, newton_relative_convergence=tiny_):
        super().__init__(
            SelfConsistentLogCosh(
                compromise_coefficient, newton_relative_convergence=newton_relative_convergence
            )
        )
