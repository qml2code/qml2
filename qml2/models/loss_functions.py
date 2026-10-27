# K.Karan: The reason for "Rescaled" in "RescaledHuber" and "RescaledLogCosh" is to underline
# that they differ from standard definitions of Huber and LogCosh functions to conform to limiting behavior
# discussed in the mSORF paper.

import warnings

from scipy.optimize import minimize

from ..basic_utils import display_scipy_convergence
from ..jit_interfaces import (
    LinAlgError_,
    abs_,
    all_,
    any_,
    array_,
    dbool_,
    dot_,
    empty_,
    exp_,
    isinf_,
    isnan_,
    jit_,
    log_,
    logical_not_,
    mean_,
    ones_,
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
        # Deliberately NOT super().calc_wders(): MSE.calc_wders returns `self(err_vals)` as
        # its value, and `self` is this RMSE instance, so __call__ dispatched back here and
        # the square root was taken TWICE. The returned value was sqrt(RMSE) rather than
        # RMSE, and the gradient was divided by that same wrong number. Computing the mean
        # square directly is what makes the value agree with __call__ and the gradient agree
        # with finite differences.
        mean_square = mean_(err_vals**2)
        RMSE_val = sqrt_(mean_square)
        if RMSE_val <= tiny_:
            # d(RMSE)/d(err) = err / (n * RMSE) is undefined at the origin -- RMSE has a
            # cusp there, like MAE -- and dividing anyway gave inf or nan, which then
            # propagated silently into whatever consumed the gradient. Zero is the
            # subgradient at the kink, and what the self-consistent losses already return.
            return RMSE_val, zeros_(err_vals.shape)
        return RMSE_val, err_vals / (err_vals.shape[0] * RMSE_val)


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
def self_consistent_huber_sorted_search_results(err_vals, compromise_coefficient):
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

    any_geq = geq_num != 0
    geq_ratio = geq_num / tot_num
    geq_mean_abs_err = geq_sum_abs_err
    if any_geq:
        geq_mean_abs_err /= geq_num

    any_lt = geq_num != tot_num
    lt_mean_err2 = lt_sum_err2
    if any_lt:
        lt_mean_err2 /= tot_num - geq_num

    return geq_mean_abs_err, any_geq, lt_mean_err2, any_lt, geq_ratio


@jit_
def self_consistent_huber_from_precalc(
    geq_mean_abs_err, any_geq, lt_mean_err2, any_lt, geq_ratio, compromise_coefficient
):
    if not any_geq:
        return (
            sqrt_(
                lt_mean_err2
                * (1 - geq_ratio)
                / compromise_coefficient
                / (1 + compromise_coefficient * geq_ratio)
            )
            / 2
        )
    if not any_lt:
        return geq_mean_abs_err * geq_ratio / (1 + compromise_coefficient * geq_ratio)
    determinant = sqrt_(
        geq_mean_abs_err**2
        + lt_mean_err2 * (1 + 1 / geq_ratio / compromise_coefficient) * (1 - geq_ratio) / geq_ratio
    )
    return (geq_mean_abs_err + determinant) / 2 / (1 / geq_ratio + compromise_coefficient)


@jit_
def self_consistent_huber_analytic(err_vals, compromise_coefficient):
    (
        geq_mean_abs_err,
        any_geq,
        lt_mean_err2,
        any_lt,
        geq_ratio,
    ) = self_consistent_huber_sorted_search_results(err_vals, compromise_coefficient)
    return self_consistent_huber_from_precalc(
        geq_mean_abs_err, any_geq, lt_mean_err2, any_lt, geq_ratio, compromise_coefficient
    )


class SelfConsistentHuberAnalytic(SelfConsistentLoss):
    def init_calculator(self):
        self.calculator = self.get_val_calculator()
        self.der_calculator = rescaled_huber_ders
        self.calculator_args = (self.compromise_coefficient,)

    def get_val_calculator(self):
        return self_consistent_huber_analytic

    def get_der_calculator_args(self, val):
        delta_value = self.compromise_coefficient * val
        return (delta_value,)


@jit_
def self_consistent_huber(err_vals, compromise_coefficient):
    abs_errs = abs_(err_vals)
    previous_geq = ones_(abs_errs.shape, dtype=dbool_)
    current_geq = zeros_(abs_errs.shape, dtype=dbool_)
    # the first iteration can be calculated without sqrt
    current_l = mean_(abs_(err_vals)) / (1 + compromise_coefficient)
    # iterate until convergence w.r.t. geq values
    while True:
        current_divisor = 2 * compromise_coefficient * current_l
        current_geq[:] = abs_errs >= current_divisor
        any_geq = any_(current_geq)
        any_lt = any_(logical_not_(current_geq))
        geq_ratio = mean_(current_geq)
        if any_geq:
            geq_mean_abs_err = mean_(abs_errs[current_geq])
        else:
            geq_mean_abs_err = 0.0
        if any_lt:
            lt_mean_err2 = mean_(err_vals[logical_not_(current_geq)] ** 2)
        else:
            lt_mean_err2 = 0.0
        current_l = self_consistent_huber_from_precalc(
            geq_mean_abs_err, any_geq, lt_mean_err2, any_lt, geq_ratio, compromise_coefficient
        )
        if all_(current_geq == previous_geq):
            return current_l
        previous_geq[:] = current_geq[:]


class SelfConsistentHuber(SelfConsistentHuberAnalytic):
    def get_val_calculator(self):
        return self_consistent_huber


class SquaredSelfConsistentHuberAnalytic(SquaredLoss):
    def __init__(self, compromise_coefficient):
        super().__init__(SelfConsistentHuberAnalytic(compromise_coefficient))


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


# ---------------------------------------------------------------------------
# Aggregating grouped (leave-one-group-out) cross-validation residuals.
# ---------------------------------------------------------------------------

GROUPED_CV_MODES = ("micro", "macro")
SUPPORT_WEIGHT_SCHEMES = ("support", "sqrt_support")


def group_residual_indices(group_ids):
    """Member indices of each group, in order of first appearance.

    Negative labels raise rather than being dropped: in a cross-validation context every
    residual came out of some fold, so an ungrouped one means the caller lost track of the
    alignment between residuals and folds.
    """
    order = {}
    members = []
    for index, label in enumerate(group_ids):
        label = int(label)
        if label < 0:
            raise ValueError(
                f"group_ids[{index}] is {label}; every cross-validation residual must "
                "belong to a fold, so negative (ungrouped) labels are not allowed here"
            )
        if label not in order:
            order[label] = len(members)
            members.append([])
        members[order[label]].append(index)
    return members


def support_weight_vector(group_sizes, aggregation_weights):
    """Resolve ``aggregation_weights`` into one multiplier per group."""
    if aggregation_weights is None:
        return None
    if isinstance(aggregation_weights, str):
        if aggregation_weights not in SUPPORT_WEIGHT_SCHEMES:
            raise ValueError(
                f"aggregation_weights must be one of {SUPPORT_WEIGHT_SCHEMES}, an array, "
                f"or None; got {aggregation_weights!r}"
            )
        sizes = array_([float(size) for size in group_sizes])
        return sizes if aggregation_weights == "support" else sqrt_(sizes)
    weights = array_(aggregation_weights)
    if weights.shape[0] != len(group_sizes):
        raise ValueError(
            f"aggregation_weights has {weights.shape[0]} entries for " f"{len(group_sizes)} groups"
        )
    return weights


def grouped_cv_loss(
    residuals,
    group_ids,
    point_loss,
    mode="micro",
    across_group_loss=None,
    aggregation_weights=None,
):
    """Scalar objective from grouped-cross-validation residuals.

    Sits between the per-point out-of-fold residuals a grouped-CV pass produces and the
    scalar a hyperparameter optimizer minimizes, when the held-out folds are groups of
    DIFFERENT sizes (scaffolds, in practice).

    The two modes are the standard micro/macro averaging vocabulary:

    ``mode="micro"`` (default)
        Pool every residual into one vector and apply ``point_loss`` once. Weights each
        MOLECULE equally, and hence each group in proportion to its size. This is the group
        generalization of the leave-one-out objective already in use, and the
        self-consistent-scale construction applies to the pooled vector unchanged.

    ``mode="macro"``
        Compute a per-group value ``s_g = point_loss(residuals in g)`` (the FIRST-LEVEL, or
        per-point, loss), optionally scale those values, then apply ``across_group_loss``
        (the SECOND-LEVEL, or across-group, loss) to the resulting vector. Weights each
        GROUP equally regardless of size. ``across_group_loss=None`` means the plain mean
        over groups, which is textbook macro-averaging.

    These are endpoints of one weighting axis rather than unrelated options: support-weighted
    macro-averaging -- per-group values weighted by group size -- recovers micro-averaging.
    With the default across-group mean that recovery is EXACT for any ``point_loss`` that is
    a plain mean of a per-point function (MAE, MSE), and approximate otherwise (RMSE, which
    is a nonlinear function of such a mean). Verified both ways in the tests.

    Note that "first-level" and "second-level" here refer only to the two stages of this
    aggregation; they are unrelated to the two-level hyperparameter/coefficient optimization
    elsewhere in the model.

    Args:
        residuals: 1D array of out-of-fold residuals, one per data point.
        group_ids: 1D int array aligning each residual to its fold. Negative labels raise.
        point_loss: callable(residual_vector) -> scalar. In micro mode this is the ONLY
            loss applied.
        mode: "micro" or "macro".
        across_group_loss: macro only; callable over the vector of per-group values.
            Defaults to the mean.
        aggregation_weights: macro only. The "support weights" of the averaging literature.
            ``None`` (uniform), ``"support"`` (w_g = group size), ``"sqrt_support"``
            (w_g = sqrt of group size), or an explicit array, one entry per group in order
            of first appearance.

    IMPORTANT, and the two places this is easy to get silently wrong:

    * ``aggregation_weights`` multiply the per-group VALUES, outside the first-level loss.
      For a nonlinear ``point_loss``, ``w * point_loss(r)`` and ``point_loss(w * r)``
      differ, and the outside form is what support weighting means. The inside form is not
      offered here; it would need its own clearly-named parameter.
    * Small groups give noisy first-level values. A singleton's ``s_g`` is one residual --
      pure noise -- and a size-3 group's is high variance, so a macro loss over UNWEIGHTED
      per-group values trusts the least reliable estimates most. Uniform weights combined
      with a robust ``across_group_loss`` is the configuration most likely to destabilize
      hyperparameter selection; precision-style weights ("support", or "sqrt_support") are
      the principled choice, and "support" recovers micro exactly for mean-type losses.

    On the self-consistent losses as ``point_loss``: they are well defined on a
    one-element input, so macro mode neither forbids singleton groups nor falls back to a
    plain statistic for them. Checked rather than assumed -- on a single residual they stay
    finite, keep exact positive homogeneity (L(c*r) = c*L(r)), and reduce to a fixed
    multiple of |r| set by the compromise coefficient. What remains is the statistical
    caveat above, so macro mode warns when singletons are present rather than silently
    resting the objective on them.
    """
    if mode not in GROUPED_CV_MODES:
        raise ValueError(f"mode must be one of {GROUPED_CV_MODES}, got {mode!r}")
    if residuals.shape[0] != len(group_ids):
        raise ValueError(f"{residuals.shape[0]} residuals against {len(group_ids)} group labels")

    if mode == "micro":
        # Refused rather than ignored: a second-level loss or a set of support weights that
        # silently does nothing is a foot-gun, and reads as if the caller got macro
        # behaviour.
        if across_group_loss is not None or aggregation_weights is not None:
            raise ValueError(
                "across_group_loss and aggregation_weights apply to mode='macro' only; "
                "in micro mode every residual is pooled into a single point_loss call, so "
                "passing them would have no effect"
            )
        # group_ids still validated, so a misaligned call fails the same way in both modes.
        group_residual_indices(group_ids)
        return point_loss(residuals)

    members = group_residual_indices(group_ids)
    group_sizes = [len(indices) for indices in members]

    nsingletons = sum(1 for size in group_sizes if size == 1)
    if nsingletons:
        warnings.warn(
            f"macro-averaging over {len(group_sizes)} groups of which {nsingletons} hold a "
            "single point: those per-group values are single residuals, so that much of "
            "the objective rests on pure noise. Consider aggregation_weights='support' "
            "(or 'sqrt_support'), which down-weights them.",
            stacklevel=2,
        )

    group_values = array_([point_loss(residuals[indices]) for indices in members])

    weights = support_weight_vector(group_sizes, aggregation_weights)
    if weights is not None:
        # Outside the first-level loss; see the docstring.
        group_values = group_values * weights / mean_(weights)

    if across_group_loss is None:
        return mean_(group_values)
    return across_group_loss(group_values)
