import importlib

import numpy as np
from conftest import add_checksum_to_dict, compare_or_create, int2rng, str2rng

from qml2.jit_interfaces import empty_

loss_functions_dict = importlib.import_module("qml2.models.loss_functions").__dict__

nerror_instances = 16
nerrors = 256


def get_error_vectors(rng):
    return rng.random((nerror_instances, nerrors)) - 0.5


def get_delta(error_vectors, compromise_coeff):
    return np.mean(np.abs(error_vectors)) * compromise_coeff


def get_several_loss_func_vals(loss_function, error_vectors, grad=False):
    val_arr = empty_(error_vectors.shape[0])
    if grad:
        grad_arr = empty_(error_vectors.shape)
    for error_vector_id, error_vector in enumerate(error_vectors):
        if grad:
            val, val_grad = loss_function.calc_wders(error_vector)
        else:
            val = loss_function(error_vector)
        val_arr[error_vector_id] = val
        if grad:
            grad_arr[error_vector_id, :] = val_grad
    if grad:
        return val_arr, grad_arr
    else:
        return val_arr


def generate_loss_func_wargs_error_vectors(seed):
    error_rng = int2rng(seed)
    error_vectors = get_error_vectors(error_rng)
    compromise_coeff = 0.25
    delta = get_delta(error_vectors, compromise_coeff)

    standard_args = ()

    smooth_args = (delta,)

    self_consistent_args = (compromise_coeff,)

    loss_funcs_wargs = [
        ("MAE", standard_args),
        ("MSE", standard_args),
        ("RMSE", standard_args),
        ("RescaledHuber", smooth_args),
        ("RescaledLogCosh", smooth_args),
        ("SelfConsistentHuber", self_consistent_args),
        ("SelfConsistentHuberAnalytic", self_consistent_args),
        ("SelfConsistentLogCosh", self_consistent_args),
        ("SquaredSelfConsistentHuber", self_consistent_args),
        ("SquaredSelfConsistentHuberAnalytic", self_consistent_args),
        ("SquaredSelfConsistentLogCosh", self_consistent_args),
    ]
    return loss_funcs_wargs, error_vectors


equivalent_losses = {
    "SquaredSelfConsistentHuberAnalytic": "SquaredSelfConsistentHuber",
    "SelfConsistentHuberAnalytic": "SelfConsistentHuber",
}


def run_loss_functions_test(grad=False):
    loss_funcs_wargs, error_vectors = generate_loss_func_wargs_error_vectors(1)
    all_passed = True
    val_checksums = {}
    grad_checksums = {}
    for loss_func_name, loss_func_args in loss_funcs_wargs:
        loss_func = loss_functions_dict[loss_func_name](*loss_func_args)
        if loss_func_name in equivalent_losses:
            true_loss_func_name = equivalent_losses[loss_func_name]
        else:
            true_loss_func_name = loss_func_name
        val_checksum_rng = str2rng(true_loss_func_name)
        if grad:
            loss_func_vals, loss_func_grads = get_several_loss_func_vals(
                loss_func, error_vectors, grad=grad
            )
            grad_checksum_rng = str2rng(true_loss_func_name)
            all_passed = add_checksum_to_dict(
                grad_checksums,
                true_loss_func_name,
                loss_func_grads,
                grad_checksum_rng,
                nstack_checksums=8,
                stacks=4,
                max_rel_difference=1.0e-9,
                compared_name=loss_func_name,
                starting_all_passed=all_passed,
            )
        else:
            loss_func_vals = get_several_loss_func_vals(loss_func, error_vectors)
        all_passed = add_checksum_to_dict(
            val_checksums,
            true_loss_func_name,
            loss_func_vals,
            val_checksum_rng,
            nstack_checksums=8,
            stacks=1,
            max_rel_difference=1.0e-10,
            compared_name=loss_func_name,
            starting_all_passed=all_passed,
        )
    compare_or_create(
        val_checksums, "loss_functions", max_rel_difference=1.0e-10, starting_all_passed=all_passed
    )
    if grad:
        compare_or_create(
            grad_checksums,
            "loss_function_gradients",
            max_rel_difference=1.0e-9,
            starting_all_passed=all_passed,
        )


def test_loss_functions():
    run_loss_functions_test(grad=False)


def test_loss_function_gradients():
    run_loss_functions_test(grad=True)


def test_loss_function_linear_error_minimization():
    loss_funcs_wargs, _ = generate_loss_func_wargs_error_vectors(1)
    # we do not consider MAE here
    loss_funcs_wargs = loss_funcs_wargs[1:]
    matrix_rng = int2rng(1)
    ndim = 8
    nprobs = 8
    As = matrix_rng.random((nprobs, nerrors, ndim))
    bs = matrix_rng.random((nprobs, nerrors))
    tol = 1.0e-9
    minima_locations = empty_((nprobs, ndim))
    minima_checksums = {}
    all_passed = True
    for loss_func_name, loss_func_args in loss_funcs_wargs:
        if loss_func_name in equivalent_losses:
            true_loss_func_name = equivalent_losses[loss_func_name]
        else:
            true_loss_func_name = loss_func_name
        minima_checksum_rng = str2rng(true_loss_func_name)
        loss_func = loss_functions_dict[loss_func_name](*loss_func_args)
        for prob_id, (A, b) in enumerate(zip(As, bs)):
            minima_locations[prob_id, :] = loss_func.find_minimum_linear_errors(A, b, tol=tol)
        all_passed = add_checksum_to_dict(
            minima_checksums,
            true_loss_func_name,
            minima_locations,
            minima_checksum_rng,
            nstack_checksums=4,
            stacks=4,
            max_rel_difference=1.0e-9,
            compared_name=loss_func_name,
            starting_all_passed=all_passed,
        )
    compare_or_create(
        minima_checksums,
        "loss_function_linear_error_minimization",
        max_rel_difference=1.0e-9,
        starting_all_passed=all_passed,
    )


# Behaviour on inputs the grouped-CV aggregation layer produces: a macro-averaged objective
# calls the loss once per group, so groups of one or two residuals reach it routinely.

single_error_losses = {
    "MAE": (loss_functions_dict["MAE"], ()),
    "MSE": (loss_functions_dict["MSE"], ()),
    "RMSE": (loss_functions_dict["RMSE"], ()),
    "RescaledHuber": (loss_functions_dict["RescaledHuber"], (1.0,)),
    "RescaledLogCosh": (loss_functions_dict["RescaledLogCosh"], (1.0,)),
    "SelfConsistentHuber": (loss_functions_dict["SelfConsistentHuber"], (0.5,)),
    "SelfConsistentHuberAnalytic": (loss_functions_dict["SelfConsistentHuberAnalytic"], (0.5,)),
    "SelfConsistentLogCosh": (loss_functions_dict["SelfConsistentLogCosh"], (0.5,)),
    "SquaredSelfConsistentHuber": (loss_functions_dict["SquaredSelfConsistentHuber"], (0.5,)),
    "SquaredSelfConsistentLogCosh": (loss_functions_dict["SquaredSelfConsistentLogCosh"], (0.5,)),
}


def instantiated_losses():
    return {name: cls(*args) for name, (cls, args) in single_error_losses.items()}


def test_losses_on_a_single_error():
    """Every loss must stay finite on one residual, value and gradient alike."""
    for name, loss_function in instantiated_losses().items():
        for errors in (np.array([0.7]), np.array([-2.5])):
            value = loss_function(errors)
            gradient_value, gradient = loss_function.calc_wders(errors)
            assert np.isfinite(value), name
            assert np.all(np.isfinite(gradient)), name
            assert np.isclose(value, gradient_value, rtol=0.0, atol=1e-14), name


def test_losses_on_zero_residuals():
    """Zero residuals are a perfect fit, not a division by zero.

    Regression: RMSE's gradient is err / (n * RMSE), which returned inf or nan at the
    origin and propagated silently into whatever consumed it.
    """
    for name, loss_function in instantiated_losses().items():
        for errors in (np.zeros(1), np.zeros(4)):
            value, gradient = loss_function.calc_wders(errors)
            assert np.isfinite(value) and abs(value) < 1.0e-30, name
            assert np.all(np.isfinite(gradient)), name


def test_loss_values_agree_with_their_gradient_calls():
    """calc_wders must report the same value as __call__.

    Regression: RMSE.calc_wders delegated to MSE.calc_wders, which returns `self(err_vals)`
    -- and `self` is the RMSE instance, so __call__ dispatched back and the square root was
    taken TWICE. The value was sqrt(RMSE) and the gradient was divided by it.
    """
    rng = int2rng(11)
    errors = rng.normal(size=6)
    for name, loss_function in instantiated_losses().items():
        value, _ = loss_function.calc_wders(errors)
        assert np.isclose(value, loss_function(errors), rtol=0.0, atol=1e-14), name


def test_loss_gradients_against_finite_differences():
    """Independent of the stored-checksum gradient test: catches a gradient that is
    self-consistently wrong, as RMSE's was."""
    rng = int2rng(12)
    errors = rng.normal(size=5)
    step = 1.0e-6
    for name, loss_function in instantiated_losses().items():
        _, gradient = loss_function.calc_wders(errors)
        numeric = np.array(
            [
                (
                    loss_function(errors + np.eye(errors.shape[0])[i] * step)
                    - loss_function(errors - np.eye(errors.shape[0])[i] * step)
                )
                / (2 * step)
                for i in range(errors.shape[0])
            ]
        )
        assert np.allclose(gradient, numeric, rtol=1e-5, atol=1e-7), name


if __name__ == "__main__":
    test_loss_functions()
    test_loss_function_gradients()
    test_loss_function_linear_error_minimization()
    test_losses_on_a_single_error()
    test_losses_on_zero_residuals()
    test_loss_values_agree_with_their_gradient_calls()
    test_loss_gradients_against_finite_differences()
