from ...jit_interfaces import dint_, empty_, jit_, zeros_
from ..utils import jitclass_
from .base import (
    basic_grad_input_components,
    basic_input_components,
    cadd_simple,
    default_int,
    get_extract_final_gradient,
    get_extract_final_result,
    inp_add_simple,
    is_grad_def,
    is_red_vs_jit_copy_def,
    is_reducable_object,
)

# Rescale with various bounds.
rescaling_lvl = "rescaling"


def add_rescaling(processed_function_definition_list, inside_routine):
    extract_final_result = get_extract_final_result()

    if is_grad_def(processed_function_definition_list):
        extract_final_gradient = get_extract_final_gradient()

        @jit_
        def rescaling(
            processed_object,
            input_object,
            grad_object,
            hyperparameters,
            input_work_array,
            grad_work_array,
        ):
            nest_grad_obj = grad_object.nested_grad_object
            nsigmas = input_object.nsigmas
            inside_routine(
                processed_object,
                input_object.nested_input_object,
                nest_grad_obj,
                hyperparameters[:-nsigmas],
                input_work_array,
                grad_work_array[: -grad_object.grad_work_gap],
            )
            final_result = extract_final_result(input_object, input_work_array)
            final_gradient = extract_final_gradient(input_object, grad_object, grad_work_array)
            nnested_hyperparameters = nest_grad_obj.nhyperparameters

            for sigma_id in range(-nsigmas, 0):
                bounds = input_object.resc_bounds[sigma_id]
                sigma = hyperparameters[sigma_id]
                final_result[bounds[0] : bounds[1]] /= sigma

                grad_row = final_gradient[sigma_id]
                grad_row[:] = 0.0
                grad_row[bounds[0] : bounds[1]] = -final_result[bounds[0] : bounds[1]] / sigma

                for nested_hyp_id in range(nnested_hyperparameters):
                    final_gradient[nested_hyp_id, bounds[0] : bounds[1]] /= sigma

    else:

        @jit_
        def rescaling(processed_object, input_object, hyperparameters, input_work_array):
            nsigmas = input_object.nsigmas
            inside_routine(
                processed_object,
                input_object.nested_input_object,
                hyperparameters[:-nsigmas],
                input_work_array,
            )
            final_result = extract_final_result(input_object, input_work_array)
            for sigma_id in range(-nsigmas, 0):
                bounds = input_object.resc_bounds[sigma_id]
                final_result[bounds[0] : bounds[1]] /= hyperparameters[sigma_id]

    return rescaling


rescaling_input_components = [("resc_bounds", default_int[:, :]), ("nsigmas", default_int)]


def inp_add_rescaling(
    processed_definition_list, nested_input_component, nested_default_input_object
):
    skip_jit = is_reducable_object(processed_definition_list)
    if is_grad_def(processed_definition_list):

        @jitclass_(
            basic_grad_input_components
            + nested_input_component
            + [("grad_work_gap", default_int)],
            skip=skip_jit,
        )
        class inp_rescaling:
            def __init__(
                self,
                nested_grad_object,
                output_size=0,
                nhyperparameters=0,
                grad_work_gap=0,
                work_size=0,
            ):
                self.nested_grad_object = nested_grad_object
                self.output_size = output_size
                self.nhyperparameters = nhyperparameters
                self.grad_work_gap = grad_work_gap
                self.work_size = work_size

    else:

        @jitclass_(
            basic_input_components + nested_input_component + rescaling_input_components,
            skip=skip_jit,
        )
        class inp_rescaling:
            def __init__(self, nested_input_object, output_size=0, nsigmas=0, work_size=0):
                self.nested_input_object = nested_input_object
                self.output_size = output_size
                self.nsigmas = nsigmas
                self.resc_bounds = zeros_((nsigmas, 2), dtype=dint_)
                self.work_size = work_size

    return inp_rescaling, inp_rescaling(nested_default_input_object)


def cadd_rescaling(processed_definition_list, inside_copy, object_class):
    skip_jit = is_red_vs_jit_copy_def(processed_definition_list)
    if is_grad_def(processed_definition_list):

        @jit_(skip=skip_jit)
        def copy(input_object):
            new_nested_input_object = inside_copy(input_object.nested_grad_object)
            return object_class(
                new_nested_input_object,
                input_object.output_size,
                input_object.nhyperparameters,
                input_object.grad_work_gap,
                input_object.work_size,
            )

    else:

        @jit_(skip=skip_jit)
        def copy(input_object):
            new_nested_input_object = inside_copy(input_object.nested_input_object)
            output_size = input_object.output_size
            nsigmas = input_object.nsigmas
            work_size = input_object.work_size
            new_input_object = object_class(
                new_nested_input_object, output_size, nsigmas, work_size
            )
            new_input_object.resc_bounds[:, :] = input_object.resc_bounds[:, :]
            return new_input_object

    return copy


# Rescale with various integer powers.
power_rescaling_lvl = "power_rescaling"


def add_power_rescaling(processed_function_definition_list, inside_routine):
    extract_final_result = get_extract_final_result()

    if is_grad_def(processed_function_definition_list):
        extract_final_gradient = get_extract_final_gradient()

        @jit_
        def power_rescaling(
            processed_object,
            input_object,
            grad_object,
            hyperparameters,
            input_work_array,
            grad_work_array,
        ):
            nest_grad_obj = grad_object.nested_grad_object
            nnested_hyperparameters = nest_grad_obj.nhyperparameters
            nsigmas = input_object.nsigmas
            output_size = input_object.output_size
            inside_routine(
                processed_object,
                input_object.nested_input_object,
                nest_grad_obj,
                hyperparameters[:nnested_hyperparameters],
                input_work_array,
                grad_work_array,
            )

            grad_size = grad_object.output_size
            sigma_mults = grad_work_array[-output_size - grad_size : -grad_size]

            final_result = extract_final_result(input_object, input_work_array)
            final_gradient = extract_final_gradient(input_object, grad_object, grad_work_array)
            if nnested_hyperparameters != 0:
                final_gradient[:nnested_hyperparameters, :] = final_gradient[
                    -nnested_hyperparameters:, :
                ]

            for sigma_id in range(nsigmas):
                resc_powers = input_object.resc_powers[sigma_id]
                true_hyp_id = sigma_id + nnested_hyperparameters
                sigma = hyperparameters[true_hyp_id]
                sigma_mults[:] = 1.0
                for i in range(input_object.resc_powers.shape[1]):
                    if resc_powers[i] == 0:
                        continue
                    sigma_mults[i] = sigma ** (-resc_powers[i])
                final_result *= sigma_mults

                final_gradient[:nnested_hyperparameters, :] *= sigma_mults

            resc_size = input_object.resc_powers.shape[1]

            for sigma_id in range(nsigmas):
                resc_powers = input_object.resc_powers[sigma_id]
                true_hyp_id = sigma_id + nnested_hyperparameters
                sigma = hyperparameters[true_hyp_id]
                final_gradient[true_hyp_id, :resc_size] = (
                    -resc_powers * final_result[:resc_size] / sigma
                )
                final_gradient[true_hyp_id, resc_size:] = 0.0

    else:

        @jit_
        def power_rescaling(processed_object, input_object, hyperparameters, input_work_array):
            nsigmas = input_object.nsigmas
            nnested_hyperparameters = hyperparameters.shape[0] - nsigmas
            inside_routine(
                processed_object,
                input_object.nested_input_object,
                hyperparameters[:-nsigmas],
                input_work_array,
            )
            final_result = extract_final_result(input_object, input_work_array)
            for sigma_id in range(nsigmas):
                sigma = hyperparameters[nnested_hyperparameters + sigma_id]
                for i in range(input_object.resc_powers.shape[1]):
                    if input_object.resc_powers[sigma_id, i] != 0:
                        final_result[i] /= sigma ** input_object.resc_powers[sigma_id, i]

    return power_rescaling


power_rescaling_input_components = [("resc_powers", default_int[:, :]), ("nsigmas", default_int)]


def inp_add_power_rescaling(
    processed_definition_list, nested_input_component, nested_default_input_object
):
    if is_grad_def(processed_definition_list):
        return inp_add_simple(
            processed_definition_list, nested_input_component, nested_default_input_object
        )
    skip_jit = is_reducable_object(processed_definition_list)

    @jitclass_(
        basic_input_components + nested_input_component + power_rescaling_input_components,
        skip=skip_jit,
    )
    class inp_power_rescaling:
        def __init__(
            self, nested_input_object, output_size=0, nsigmas=0, rescaling_size=0, work_size=0
        ):
            self.nested_input_object = nested_input_object
            self.output_size = output_size
            self.nsigmas = nsigmas
            self.work_size = work_size
            self.resc_powers = empty_((nsigmas, rescaling_size), dtype=dint_)

    return inp_power_rescaling, inp_power_rescaling(nested_default_input_object)


def cadd_power_rescaling(processed_definition_list, inside_copy, object_class):
    if is_grad_def(processed_definition_list):
        return cadd_simple(processed_definition_list, inside_copy, object_class)
    skip_jit = is_red_vs_jit_copy_def(processed_definition_list)

    @jit_(skip=skip_jit)
    def copy(input_object):
        new_nested_input_object = inside_copy(input_object.nested_input_object)
        output_size = input_object.output_size
        nsigmas, rescaling_size = input_object.resc_powers.shape
        new_input_object = object_class(
            new_nested_input_object, output_size, nsigmas, rescaling_size, input_object.work_size
        )
        new_input_object.resc_powers[:, :] = input_object.resc_powers[:, :]
        return new_input_object

    return copy


# Rescale by sigma (typically used before unscaled_sorf to create "sorf")
full_rescaling_lvl = "full_rescaling"


def add_full_rescaling(processed_function_definition_list, inside_routine):
    assert not is_grad_def(processed_function_definition_list)

    @jit_
    def full_rescaling(processed_object, input_object, hyperparameters, input_work_array):
        inside_routine(processed_object, input_object, hyperparameters[:-1], input_work_array)
        input_work_array[-input_object.output_size :] /= hyperparameters[-1]

    return full_rescaling


function_level_additions = {
    rescaling_lvl: add_rescaling,
    power_rescaling_lvl: add_power_rescaling,
    full_rescaling_lvl: add_full_rescaling,
}

copy_level_additions = {rescaling_lvl: cadd_rescaling, power_rescaling_lvl: cadd_power_rescaling}

class_level_additions = {
    rescaling_lvl: inp_add_rescaling,
    power_rescaling_lvl: inp_add_power_rescaling,
}
