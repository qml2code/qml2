from ...jit_interfaces import jit_
from .base import get_separate_grad_work_arr, get_separate_input_work_arr, is_grad_def

# Addition of different components in list.
component_sum_lvl = "component_sum"


def add_component_sum(processed_function_definition_list, inside_routine):
    separate_input_work_arr = get_separate_input_work_arr()

    if is_grad_def(processed_function_definition_list):
        separate_grad_work_arr = get_separate_grad_work_arr()

        @jit_
        def component_sum(
            processed_object,
            input_object,
            grad_object,
            hyperparameters,
            input_work_array,
            grad_work_array,
        ):
            nested_input_work_array, final_result, nested_final_result = separate_input_work_arr(
                input_object, input_work_array
            )
            nested_grad_work_array, final_gradient, nested_final_gradient = separate_grad_work_arr(
                input_object, grad_object, grad_work_array
            )

            final_result[:] = 0.0
            final_gradient[:] = 0.0

            for component in processed_object:
                inside_routine(
                    component,
                    input_object.nested_input_object,
                    grad_object.nested_grad_object,
                    hyperparameters,
                    nested_input_work_array,
                    nested_grad_work_array,
                )
                final_result[:] += nested_final_result[:]
                final_gradient[:, :] += nested_final_gradient[:, :]

    else:

        @jit_
        def component_sum(processed_object, input_object, hyperparameters, input_work_array):
            nested_input_work_array, final_result, nested_final_result = separate_input_work_arr(
                input_object, input_work_array
            )
            final_result[:] = 0.0
            for component in processed_object:
                inside_routine(
                    component,
                    input_object.nested_input_object,
                    hyperparameters,
                    nested_input_work_array,
                )
                final_result[:] += nested_final_result

    return component_sum


def cpuest_add_component_sum(processed_function_definition_list, inside_routine):
    if is_grad_def(processed_function_definition_list):

        @jit_
        def component_sum(processed_object, input_object, grad_object):
            output = 0.0
            for component in processed_object:
                output += inside_routine(
                    component, input_object.nested_input_object, grad_object.nested_grad_object
                )
            return output

    else:

        @jit_
        def component_sum(processed_object, input_object):
            output = 0.0
            for component in processed_object:
                output += inside_routine(component, input_object.nested_input_object)
            return output

    return component_sum


# Addition of component in a list augmented with weight values.
weighted_component_sum_lvl = "weighted_component_sum"


def add_weighted_component_sum(processed_function_definition_list, inside_routine):
    separate_input_work_arr = get_separate_input_work_arr()

    if is_grad_def(processed_function_definition_list):
        separate_grad_work_arr = get_separate_grad_work_arr()

        @jit_
        def weighted_component_sum(
            processed_object,
            input_object,
            grad_object,
            hyperparameters,
            input_work_array,
            grad_work_array,
        ):
            nested_input_work_array, final_result, nested_final_result = separate_input_work_arr(
                input_object, input_work_array
            )
            nested_grad_work_array, final_gradient, nested_final_gradient = separate_grad_work_arr(
                input_object, grad_object, grad_work_array
            )

            final_result[:] = 0.0
            final_gradient[:, :] = 0.0

            component_id = 0
            for component in processed_object.components:
                inside_routine(
                    component,
                    input_object.nested_input_object,
                    grad_object.nested_grad_object,
                    hyperparameters,
                    nested_input_work_array,
                    nested_grad_work_array,
                )
                rho = processed_object.rhos[component_id]
                final_result[:] += rho * nested_final_result[:]
                final_gradient[:, :] += rho * nested_final_gradient[:, :]
                component_id += 1

    else:

        @jit_
        def weighted_component_sum(
            processed_object, input_object, hyperparameters, input_work_array
        ):
            nested_input_work_array, final_result, nested_final_result = separate_input_work_arr(
                input_object, input_work_array
            )
            final_result[:] = 0.0

            component_id = 0
            for component in processed_object.components:
                inside_routine(
                    component,
                    input_object.nested_input_object,
                    hyperparameters,
                    nested_input_work_array,
                )
                final_result[:] += processed_object.rhos[component_id] * nested_final_result[:]
                component_id += 1

    return weighted_component_sum


def cpuest_add_weighted_component_sum(processed_function_definition_list, inside_routine):
    if is_grad_def(processed_function_definition_list):

        @jit_
        def weighted_component_sum(processed_object, input_object, grad_object):
            output = 0.0
            for component in processed_object.components:
                output += inside_routine(
                    component, input_object.nested_input_object, grad_object.nested_grad_object
                )
            return output

    else:

        @jit_
        def weighted_component_sum(processed_object, input_object):
            output = 0.0
            for component in processed_object.components:
                output += inside_routine(component, input_object.nested_input_object)
            return output

    return weighted_component_sum


function_level_additions = {
    component_sum_lvl: add_component_sum,
    weighted_component_sum_lvl: add_weighted_component_sum,
}
cpuest_level_additions = {
    component_sum_lvl: cpuest_add_component_sum,
    weighted_component_sum_lvl: cpuest_add_weighted_component_sum,
}
