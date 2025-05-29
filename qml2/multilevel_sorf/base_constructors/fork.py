from ...jit_interfaces import jit_
from ..utils import jitclass_
from .base import (
    basic_grad_input_components,
    basic_input_components,
    default_int,
    get_extract_final_gradient,
    is_grad_def,
    is_red_vs_jit_copy_def,
    is_reducable_object,
)

# Combining output of two different MSORF models.
fork_lvl = "fork"


def add_fork(processed_function_definition_list, inside_routine1, inside_routine2):
    if is_grad_def(processed_function_definition_list):
        extract_nested_final_gradient1 = get_extract_final_gradient()
        extract_nested_final_gradient2 = get_extract_final_gradient()
        extract_final_gradient = get_extract_final_gradient()

        @jit_
        def fork(
            processed_object,
            input_object,
            grad_object,
            hyperparameters,
            input_work_array,
            grad_work_array,
        ):
            hyp_separator = input_object.hyperparameter_separator
            result_separator = input_object.result_separator

            final_gradient = extract_final_gradient(input_object, grad_object, grad_work_array)

            nested_input_object2 = input_object.nested_input_object2
            nested_grad_object2 = grad_object.nested_grad_object2

            inside_routine2(
                processed_object,
                nested_input_object2,
                nested_grad_object2,
                hyperparameters[hyp_separator:],
                input_work_array,
                grad_work_array,
            )

            nested_input_object1 = input_object.nested_input_object1
            nested_grad_object1 = grad_object.nested_grad_object1

            grad_work_array1 = grad_work_array[: -nested_grad_object2.output_size]

            inside_routine1(
                processed_object,
                nested_input_object1,
                nested_grad_object1,
                hyperparameters[:hyp_separator],
                input_work_array[: -nested_input_object2.output_size],
                grad_work_array1,
            )

            nested_final_gradient1 = extract_nested_final_gradient1(
                nested_input_object1, nested_grad_object1, grad_work_array1
            )
            # final_gradient[:hyp_separator, :result_separator] = nested_final_gradient1[:, :]
            for i_hyp in range(hyp_separator):
                final_gradient[i_hyp, :result_separator] = nested_final_gradient1[i_hyp, :]

            nested_final_gradient2 = extract_nested_final_gradient2(
                nested_input_object2, nested_grad_object2, grad_work_array
            )
            # final_gradient[hyp_separator:, result_separator:] = nested_final_gradient2[:, :]
            for i_hyp in range(hyperparameters.shape[0] - hyp_separator):
                final_gradient[i_hyp + hyp_separator, result_separator:] = nested_final_gradient2[
                    i_hyp, :
                ]

            final_gradient[:hyp_separator, result_separator:] = 0.0
            final_gradient[hyp_separator:, :result_separator] = 0.0

    else:

        @jit_
        def fork(processed_object, input_object, hyperparameters, input_work_array):
            hyp_separator = input_object.hyperparameter_separator
            nested_input_object2 = input_object.nested_input_object2

            inside_routine2(
                processed_object,
                nested_input_object2,
                hyperparameters[hyp_separator:],
                input_work_array,
            )

            inside_routine1(
                processed_object,
                input_object.nested_input_object1,
                hyperparameters[:hyp_separator],
                input_work_array[: -nested_input_object2.output_size],
            )

    return fork


def inp_add_fork(
    processed_definition_lists,
    nested_input_component1,
    nested_default_input_object1,
    nested_input_component2,
    nested_default_input_object2,
):
    both_nested_input_components = nested_input_component1 + nested_input_component2
    skip_jit = is_reducable_object(processed_definition_lists[0])
    if is_grad_def(processed_definition_lists[0]):

        @jitclass_(basic_grad_input_components + both_nested_input_components, skip=skip_jit)
        class inp_fork:
            def __init__(
                self,
                nested_grad_object1,
                nested_grad_object2,
                output_size=0,
                nhyperparameters=0,
                work_size=0,
            ):
                self.nested_grad_object1 = nested_grad_object1
                self.nested_grad_object2 = nested_grad_object2
                self.output_size = output_size
                self.nhyperparameters = nhyperparameters
                self.work_size = work_size

    else:

        @jitclass_(
            basic_input_components
            + both_nested_input_components
            + [("result_separator", default_int), ("hyperparameter_separator", default_int)],
            skip=skip_jit,
        )
        class inp_fork:
            def __init__(
                self,
                nested_input_object1,
                nested_input_object2,
                output_size=0,
                hyperparameter_separator=0,
                work_size=0,
            ):
                self.nested_input_object1 = nested_input_object1
                self.nested_input_object2 = nested_input_object2
                self.result_separator = nested_input_object1.output_size
                self.output_size = output_size
                self.hyperparameter_separator = hyperparameter_separator
                self.work_size = work_size

    return inp_fork, inp_fork(nested_default_input_object1, nested_default_input_object2)


def cadd_fork(processed_definition_list, inside_copy1, inside_copy2, object_class):
    skip_jit = is_red_vs_jit_copy_def(processed_definition_list)

    if is_grad_def(processed_definition_list):

        @jit_(skip=skip_jit)
        def copy(input_object):
            new_nested_object1 = inside_copy1(input_object.nested_grad_object1)
            new_nested_object2 = inside_copy2(input_object.nested_grad_object2)
            new_object = object_class(
                new_nested_object1,
                new_nested_object2,
                input_object.output_size,
                input_object.nhyperparameters,
                input_object.work_size,
            )
            return new_object

    else:

        @jit_(skip=skip_jit)
        def copy(input_object):
            new_nested_object1 = inside_copy1(input_object.nested_input_object1)
            new_nested_object2 = inside_copy2(input_object.nested_input_object2)
            new_object = object_class(
                new_nested_object1,
                new_nested_object2,
                input_object.output_size,
                input_object.hyperparameter_separator,
                input_object.work_size,
            )
            return new_object

    return copy


def cpuest_add_fork(processed_definition_list, inside_routine1, inside_routine2):
    if is_grad_def(processed_definition_list):

        @jit_
        def cpuest_fork(processed_object, input_object, grad_object):
            return inside_routine1(
                processed_object,
                input_object.nested_input_object1,
                grad_object.nested_grad_object1,
            ) + inside_routine2(
                processed_object,
                input_object.nested_input_object2,
                grad_object.nested_grad_object2,
            )

    else:

        @jit_
        def cpuest_fork(processed_object, input_object):
            return inside_routine1(
                processed_object, input_object.nested_input_object1
            ) + inside_routine2(processed_object, input_object.nested_input_object2)

    return cpuest_fork


function_level_additions = {
    fork_lvl: add_fork,
}
copy_level_additions = {
    fork_lvl: cadd_fork,
}
class_level_additions = {fork_lvl: inp_add_fork}
cpuest_level_additions = {fork_lvl: cpuest_add_fork}
