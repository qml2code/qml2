from numba import float64, int64

from ...jit_interfaces import copy_, empty_, jit_, reshape_
from ..base_functions import extract_row_from_1D
from ..utils import jitclass_

# Naming convention
datatype_prefix = "dt"
input_object_prefix = "inp"
function_prefix = "f"


reducable_prefix = "red"
red_datatype_prefix = reducable_prefix + datatype_prefix
red_input_object_prefix = reducable_prefix + input_object_prefix

gradient_prefix = "g"
gradient_input_object_prefix = gradient_prefix + input_object_prefix
gradient_function_prefix = gradient_prefix + function_prefix

gradient_red_input_object_prefix = gradient_prefix + red_input_object_prefix
copy_prefix = "copy"
gradient_copy_prefix = gradient_prefix + copy_prefix
rcopy_prefix = "rcopy"
gradient_rcopy_prefix = gradient_prefix + rcopy_prefix
jcopy_prefix = "jcopy"
gradient_jcopy_prefix = gradient_prefix + jcopy_prefix

cpuest_prefix = "cpuest"
gradient_cpuest_prefix = gradient_prefix + "cpuest"


# Base datatype building blocks
default_float = float64
default_int = int64

array_1D_ = default_float[:]

array_2D_ = default_float[:, :]

array_3D_ = default_float[:, :, :]


def default_array(ndim):
    return empty_(tuple(1 for _ in range(ndim)))


default_class_instances = {}
for dim in range(1, 4):
    ending = str(dim) + "D_"
    # default array instances
    for pref in [datatype_prefix, red_datatype_prefix]:
        datatype_name = pref + "_array_" + ending
        globals()[datatype_name] = globals()["array_" + ending]
        default_class_instances[datatype_name] = default_array(dim)
    # default copying procedures
    for copy_pref in ["", "r", "j"]:
        globals()[copy_pref + "copy_array_" + ending] = copy_


# Naming convention
def is_grad_def(processed_function_definition_list):
    return processed_function_definition_list[0] in [
        gradient_input_object_prefix,
        gradient_function_prefix,
        gradient_copy_prefix,
        gradient_red_input_object_prefix,
        gradient_rcopy_prefix,
        gradient_jcopy_prefix,
        gradient_cpuest_prefix,
    ]


def is_copy_def(processed_function_definition_list):
    if isinstance(processed_function_definition_list, tuple):
        return is_copy_def(processed_function_definition_list[0])
    return processed_function_definition_list[0] in [
        copy_prefix,
        gradient_copy_prefix,
        rcopy_prefix,
        gradient_rcopy_prefix,
        jcopy_prefix,
        gradient_jcopy_prefix,
    ]


def is_cpuest_def(processed_function_definition_list):
    if isinstance(processed_function_definition_list, tuple):
        return is_copy_def(processed_function_definition_list[0])
    return processed_function_definition_list[0] in [cpuest_prefix, gradient_cpuest_prefix]


def is_reducable_object(processed_function_definition_list):
    return processed_function_definition_list[0] in [
        red_datatype_prefix,
        red_input_object_prefix,
        gradient_red_input_object_prefix,
    ]


def is_red_vs_jit_copy_def(processed_function_definition_list):
    return processed_function_definition_list[0] in [
        rcopy_prefix,
        gradient_rcopy_prefix,
        jcopy_prefix,
        gradient_jcopy_prefix,
    ]


def is_red2jit_copy_def(processed_function_definition_list):
    return processed_function_definition_list[0] in [rcopy_prefix, gradient_rcopy_prefix]


def is_data_definition(processed_function_definition_list):
    for comp in ["array_1D", "array_2D", "rhos", "list"]:
        if comp in processed_function_definition_list:
            return True
    return False


def change_nested_component_name(nested_component_def, new_name):
    return [(new_name, nested_component_def[0][1])]


# Shorthands for extracting final results and final gradients.
def get_extract_final_result():
    @jit_
    def extract_final_result(input_object, input_work_array):
        return input_work_array[-input_object.output_size :]

    return extract_final_result


def get_extract_final_gradient():
    @jit_
    def extract_final_gradient(input_object, grad_object, grad_work_array):
        if grad_object.nhyperparameters != 0:
            return reshape_(
                grad_work_array[-grad_object.output_size :],
                (grad_object.nhyperparameters, input_object.output_size),
            )
        else:
            return empty_((0, input_object.output_size))

    return extract_final_gradient


def get_separate_input_work_arr():
    extract_final = get_extract_final_result()
    extract_nested = get_extract_final_result()

    @jit_
    def separate_input_work_arr(input_object, input_work_array):
        nested_input_work_array = input_work_array[: -input_object.output_size]
        final_result = extract_final(input_object, input_work_array)
        nested_final_result = extract_nested(
            input_object.nested_input_object, nested_input_work_array
        )
        return nested_input_work_array, final_result, nested_final_result

    return separate_input_work_arr


def get_separate_grad_work_arr():
    extract_final = get_extract_final_gradient()
    extract_nested = get_extract_final_gradient()

    @jit_
    def separate_grad_work_arr(input_object, grad_object, grad_work_array):
        nested_grad_work_array = grad_work_array[: -grad_object.output_size]
        final_gradient = extract_final(input_object, grad_object, grad_work_array)
        nested_final_gradient = extract_nested(
            input_object.nested_input_object,
            grad_object.nested_grad_object,
            nested_grad_work_array,
        )
        return nested_grad_work_array, final_gradient, nested_final_gradient

    return separate_grad_work_arr


basic_input_components = [("output_size", default_int), ("work_size", default_int)]
basic_grad_input_components = basic_input_components + [("nhyperparameters", default_int)]


# No extra items are added during nesting.
def inp_add_simple(processed_definition_list, nested_input_component, nested_default_input_object):
    skip_jit = is_reducable_object(processed_definition_list)
    if is_grad_def(processed_definition_list):

        @jitclass_(basic_grad_input_components + nested_input_component, skip=skip_jit)
        class simple:
            def __init__(self, nested_grad_object, output_size=0, nhyperparameters=0, work_size=0):
                self.nested_grad_object = nested_grad_object
                self.output_size = output_size
                self.nhyperparameters = nhyperparameters
                self.work_size = work_size

    else:

        @jitclass_(basic_input_components + nested_input_component, skip=skip_jit)
        class simple:
            def __init__(self, nested_input_object, output_size=0, work_size=0):
                self.nested_input_object = nested_input_object
                self.output_size = output_size
                self.work_size = work_size

    return simple, simple(nested_default_input_object)


def cadd_simple(processed_definition_list, inside_copy, object_class):
    skip_jit = is_red_vs_jit_copy_def(processed_definition_list)
    if is_grad_def(processed_definition_list):

        @jit_(skip=skip_jit)
        def get_copy(other_object):
            new_nested_object = inside_copy(other_object.nested_grad_object)
            return object_class(
                new_nested_object,
                other_object.output_size,
                other_object.nhyperparameters,
                other_object.work_size,
            )

    else:

        @jit_(skip=skip_jit)
        def get_copy(other_object):
            new_nested_object = inside_copy(other_object.nested_input_object)
            return object_class(
                new_nested_object, other_object.output_size, other_object.work_size
            )

    return get_copy


def cpuest_add_simple(processed_definition_list, routine):
    if is_grad_def(processed_definition_list):

        @jit_
        def cpuest(processed_object, input_object, grad_object):
            return routine(
                processed_object, input_object.nested_input_object, grad_object.nested_grad_object
            )

    else:

        @jit_
        def cpuest(processed_object, input_object):
            return routine(processed_object, input_object.nested_input_object)

    return cpuest


def cpuest_add_single(processed_definition_list, routine):
    if is_grad_def(processed_definition_list):

        @jit_
        def cpuest(processed_object, input_object, grad_object):
            return (
                routine(
                    processed_object,
                    input_object.nested_input_object,
                    grad_object.nested_grad_object,
                )
                + grad_object.nhyperparameters
                + 1
            )

    else:

        @jit_
        def cpuest(processed_object, input_object):
            return routine(processed_object, input_object.nested_input_object) + 1

    return cpuest


# for adding resize in the middle. Memory usage can be optimized if necessary.
def add_resize(processed_function_definition_list, inside_routine):
    if is_grad_def(processed_function_definition_list):

        @jit_
        def resize(
            processed_object,
            input_object,
            grad_object,
            hyperparameters,
            input_work_array,
            grad_work_array,
        ):
            nest_inp_obj = input_object.nested_input_object
            nest_grad_obj = grad_object.nested_grad_object
            output_size = input_object.output_size
            output_size_gap = output_size - nest_inp_obj.output_size
            grad_size_gap = grad_object.output_size - nest_grad_obj.output_size
            inside_routine(
                processed_object,
                nest_inp_obj,
                nest_grad_obj,
                hyperparameters,
                input_work_array[:-output_size_gap],
                grad_work_array[:-grad_size_gap],
            )
            input_work_array[-output_size_gap:] = 0.0

            for hyp_id in range(grad_object.nhyperparameters - 1, -1, -1):
                old_storage = extract_row_from_1D(
                    grad_work_array[:-grad_size_gap],
                    hyp_id,
                    nest_inp_obj.output_size,
                    nest_grad_obj.nhyperparameters,
                )
                new_storage = extract_row_from_1D(
                    grad_work_array, hyp_id, output_size, grad_object.nhyperparameters
                )
                new_storage[: nest_inp_obj.output_size] = old_storage[:]
                new_storage[nest_inp_obj.output_size :] = 0.0

    else:

        @jit_
        def resize(processed_object, input_object, hyperparameters, input_work_array):
            nest_inp_obj = input_object.nested_input_object
            output_size_gap = input_object.output_size - nest_inp_obj.output_size
            inside_routine(
                processed_object,
                nest_inp_obj,
                hyperparameters,
                input_work_array[:-output_size_gap],
            )
            input_work_array[-output_size_gap:] = 0.0

    return resize
