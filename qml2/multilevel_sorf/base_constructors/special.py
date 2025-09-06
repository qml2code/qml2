from numba import typed

from ...jit_interfaces import copy_, dot_, empty_, jit_, l2_norm_
from ..base_functions import compress, extract_row_from_1D, inplace_normalization_wgrad
from ..utils import jitclass_
from .base import (
    array_2D_,
    basic_grad_input_components,
    basic_input_components,
    cadd_simple,
    cpuest_add_single,
    default_int,
    get_extract_final_gradient,
    get_extract_final_result,
    inp_add_simple,
    is_grad_def,
    is_red_vs_jit_copy_def,
    is_reducable_object,
)
from .datatypes import cadd_list, create_list_of

# Resize to fit a 2**int buffer.
resize_lvl = "resize"


@jit_
def f_resize_(processed_object, input_object, hyperparameters, input_work_array):
    input_size = processed_object.shape[0]
    work_size = input_work_array.shape[0]
    output_start_id = work_size - input_object.output_size
    output_finish_id = output_start_id + input_size
    input_work_array[output_start_id:output_finish_id] = processed_object[:]
    input_work_array[output_finish_id:] = 0.0


@jit_
def gf_resize_(
    processed_object, input_object, grad_object, hyperparameters, input_work_array, grad_work_array
):
    f_resize_(processed_object, input_object, hyperparameters, input_work_array)


class redinp_resize_:
    def __init__(self, output_size=0, work_size=0):
        self.output_size = output_size
        self.work_size = work_size


# NOTE K.Karan.: writing `inp_resize_ = jitclass_(basic_input_components)(redinp_resize_)` seems to mess up picklability of redinp_resize_.
@jitclass_(basic_input_components)
class inp_resize_(redinp_resize_):
    pass


class gredinp_resize_:
    def __init__(self, output_size=0, nhyperparameters=0, work_size=0):
        self.output_size = output_size
        self.nhyperparameters = nhyperparameters
        self.work_size = work_size


@jitclass_(basic_grad_input_components)
class ginp_resize_(gredinp_resize_):
    pass


def jcopy_resize_(inp_object):
    return inp_resize_(inp_object.output_size, inp_object.work_size)


def rcopy_resize_(inp_object):
    return redinp_resize_(inp_object.output_size, inp_object.work_size)


copy_resize_ = jit_(jcopy_resize_)


def gjcopy_resize_(ginp_object):
    return ginp_resize_(
        ginp_object.output_size, ginp_object.nhyperparameters, ginp_object.work_size
    )


def grcopy_resize_(ginp_object):
    return gredinp_resize_(
        ginp_object.output_size, ginp_object.nhyperparameters, ginp_object.work_size
    )


gcopy_resize_ = jit_(gjcopy_resize_)


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


@jit_
def cpuest_resize_(processed_object, input_object):
    return 0.0


@jit_
def gcpuest_resize_(processed_object, input_object, grad_object):
    return 0.0


# Projecting on principal components before resizing.
project_resize_lvl = "project_resize"


@jit_
def f_project_resize_(processed_object, input_object, hyperparameters, input_work_array):
    reductor = input_object.reductor
    projected_vector_size = reductor.shape[1]
    work_size = input_work_array.shape[0]

    output_start_id = work_size - input_object.output_size
    output_finish_id = output_start_id + projected_vector_size

    input_work_array[output_start_id:output_finish_id] = dot_(reductor.T, processed_object[:])
    input_work_array[output_finish_id:] = 0.0


@jit_
def gf_project_resize_(
    processed_object, input_object, grad_object, hyperparameters, input_work_array, grad_work_array
):
    f_project_resize_(processed_object, input_object, hyperparameters, input_work_array)


class redinp_project_resize_:
    def __init__(self, output_size=0, work_size=0, shape=(1, 1), reductor=None):
        self.output_size = output_size
        self.work_size = work_size
        if reductor is None:
            self.reductor = empty_(shape)
        else:
            self.reductor = reductor


# NOTE K.Karan.: writing `inp_resize_ = jitclass_(basic_input_components)(redinp_resize_)` seems to mess up picklability of redinp_resize_.
@jitclass_(basic_input_components + [("reductor", array_2D_)])
class inp_project_resize_(redinp_project_resize_):
    pass


class gredinp_project_resize_:
    def __init__(self, output_size=0, nhyperparameters=0, work_size=0):
        self.output_size = output_size
        self.nhyperparameters = nhyperparameters
        self.work_size = work_size


@jitclass_(basic_grad_input_components)
class ginp_project_resize_(gredinp_resize_):
    pass


def jcopy_project_resize_(inp_object):
    return inp_project_resize_(
        inp_object.output_size,
        inp_object.work_size,
        inp_object.reductor.shape,
        copy_(inp_object.reductor),
    )


def rcopy_project_resize_(inp_object):
    return redinp_resize_(
        inp_object.output_size,
        inp_object.work_size,
        inp_object.reductor.shape,
        copy_(inp_object.reductor),
    )


copy_project_resize_ = jit_(jcopy_project_resize_)


def gjcopy_project_resize_(ginp_object):
    return ginp_project_resize_(
        ginp_object.output_size, ginp_object.nhyperparameters, ginp_object.work_size
    )


def grcopy_project_resize_(ginp_object):
    return gredinp_project_resize_(
        ginp_object.output_size, ginp_object.nhyperparameters, ginp_object.work_size
    )


gcopy_project_resize_ = jit_(gjcopy_project_resize_)


@jit_
def cpuest_project_resize_(processed_object, input_object):
    return 0.0


@jit_
def gcpuest_project_resize_(processed_object, input_object, grad_object):
    return 0.0


# Normalization of the vectors.
normalization_lvl = "normalization"


def add_normalization(processed_function_definition_list, inside_routine):
    extract_final_result = get_extract_final_result()
    if is_grad_def(processed_function_definition_list):
        extract_final_gradient = get_extract_final_gradient()

        @jit_
        def normalization(
            processed_object,
            input_object,
            grad_object,
            hyperparameters,
            input_work_array,
            grad_work_array,
        ):
            final_result = extract_final_result(input_object, input_work_array)
            final_gradient = extract_final_gradient(input_object, grad_object, grad_work_array)

            inside_routine(
                processed_object,
                input_object.nested_input_object,
                grad_object.nested_grad_object,
                hyperparameters,
                input_work_array,
                grad_work_array,
            )
            inplace_normalization_wgrad(final_result, final_gradient, hyperparameters.shape[0])

    else:

        @jit_
        def normalization(processed_object, input_object, hyperparameters, input_work_array):
            inside_routine(
                processed_object,
                input_object.nested_input_object,
                hyperparameters,
                input_work_array,
            )
            final_result = extract_final_result(input_object, input_work_array)
            final_result[:] /= l2_norm_(final_result)

    return normalization


# Concatenating two vectors obtained from different sources.
concatenation_lvl = "concatenation"


def add_concatenation(processed_function_definition_list, inside_routine):
    if is_grad_def(processed_function_definition_list):
        extract_final_gradient = get_extract_final_gradient()

        @jit_
        def concatenation(
            processed_object,
            input_object,
            grad_object,
            hyperparameters,
            input_work_array,
            grad_work_array,
        ):
            nested_input_work_array_ub = input_work_array.shape[0]
            grad_size = grad_object.output_size
            nested_grad_work_array_ub = grad_work_array.shape[0] - grad_size
            nested_grad_object = grad_object.nested_grad_object
            num_components = input_object.num_components
            for component_id in range(num_components - 1, -1, -1):
                nested_input_object = input_object.nested_input_objects[component_id]
                inside_routine(
                    processed_object[component_id],
                    nested_input_object,
                    nested_grad_object,
                    hyperparameters,
                    input_work_array[:nested_input_work_array_ub],
                    grad_work_array[:nested_grad_work_array_ub],
                )
                nested_input_work_array_ub -= nested_input_object.output_size
                nested_grad_work_array_ub -= nested_grad_object.output_size

            nested_output_size = nested_input_object.output_size
            final_gradient_1D = grad_work_array[-2 * grad_size : -grad_size]

            final_gradient = extract_final_gradient(input_object, grad_object, grad_work_array)

            lb = 0
            comp_lb = 0
            for component_id in range(num_components):
                comp_ub = comp_lb + nested_output_size
                for hyp_id in range(grad_object.nhyperparameters):
                    ub = lb + nested_output_size
                    final_gradient[hyp_id, comp_lb:comp_ub] = final_gradient_1D[lb:ub]
                    lb = ub
                comp_lb = comp_ub

    else:

        @jit_
        def concatenation(processed_object, input_object, hyperparameters, input_work_array):
            nested_input_work_array_ub = input_work_array.shape[0]
            num_components = input_object.num_components
            for component_id in range(num_components - 1, -1, -1):
                nested_input_work_array = input_work_array[:nested_input_work_array_ub]
                nested_input_object = input_object.nested_input_objects[component_id]
                inside_routine(
                    processed_object[component_id],
                    nested_input_object,
                    hyperparameters,
                    nested_input_work_array,
                )
                nested_input_work_array_ub -= nested_input_object.output_size

    return concatenation


def inp_add_concatenation(
    processed_definition_list, nested_input_object, nested_default_input_object
):
    if is_grad_def(processed_definition_list):
        return inp_add_simple(
            processed_definition_list,
            nested_input_object,
            nested_default_input_object,
        )

    nested_input_objects, nested_default_input_objects = create_list_of(
        processed_definition_list, nested_input_object, nested_default_input_object
    )

    @jitclass_(
        basic_input_components
        + [("nested_input_objects", nested_input_objects), ("num_components", default_int)],
        skip=is_reducable_object(processed_definition_list),
    )
    class inp_concatenation:
        def __init__(self, nested_input_objects, output_size=0, num_components=0, work_size=0):
            self.nested_input_objects = nested_input_objects
            self.output_size = output_size
            self.num_components = num_components
            self.work_size = work_size

    return inp_concatenation, inp_concatenation(nested_default_input_objects)


def cadd_concatenation(processed_definition_list, inside_copy, object_class):
    if is_grad_def(processed_definition_list):
        return cadd_simple(processed_definition_list, inside_copy, object_class)

    @jit_(skip=is_red_vs_jit_copy_def(processed_definition_list))
    def copy(input_object):
        nested_input_objects = typed.List()
        for comp_id in range(len(input_object.nested_input_objects)):
            comp = input_object.nested_input_objects[comp_id]
            nested_input_objects.append(inside_copy(comp))
        output_size = input_object.output_size
        return object_class(
            nested_input_objects, output_size, input_object.num_components, input_object.work_size
        )

    return copy


def cpuest_add_concatenation(processed_definition_list, routine):
    if is_grad_def(processed_definition_list):

        @jit_
        def cpuest_concatenation(processed_object, input_object, grad_object):
            num_components = input_object.num_components
            output = 0.0
            for component_id in range(num_components):
                nested_input_object = input_object.nested_input_objects[component_id]
                output += routine(
                    processed_object[component_id],
                    nested_input_object,
                    grad_object.nested_grad_object,
                )
            return output

    else:

        @jit_
        def cpuest_concatenation(processed_object, input_object):
            num_components = input_object.num_components
            output = 0.0
            for component_id in range(num_components):
                nested_input_object = input_object.nested_input_objects[component_id]
                output += routine(
                    processed_object[component_id],
                    nested_input_object,
                )
            return output

    return cpuest_concatenation


# Switching to different iterations of the same SORF generator depending on the element id.
# Mainly used with SORF for effects analogous to using local_dn kernel.
element_id_switch_lvl = "element_id_switch"


def add_element_id_switch(processed_function_definition_list, inside_routine):
    if is_grad_def(processed_function_definition_list):

        @jit_
        def element_id_switch(
            processed_object,
            input_object,
            grad_object,
            hyperparameters,
            input_work_array,
            grad_work_array,
        ):
            used_nested_input_object = input_object.nested_input_objects[
                processed_object.element_id
            ]
            inside_routine(
                processed_object.representation,
                used_nested_input_object,
                grad_object.nested_grad_object,
                hyperparameters,
                input_work_array,
                grad_work_array,
            )

    else:

        @jit_
        def element_id_switch(processed_object, input_object, hyperparameters, input_work_array):
            used_nested_input_object = input_object.nested_input_objects[
                processed_object.element_id
            ]
            inside_routine(
                processed_object.representation,
                used_nested_input_object,
                hyperparameters,
                input_work_array,
            )

    return element_id_switch


def inp_add_element_id_switch(
    processed_definition_list, nested_input_object, nested_default_input_object
):
    if is_grad_def(processed_definition_list):
        return inp_add_simple(
            processed_definition_list, nested_input_object, nested_default_input_object
        )
    nested_input_objects, nested_default_input_objects = create_list_of(
        processed_definition_list, nested_input_object, nested_default_input_object
    )

    @jitclass_(
        basic_input_components
        + [("nested_input_objects", nested_input_objects), ("num_components", default_int)],
        skip=is_reducable_object(processed_definition_list),
    )
    class inp_element_id_switch:
        def __init__(self, nested_input_objects, output_size=0, work_size=0):
            self.nested_input_objects = nested_input_objects
            self.output_size = output_size
            self.work_size = work_size

    return inp_element_id_switch, inp_element_id_switch(nested_default_input_objects)


def cadd_element_id_switch(processed_definition_list, inside_copy, object_class):
    if is_grad_def(processed_definition_list):
        return cadd_simple(processed_definition_list, inside_copy, object_class)
    list_copy = cadd_list(processed_definition_list, inside_copy, object_class)

    @jit_(skip=is_red_vs_jit_copy_def(processed_definition_list))
    def switch_copy(input_object):
        nested_input_objects = list_copy(input_object.nested_input_objects)
        return object_class(nested_input_objects, input_object.output_size, input_object.work_size)

    return switch_copy


def cpuest_add_element_id_switch(processed_definition_list, routine):
    if is_grad_def(processed_definition_list):

        @jit_
        def cpuest_element_id_switch(processed_object, input_object, grad_object):
            used_nested_input_object = input_object.nested_input_objects[
                processed_object.element_id
            ]
            return routine(
                processed_object.representation,
                used_nested_input_object,
                grad_object.nested_grad_object,
            )

    else:

        @jit_
        def cpuest_element_id_switch(processed_object, input_object):
            used_nested_input_object = input_object.nested_input_objects[
                processed_object.element_id
            ]
            return routine(
                processed_object.representation,
                used_nested_input_object,
            )

    return cpuest_element_id_switch


# Compressing input into output which is half as large.
compression_lvl = "compression"


def add_compression(processed_function_definition_list, inside_routine):
    if is_grad_def(processed_function_definition_list):
        extract_final_result = get_extract_final_result()
        extract_final_gradient = get_extract_final_gradient()

        @jit_
        def compression(
            processed_object,
            input_object,
            grad_object,
            hyperparameters,
            input_work_array,
            grad_work_array,
        ):
            nested_input_object = input_object.nested_input_object
            nested_grad_object = grad_object.nested_grad_object
            nested_output_size = nested_input_object.output_size
            nested_grad_size = nested_grad_object.output_size
            inside_routine(
                processed_object,
                nested_input_object,
                nested_grad_object,
                hyperparameters[:-nested_output_size],
                input_work_array,
                grad_work_array,
            )
            compression_hyperparameters = hyperparameters[-nested_output_size:]
            compression_ratio = input_object.compression_ratio
            compress(
                grad_work_array[-nested_grad_size:], compression_hyperparameters, compression_ratio
            )

            final_gradient = extract_final_gradient(input_object, grad_object, grad_work_array)
            nested_final_result = extract_final_result(nested_input_object, input_work_array)

            nnested_hyperparameters = nested_grad_object.nhyperparameters
            final_gradient[:nnested_hyperparameters, :] = final_gradient[
                -nnested_hyperparameters:, :
            ]
            final_gradient[nnested_hyperparameters:, :] = 0.0
            hyperparameter_lb = nnested_hyperparameters
            vec_lb = 0
            for new_val_id in range(input_object.output_size):
                hyperparameter_ub = hyperparameter_lb + compression_ratio
                vec_ub = vec_lb + compression_ratio
                final_gradient[
                    hyperparameter_lb:hyperparameter_ub, new_val_id
                ] = nested_final_result[vec_lb:vec_ub]
                hyperparameter_lb = hyperparameter_ub
                vec_lb = vec_ub

            compress(nested_final_result, compression_hyperparameters, compression_ratio)

    else:

        @jit_
        def compression(processed_object, input_object, hyperparameters, input_work_array):
            nested_input_object = input_object.nested_input_object
            nested_output_size = nested_input_object.output_size
            inside_routine(
                processed_object,
                input_object.nested_input_object,
                hyperparameters[:-nested_output_size],
                input_work_array,
            )
            compress(
                input_work_array[-nested_output_size:],
                hyperparameters[-nested_output_size:],
                input_object.compression_ratio,
            )

    return compression


def inp_add_compression(
    processed_definition_list, nested_input_component, nested_default_input_object
):
    if is_grad_def(processed_definition_list):
        return inp_add_simple(
            processed_definition_list, nested_input_component, nested_default_input_object
        )

    @jitclass_(
        basic_input_components + nested_input_component + [("compression_ratio", default_int)],
        skip=is_reducable_object(processed_definition_list),
    )
    class inp_compression:
        def __init__(self, nested_input_object, output_size=0, work_size=0, compression_ratio=0):
            self.nested_input_object = nested_input_object
            self.output_size = output_size
            self.work_size = work_size
            self.compression_ratio = compression_ratio

    return inp_compression, inp_compression(nested_default_input_object)


def cadd_compression(processed_definition_list, inside_copy, object_class):
    if is_grad_def(processed_definition_list):
        return cadd_simple(processed_definition_list, inside_copy, object_class)

    @jit_(skip=is_red_vs_jit_copy_def(processed_definition_list))
    def copy(input_object):
        new_nested_object = inside_copy(input_object.nested_input_object)
        new_object = object_class(
            new_nested_object,
            input_object.output_size,
            input_object.work_size,
            input_object.compression_ratio,
        )
        return new_object

    return copy


function_level_additions = {
    resize_lvl: add_resize,
    normalization_lvl: add_normalization,
    concatenation_lvl: add_concatenation,
    compression_lvl: add_compression,
    element_id_switch_lvl: add_element_id_switch,
}
copy_level_additions = {
    concatenation_lvl: cadd_concatenation,
    compression_lvl: cadd_compression,
    element_id_switch_lvl: cadd_element_id_switch,
}
class_level_additions = {
    concatenation_lvl: inp_add_concatenation,
    compression_lvl: inp_add_compression,
    element_id_switch_lvl: inp_add_element_id_switch,
}
cpuest_level_additions = {
    concatenation_lvl: cpuest_add_concatenation,
    compression_lvl: cpuest_add_single,
    element_id_switch_lvl: cpuest_add_element_id_switch,
}
