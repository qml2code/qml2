from ...jit_interfaces import abs_, jit_, l2_norm_
from ..utils import jitclass_
from .base import basic_input_components, get_separate_input_work_arr

# Special procedures for standard deviation evaluation.
calc_variance_lvl = "calc_variance"
calc_variance_normalized_lvl = "calc_variance_normalized"


def add_calc_variance(processed_definition_list, inside_routine):
    separate_input_work_arr = get_separate_input_work_arr()

    @jit_
    def calc_variance(processed_object, input_object, hyperparameters, input_work_array):
        nested_input_work_array, final_result, nested_final_result = separate_input_work_arr(
            input_object, input_work_array
        )
        nested_input_object = input_object.nested_input_object
        nested_output_size = nested_input_object.output_size
        inside_routine(
            processed_object, nested_input_object, hyperparameters, nested_input_work_array
        )
        final_result[-1] += 1.0
        final_result[:nested_output_size] += nested_final_result[:]
        final_result[nested_output_size:-1] += nested_final_result[:] ** 2

    return calc_variance


def add_calc_variance_normalized(processed_definition_list, inside_routine):
    @jit_
    def new_inside_routine(
        processed_object, nested_input_object, hyperparameters, nested_input_work_array
    ):
        inside_routine(
            processed_object, nested_input_object, hyperparameters, nested_input_work_array
        )
        final_result = nested_input_work_array[-nested_input_object.output_size :]
        final_result /= l2_norm_(final_result)

    return add_calc_variance(processed_definition_list, new_inside_routine)


def inp_add_calc_variance(
    processed_definition_list, nested_input_component, nested_default_input_object
):
    @jitclass_(basic_input_components + nested_input_component)
    class inp_calc_variance:
        def __init__(self, nested_input_object):
            self.nested_input_object = nested_input_object

            nested_output_size = nested_input_object.output_size
            self.output_size = 2 * nested_output_size + 1
            self.work_size = self.nested_input_object.work_size + self.output_size

    return inp_calc_variance, inp_calc_variance(nested_default_input_object)


def cadd_calc_variance(processed_definition_list, inside_copy, object_class):
    @jit_
    def copy(input_object):
        new_nested_object = inside_copy(input_object.nested_input_object)
        output = object_class(new_nested_object)
        return output

    return copy


pass_variance_lvl = "pass_variance"


def add_pass_variance(processed_definition_list, inside_routine):
    @jit_
    def pass_variance(processed_object, input_object, hyperparameters, input_work_array):
        inside_routine(
            processed_object, input_object.nested_input_object, hyperparameters, input_work_array
        )

    return pass_variance


def inp_add_pass_variance(
    processed_definition_list, nested_input_component, nested_default_input_object
):
    @jitclass_(basic_input_components + nested_input_component)
    class inp_pass_variance:
        def __init__(self, nested_input_object):
            self.nested_input_object = nested_input_object
            self.output_size = nested_input_object.output_size
            self.work_size = nested_input_object.work_size

    return inp_pass_variance, inp_pass_variance(nested_default_input_object)


# TODO: unify with cadd_simple?
def cadd_pass_variance(processed_definition_list, inside_copy, object_class):
    @jit_
    def copy(input_object):
        new_nested_object = inside_copy(input_object.nested_input_object)
        return object_class(new_nested_object)

    return copy


component_cycle_lvl = "component_cycle"


def add_component_cycle(processed_definition_list, inside_routine):
    @jit_
    def pass_variance(processed_object, input_object, hyperparameters, input_work_array):
        for i in range(len(processed_object)):
            inside_routine(
                processed_object[i],
                input_object.nested_input_object,
                hyperparameters,
                input_work_array,
            )

    return pass_variance


weighted_component_cycle_lvl = "weighted_component_cycle"


def add_weighted_component_cycle(processed_definition_list, inside_routine):
    separate_input_work_arr = get_separate_input_work_arr()

    @jit_
    def weighted_component_cycle(
        processed_object, input_object, hyperparameters, input_work_array
    ):
        nest_inp_obj = input_object.nested_input_object
        component_id = 0

        nested_input_work_array, final_result, nested_final_result = separate_input_work_arr(
            input_object, input_work_array
        )

        for component in processed_object.components:
            nested_final_result[:] = 0.0
            inside_routine(component, nest_inp_obj, hyperparameters, nested_input_work_array)
            final_result += abs_(processed_object.rhos[component_id]) * nested_final_result
            component_id += 1

    return weighted_component_cycle


def inp_add_weighted_component_cycle(
    processed_definition_list, nested_input_component, nested_default_input_object
):
    @jitclass_(basic_input_components + nested_input_component)
    class inp_weighed_component_cycle:
        def __init__(self, nested_input_object):
            self.nested_input_object = nested_input_object
            self.output_size = nested_input_object.output_size
            self.work_size = nested_input_object.work_size + self.output_size

    return inp_weighed_component_cycle, inp_weighed_component_cycle(nested_default_input_object)


concatenation_variance_lvl = "concatenation_variance"


def add_concatenation_variance(processed_definition_list, inside_routine):
    @jit_
    def concatenation_variance(processed_object, input_object, hyperparameters, input_work_array):
        for component_id in range(len(processed_object)):
            inside_routine(
                processed_object[component_id],
                input_object.nested_input_object,
                hyperparameters,
                input_work_array,
            )

    return concatenation_variance


element_id_switch_variance_lvl = "element_id_switch_variance"


def add_element_id_switch_variance(processed_definition_list, inside_routine):
    @jit_
    def element_id_switch_variance(
        processed_object, input_object, hyperparameters, input_work_array
    ):
        inside_routine(
            processed_object.representation,
            input_object.nested_input_object,
            hyperparameters,
            input_work_array,
        )

    return element_id_switch_variance


function_level_additions = {
    calc_variance_lvl: add_calc_variance,
    calc_variance_normalized_lvl: add_calc_variance_normalized,
    pass_variance_lvl: add_pass_variance,
    component_cycle_lvl: add_component_cycle,
    weighted_component_cycle_lvl: add_weighted_component_cycle,
    element_id_switch_variance_lvl: add_element_id_switch_variance,
    concatenation_variance_lvl: add_concatenation_variance,
}
copy_level_additions = {
    calc_variance_lvl: cadd_calc_variance,
    calc_variance_normalized_lvl: cadd_calc_variance,
    pass_variance_lvl: cadd_pass_variance,
    component_cycle_lvl: cadd_pass_variance,
    weighted_component_cycle_lvl: cadd_pass_variance,
    element_id_switch_variance_lvl: cadd_pass_variance,
    concatenation_variance_lvl: cadd_pass_variance,
}
class_level_additions = {
    calc_variance_lvl: inp_add_calc_variance,
    calc_variance_normalized_lvl: inp_add_calc_variance,
    pass_variance_lvl: inp_add_pass_variance,
    component_cycle_lvl: inp_add_pass_variance,
    weighted_component_cycle_lvl: inp_add_weighted_component_cycle,
    element_id_switch_variance_lvl: inp_add_pass_variance,
    concatenation_variance_lvl: inp_add_pass_variance,
}

for i in ["1", "2"]:
    pvf_name = "pass_variance_fork" + i
    globals()[pvf_name + "_lvl"] = pvf_name
    function_level_additions[pvf_name] = add_pass_variance
    copy_level_additions[pvf_name] = cadd_pass_variance
    class_level_additions[pvf_name] = inp_add_pass_variance
