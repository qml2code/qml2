# Create and add basic building blocks for SORF evaluation.

# K.Karan: A lot of this is sloppy, should be made less sloppy as jitclass moves out of numba.experimental.
# Also is probably worth trying to make work with TORCH.
import importlib
from copy import deepcopy

from numba import typed, typeof

from ...basic_utils import checked_dict_entry, convert_dict_list, recursive_class_dict
from ...jit_interfaces import all_
from ...utils import is_power2
from ..utils import get_numba_list
from .base import (
    cadd_simple,
    copy_prefix,
    cpuest_add_simple,
    cpuest_prefix,
    datatype_prefix,
    default_class_instances,
    function_prefix,
    gradient_copy_prefix,
    gradient_cpuest_prefix,
    gradient_function_prefix,
    gradient_input_object_prefix,
    gradient_jcopy_prefix,
    gradient_rcopy_prefix,
    gradient_red_input_object_prefix,
    inp_add_simple,
    input_object_prefix,
    is_copy_def,
    is_cpuest_def,
    is_data_definition,
    is_grad_def,
    is_red2jit_copy_def,
    is_reducable_object,
    jcopy_prefix,
    rcopy_prefix,
    red_datatype_prefix,
    red_input_object_prefix,
)
from .fork import fork_lvl, inp_add_fork
from .rescalings import power_rescaling_lvl, rescaling_lvl
from .sorf_related import (
    mixed_extensive_sorf_lvl,
    sign_invariant_sorf_lvl,
    sorf_lvl,
    unscaled_sign_invariant_sorf_lvl,
    unscaled_sorf_lvl,
)
from .special import compression_lvl, concatenation_lvl, element_id_switch_lvl, resize_lvl
from .summation import component_sum_lvl, weighted_component_sum_lvl

# Everything related to adding levels to objects, functions, and copy functions.
class_level_additions = {}
copy_level_additions = {}
function_level_additions = {}
cpuest_level_additions = {}


def set_add_dicts():
    addition_sources = [
        "datatypes",
        "fork",
        "rescalings",
        "sorf_related",
        "special",
        "summation",
        "variance",
    ]
    for dict_prefix in ["class", "copy", "function", "cpuest"]:
        comb_dict_name = dict_prefix + "_level_additions"
        combined_dict = globals()[dict_prefix + "_level_additions"]
        for add_source in addition_sources:
            submod = importlib.import_module(".." + add_source, package=__name__)
            imported_vars = submod.__dict__
            if comb_dict_name in imported_vars:
                combined_dict.update(imported_vars[comb_dict_name])


def import_bottom_level():
    for submod_name in ["special", "base"]:
        submod = importlib.import_module(".." + submod_name, package=__name__)
        imported_vars = submod.__dict__
        for k, val in imported_vars.items():
            if k[-1] == "_":
                globals()[k] = val


def get_processed_definition_list(input_definition_list, prefix):
    return [prefix] + input_definition_list


def fork_here(processed_definition_list):
    return isinstance(processed_definition_list[-1], tuple)


def get_fork_processed_lists(processed_definition_list):
    fork_tuple = processed_definition_list[1]
    assert isinstance(fork_tuple, tuple)
    prefix = processed_definition_list[0]
    return tuple([[prefix] + l for l in fork_tuple])


def get_name_from_definition_list(processed_definition_list):
    if not isinstance(processed_definition_list[1], tuple):
        return join_wtail(processed_definition_list)
    output = ""
    for l in get_fork_processed_lists(processed_definition_list):
        output = output + get_name_from_definition_list(l)
    return output + join_wtail(processed_definition_list[2:])


def join_wtail(str_list):
    return "_".join(str_list) + "_"


# Procedures for adding class levels.
def get_nested_component_list_class_instance(processed_definition_list, component_id=None):
    if component_id is None:
        used_definition_list = processed_definition_list[:-1]
    else:
        used_definition_list = processed_definition_list
    nested_input_class_instance = get_default_class_instance(used_definition_list)
    if is_grad_def(processed_definition_list):
        name = "nested_grad_object"
    else:
        name = "nested_input_object"
    if component_id is not None:
        name = name + str(component_id + 1)
    if is_reducable_object(used_definition_list):
        nested_input_class = get_class(used_definition_list)
    else:
        nested_input_class = typeof(nested_input_class_instance)

    return [(name, nested_input_class)], nested_input_class_instance


def setup_class(processed_definition_list):
    class_name = get_name_from_definition_list(processed_definition_list)
    gl = globals()
    global default_class_instances
    is_fork = fork_here(processed_definition_list)
    if len(processed_definition_list) == 2 and not is_fork:
        class_def = gl[class_name]
        assert class_name not in default_class_instances
        default_class_instances[class_name] = class_def()
        return

    if is_fork:
        # forks only support branching into two
        assert len(processed_definition_list[-1]) == 2
        # we must've reached bottom level of the definition list
        assert len(processed_definition_list) == 2
        forked_definition_lists = get_fork_processed_lists(processed_definition_list)
        all_nested_comps = [
            get_nested_component_list_class_instance(forked_definition_list, component_id=comp_id)
            for comp_id, forked_definition_list in enumerate(forked_definition_lists)
        ]
        new_class, new_class_example = inp_add_fork(
            forked_definition_lists, *all_nested_comps[0], *all_nested_comps[1]
        )
    else:
        (
            nested_component_list,
            nested_component_instance,
        ) = get_nested_component_list_class_instance(processed_definition_list)
        class_level_addition = checked_dict_entry(
            class_level_additions, processed_definition_list[-1], inp_add_simple
        )
        new_class, new_class_example = class_level_addition(
            processed_definition_list, nested_component_list, nested_component_instance
        )

    gl[class_name] = new_class
    default_class_instances[class_name] = new_class_example


def get_precomputed_quant(processed_definition_list, storage_dict, setup_routine, **other_kwargs):
    class_name = get_name_from_definition_list(processed_definition_list)
    if class_name not in storage_dict:
        setup_routine(processed_definition_list, **other_kwargs)
    return storage_dict[class_name]


def get_default_class_instance(processed_definition_list):
    return get_precomputed_quant(processed_definition_list, default_class_instances, setup_class)


def get_class(processed_definition_list):
    return get_precomputed_quant(processed_definition_list, globals(), setup_class)


def get_datatype(definition_list):
    processed_definition_list = get_processed_definition_list(definition_list, datatype_prefix)
    return get_class(processed_definition_list)


# For setting up copying
def change_to_object_prefix(definition_list):
    is_dt = is_data_definition(definition_list)
    is_red2jit_copy = is_red2jit_copy_def(definition_list)
    if is_grad_def(definition_list):
        assert not is_dt
        if is_red2jit_copy:
            new_prefix = gradient_red_input_object_prefix
        else:
            new_prefix = gradient_input_object_prefix
    else:
        if is_red2jit_copy:
            if is_dt:
                new_prefix = red_datatype_prefix
            else:
                new_prefix = red_input_object_prefix
        else:
            if is_dt:
                new_prefix = datatype_prefix
            else:
                new_prefix = input_object_prefix
    definition_list[0] = new_prefix


def get_copied_obj_class(processed_definition_list):
    class_definition_list = deepcopy(processed_definition_list)
    change_to_object_prefix(class_definition_list)
    return get_class(class_definition_list)


def get_inside_copy_obj_class(processed_definition_list):
    assert is_copy_def(processed_definition_list)
    is_fork = fork_here(processed_definition_list)
    if is_fork:
        output = [
            get_routine(forked_def)
            for forked_def in get_fork_processed_lists(processed_definition_list)
        ]
        output.append(get_copied_obj_class(processed_definition_list))
        return tuple(output)
    else:
        return get_routine(processed_definition_list[:-1]), get_copied_obj_class(
            processed_definition_list
        )


def get_nested_routine_tuple(processed_definition_list):
    is_fork = fork_here(processed_definition_list)
    if is_fork:
        return tuple(
            [
                get_routine(forked_def)
                for forked_def in get_fork_processed_lists(processed_definition_list)
            ]
        )
    else:
        return (get_routine(processed_definition_list[:-1]),)


# final procedures for getting routines.
def setup_routine(processed_function_definition_list):
    assert len(processed_function_definition_list) != 1, processed_function_definition_list
    function_name = get_name_from_definition_list(processed_function_definition_list)

    is_fork = fork_here(processed_function_definition_list)

    if is_copy_def(processed_function_definition_list):
        nested_fetcher = get_inside_copy_obj_class
    else:
        nested_fetcher = get_nested_routine_tuple

    if is_fork:
        lvl_name = fork_lvl
    else:
        lvl_name = processed_function_definition_list[-1]
    if is_copy_def(processed_function_definition_list):
        function_level_addition = checked_dict_entry(copy_level_additions, lvl_name, cadd_simple)
    elif is_cpuest_def(processed_function_definition_list):
        function_level_addition = checked_dict_entry(
            cpuest_level_additions, lvl_name, cpuest_add_simple
        )
    else:
        function_level_addition = function_level_additions[lvl_name]
    nested_args = nested_fetcher(processed_function_definition_list)
    new_function = function_level_addition(processed_function_definition_list, *nested_args)

    globals()[function_name] = new_function


def get_routine(processed_function_definition_list):
    return get_precomputed_quant(processed_function_definition_list, globals(), setup_routine)


def get_class_from_def(function_definition_list, prefix):
    return get_class(get_processed_definition_list(function_definition_list, prefix))


def get_routine_from_def(function_definition_list, gradient=False):
    if gradient:
        prefix = gradient_function_prefix
    else:
        prefix = function_prefix
    return get_routine(get_processed_definition_list(function_definition_list, prefix))


def get_copy_from_def(function_definition_list, gradient=False):
    if gradient:
        prefix = gradient_copy_prefix
    else:
        prefix = copy_prefix
    return get_routine(get_processed_definition_list(function_definition_list, prefix))


def get_cpuest_from_def(function_definition_list, gradient=False):
    if gradient:
        prefix = gradient_cpuest_prefix
    else:
        prefix = cpuest_prefix
    return get_routine(get_processed_definition_list(function_definition_list, prefix))


# Creating initialized input objects.
def create_nested_input_object(processed_definition_list, parameter_list, level_parameter_list):
    return create_input_object(
        processed_definition_list[:-1], parameter_list[:-1], level_parameter_list[:-1]
    )


def create_input_object(processed_definition_list, size_parameter_list, level_parameter_list):
    assert len(processed_definition_list) == len(size_parameter_list) + 1
    assert len(size_parameter_list) == len(level_parameter_list)
    created_class = get_class(processed_definition_list)
    if fork_here(processed_definition_list):
        # it is a fork
        assert len(processed_definition_list) == 2
        processed_definition_lists = get_fork_processed_lists(processed_definition_list)
        nested_objects = []
        for definition, size, level in zip(
            processed_definition_lists,
            size_parameter_list[0]["forked"],
            level_parameter_list[0],
        ):
            nested_objects.append(create_input_object(definition, size, level))
        assert len(nested_objects) == 2
        return created_class(*nested_objects, **size_parameter_list[0]["params"])

    if len(processed_definition_list) == 2:
        return created_class(**size_parameter_list[0])
    last_step = processed_definition_list[-1]
    last_size_parameters = size_parameter_list[-1]
    if (last_step in [concatenation_lvl, element_id_switch_lvl]) and (
        not is_grad_def(processed_definition_list)
    ):
        last_level_parameters = level_parameter_list[-1]
        nested_list = [
            create_nested_input_object(
                processed_definition_list, size_parameter_list, level_parameter_list
            )
        ]
        if last_step == concatenation_lvl:
            num_components = last_level_parameters["num_components"]
        else:
            num_components = last_level_parameters["num_element_ids"]
        nested_list = [
            create_nested_input_object(
                processed_definition_list, size_parameter_list, level_parameter_list
            )
            for _ in range(num_components)
        ]
        return created_class(get_numba_list(nested_list), **last_size_parameters)
    nested_input_object = create_nested_input_object(
        processed_definition_list, size_parameter_list, level_parameter_list
    )
    return created_class(nested_input_object, **last_size_parameters)


def create_input_object_from_def(
    definition_list, size_parameter_list, level_parameter_list, gradient=False
):
    if gradient:
        prefix = gradient_input_object_prefix
    else:
        prefix = input_object_prefix
    return create_input_object(
        get_processed_definition_list(definition_list, prefix),
        size_parameter_list,
        level_parameter_list,
    )


# For calculating sizes of all arrays appearing at each level.
def calc_hyperparameter_num(step_str, parameter_dict, previous_input_size_parameters={}):
    if step_str in [sorf_lvl, sign_invariant_sorf_lvl]:
        return 1
    elif step_str == rescaling_lvl:
        return parameter_dict["resc_bounds"].shape[0]
    elif step_str == power_rescaling_lvl:
        return parameter_dict["resc_powers"].shape[0]
    elif step_str == compression_lvl:
        return previous_input_size_parameters["output_size"]
    elif step_str == mixed_extensive_sorf_lvl:
        return 2
    else:
        return 0


def calc_output_size(previous_input_size_parameters, level_parameters, level_type):
    if level_type in [resize_lvl, compression_lvl]:
        return level_parameters["output_size"]

    previous_output_size = previous_input_size_parameters["output_size"]
    if level_type in [
        sorf_lvl,
        unscaled_sorf_lvl,
        mixed_extensive_sorf_lvl,
        sign_invariant_sorf_lvl,
        unscaled_sign_invariant_sorf_lvl,
    ]:
        assert is_power2(
            previous_output_size
        ), "Attempting to run SORF not on a 2**n input; check parameters."
        return level_parameters["nfeature_stacks"] * previous_output_size
    elif level_type == concatenation_lvl:
        return level_parameters["num_components"] * previous_output_size
    else:
        return previous_output_size


def resize_work_size(output_size, previous_size_parameters):
    if "work_size" not in previous_size_parameters:
        return output_size
    previous_work_size = previous_size_parameters["work_size"]
    previous_output_size = previous_size_parameters["output_size"]
    output_gap = output_size - previous_output_size
    return previous_work_size + output_gap


def calc_input_work_size(
    output_size, level_type, other_input_size_parameters={}, previous_input_size_parameters={}
):
    if level_type == resize_lvl:
        return resize_work_size(output_size, previous_input_size_parameters)
    previous_work_size = previous_input_size_parameters["work_size"]

    if level_type in [
        component_sum_lvl,
        weighted_component_sum_lvl,
        sorf_lvl,
        unscaled_sorf_lvl,
        mixed_extensive_sorf_lvl,
        sign_invariant_sorf_lvl,
        unscaled_sign_invariant_sorf_lvl,
    ]:
        return previous_work_size + output_size
    elif level_type == resize_lvl:
        return max(previous_work_size, output_size)
    elif level_type == concatenation_lvl:
        previous_output_size = previous_input_size_parameters["output_size"]
        return (
            previous_work_size
            + (other_input_size_parameters["num_components"] - 1) * previous_output_size
        )
    else:
        return previous_work_size


def calc_grad_work_size(
    grad_output_size,
    level_type,
    previous_grad_input_size_parameters={},
    input_size_parameters={},
    previous_input_size_parameters={},
):
    if level_type == resize_lvl:
        return resize_work_size(grad_output_size, previous_grad_input_size_parameters)

    output_size = input_size_parameters["output_size"]
    previous_grad_work_size = previous_grad_input_size_parameters["work_size"]
    previous_grad_output_size = previous_grad_input_size_parameters["output_size"]

    if level_type in [component_sum_lvl, weighted_component_sum_lvl]:
        return previous_grad_work_size + grad_output_size
    elif level_type in [
        sorf_lvl,
        unscaled_sorf_lvl,
        mixed_extensive_sorf_lvl,
        sign_invariant_sorf_lvl,
        unscaled_sign_invariant_sorf_lvl,
    ]:
        # need to make space for "phases" array
        additional_work_size = previous_grad_output_size + output_size
        if level_type == mixed_extensive_sorf_lvl:
            # need somewhere to store derivatives of vector norm w.r.t. hyperparameters
            additional_work_size += previous_grad_input_size_parameters["nhyperparameters"]
        final_previous_grad_work_size = max(additional_work_size, previous_grad_work_size)
        return final_previous_grad_work_size + grad_output_size
    elif level_type == power_rescaling_lvl:
        new_grad_work_size = grad_output_size + output_size
    elif level_type == rescaling_lvl:
        return previous_grad_work_size + input_size_parameters["nsigmas"] * output_size
    elif level_type == concatenation_lvl:
        return (
            previous_grad_work_size
            + (input_size_parameters["num_components"] - 1) * previous_grad_output_size
            + grad_output_size
        )
    elif level_type == element_id_switch_lvl:
        return previous_grad_work_size
    else:
        new_grad_work_size = grad_output_size
    return max(previous_grad_work_size, new_grad_work_size)


def calc_input_size_parameters(level_parameters, level_type, previous_input_size_parameters={}):
    output_size = calc_output_size(previous_input_size_parameters, level_parameters, level_type)
    if "output_size" in previous_input_size_parameters:
        previous_output_size = previous_input_size_parameters["output_size"]
    else:
        previous_output_size = None
    if level_type in [
        sorf_lvl,
        unscaled_sorf_lvl,
        mixed_extensive_sorf_lvl,
        sign_invariant_sorf_lvl,
        unscaled_sign_invariant_sorf_lvl,
    ]:
        assert previous_output_size is not None
        final_dict = {
            "init_size": previous_output_size,
            "nfeature_stacks": level_parameters["nfeature_stacks"],
            "ntransforms": level_parameters["ntransforms"],
        }
    elif level_type == rescaling_lvl:
        resc_bounds = level_parameters["resc_bounds"]
        nsigmas = level_parameters["resc_bounds"].shape[0]
        assert all_(resc_bounds) <= output_size
        final_dict = {"nsigmas": nsigmas}
    elif level_type == power_rescaling_lvl:
        nsigmas, rescaling_size = level_parameters["resc_powers"].shape
        assert rescaling_size <= output_size
        final_dict = {
            "rescaling_size": rescaling_size,
            "nsigmas": nsigmas,
        }
    elif level_type == compression_lvl:
        assert previous_output_size is not None
        assert previous_output_size % output_size == 0
        final_dict = {"compression_ratio": previous_output_size // output_size}
    elif level_type == concatenation_lvl:
        num_components = level_parameters["num_components"]
        final_dict = {"num_components": num_components}
        output_size = previous_output_size * num_components
    else:
        final_dict = {}
    final_dict["output_size"] = output_size
    final_dict["work_size"] = calc_input_work_size(
        output_size,
        level_type,
        previous_input_size_parameters=previous_input_size_parameters,
        other_input_size_parameters=final_dict,
    )
    return final_dict


def calc_grad_input_size_parameters(
    level_type,
    nhyperparameters,
    input_size_parameters={},
    previous_input_size_parameters={},
    previous_grad_input_size_parameters={},
):
    output_size = input_size_parameters["output_size"]
    grad_output_size = output_size * nhyperparameters
    final_dict = {"output_size": grad_output_size, "nhyperparameters": nhyperparameters}
    work_size = calc_grad_work_size(
        grad_output_size,
        level_type,
        previous_input_size_parameters=previous_input_size_parameters,
        previous_grad_input_size_parameters=previous_grad_input_size_parameters,
        input_size_parameters=input_size_parameters,
    )
    final_dict["work_size"] = work_size
    if level_type == "rescaling":
        final_dict["grad_work_gap"] = output_size * input_size_parameters["nsigmas"]
    return final_dict


# For converting classes into something serializable.
def get_class2dict(processed_definition_list):
    copy_func_def = deepcopy(processed_definition_list)
    if is_grad_def(processed_definition_list):
        rcopy_pref = gradient_rcopy_prefix
    else:
        rcopy_pref = rcopy_prefix
    copy_func_def[0] = rcopy_pref
    jit2red_copy = get_routine(copy_func_def)

    def make_dict(obj):
        obj_cl = jit2red_copy(obj)
        return recursive_class_dict(obj_cl)

    return make_dict


def get_dict2class(processed_definition_list):
    copy_func_def = deepcopy(processed_definition_list)
    if is_grad_def(processed_definition_list):
        jcopy_pref = gradient_jcopy_prefix
    else:
        jcopy_pref = jcopy_prefix
    copy_func_def[0] = jcopy_pref
    red2jit_copy = get_routine(copy_func_def)

    def make_inst(d):
        obj_cl = convert_dict_list(d)
        return red2jit_copy(obj_cl)

    return make_inst


def get_datatype2dict(definition_list):
    return get_class2dict(get_processed_definition_list(definition_list, datatype_prefix))


def get_dict2datatype(definition_list):
    return get_dict2class(get_processed_definition_list(definition_list, datatype_prefix))


def get_transform_list_dict2datatype(definition_list):
    dict2datatype = get_dict2datatype(definition_list)

    def list_create(dict_list):
        output = typed.List()
        for d in dict_list:
            output.append(dict2datatype(d))
        return output

    return list_create


if __name__ != "__main__":
    set_add_dicts()
    import_bottom_level()
