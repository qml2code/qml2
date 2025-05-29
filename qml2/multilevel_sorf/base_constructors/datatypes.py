import typing
from copy import deepcopy

from numba import typed, typeof

from ...jit_interfaces import empty_, jit_
from ..utils import jitclass_
from .base import (
    array_1D_,
    change_nested_component_name,
    default_int,
    is_red_vs_jit_copy_def,
    is_reducable_object,
)

# For adding datatype levels on top of the base.
list_lvl = "list"


def create_list_of(processed_definition_list, nested_def_list, default_item):
    if is_reducable_object(processed_definition_list):
        return typing.List, [default_item]
    else:
        list_instance = typed.List()
        list_instance.append(default_item)
        return typeof(list_instance), list_instance


def cadd_list(processed_definition_list, inside_copy, object_class):
    if is_reducable_object(processed_definition_list):
        return deepcopy

    @jit_(skip=is_red_vs_jit_copy_def(processed_definition_list))
    def get_copy(other_list):
        output = typed.List()
        for other_element in other_list:
            output.append(inside_copy(other_element))
        return output

    return get_copy


rhos_lvl = "rhos"


def augment_with_rhos(processed_definition_list, nested_def_list, default_list):
    new_nested_def_list = change_nested_component_name(nested_def_list, "components")
    jitclass_def = [
        ("rhos", array_1D_),
    ] + new_nested_def_list

    @jitclass_(jitclass_def, skip=is_reducable_object(processed_definition_list))
    class RhoAugmentedClass:
        def __init__(self, components, rhos=None):
            self.components = components
            self.rhos = empty_((len(components),))
            if rhos is not None:
                self.rhos[:] = rhos[:]

    default_rho_augmented_class = RhoAugmentedClass(default_list)
    return RhoAugmentedClass, default_rho_augmented_class


def cadd_rhos(processed_definition_list, inside_copy, object_class):
    @jit_(skip=is_red_vs_jit_copy_def(processed_definition_list))
    def get_copy(other_object):
        new_components = inside_copy(other_object.components)
        return object_class(
            new_components,
            other_object.rhos,
        )

    return get_copy


element_id_lvl = "element_id"


def augment_with_element_id(processed_definition_list, nested_def, default_list):
    new_nested_def_list = change_nested_component_name(nested_def, "representation")
    jitclass_def = [
        ("element_id", default_int),
    ] + new_nested_def_list

    @jitclass_(jitclass_def, skip=is_reducable_object(processed_definition_list))
    class ElementIDAugmentedClass:
        def __init__(self, representation, element_id=0):
            self.representation = representation
            self.element_id = element_id

    default_element_id_augmented_class = ElementIDAugmentedClass(default_list)
    return ElementIDAugmentedClass, default_element_id_augmented_class


def cadd_element_id(processed_definition_list, inside_copy, object_class):
    @jit_(skip=is_red_vs_jit_copy_def(processed_definition_list))
    def get_copy(other_object):
        new_representation = inside_copy(other_object.representation)
        return object_class(
            new_representation,
            other_object.element_id,
        )

    return get_copy


class_level_additions = {
    list_lvl: create_list_of,
    rhos_lvl: augment_with_rhos,
    element_id_lvl: augment_with_element_id,
}
copy_level_additions = {list_lvl: cadd_list, rhos_lvl: cadd_rhos, element_id_lvl: cadd_element_id}
