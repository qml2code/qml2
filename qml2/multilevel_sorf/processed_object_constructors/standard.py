from numba import typed

from ...jit_interfaces import all_, array_, concatenate_, dint_
from ...parallelization import embarrassingly_parallel
from ...representations import generate_coulomb_matrix, generate_fchl19
from ...representations.element_representations import period_valence_representation
from ...utils import get_element_ids_from_sorted, get_sorted_elements
from ..base_constructors import get_dict2datatype, get_transform_list_dict2datatype

el_aug_rep_def = ["array_1D", "element_id", "list"]
el_aug_rep_dict_list2reps = get_transform_list_dict2datatype(el_aug_rep_def)
el_aug_rep_dict2rep = get_dict2datatype(el_aug_rep_def)


class LocalRepresentationCalc:
    def __init__(
        self,
        representation_function=generate_fchl19,
        representation_function_kwargs={},
        element_representation_function=None,
        element_representation_function_kwargs={},
    ):
        self.representation_function = representation_function
        self.representation_function_kwargs = representation_function_kwargs
        self.element_representation_function = element_representation_function
        self.element_representation_function_kwargs = element_representation_function_kwargs

    def calc_with_component_bounds(self, compound):
        representation = self.representation_function(
            compound.nuclear_charges, compound.coordinates, **self.representation_function_kwargs
        )
        if self.element_representation_function is None:
            component_bounds = array_([[0, representation.shape[1]]])
        else:
            element_representation = self.element_representation_function(
                compound.nuclear_charges, **self.element_representation_function_kwargs
            )
            representation = concatenate_((element_representation, representation), axis=1)
            component_bounds = array_(
                [
                    [0, element_representation.shape[1]],
                    [element_representation.shape[1], representation.shape[1]],
                ]
            )
        return representation, component_bounds


def ElementRepAugmentedLocalRepresentationCalc(
    representation_function=generate_fchl19,
    representation_function_kwargs={},
    element_rep_length=4,
):
    return LocalRepresentationCalc(
        representation_function=representation_function,
        representation_function_kwargs=representation_function_kwargs,
        element_representation_function=period_valence_representation,
        element_representation_function_kwargs={"rep_length": element_rep_length},
    )


class ElementAugmentedLocalRepresentationCalc:
    def __init__(
        self,
        representation_function=generate_fchl19,
        representation_function_kwargs={},
        possible_nuclear_charges=array_([1, 6, 7, 8, 16], dtype=dint_),
    ):
        self.representation_function = representation_function
        self.representation_function_kwargs = representation_function_kwargs
        self.sorted_elements = get_sorted_elements(possible_nuclear_charges)

    def calc_with_component_bounds(self, compound):
        ncharges = compound.nuclear_charges
        representation = self.representation_function(
            ncharges, compound.coordinates, **self.representation_function_kwargs
        )
        element_ids = get_element_ids_from_sorted(ncharges, sorted_elements=self.sorted_elements)
        output = [
            {"representation": rep, "element_id": el_id}
            for el_id, rep in zip(element_ids, representation)
        ]
        return output, array_([[0, representation.shape[1]]])


class ProcessedRepresentationListCalc:
    def __init__(self, local_representation_object: LocalRepresentationCalc):
        self.local_representation_object = local_representation_object
        self.component_bounds = None

    def __call__(
        self, compound_list, num_procs=None, fixed_num_threads=1, allow_None_entries=False
    ):
        lr_output = embarrassingly_parallel(
            self.local_representation_object.calc_with_component_bounds,
            compound_list,
            (),
            num_procs=num_procs,
            fixed_num_threads=fixed_num_threads,
        )
        output = None
        if not allow_None_entries:
            check_component_bounds = True

        for representation, component_bounds in lr_output:
            if output is None:
                if (type(representation) in [dict, list]) or allow_None_entries:
                    output = []
                else:
                    output = typed.List()

            # Check that representation component shapes are consistent.
            if allow_None_entries:
                check_component_bounds = component_bounds is not None

            if check_component_bounds:
                if self.component_bounds is None:
                    self.component_bounds = component_bounds
                else:
                    assert self.component_bounds.shape == component_bounds.shape
                    assert all_(self.component_bounds == component_bounds)
            output.append(representation)
        return output

    def clear_component_bounds(self):
        self.component_bounds = None

    def max_component_bound(self):
        assert self.component_bounds is not None
        return self.component_bounds[-1][1]


# K.Karan.: If needed in the future can write a calculator that combines more than one representation together.
class GlobalRepresentationCalc:
    def __init__(
        self,
        representation_function=generate_coulomb_matrix,
        representation_function_kwargs={},
    ):
        self.representation_function = representation_function
        self.representation_function_kwargs = representation_function_kwargs

    def calc_with_component_bounds(self, compound):
        representation = self.representation_function(
            compound.nuclear_charges, compound.coordinates, **self.representation_function_kwargs
        )
        return representation, array_([[0, representation.shape[0]]])
