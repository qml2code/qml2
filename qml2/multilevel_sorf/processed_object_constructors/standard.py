from numba import typed

from ...compound import Compound
from ...jit_interfaces import all_, array_, concatenate_, dint_, unique_
from ...parallelization import default_num_procs, embarrassingly_parallel
from ...representations import generate_coulomb_matrix, generate_fchl19
from ...representations.element_representations import period_valence_representation
from ...utils import get_element_ids_from_sorted
from ..base_constructors import get_dict2datatype, get_transform_list_dict2datatype

el_aug_rep_def = ["array_1D", "element_id", "list"]
el_aug_rep_dict_list2reps = get_transform_list_dict2datatype(el_aug_rep_def)
el_aug_rep_dict2rep = get_dict2datatype(el_aug_rep_def)


class RepresentationCalc:
    def __init__(
        self,
        representation_function=None,
        representation_function_kwargs=None,
    ):
        self.representation_function = representation_function
        self.representation_function_kwargs = representation_function_kwargs

    def get_representation(self, compound: Compound, **additional_kwargs):
        if self.representation_function is None:
            assert (
                compound.representation is not None
            ), "Attempting recovery of uninitialized representation!"
            return compound.representation
        return self.representation_function(
            compound.nuclear_charges,
            compound.coordinates,
            **self.representation_function_kwargs,
            **additional_kwargs,
        )

    def calc_with_component_bounds(self, compound: Compound, **additional_kwargs):
        pass

    def calc_rep(self, compound: Compound, **additional_kwargs):
        return self.calc_with_component_bounds(compound, **additional_kwargs)[0]


class LocalRepresentationCalc(RepresentationCalc):
    def __init__(
        self,
        representation_function=generate_fchl19,
        representation_function_kwargs={},
        element_representation_function=None,
        element_representation_function_kwargs={},
    ):
        super().__init__(
            representation_function=representation_function,
            representation_function_kwargs=representation_function_kwargs,
        )
        self.element_representation_function = element_representation_function
        self.element_representation_function_kwargs = element_representation_function_kwargs

    def calc_with_component_bounds(self, compound: Compound, **additional_kwargs):
        representation = self.get_representation(compound, **additional_kwargs)
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


def gen_element_augmented_local_rep_object(nuclear_charges, representation, sorted_elements):
    element_ids = get_element_ids_from_sorted(nuclear_charges, sorted_elements=sorted_elements)
    return [
        {"representation": rep, "element_id": el_id}
        for el_id, rep in zip(element_ids, representation)
    ]


class ElementAugmentedLocalRepresentationCalc(RepresentationCalc):
    def __init__(
        self,
        representation_function=generate_fchl19,
        representation_function_kwargs={},
        possible_nuclear_charges=array_([1, 6, 7, 8, 16], dtype=dint_),
    ):
        super().__init__(
            representation_function=representation_function,
            representation_function_kwargs=representation_function_kwargs,
        )
        self.sorted_elements = unique_(possible_nuclear_charges)

    def calc_with_component_bounds(self, compound: Compound, **additional_kwargs):
        representation = self.get_representation(compound, **additional_kwargs)
        output = gen_element_augmented_local_rep_object(
            compound.nuclear_charges, representation, self.sorted_elements
        )
        return output, array_([[0, representation.shape[1]]])


# K.Karan.: If needed in the future can write a calculator that combines more than one representation together.
class GlobalRepresentationCalc(RepresentationCalc):
    def __init__(
        self,
        representation_function=generate_coulomb_matrix,
        representation_function_kwargs={},
    ):
        super().__init__(
            representation_function=representation_function,
            representation_function_kwargs=representation_function_kwargs,
        )

    def calc_with_component_bounds(self, compound: Compound, **additional_kwargs):
        representation = self.representation_function(
            compound.nuclear_charges,
            compound.coordinates,
            **self.representation_function_kwargs,
            **additional_kwargs,
        )
        return representation, array_([[0, representation.shape[0]]])


class ProcessedRepresentationListCalc:
    def __init__(self, representation_object: RepresentationCalc):
        self.representation_object = representation_object
        self.component_bounds = None

    def single_calc_wkwargs(self, comp_kwargs_tuple):
        comp = comp_kwargs_tuple[0]
        kwargs = comp_kwargs_tuple[1]
        return self.representation_object.calc_with_component_bounds(comp, **kwargs)

    def _ingest(self, representation, component_bounds, output, allow_None_entries):
        if output is None:
            if (type(representation) in [dict, list]) or allow_None_entries:
                output = []
            else:
                output = typed.List()
        if allow_None_entries:
            check_component_bounds = component_bounds is not None
        else:
            check_component_bounds = True
        if check_component_bounds:
            if self.component_bounds is None:
                self.component_bounds = component_bounds
            else:
                assert self.component_bounds.shape == component_bounds.shape
                assert all_(self.component_bounds == component_bounds)
        output.append(representation)
        return output

    def __call__(
        self,
        compound_list,
        num_procs=None,
        fixed_num_threads=None,
        allow_None_entries=False,
        additional_kwarg_list=None,
        vacate_list=False,
    ):
        output = None

        # Single-process + vacate: inline the calc so peak memory is
        # max_N(|compound_list[N:]| + |output[:N]|) rather than
        # |compound_list| + |all results|. Requires fixed_num_threads is None
        # because thread-env reset only happens in embarrassingly_parallel's
        # child-exec path.
        if vacate_list and fixed_num_threads is None and default_num_procs(num_procs) == 1:
            for compound_id in range(len(compound_list)):
                compound = compound_list[compound_id]
                if additional_kwarg_list is None:
                    (
                        representation,
                        component_bounds,
                    ) = self.representation_object.calc_with_component_bounds(compound)
                else:
                    (
                        representation,
                        component_bounds,
                    ) = self.representation_object.calc_with_component_bounds(
                        compound, **additional_kwarg_list[compound_id]
                    )
                output = self._ingest(representation, component_bounds, output, allow_None_entries)
                compound_list[compound_id] = None
            return output

        if additional_kwarg_list is None:
            lr_output = embarrassingly_parallel(
                self.representation_object.calc_with_component_bounds,
                compound_list,
                (),
                num_procs=num_procs,
                fixed_num_threads=fixed_num_threads,
            )
        else:
            lr_output = embarrassingly_parallel(
                self.single_calc_wkwargs,
                zip(compound_list, additional_kwarg_list),
                (),
                num_procs=num_procs,
                fixed_num_threads=fixed_num_threads,
            )

        for compound_id in range(len(lr_output)):
            representation, component_bounds = lr_output[compound_id]
            output = self._ingest(representation, component_bounds, output, allow_None_entries)
            if vacate_list:
                compound_list[compound_id] = None
                lr_output[compound_id] = None
        return output

    def clear_component_bounds(self):
        self.component_bounds = None

    def max_component_bound(self):
        assert self.component_bounds is not None
        return self.component_bounds[-1][1]
