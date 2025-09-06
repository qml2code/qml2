"""
Procedures for processing of tuples (List[X],numpy.array), where X can be Compound or Ensemble, and numpy.array is an array of weights corresponding to each X instance.

Possible usecases:

- representing mixture of molecules (each represented by X instance), the numpy.array being e.g. molar proportion.

- representing a chemical-physical process (e.g. docking, absorption), with numpy.array equalling (-1, 1) (similar to FJK kernel) and List[X] being a pair of representations "before" and "after" (e.g. molecule in vacuum vs docked molecule).
"""
# TODO: should this be combined with Ensemble?
from ...jit_interfaces import all_, empty_
from ..base_constructors import get_transform_list_dict2datatype
from .standard import el_aug_rep_def


def comp2weighted_list_rep_def(comp_rep_def):
    return comp_rep_def + ["list", "rhos"]


global_rep_weighted_list_rep_def = ["array_2D", "rhos"]
global_rep_weighted_list_dict2datatype = get_transform_list_dict2datatype(
    global_rep_weighted_list_rep_def
)

local_dn_rep_weighted_list_rep_def = comp2weighted_list_rep_def(el_aug_rep_def)
local_dn_rep_weighted_list_list_dict2datatype = get_transform_list_dict2datatype(
    local_dn_rep_weighted_list_rep_def
)


class WeightedListRepresentationCalc:
    def __init__(self, rep_calc):
        self.rep_calc = rep_calc

    def calc_with_component_bounds(self, weighted_tuple):
        compounds = weighted_tuple[0]
        rhos = weighted_tuple[1]
        processed_compounds = []
        output_bounds = None
        for compound in compounds:
            processed_compound, bounds = self.rep_calc.calc_with_component_bounds(compound)
            if output_bounds is None:
                output_bounds = bounds
            else:
                assert all_(output_bounds == bounds)
            processed_compounds.append(processed_compound)
        return {"components": processed_compounds, "rhos": rhos}, output_bounds


class WeightedListGlobalRepresentationCalc(WeightedListRepresentationCalc):
    def calc_with_component_bounds(self, weighted_tuple):
        compounds = weighted_tuple[0]
        rhos = weighted_tuple[1]
        ncompounds = len(compounds)
        processed_compounds = None
        output_bounds = None
        for i, compound in enumerate(compounds):
            processed_compound, bounds = self.rep_calc.calc_with_component_bounds(compound)
            if output_bounds is None:
                output_bounds = bounds
            else:
                assert all_(output_bounds == bounds)
            if processed_compounds is None:
                processed_compounds = empty_((ncompounds, *processed_compound.shape))
            processed_compounds[i, :] = processed_compound[:]
        return {"components": processed_compounds, "rhos": rhos}, output_bounds
