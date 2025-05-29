from ...ensemble import Ensemble
from ...jit_interfaces import all_, empty_
from ..base_constructors import get_transform_list_dict2datatype
from .fjk import comp_rep_def
from .standard import el_aug_rep_def


def comp2ensemble_rep_def(comp_rep_def):
    return comp_rep_def + ["list", "rhos", "list"]


fjk_comp_ensemble_rep_def = comp2ensemble_rep_def(comp_rep_def)
fjk_comp_ensemble_list_dict2datatype = get_transform_list_dict2datatype(fjk_comp_ensemble_rep_def)

global_rep_ensemble_rep_def = ["array_2D", "rhos", "list"]
global_rep_ensemble_list_dict2datatype = get_transform_list_dict2datatype(
    global_rep_ensemble_rep_def
)

local_rep_ensemble_rep_def = ["array_3D", "rhos", "list"]
local_rep_ensemble_list_dict2datatype = get_transform_list_dict2datatype(
    local_rep_ensemble_rep_def
)

local_dn_rep_ensemble_rep_def = comp2ensemble_rep_def(el_aug_rep_def)
local_dn_rep_ensemble_list_dict2datatype = get_transform_list_dict2datatype(
    local_dn_rep_ensemble_rep_def
)
local_dn_rep_conformers_list_dict2data = get_transform_list_dict2datatype(
    el_aug_rep_def + ["list"]
)


class EnsembleRepresentationCalc:
    def __init__(self, rep_calc):
        self.rep_calc = rep_calc

    def calc_processed_conf_list_wrhos(self, conf_list, output_bounds):
        nconfs = len(conf_list)
        rhos = empty_(nconfs)
        processed_conf_list = []
        for i, wcomp in enumerate(conf_list):
            processed_comp, bounds = self.rep_calc.calc_with_component_bounds(wcomp.compound)
            if output_bounds is None:
                output_bounds = bounds
            else:
                assert all_(output_bounds == bounds)
            processed_conf_list.append(processed_comp)
            rhos[i] = wcomp.rho
        return processed_conf_list, rhos, output_bounds

    def calc_with_component_bounds(self, ensemble: Ensemble):
        ensemble.run_calcs()
        output = []
        output_bounds = None
        for conf_list in ensemble.filtered_conformers:
            processed_conf_list, rhos, output_bounds = self.calc_processed_conf_list_wrhos(
                conf_list, output_bounds
            )
            output.append({"components": processed_conf_list, "rhos": rhos})
        return output, output_bounds


class MinConformerRepresentationCalc(EnsembleRepresentationCalc):
    def calc_with_component_bounds(self, ensemble: Ensemble):
        ensemble.run_calcs()
        min_conformers = [conf_list[0] for conf_list in ensemble.filtered_conformers]
        processed_conformers, _, output_bounds = self.calc_processed_conf_list_wrhos(
            min_conformers, None
        )
        return processed_conformers, output_bounds


class EnsembleArrayRepresentationCalc(EnsembleRepresentationCalc):
    def calc_processed_conf_list_wrhos(self, conf_list, output_bounds):
        nconfs = len(conf_list)
        rhos = empty_(nconfs)
        processed_conf_list = None
        for i, wcomp in enumerate(conf_list):
            processed_comp, bounds = self.rep_calc.calc_with_component_bounds(wcomp.compound)
            if output_bounds is None:
                output_bounds = bounds
            else:
                assert all_(output_bounds == bounds)
            if processed_conf_list is None:
                processed_conf_list = empty_((nconfs, *processed_comp.shape))
            processed_conf_list[i, :] = processed_comp[:]
            rhos[i] = wcomp.rho
        return processed_conf_list, rhos, output_bounds


# TODO K.Karan: Originally, only EnsembleGlobalRepresentationCalc existed. Then came last-minute decision to add SOAP requiring EnsembleLocalRepresentationCalc, but they ended up sharing the same code. I left them like that for now for paper script compatibility.
class EnsembleGlobalRepresentationCalc(EnsembleArrayRepresentationCalc):
    pass


class EnsembleLocalRepresentationCalc(EnsembleArrayRepresentationCalc):
    pass
