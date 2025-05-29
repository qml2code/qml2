from ...jit_interfaces import all_, array_, concatenate_, dint_, empty_
from ...orb_ml.oml_compound import OML_Compound, OML_Slater_pair, OML_Slater_pairs
from ...orb_ml.representations import OML_orb_rep, OML_rep_params
from ...utils import get_element_ids_from_sorted, get_sorted_elements
from ..base_constructors import (
    get_datatype,
    get_datatype2dict,
    get_dict2datatype,
    get_transform_list_dict2datatype,
)

"""
FJK Compound is converted into a datatype where:
1. each atomic contribution to a localized orbital is given a vector.
2. atomic contributions are grouped into weighted lists that represent a localized orbital
3. a compound is represented by a weighed list of localized orbitals.
"""
orb_rep_def = ["array_2D", "rhos"]
comp_rep_def = orb_rep_def + ["list", "rhos"]
slater_pairs_rep_def = comp_rep_def + ["list"]

orb_rep_type = get_datatype(orb_rep_def)
comp_rep_type = get_datatype(comp_rep_def)

comp_rep2dict = get_datatype2dict(comp_rep_def)
comp_dict2rep = get_dict2datatype(comp_rep_def)

slater_pairs_rep2dict = get_datatype2dict(slater_pairs_rep_def)
slater_pairs_dict2rep = get_dict2datatype(slater_pairs_rep_def)

comp_dict_list2reps = get_transform_list_dict2datatype(comp_rep_def)
slater_pairs_dict_list2reps = get_transform_list_dict2datatype(slater_pairs_rep_def)


def fjk_rep_component_bounds(oml_rep_params=OML_rep_params()):
    max_angular_momentum = oml_rep_params.max_angular_momentum
    if max_angular_momentum == 0:
        nregions = 3  # for F, J, and K matrices
        lb_mat_range = 0
        lb_mat_id = 0
    else:
        nregions = 4  # + angular moment distribution
        lb_mat_range = max_angular_momentum
        lb_mat_id = 1
    num_ang_mom_vals = max_angular_momentum + 1
    bounds = empty_((nregions, 2), dtype=dint_)
    for mat_id in range(3):
        ub_mat_range = (
            lb_mat_range + num_ang_mom_vals**2 + (num_ang_mom_vals + 1) * num_ang_mom_vals // 2
        )
        bounds[mat_id + lb_mat_id, 0] = lb_mat_range
        bounds[mat_id + lb_mat_id, 1] = ub_mat_range
        lb_mat_range = ub_mat_range
    if max_angular_momentum != 0:
        bounds[0, 0] = 0
        bounds[0, 1] = max_angular_momentum
    return bounds


def processed_orb_rep(orb_rep: OML_orb_rep, local_representations=None):
    nareps = len(orb_rep.orb_atom_reps)
    orb_rep_size = orb_rep.orb_atom_reps[0].scalar_reps.shape[0]
    tot_rep_size = orb_rep_size
    if local_representations is not None:
        tot_rep_size += local_representations.shape[1]
    representations = empty_((nareps, tot_rep_size))
    rhos = empty_(nareps)
    for arep_id, orb_atom_rep in enumerate(orb_rep.orb_atom_reps):
        representations[arep_id, -orb_rep_size:] = orb_atom_rep.scalar_reps[:]
        rhos[arep_id] = orb_atom_rep.rho
        if local_representations is not None:
            atom_id = orb_atom_rep.atom_id
            representations[arep_id, :-orb_rep_size] = local_representations[atom_id, :]
    final_output = {"components": representations, "rhos": rhos}
    return final_output


class FJKRepresentationCalc:
    def __init__(self, oml_rep_params=OML_rep_params(), local_representation_processor=None):
        self.oml_rep_params = oml_rep_params
        self.local_representation_processor = local_representation_processor
        self.local_rep_added = self.local_representation_processor is not None

    def append_orb_representations(self, list_in, rho_list, compound: OML_Compound):
        if self.local_rep_added:
            (
                local_representations,
                local_representation_component_bounds,
            ) = self.local_representation_processor.calc_with_component_bounds(compound)
        else:
            local_representations = None
            local_representation_component_bounds = None
        for orb_rep in compound.orb_reps:
            rho_list.append(orb_rep.rho)
            list_in.append(processed_orb_rep(orb_rep, local_representations=local_representations))

        return local_representation_component_bounds

    def calc_with_component_bounds(
        self, compound: OML_Compound | OML_Slater_pair | OML_Slater_pairs
    ):
        if isinstance(compound, OML_Slater_pairs):
            output = []
            prev_bounds = None
            for slater_pair in compound.get_Slater_pairs():
                slater_pair_rep, bounds = self.calc_with_component_bounds(slater_pair)
                if prev_bounds is None:
                    prev_bounds = bounds
                else:
                    assert all_(prev_bounds == bounds)
                output.append(slater_pair_rep)
            return output, bounds

        compound.generate_orb_reps(rep_params=self.oml_rep_params)
        orb_list = []
        rho_list = []
        if isinstance(compound, OML_Compound):
            local_rep_calc_bounds = self.append_orb_representations(orb_list, rho_list, compound)
        elif isinstance(compound, OML_Slater_pair):
            local_rep_calc_bounds = self.append_orb_representations(
                orb_list, rho_list, compound.comps[0]
            )
            nfirst = len(rho_list)
            local_rep_calc_bounds2 = self.append_orb_representations(
                orb_list, rho_list, compound.comps[1]
            )
            if self.local_rep_added:
                assert all_(local_rep_calc_bounds == local_rep_calc_bounds2)
        else:
            raise Exception
        rhos = array_(rho_list)
        if isinstance(compound, OML_Slater_pair):
            rhos[:nfirst] *= -1
        comp_rep = {"components": orb_list, "rhos": rhos}

        bounds = fjk_rep_component_bounds(oml_rep_params=self.oml_rep_params)
        if self.local_rep_added:
            bounds += local_rep_calc_bounds[-1, -1]
            bounds = concatenate_([local_rep_calc_bounds, bounds])

        return comp_rep, bounds


# Everything related to delta-nuclear-charge:
orb_dn_rep_def = ["array_1D", "element_id", "list", "rhos"]
comp_dn_rep_def = orb_dn_rep_def + ["list", "rhos"]
slater_pairs_dn_rep_def = comp_dn_rep_def + ["list"]

orb_dn_rep_type = get_datatype(orb_dn_rep_def)
comp_dn_rep_type = get_datatype(comp_dn_rep_def)

comp_dn_rep2dict = get_datatype2dict(comp_dn_rep_def)
comp_dn_dict2rep = get_dict2datatype(comp_dn_rep_def)

slater_pairs_dn_rep2dict = get_datatype2dict(slater_pairs_dn_rep_def)
slater_pairs_dn_dict2rep = get_dict2datatype(slater_pairs_dn_rep_def)

comp_dn_dict_list2reps = get_transform_list_dict2datatype(comp_dn_rep_def)
slater_pairs_dn_dict_list2reps = get_transform_list_dict2datatype(slater_pairs_dn_rep_def)


def processed_atom_orb_rep_dn(
    orb_atom_rep, element_ids, orb_rep_size, tot_rep_size, local_representations=None
):
    representation = empty_((tot_rep_size,))
    atom_id = orb_atom_rep.atom_id
    if local_representations is not None:
        representation[: local_representations.shape[1]] = local_representations[atom_id, :]
    representation[-orb_rep_size:] = orb_atom_rep.scalar_reps[:]
    return {"representation": representation, "element_id": element_ids[atom_id]}


def processed_orb_rep_dn(orb_rep: OML_orb_rep, element_ids, local_representations=None):
    nareps = len(orb_rep.orb_atom_reps)
    orb_rep_size = orb_rep.orb_atom_reps[0].scalar_reps.shape[0]
    tot_rep_size = orb_rep_size
    if local_representations is not None:
        tot_rep_size += local_representations.shape[1]
    representations = []
    rhos = empty_(nareps)
    for arep_id, orb_atom_rep in enumerate(orb_rep.orb_atom_reps):
        representations.append(
            processed_atom_orb_rep_dn(
                orb_atom_rep,
                element_ids,
                orb_rep_size,
                tot_rep_size,
                local_representations=local_representations,
            )
        )
        rhos[arep_id] = orb_atom_rep.rho
    final_output = {"components": representations, "rhos": rhos}
    return final_output


class FJKdnRepresentationCalc(FJKRepresentationCalc):
    def __init__(
        self,
        oml_rep_params=OML_rep_params(),
        representation_function=None,
        representation_function_kwargs={},
        possible_nuclear_charges=array_([1, 6, 7, 8, 16], dtype=dint_),
    ):
        self.oml_rep_params = oml_rep_params
        self.representation_function = representation_function
        self.representation_function_kwargs = representation_function_kwargs
        self.sorted_elements = get_sorted_elements(possible_nuclear_charges)
        self.local_rep_added = self.representation_function is not None

    def append_orb_representations(self, list_in, rho_list, compound: OML_Compound):
        if self.local_rep_added:
            local_representations = self.representation_function(
                compound.nuclear_charges,
                compound.coordinates,
                **self.representation_function_kwargs
            )
        else:
            local_representations = None
        element_ids = get_element_ids_from_sorted(
            compound.nuclear_charges, sorted_elements=self.sorted_elements
        )
        for orb_rep in compound.orb_reps:
            rho_list.append(orb_rep.rho)
            list_in.append(
                processed_orb_rep_dn(
                    orb_rep, element_ids, local_representations=local_representations
                )
            )

        if self.local_rep_added:
            local_representation_component_bounds = array_([[0, local_representations.shape[1]]])
        else:
            return None

        return local_representation_component_bounds
