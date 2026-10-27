from ...basic_utils import OptionUnavailableError
from ...jit_interfaces import all_, array_, concatenate_, dint_, empty_, unique_
from ...utils import get_element_ids_from_sorted
from ..base_constructors import (
    get_datatype,
    get_datatype2dict,
    get_dict2datatype,
    get_transform_list_dict2datatype,
)

# all the orb_ml imports are made optional to not cause problems in ensemble.
try:
    from ...orb_ml.oml_compound import OML_Compound, OML_Slater_pair, OML_Slater_pairs
    from ...orb_ml.representations import OML_orb_rep, OML_rep_params

    default_OML_rep_params = OML_rep_params()
except ModuleNotFoundError as ex:
    from ...basic_utils import ExceptionRaisingClass

    OML_rep_params = ExceptionRaisingClass(ex)
    OML_orb_rep = ExceptionRaisingClass(ex)
    OML_Compound = ExceptionRaisingClass(ex)
    OML_Slater_pair = ExceptionRaisingClass(ex)
    OML_Slater_pairs = ExceptionRaisingClass(ex)
    default_OML_rep_params = None

orb_rep_def = ["array_2D", "rhos"]
comp_rep_def = orb_rep_def + ["list", "rhos"]
slater_pairs_rep_def = comp_rep_def + ["list"]
# A Slater pair whose determinants are kept apart rather than summed with opposite signs:
# structurally a plain list of compound representations, which is what a "concatenation"
# level consumes. Same shape as slater_pairs_rep_def, named separately because it means
# something different -- two determinants of ONE pair rather than several pairs.
concatenated_pair_rep_def = comp_rep_def + ["list"]

orb_rep_type = get_datatype(orb_rep_def)
comp_rep_type = get_datatype(comp_rep_def)

comp_rep2dict = get_datatype2dict(comp_rep_def)
comp_dict2rep = get_dict2datatype(comp_rep_def)

slater_pairs_rep2dict = get_datatype2dict(slater_pairs_rep_def)
slater_pairs_dict2rep = get_dict2datatype(slater_pairs_rep_def)

comp_dict_list2reps = get_transform_list_dict2datatype(comp_rep_def)
slater_pairs_dict_list2reps = get_transform_list_dict2datatype(slater_pairs_rep_def)
concatenated_pair_dict_list2reps = get_transform_list_dict2datatype(concatenated_pair_rep_def)


def fjk_rep_component_bounds(oml_rep_params=default_OML_rep_params):
    """Index ranges of the regions that are rescaled independently of one another.

    One region per coupling channel AS STORED, preceded by the angular-momentum
    distribution block unless there is a single angular momentum, where the distribution
    carries no information. Each region receives its own length scale --
    ``calc_hyperparameter_num`` returns one sigma per region -- so the list has to tile the
    feature vector exactly.

    "As stored" is the part worth stating. For FJD the channels are not the matrices that
    went in: its three shifted-dipole components are contracted to one by the l2 merge, and
    its bond projection carries no self block. A region list built from the matrix count
    would leave the tail of every feature vector outside every region, and nothing
    downstream re-checks the coverage -- so those features would silently lose their length
    scale rather than raise.
    """
    max_angular_momentum = oml_rep_params.max_angular_momentum
    num_ang_mom_vals = max_angular_momentum + 1
    self_block = (num_ang_mom_vals + 1) * num_ang_mom_vals // 2
    other_block = num_ang_mom_vals**2

    if oml_rep_params.fjd_coup_mats:
        # F, J and the merged shifted-dipole channel, then the bond projection, whose self
        # coupling vanishes identically and is not stored.
        channel_sizes = [self_block + other_block] * 3 + [other_block]
    else:
        channel_sizes = [self_block + other_block] * 3  # F, J and K

    distribution_component_present = max_angular_momentum != 0
    nregions = len(channel_sizes) + int(distribution_component_present)
    bounds = empty_((nregions, 2), dtype=dint_)

    region_id = 0
    lower_bound = 0
    if distribution_component_present:
        bounds[0, 0] = 0
        bounds[0, 1] = max_angular_momentum
        lower_bound = max_angular_momentum
        region_id = 1
    for channel_size in channel_sizes:
        bounds[region_id, 0] = lower_bound
        bounds[region_id, 1] = lower_bound + channel_size
        lower_bound += channel_size
        region_id += 1
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
    def __init__(
        self,
        oml_rep_params=default_OML_rep_params,
        local_representation_processor=None,
        concatenate_components=False,
    ):
        """
        Converts an OML Compound with FJK representation into a datatype where:
        1. each atomic contribution to a localized orbital is given a vector.
        2. atomic contributions are grouped into weighted lists that represent a localized orbital
        3. a compound is represented by a weighed list of localized orbitals.

        A Slater pair's two determinants can be combined in either of two ways, and the
        choice is made here rather than in the SORF stack:

        * `concatenate_components=False` (the default) puts both determinants' orbitals in
          ONE weighted list and negates the first determinant's weights, so that whatever
          sums over that list returns a DIFFERENCE of the two;
        * `concatenate_components=True` keeps them apart, returning a list of two compound
          representations for a "concatenation" level to lay end to end. The output is then
          twice as long and the SORF stack needs that level with `num_components=2`.

        Either way the ORDER of the determinants is part of the representation. The
        difference is antisymmetric, so swapping them flips its sign; the concatenation is
        not symmetric at all, so swapping them produces an unrelated vector. In both cases
        the pairs must be built the same way round everywhere -- a training set assembled
        as (gas, solvated) cannot be used to predict pairs assembled as (solvated, gas).

        `concatenate_components=True` accepts nothing but an OML_Slater_pair: a single
        compound has no components to concatenate, and OML_Slater_pairs is not supported.
        """

        self.oml_rep_params = oml_rep_params
        self.local_representation_processor = local_representation_processor
        self.local_rep_added = self.local_representation_processor is not None
        self.concatenate_components = concatenate_components

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

    def single_compound_representation(self, compound: OML_Compound):
        """One compound as a weighted list of its localized orbitals."""
        orb_list = []
        rho_list = []
        local_rep_calc_bounds = self.append_orb_representations(orb_list, rho_list, compound)
        return {"components": orb_list, "rhos": array_(rho_list)}, local_rep_calc_bounds

    def component_bounds(self, local_rep_calc_bounds):
        """Where one COMPONENT's rescaling regions are.

        Unchanged by concatenation: the rescaling happens before the components are laid
        end to end, so the regions describe one determinant either way and the number of
        sigmas does not double.
        """
        bounds = fjk_rep_component_bounds(oml_rep_params=self.oml_rep_params)
        if self.local_rep_added:
            bounds += local_rep_calc_bounds[-1, -1]
            bounds = concatenate_([local_rep_calc_bounds, bounds])
        return bounds

    def calc_with_component_bounds(
        self, compound: OML_Compound | OML_Slater_pair | OML_Slater_pairs
    ):
        if self.concatenate_components:
            if not isinstance(compound, OML_Slater_pair) or isinstance(compound, OML_Slater_pairs):
                raise OptionUnavailableError(
                    "concatenate_components needs an OML_Slater_pair to have components to "
                    f"concatenate; got {type(compound).__name__}"
                )
            compound.generate_orb_reps(rep_params=self.oml_rep_params)
            # A plain list of the two determinants, in the order the pair was built. Nothing
            # is negated and nothing is merged -- the "concatenation" level lays them end to
            # end, so the order they appear in here is the order they appear in the features.
            component_representations = []
            local_rep_calc_bounds = None
            for determinant in compound.comps:
                representation, determinant_bounds = self.single_compound_representation(
                    determinant
                )
                if self.local_rep_added:
                    if local_rep_calc_bounds is None:
                        local_rep_calc_bounds = determinant_bounds
                    else:
                        assert all_(local_rep_calc_bounds == determinant_bounds)
                component_representations.append(representation)
            return component_representations, self.component_bounds(local_rep_calc_bounds)

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
            # The first determinant enters negatively, so that summing over the merged list
            # returns comps[1] - comps[0]. Antisymmetric: building the pair the other way
            # round flips the sign of every feature.
            rhos[:nfirst] *= -1
        comp_rep = {"components": orb_list, "rhos": rhos}

        return comp_rep, self.component_bounds(local_rep_calc_bounds)


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
        oml_rep_params=default_OML_rep_params,
        representation_function=None,
        representation_function_kwargs={},
        possible_nuclear_charges=array_([1, 6, 7, 8, 16], dtype=dint_),
    ):
        """
        Same as FJKRepresentationCalc, but the atomic component kernel elements contain delta function.
        """
        self.oml_rep_params = oml_rep_params
        self.representation_function = representation_function
        self.representation_function_kwargs = representation_function_kwargs
        self.sorted_elements = unique_(possible_nuclear_charges)
        self.local_rep_added = self.representation_function is not None

    def append_orb_representations(self, list_in, rho_list, compound: OML_Compound):
        if self.local_rep_added:
            local_representations = self.representation_function(
                compound.nuclear_charges,
                compound.coordinates,
                **self.representation_function_kwargs,
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
