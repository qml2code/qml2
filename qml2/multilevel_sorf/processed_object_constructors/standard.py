from numba import typed

from ...compound import Compound
from ...jit_interfaces import all_, array_, concatenate_, dint_, unique_
from ...parallelization import default_num_procs, embarrassingly_parallel
from ...representations import generate_coulomb_matrix, generate_fchl19
from ...representations.element_representations import period_valence_representation
from ...representations.numba_calculators import CONFORMER_CLASS, CONFORMER_WLOCALREP_CLASS
from ...utils import get_element_ids_from_sorted, get_numba_list
from ..base_constructors import get_dict2datatype, get_transform_list_dict2datatype

# Datatype level names for the numba conformer classes, registered in base_constructors/base.py.
conformer_lvl = "conformer"
conformer_wlocalrep_lvl = "conformer_wlocalrep"

el_aug_rep_def = ["array_1D", "element_id", "list"]
el_aug_rep_dict_list2reps = get_transform_list_dict2datatype(el_aug_rep_def)
el_aug_rep_dict2rep = get_dict2datatype(el_aug_rep_def)


# Python Compound -> numba conformer
#
# The objects every other constructor in this module builds are FEATURE containers: a Compound goes
# in, a representation comes out, and the geometry is gone by the time jitted code sees it. A level
# that computes its representation inside the parallel loop needs the opposite -- the geometry
# itself, in a form numba can hold.
#
# These are calculator-free on purpose. `JITRepresentationCalculator.compound2numba_conformer` does
# the same job, but only if you already have a calculator; importing an ensemble and storing it as
# conformers should not require one.
#
# They use the CANONICAL classes rather than calling the factories. That is not a style preference:
# two factory calls give two distinct numba types, and a conformer of the wrong one is a hard
# TypingError wherever the type has been baked in (a typed.List of conformers, a compiled level).
# See CONFORMER_CLASS in qml2.representations.numba_calculators.


def compound2conformer(compound: Compound):
    """The geometry alone: nuclear charges and coordinates."""
    return CONFORMER_CLASS(compound.nuclear_charges, compound.coordinates)


def compound2conformer_wlocalrep(compound: Compound, representation=None):
    """Geometry PLUS a pre-calculated local representation, one row per atom.

    `representation` defaults to `compound.representation`, which is where a generate_* method and
    the ensemble importers leave it -- so a compound carrying pre-computed tight-binding charges
    converts with no extra argument.

    Both checks below exist because the alternative is a numba typing error several frames deeper
    that names neither the compound nor the field.
    """
    if representation is None:
        representation = compound.representation
    if representation is None:
        raise ValueError(
            "compound2conformer_wlocalrep needs a pre-calculated local representation, either "
            "passed explicitly or on compound.representation; both are None. Use "
            "compound2conformer if the conformer should carry geometry only."
        )
    representation = array_(representation)
    if representation.ndim != 2:
        raise ValueError(
            "the pre-calculated representation must be LOCAL, i.e. (natoms, nfeatures); got shape "
            f"{representation.shape}. A global (1-D) representation has no per-atom rows to append."
        )
    if representation.shape[0] != len(compound.nuclear_charges):
        raise ValueError(
            f"representation has {representation.shape[0]} rows but the compound has "
            f"{len(compound.nuclear_charges)} atoms; a local representation needs one row per atom."
        )
    return CONFORMER_WLOCALREP_CLASS(
        compound.nuclear_charges, compound.coordinates, representation
    )


def compounds2conformers(compounds, local_representations=None):
    """A numba list of conformers, which is what a jitted loop iterates over.

    Without `local_representations` the conformers carry geometry only; with it (one entry per
    compound, or None to take each compound's own `representation`) they carry the pre-calculated
    block too. A numba list is homogeneous, so this deliberately does not mix the two kinds -- a
    mixed list has no single element type and cannot be built.
    """
    output = get_numba_list()
    if local_representations is None:
        for compound in compounds:
            output.append(compound2conformer(compound))
        return output
    if local_representations is not None and len(local_representations) != len(compounds):
        raise ValueError(
            f"got {len(local_representations)} local representations for {len(compounds)} "
            "compounds; there must be exactly one per compound."
        )
    for compound, representation in zip(compounds, local_representations):
        output.append(compound2conformer_wlocalrep(compound, representation))
    return output


def compounds2conformers_wlocalrep(compounds):
    """`compounds2conformers` taking each compound's own `representation`, which is the shape the
    ensemble importers leave the data in."""
    return compounds2conformers(compounds, [None] * len(compounds))


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
        additional_comp_rep_functions=None,
        additional_comp_rep_function_kwargs=None,
        element_representation_function=None,
        element_representation_function_kwargs={},
    ):
        super().__init__(
            representation_function=representation_function,
            representation_function_kwargs=representation_function_kwargs,
        )
        self.additional_comp_rep_functions = additional_comp_rep_functions
        # One entry per function when given at all: a caller wanting no keyword arguments
        # for some of them passes {} for those, rather than a short list. Checked here
        # because the alternative is an IndexError from inside the per-compound loop, which
        # surfaces in a worker process with nothing pointing at the constructor call.
        if additional_comp_rep_function_kwargs is not None:
            assert additional_comp_rep_functions is not None, (
                "additional_comp_rep_function_kwargs given without "
                "additional_comp_rep_functions"
            )
            assert len(additional_comp_rep_function_kwargs) == len(
                additional_comp_rep_functions
            ), (
                f"{len(additional_comp_rep_function_kwargs)} keyword dictionaries for "
                f"{len(additional_comp_rep_functions)} additional representation functions; "
                "pass {} for the ones that need none"
            )
        self.additional_comp_rep_function_kwargs = additional_comp_rep_function_kwargs
        self.element_representation_function = element_representation_function
        self.element_representation_function_kwargs = element_representation_function_kwargs

    def calc_with_component_bounds(self, compound: Compound, **additional_kwargs):
        representation = self.get_representation(compound, **additional_kwargs)
        component_bounds = [(0, representation.shape[1])]
        if self.additional_comp_rep_functions is not None:
            for add_comp_rep_func_id, add_comp_rep_func in enumerate(
                self.additional_comp_rep_functions,
            ):
                if self.additional_comp_rep_function_kwargs is None:
                    add_rep_func_kwargs = {}
                else:
                    add_rep_func_kwargs = self.additional_comp_rep_function_kwargs[
                        add_comp_rep_func_id
                    ]
                add_rep = add_comp_rep_func(compound, **add_rep_func_kwargs)
                representation = concatenate_((representation, add_rep), axis=1)
                previous_component_bound = component_bounds[-1][1]
                component_bounds.append(
                    (previous_component_bound, previous_component_bound + add_rep.shape[1])
                )
        if self.element_representation_function is not None:
            element_representation = self.element_representation_function(
                compound.nuclear_charges, **self.element_representation_function_kwargs
            )
            representation = concatenate_((representation, element_representation), axis=1)
            component_bounds.append(
                (
                    component_bounds[-1][1],
                    component_bounds[-1][1] + element_representation.shape[1],
                )
            )
        component_bounds = array_(component_bounds, dtype=dint_)
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


class ElementAugmentedLocalRepresentationCalc(LocalRepresentationCalc):
    def __init__(
        self,
        *args,
        possible_nuclear_charges=array_([1, 6, 7, 8, 16], dtype=dint_),
        **kwargs,
    ):
        """As LocalRepresentationCalc, but each atom is tagged with its element id.

        `possible_nuclear_charges` is keyword-ONLY on purpose. It used to be the third
        positional parameter, after representation_function and its kwargs, and several
        callers pass the representation function positionally -- both CPU_times scripts do.
        Taking the first position here would bind the representation function to it, which
        raises "got multiple values for argument 'possible_nuclear_charges'" when the real
        one arrives by keyword, and would silently fall back to generate_fchl19 if it did
        not. The star keeps the old positional interface intact.
        """
        super().__init__(*args, **kwargs)
        self.sorted_elements = unique_(possible_nuclear_charges)

    def calc_with_component_bounds(self, compound: Compound, **kwargs):
        representation, component_bounds = super().calc_with_component_bounds(compound, **kwargs)
        output = gen_element_augmented_local_rep_object(
            compound.nuclear_charges, representation, self.sorted_elements
        )
        return output, component_bounds


# Compound -> numba conformer, for a level that computes its representation IN the parallel loop.
#
# These are the inverse of every other calculator in this module. Those return features and drop the
# geometry; these return the geometry and compute no features, leaving multilevel_sorf's
# `calculate_representation` level to compute them inside the loop and release them again per atom.
#
# Use them where the representation is LARGE relative to the geometry it comes from. The peak memory of
# the ordinary path is one representation per conformer per molecule, held for the whole SORF pass; here
# it is one per worker at a time, and what is stored per atom is a few numbers rather than a feature
# vector. The saving scales with the representation's width and with how many conformers, and how many
# states per conformer, each molecule contributes -- so it is largest exactly where the ordinary path
# is least affordable. In exchange the representation is recomputed on every pass over a conformer.
#
# Only the CONFORMER level is written here on purpose. PairLocalRepresentationCalc,
# EnsembleRepresentationCalc, MinConformerRepresentationCalc and ProcessedRepresentationListCalc need
# nothing from a calculator but `calc_with_component_bounds`, and the dict->datatype transforms accept
# a jitclass as the leaf they would otherwise build from a dict -- so every ensemble and pair shape
# composes from what already exists, with no parallel hierarchy to keep in step.


class NumbaConformerGenerator(RepresentationCalc):
    """Geometry only: nuclear charges and coordinates, in a form numba can hold.

    `component_bounds` is a constructor argument rather than something measured, and has to be:
    nothing here computes the representation those regions describe. It is passed through unchanged so
    `ProcessedRepresentationListCalc.component_bounds` still gives the `rescaling` level its regions,
    and so `_ingest`'s shape-and-equality check still has something to check. The caller is the only
    party that knows how wide the baked-in representation function is.
    """

    def __init__(self, component_bounds):
        super().__init__(representation_function=None, representation_function_kwargs={})
        if component_bounds is None:
            raise ValueError(
                "NumbaConformerGenerator needs component_bounds: it computes no representation, so "
                "it cannot measure the regions the rescaling level rescales. Pass the bounds of the "
                "representation the calculate_representation level is given."
            )
        self.component_bounds = array_(component_bounds)

    def processed_representation_def(self):
        """The datatype ONE conformer has, for a caller building a dict->datatype transform.

        Asked of the generator for the same reason PairLocalRepresentationCalc asks its inner
        calculator: which container it produces is a fact about it, and deciding that twice is how a
        transform and its data drift apart.
        """
        return [conformer_lvl]

    def calc_with_component_bounds(self, compound: Compound, **additional_kwargs):
        # A DICT, not a conformer. These calculators run in worker processes, and a jitclass cannot be
        # pickled, so a conformer built here could never reach the parent -- which is what the `reddt_`
        # half of the datatype and its un-jitted `jcopy_` exist for. The parent's
        # `get_transform_list_dict2datatype` turns this into the real jitclass. Same contract every
        # other calculator in this module follows: plain data out, datatype built on the far side.
        return {
            "nuclear_charges": compound.nuclear_charges,
            "coordinates": compound.coordinates,
        }, self.component_bounds


class NumbaConformerWLocalRepGenerator(NumbaConformerGenerator):
    """Geometry PLUS the per-atom quantities that no jitted function could recompute.

    The motivating one is the tight-binding atomic charge. `atomic_charge_representation` reads it off
    the converged calculation, so nothing inside numba can reproduce it from nuclear charges and
    coordinates -- which is precisely why it carries information the base representation does not. It
    rides along on the conformer, and `JITLocalRepresentationAdditionCalculator` concatenates it onto
    whatever the in-loop base representation returns.

    `comp_rep_functions` has the same shape as `LocalRepresentationCalc.additional_comp_rep_functions`:
    each takes the COMPOUND and returns (natoms, k), and they are concatenated in order. The charge
    column is one of these rather than a special case, so a second precomputed per-atom quantity needs
    no new class.
    """

    def __init__(self, comp_rep_functions, component_bounds, comp_rep_function_kwargs=None):
        super().__init__(component_bounds)
        if not comp_rep_functions:
            raise ValueError(
                "NumbaConformerWLocalRepGenerator with no comp_rep_functions carries nothing the "
                "in-loop representation could not compute itself; use NumbaConformerGenerator."
            )
        self.comp_rep_functions = list(comp_rep_functions)
        # One entry per function when given at all, checked here for the same reason
        # LocalRepresentationCalc checks it: otherwise it is an IndexError from inside a worker
        # process with nothing pointing back at the constructor call.
        if comp_rep_function_kwargs is None:
            comp_rep_function_kwargs = [{} for _ in self.comp_rep_functions]
        elif len(comp_rep_function_kwargs) != len(self.comp_rep_functions):
            raise ValueError(
                f"{len(self.comp_rep_functions)} comp_rep_functions but "
                f"{len(comp_rep_function_kwargs)} kwargs entries; pass {{}} for the ones that take "
                "no keyword arguments rather than a shorter list."
            )
        self.comp_rep_function_kwargs = list(comp_rep_function_kwargs)

    def local_representation(self, compound: Compound):
        columns = [
            array_(function(compound, **kwargs))
            for function, kwargs in zip(self.comp_rep_functions, self.comp_rep_function_kwargs)
        ]
        return concatenate_(columns, axis=1)

    def processed_representation_def(self):
        return [conformer_wlocalrep_lvl]

    def calc_with_component_bounds(self, compound: Compound, **additional_kwargs):
        # Plain data, for the reason given on NumbaConformerGenerator. The validation that
        # `compound2conformer_wlocalrep` performs is done here instead, so a malformed local
        # representation is still caught next to the compound that produced it rather than in the
        # parent with nothing to point at.
        representation = self.local_representation(compound)
        natoms = len(compound.nuclear_charges)
        if representation.ndim != 2 or representation.shape[0] != natoms:
            raise ValueError(
                f"the per-atom quantities came out with shape {representation.shape}, but the "
                f"compound has {natoms} atoms; each comp_rep_function must return (natoms, k)."
            )
        return {
            "nuclear_charges": compound.nuclear_charges,
            "coordinates": compound.coordinates,
            "representation": representation,
        }, self.component_bounds


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
