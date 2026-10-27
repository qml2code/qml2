"""
Generators of Numba representation functions that can be used inside other Numba JIT-created functions.
"""
import inspect
from typing import List

from numba import float64, int64
from numba.experimental import jitclass

from ..compound import Compound
from ..jit_interfaces import array_, concatenate_, dint_, jit_, sort_, zeros_
from ..utils import get_numba_list
from .basic_utilities import calculate_distances_wrcut
from .calculators import get_check_all_nuclear_charges, get_check_elements
from .fchl19 import generate_fchl19_no_gradients
from .slatm import generate_slatm_global, generate_slatm_local, get_slatm_mbtypes


def get_standard_numba_compound_class():
    """
    Creates a Numba class for a compound containing nuclear charges and coordinates.

    NOTE: kept in a function because jitclass is not picklable yet.
    """

    @jitclass([("nuclear_charges", int64[:]), ("coordinates", float64[:, :])])
    class Conformer:
        def __init__(self, nuclear_charges, coordinates):
            """
            Analogue of Compound to be used inside Numba-JIT-ed functions.
            """
            self.nuclear_charges = nuclear_charges
            self.coordinates = coordinates

    return Conformer


def get_numba_compound_class_with_local_representation():
    """
    Creates a Numba class for a compound containing nuclear charges, coordinates, and representation which had to be pre-calculated outside Numba JIT.

    """

    @jitclass(
        [
            ("nuclear_charges", int64[:]),
            ("coordinates", float64[:, :]),
            ("representation", float64[:, :]),
        ]
    )
    class ConformerWLocalRepresentation:
        def __init__(self, nuclear_charges, coordinates, representation):
            """
            Analogue of Compound to be used inside Numba-JIT-ed functions.
            """
            self.nuclear_charges = nuclear_charges
            self.coordinates = coordinates
            self.representation = representation

    return ConformerWLocalRepresentation


# THE canonical conformer types. Everything that hands a conformer to jitted code -- the calculators
# here, the processed-object constructors, and the multilevel_sorf datatypes -- must use these and
# not call the factories again.
#
# Two calls to a factory produce two DISTINCT numba types with identical fields. A generic @jit_
# function silently recompiles for the second one (a cost, not an error), but anywhere a type is
# BAKED IN -- a typed.List of conformers, a jitclass field, a level compiled against one -- a foreign
# conformer is a hard TypingError. Holding one of each here is what makes "a conformer" mean one
# thing process-wide. The factories stay for a caller that deliberately wants a fresh type.
CONFORMER_CLASS = get_standard_numba_compound_class()
CONFORMER_WLOCALREP_CLASS = get_numba_compound_class_with_local_representation()


def generate_trailing_arg_tuple(function, **kwargs):
    sig = inspect.signature(function)
    function_keys = []
    out = []
    for name, p in sig.parameters.items():
        # skip *args/**kwargs containers
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD) or p.default is inspect._empty:
            continue
        function_keys.append(name)
        if name in kwargs:
            val = kwargs[name]
        else:
            val = p.default
        out.append(val)
    for k in kwargs.keys():
        assert k in function_keys
    return tuple(out)


class JITRepresentationCalculator:
    def __init__(self, jitted_rep_function, *extra_args, **extra_kwargs):
        self.jitted_rep_function = jitted_rep_function
        self.extra_args = extra_args
        self.extra_kwargs = extra_kwargs
        self.numba_compound_class = None
        self.init_numba_calculator()

    def init_numba_calculator(self):
        self.numba_calculator = self.get_single_compound_calculator()

    def generate_numba_compound_class(self):
        """The canonical class, NOT a fresh one from the factory -- see CONFORMER_CLASS on why a
        per-calculator type breaks anything that bakes the type in."""
        return CONFORMER_CLASS

    def compound2numba_conformer(self, comp: Compound):
        if self.numba_compound_class is None:
            self.numba_compound_class = self.generate_numba_compound_class()
        return self.numba_compound_class(comp.nuclear_charges, comp.coordinates)

    def compounds2numba_conformers(self, comps: List[Compound]):
        return get_numba_list([self.compound2numba_conformer(comp) for comp in comps])

    def generate_trailing_arg_tuple(self):
        return generate_trailing_arg_tuple(self.jitted_rep_function, **self.extra_kwargs)

    def get_single_compound_calculator(self, parallel=False):
        trailing_arg_tuple = self.generate_trailing_arg_tuple()
        rep_func = self.jitted_rep_function
        final_arg_tuple = (*self.extra_args, *trailing_arg_tuple)

        @jit_(numba_parallel=parallel)
        def final_function(conformer):
            return rep_func(conformer.nuclear_charges, conformer.coordinates, *final_arg_tuple)

        return final_function

    def single_coordinate_calc(self, nuclear_charges, coordinates):
        """
        Suboptimal, introduced for testing purposes.
        """
        conformer = self.compound2numba_conformer(
            Compound(nuclear_charges=nuclear_charges, coordinates=coordinates)
        )
        return self.numba_calculator(conformer)

    def __getstate__(self):
        output = self.__dict__
        output["numba_compound_class"] = None
        return output


class FCHL19JITCalculator(JITRepresentationCalculator):
    def __init__(
        self, elements=None, all_nuclear_charges=None, all_compounds=None, **other_rep_kwargs
    ):
        """
        Calculator for FCHL19 representation.
        """
        self.elements = get_check_elements(
            elements=elements, all_nuclear_charges=all_nuclear_charges, all_compounds=all_compounds
        )
        super().__init__(generate_fchl19_no_gradients, elements=self.elements, **other_rep_kwargs)


class SLATMJITCalculator(JITRepresentationCalculator):
    def __init__(
        self,
        all_nuclear_charges=None,
        all_compounds=None,
        local=False,
        rcut=4.8,
        **other_rep_kwargs,
    ):
        """
        Calculator for SLATM representation. Use keyword argument `local=True` to generate aSLATM instead.
        """
        all_nuclear_charges = get_check_all_nuclear_charges(
            all_nuclear_charges=all_nuclear_charges, all_compounds=all_compounds
        )
        slatm_mbtypes = get_slatm_mbtypes(all_nuclear_charges)
        self.local = local
        self.rcut = rcut
        if self.local:
            inner_slatm_func = generate_slatm_local
        else:
            inner_slatm_func = generate_slatm_global
        trailing_arg_tuple = generate_trailing_arg_tuple(
            inner_slatm_func, rcut=rcut, **other_rep_kwargs
        )

        @jit_
        def jitted_rep_function(nuclear_charges, coordinates):
            distances, neighbor_ids, num_neighbors = calculate_distances_wrcut(
                coordinates, len(coordinates), rcut * 2
            )
            return inner_slatm_func(
                nuclear_charges,
                distances,
                neighbor_ids,
                num_neighbors,
                slatm_mbtypes,
                len(nuclear_charges),
                *trailing_arg_tuple,
            )

        super().__init__(jitted_rep_function)


class aSLATMJITCalculator(SLATMJITCalculator):
    def __init__(self, **kwargs):
        """
        Calculator for aSLATM representation.
        """
        super().__init__(local=True, **kwargs)


class cMBDFJITCalculator(JITRepresentationCalculator):
    def __init__(
        self,
        local=True,
        all_nuclear_charges=None,
        all_compounds=None,
        rcut=10.0,
        n_atm=2.0,
        convolutions=None,
        **convolution_kwargs,
    ):
        """
        Calculator for the cMBDF representation. Use keyword argument
        `local=True` (default) for the local (atomic) representation or
        `local=False` for the global (bagged) molecular representation.

        The global representation is bagged per element and therefore
        depends on the dataset-wide largest per-element atom count, so
        `all_nuclear_charges` (or `all_compounds`) must be supplied when
        `local=False`; both are ignored when `local=True`.

        The FFT-based convolution grids (`get_convolutions`) are
        dataset-independent; they are computed once here (or supplied
        via `convolutions`). The per-molecule kernel is fully
        JIT-compiled, so cMBDF can be used inside fcRFLocalModel
        exactly like FCHL19/aSLATM, with no multiprocessing.
        """
        # Imported here rather than at module level: cMBDF pulls in
        # scipy, and qml2/representations/__init__.py already guards that
        # import behind a try/except.
        from .cMBDF import generate_data, get_asize, get_convolutions

        if convolutions is None:
            convolutions = get_convolutions(rcut=rcut, **convolution_kwargs)
        else:
            assert (
                not convolution_kwargs
            ), "convolution_kwargs only apply when `convolutions` is computed here"
        self.local = local
        self.rcut = rcut
        self.n_atm = n_atm
        rconvs, aconvs = convolutions

        if local:

            @jit_
            def jitted_rep_function(nuclear_charges, coordinates):
                size = len(nuclear_charges)
                twob, threeb = generate_data(
                    size, nuclear_charges, coordinates, rconvs, aconvs, rcut, n_atm
                )
                # Local cMBDF (cf. generate_cmbdf with local=True): collapse
                # the two-body block over the neighbour index and the
                # three-body block over both neighbour indices, leaving one
                # representation row per atom.
                nrs = twob.shape[2]
                nas = threeb.shape[3]
                representation = zeros_((size, nrs + nas))
                representation[:, :nrs] = twob.sum(axis=1)
                representation[:, nrs:] = threeb.sum(axis=1).sum(axis=1)
                return representation

        else:
            all_nuclear_charges = get_check_all_nuclear_charges(
                all_nuclear_charges=all_nuclear_charges, all_compounds=all_compounds
            )
            self.asize = get_asize(all_nuclear_charges)
            # `get_asize` keeps elements in ascending-charge order; the
            # bags are laid out contiguously in that same order.
            bag_charges = array_(list(self.asize.keys()), dtype=dint_)
            bag_sizes = array_(list(self.asize.values()), dtype=dint_)

            @jit_
            def jitted_rep_function(nuclear_charges, coordinates):
                size = len(nuclear_charges)
                twob, threeb = generate_data(
                    size, nuclear_charges, coordinates, rconvs, aconvs, rcut, n_atm
                )
                # Per-atom (local) cMBDF rows, exactly as in the `local`
                # branch above.
                nrs = twob.shape[2]
                nas = threeb.shape[3]
                desc_size = nrs + nas
                local_rep = zeros_((size, desc_size))
                local_rep[:, :nrs] = twob.sum(axis=1)
                local_rep[:, nrs:] = threeb.sum(axis=1).sum(axis=1)
                # Global cMBDF (cf. generate_cmbdf with local=False): bag
                # the per-atom rows by element, sort each bag's columns
                # in descending order, zero-pad every bag to the
                # dataset-wide per-element atom count, then flatten.
                rep_size = 0
                for ki in range(len(bag_sizes)):
                    rep_size += bag_sizes[ki]
                mat = zeros_((rep_size, desc_size))
                ind = 0
                for ki in range(len(bag_charges)):
                    key = bag_charges[ki]
                    num = 0
                    for i in range(size):
                        if nuclear_charges[i] == key:
                            num += 1
                    if num != 0:
                        bag = zeros_((num, desc_size))
                        bi = 0
                        for i in range(size):
                            if nuclear_charges[i] == key:
                                bag[bi] = local_rep[i]
                                bi += 1
                        for c in range(desc_size):
                            mat[ind : ind + num, c] = -sort_(-bag[:, c])
                    ind += bag_sizes[ki]
                return mat.ravel()

        super().__init__(jitted_rep_function)


class cMBDFLocalJITCalculator(cMBDFJITCalculator):
    def __init__(self, **kwargs):
        """
        Calculator for the local (atomic) cMBDF representation.
        """
        super().__init__(local=True, **kwargs)


class JITLocalRepresentationAdditionCalculator(JITRepresentationCalculator):
    """Appends a PRE-CALCULATED local representation to a freshly calculated one, per atom.

    The pre-calculated block is whatever could not be produced inside numba -- tight-binding charges
    being the motivating case. It rides along on the conformer object
    (:func:`get_numba_compound_class_with_local_representation`), so the combination happens inside
    the jitted body and nothing has to leave the parallel loop to fetch it.

    Both halves are LOCAL: the result is ``(natoms, n_fresh + n_precalculated)``, one row per atom.
    That is why the concatenation is along axis 1, and why a global representation cannot be used on
    either side.
    """

    def __init__(self, calculator: JITRepresentationCalculator):
        # Set before super().__init__, which ends by calling init_numba_calculator ->
        # get_single_compound_calculator, and that needs the nested calculator to already be here.
        self.calculator = calculator
        # The parent's constructor contract describes ONE representation function, while this class
        # wraps another calculator. Borrowing the nested calculator's function and arguments keeps
        # the inherited introspection (generate_trailing_arg_tuple, __getstate__) describing
        # something true -- the nested representation -- rather than leaving those attributes unset,
        # which is what made every inherited method fail.
        super().__init__(
            calculator.jitted_rep_function, *calculator.extra_args, **calculator.extra_kwargs
        )

    def generate_numba_compound_class(self):
        """Not the standard Conformer: this calculator reads ``conformer.representation``, so the
        conformer has to be the variant that carries it -- and the canonical one, per
        CONFORMER_WLOCALREP_CLASS."""
        return CONFORMER_WLOCALREP_CLASS

    def compound2numba_conformer(self, comp: Compound):
        """``comp.representation`` must already hold the PRE-CALCULATED block -- this calculator
        appends to it, it does not produce it. Checked here because the alternative is a numba
        typing error several frames deeper, naming neither the compound nor the missing field."""
        if comp.representation is None:
            raise ValueError(
                "JITLocalRepresentationAdditionCalculator needs comp.representation to hold the "
                "pre-calculated local representation (e.g. tight-binding charges) that is appended "
                "to the freshly calculated one; it is None. Compute it before converting the "
                "compound."
            )
        representation = array_(comp.representation)
        if representation.ndim != 2:
            raise ValueError(
                "the pre-calculated representation must be LOCAL, i.e. (natoms, nfeatures); got "
                f"shape {representation.shape}. A global (1-D) representation cannot be appended "
                "per atom."
            )
        if self.numba_compound_class is None:
            self.numba_compound_class = self.generate_numba_compound_class()
        return self.numba_compound_class(comp.nuclear_charges, comp.coordinates, representation)

    def single_coordinate_calc(self, nuclear_charges, coordinates, representation=None):
        """Diverges from the parent by requiring the pre-calculated block, which the parent's
        two-argument version has no way to supply."""
        if representation is None:
            raise ValueError(
                "single_coordinate_calc on this calculator also needs the pre-calculated local "
                "representation; the parent's two-argument form cannot express it."
            )
        comp = Compound(nuclear_charges=nuclear_charges, coordinates=coordinates)
        comp.representation = representation
        return self.numba_calculator(self.compound2numba_conformer(comp))

    def get_single_compound_calculator(self, parallel=False):
        # Close over the nested JITTED FUNCTION, not over the calculator object. numba cannot type a
        # Python instance, so referring to `self.calculator` (or a local alias of it) inside the
        # jitted body fails with "Untyped global name 'calculator'". This mirrors the parent, which
        # binds `rep_func = self.jitted_rep_function` before defining its body.
        nested_calculator = self.calculator.numba_calculator

        @jit_(numba_parallel=parallel)
        def final_function(conformer):
            rep = nested_calculator(conformer)
            # axis=1: one row per atom, features side by side. axis=0 would stack the two
            # along the ATOM axis -- it raises whenever the widths differ (the normal case: aSLATM
            # is thousands of features, charges are one or two) and, when they happen to match,
            # silently returns twice as many atoms.
            final_rep = concatenate_((rep, conformer.representation), axis=1)
            return final_rep

        return final_function
