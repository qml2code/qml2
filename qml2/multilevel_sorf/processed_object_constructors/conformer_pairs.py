"""Two states of one geometry, combined the way a Slater pair's determinants are.

The FJK/FJD machinery has two ways of putting a pair of states together, chosen in
`fjk.FJKRepresentationCalc`: subtract them, by merging both into one weighted list with the
first one's weights negated, or concatenate them, by keeping them apart for a
"concatenation" level to lay end to end. This module offers the same choice one level
shallower, for LOCAL representations that have no orbitals underneath them.

The case it exists for: two states over ONE geometry whose base representation is identical
and whose per-atom augmentation is not -- tight-binding atomic charges, say, which move
under charging, spin excitation and solvation while FCHL19 and aSLATM do not move at all.
Subtracting then isolates what the second state did to the charges; concatenating keeps both
and lets the model weigh them.

Because the geometry is shared, the base representation is computed TWICE and is expected to
come out the same both times. That is wasteful and deliberate: the alternative is to compute
it once and trust that the two states agree, which is exactly the assumption this module
should be checking rather than relying on. `calc_with_component_bounds` asserts it.
"""

from ...jit_interfaces import all_, array_, concatenate_, empty_
from .standard import (
    ElementAugmentedLocalRepresentationCalc,
    LocalRepresentationCalc,
    NumbaConformerGenerator,
    el_aug_rep_def,
)

# The datatype of ONE state, which is what the inner calculator returns and the only thing
# that differs between the two families:
#
#   LocalRepresentationCalc                 -> a (natoms x nfeatures) array
#   ElementAugmentedLocalRepresentationCalc -> a list of per-atom dicts carrying element ids
#
# Everything below combines them identically; only the container changes.
single_state_rep_def = ["array_2D"]

# Both states in one weighted list whose weights are -1 and +1, so whatever sums over it
# returns state 1 minus state 0.
pair_local_rep_def = ["array_2D", "rhos"]
el_aug_pair_local_rep_def = el_aug_rep_def + ["rhos"]

# The two states kept apart, for a "concatenation" level with num_components=2. Each state's
# atoms are summed first, so this sits one level up from the difference.
concatenated_pair_local_rep_def = ["array_2D", "list"]
el_aug_concatenated_pair_local_rep_def = el_aug_rep_def + ["list"]


# What container one state's processed form arrives in. Resolved from the calculator passed in rather
# than asked for, so the description cannot contradict the thing it describes.
STATE_ARRAY = "array"  # LocalRepresentationCalc: one (natoms, nfeatures) block
STATE_ELEMENT_AUGMENTED = "element_augmented"  # ...Augmented...: a list of per-atom dicts
STATE_NUMBA_CONFORMER = "numba_conformer"  # NumbaConformerGenerator: a jitclass, no features yet


class PairLocalRepresentationCalc:
    def __init__(
        self,
        local_representation_calc: LocalRepresentationCalc,
        concatenate_components: bool = False,
    ):
        """Combine the local representations of a two-state pair.

        Args:
            local_representation_calc: the calculator applied to each state separately.
                Its `component_bounds` describe ONE state and are passed through unchanged.
            concatenate_components (bool): False subtracts the two states, True keeps them
                apart. Mirrors the option of the same name on
                `fjk.FJKRepresentationCalc`, including its consequences:

                * `False` merges both states' atoms into one weighted list and negates the
                  first state's weights, so a `weighted_component_sum` over it returns
                  state 1 minus state 0. Antisymmetric: building the pair the other way
                  round flips the sign of every feature.
                * `True` returns a list of the two states' representations for a
                  "concatenation" level to lay end to end. The output is then twice as long
                  and the SORF stack needs that level with `num_components=2`, after the
                  level that sums each state's atoms.

                Either way the ORDER of the states is part of the representation, and a
                training set assembled as (gas, solvated) cannot be used to predict pairs
                assembled as (solvated, gas).
        """
        self.local_representation_calc = local_representation_calc
        self.concatenate_components = concatenate_components
        # Derived, not asked for. Which container the states arrive in is a fact about the
        # calculator already passed in, so taking it as a second argument would create a
        # description that can contradict the thing it describes -- where
        # `concatenate_components` above is a genuine choice, both modes being valid for
        # one calculator.
        #
        # Each subclass test MUST precede the base it derives from, or the base check matches it too
        # and silently routes it down the wrong path.
        if isinstance(local_representation_calc, NumbaConformerGenerator):
            self.state_container = STATE_NUMBA_CONFORMER
        elif isinstance(local_representation_calc, ElementAugmentedLocalRepresentationCalc):
            self.state_container = STATE_ELEMENT_AUGMENTED
        elif isinstance(local_representation_calc, LocalRepresentationCalc):
            self.state_container = STATE_ARRAY
        else:
            raise TypeError(
                "PairLocalRepresentationCalc needs a LocalRepresentationCalc, an "
                "ElementAugmentedLocalRepresentationCalc or a NumbaConformerGenerator; got "
                f"{type(local_representation_calc).__name__}"
            )

    def processed_representation_def(self):
        """The datatype one molecule's representation has, before the conformer levels.

        Returned from the same resolution that chooses the container, so a caller building
        the `get_transform_list_dict2datatype` transform never repeats the decision.
        """
        if self.state_container == STATE_NUMBA_CONFORMER:
            # Whichever conformer class the generator produces, plus the container the two states sit
            # in. Nothing below it: the per-atom levels get their datatype from the constructor the
            # calculate_representation level is given, not from the input object.
            inner = self.local_representation_calc.processed_representation_def()
            if self.concatenate_components:
                return inner + ["list"]
            return inner + ["list", "rhos"]
        if self.state_container == STATE_ELEMENT_AUGMENTED:
            if self.concatenate_components:
                return el_aug_concatenated_pair_local_rep_def
            return el_aug_pair_local_rep_def
        if self.concatenate_components:
            return concatenated_pair_local_rep_def
        return pair_local_rep_def

    def state_size(self, state_representation):
        """How many atoms one state contributed, whichever container it came in."""
        if self.state_container == STATE_NUMBA_CONFORMER:
            # A dict, not a conformer: the generator emits plain data so it can cross a process
            # boundary, and the jitclass is built in the parent. See NumbaConformerGenerator.
            return len(state_representation["nuclear_charges"])
        if self.state_container == STATE_ELEMENT_AUGMENTED:
            return len(state_representation)
        return state_representation.shape[0]

    def state_representations(self, pair, **additional_kwargs):
        """Each state's (natoms x nfeatures) representation, and the bounds they share.

        Raises rather than returns on the two ways the pair can fail to be a pair: not
        having exactly two states, and the two states not sharing a geometry. Neither is
        detectable downstream -- a pair of different molecules produces a perfectly
        well-formed difference of two unrelated things.
        """
        states = getattr(pair, "comps", None)
        if states is None:
            raise TypeError(
                "PairLocalRepresentationCalc needs an object with `comps` (an "
                f"OML_Slater_pair or equivalent); got {type(pair).__name__}"
            )
        if len(states) != 2:
            raise ValueError(
                f"only two-state pairs are supported, got one with {len(states)} states"
            )

        representations = []
        component_bounds = None
        for state in states:
            representation, bounds = self.local_representation_calc.calc_with_component_bounds(
                state, **additional_kwargs
            )
            if component_bounds is None:
                component_bounds = bounds
            else:
                # Differing bounds mean the two states were represented by different
                # things, so nothing below compares like with like.
                assert all_(component_bounds == bounds), (
                    "the two states produced different component bounds: "
                    f"{component_bounds} against {bounds}"
                )
            representations.append(representation)

        sizes = [self.state_size(representation) for representation in representations]
        if sizes[0] != sizes[1]:
            raise ValueError(
                f"the two states hold {sizes[0]} and {sizes[1]} atoms: they are not two "
                "states of one geometry"
            )
        # The premise of this module, checked rather than assumed. A pair whose states sit
        # on different geometries still produces a well-formed difference -- of two
        # unrelated molecules.
        assert all_(states[0].coordinates == states[1].coordinates), (
            "the two states do not share a geometry, so their base representations describe "
            "different molecules and their difference means nothing"
        )
        return representations, component_bounds

    def component_bounds(self, inner_component_bounds):
        """Where one STATE's rescaling regions are.

        Unchanged by either combination: the rescaling happens before the states are
        combined, so the regions describe one state and the number of sigmas does not
        double. The same reasoning as `fjk.FJKRepresentationCalc.component_bounds`.
        """
        return inner_component_bounds

    def calc_with_component_bounds(self, pair, **additional_kwargs):
        representations, inner_bounds = self.state_representations(pair, **additional_kwargs)
        bounds = self.component_bounds(inner_bounds)

        if self.concatenate_components:
            # Nothing negated and nothing merged: the "concatenation" level lays the two
            # end to end, so the order here is the order they appear in the features.
            return representations, bounds

        if self.state_container == STATE_NUMBA_CONFORMER:
            # Same difference, with the weights on the two STATES instead of on every atom. Exact
            # rather than approximate: the atom weight is constant within a state and the atom sum is
            # linear, so sum_a w_a f(a) == sum_s w_s sum_{a in s} f(a). It also HAS to move here --
            # the per-atom features do not exist until the calculate_representation level runs, which
            # sits below this one in the stack.
            #
            # The exactness needs one condition, met by every stack that uses this: no NONLINEAR level
            # may sit between the per-atom sum and this subtraction. Inserting `normalization` there
            # would make the two orderings genuinely different, and this one wrong.
            return (
                {"components": representations, "rhos": array_([-1.0, 1.0])},
                bounds,
            )

        # The first state enters negatively, so that summing over the merged list returns
        # state 1 minus state 0 -- the same sign convention as an OML_Slater_pair.
        natoms = self.state_size(representations[0])
        rhos = empty_(2 * natoms)
        rhos[:natoms] = -1.0
        rhos[natoms:] = 1.0
        if self.state_container == STATE_ELEMENT_AUGMENTED:
            # Lists of per-atom dicts: joined end to end, each atom keeping its element id.
            merged = list(representations[0]) + list(representations[1])
        else:
            merged = concatenate_(representations)
        return {"components": merged, "rhos": rhos}, bounds

    def calc_rep(self, pair, **additional_kwargs):
        return self.calc_with_component_bounds(pair, **additional_kwargs)[0]


def atomic_charge_representation(compound):
    """The tight-binding atomic charges as a (natoms x 1) per-atom representation.

    An "additional representation function" for `LocalRepresentationCalc`: one column, so it
    lands in its own `component_bounds` region and receives its own sigma. Takes the compound
    rather than its nuclear charges because the quantity is a property of the converged
    calculation, not of the element list -- which is the whole reason it carries information
    the base representation does not.
    """
    charges = compound.get_xtb_charges()
    return array_(charges).reshape((-1, 1))
