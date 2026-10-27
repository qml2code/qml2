"""The `calculate_representation` level: compute a representation in-loop and feed it downward.

Every other level takes a processed object that was built outside jitted code. This one is handed a
CONFORMER -- nuclear charges and coordinates -- computes its representation inside the parallel loop,
builds the processed object from it, and calls the nested routine on that. The levels below it then
see exactly what they see when a pre-processed compound is used, and need no knowledge that anything
changed.

Where it sits, which is the confusing part
------------------------------------------
The LAST entry of a `function_definition_list` is the OUTERMOST level and the first is the innermost
(`get_nested_routine_tuple` recurses on `processed_definition_list[:-1]`). This level goes in the
MIDDLE -- above the per-atom stack, below the levels that handle ensembles::

    ["resize", "sorf", "element_id_switch", "component_sum",
     "calculate_representation", "weighted_component_sum", "component_sum"]

Reading outside in: `weighted_component_sum` and the outer `component_sum` work on ensembles of
conformers and hand a SINGLE conformer down; `calculate_representation` turns it into a processed
object; `component_sum -> element_id_switch -> sorf -> resize` consume that per-atom, unchanged from
today. The level is precisely the boundary where a conformer becomes per-atom representations.

The trade this makes: a conformer reached more than once -- through two ensemble members, or as a
state of a pair whose other state shares its geometry -- has its representation computed again each
time, because none of them is kept. That is the point rather than a defect, and it is worth taking
when the representation is wide enough that holding one per conformer for the whole pass costs more
than recomputing it. For a narrow representation it is a straight loss; use the ordinary
precomputed path there.

Sizes and hyperparameters
-------------------------
It has neither of its own. `output_size` and `work_size` come from the nested level, and the
hyperparameter count is 0, exactly as for `normalization`: both fall out of the existing defaults in
`sorf_calculation.py`, so nothing needs registering beyond `function_level_additions`. The level
never touches `input_work_array` -- it only substitutes the object passed down.

Gradients pass straight through, and that is safe for a structural reason rather than by assumption:
the representation function is baked in at COMPILE time and takes only the conformer, while
hyperparameters arrive as a call-time array. A representation therefore cannot depend on the
hyperparameters being differentiated, so it contributes no term. (These are hyperparameter
gradients; force calculations are deliberately out of scope for `multilevel_sorf`.)

What it is handed
-----------------
Both compile-time parameters are required::

    {"compilation_parameters": {
        "representation_function": aSLATMJITCalculator(...).numba_calculator,
        "processed_object_constructor": get_el_aug_rep_constructor(elements),
    }}

`representation_function(conformer) -> representation` comes from
`qml2.representations.numba_calculators`; `processed_object_constructor(representation, conformer)`
from `qml2.multilevel_sorf.processed_object_constructors.numba_constructors`. Keeping the constructor
a parameter is what lets this one level serve any processed-object shape without importing a datatype
or a jitclass -- see that module's docstring.
"""

from ...jit_interfaces import jit_, zeros_
from .base import is_grad_def

calculate_representation_lvl = "calculate_representation"

REPRESENTATION_FUNCTION_KEY = "representation_function"
PROCESSED_OBJECT_CONSTRUCTOR_KEY = "processed_object_constructor"


def _required(compilation_parameters, key, level_name):
    """Compile-time parameters are not optional here, and a missing one must say which."""
    if compilation_parameters is None:
        raise ValueError(
            f"level {level_name!r} requires compilation_parameters "
            f"{{{REPRESENTATION_FUNCTION_KEY!r}: ..., {PROCESSED_OBJECT_CONSTRUCTOR_KEY!r}: ...}} in "
            "its parameter_list entry; it has no default representation to fall back on."
        )
    if key not in compilation_parameters:
        raise ValueError(
            f"level {level_name!r} is missing compilation parameter {key!r}; got "
            f"{sorted(compilation_parameters)}."
        )
    value = compilation_parameters[key]
    if not callable(value):
        raise ValueError(
            f"compilation parameter {key!r} of level {level_name!r} must be a jitted callable, got "
            f"{type(value).__name__}."
        )
    return value


def add_calculate_representation(
    processed_function_definition_list, inside_routine, compilation_parameters=None
):
    level_name = calculate_representation_lvl
    representation_function = _required(
        compilation_parameters, REPRESENTATION_FUNCTION_KEY, level_name
    )
    construct_processed_object = _required(
        compilation_parameters, PROCESSED_OBJECT_CONSTRUCTOR_KEY, level_name
    )

    if is_grad_def(processed_function_definition_list):

        @jit_
        def calculate_representation(
            conformer,
            input_object,
            grad_object,
            hyperparameters,
            input_work_array,
            grad_work_array,
        ):
            representation = representation_function(conformer)
            processed_object = construct_processed_object(representation, conformer)
            inside_routine(
                processed_object,
                input_object.nested_input_object,
                grad_object.nested_grad_object,
                hyperparameters,
                input_work_array,
                grad_work_array,
            )

    else:

        @jit_
        def calculate_representation(conformer, input_object, hyperparameters, input_work_array):
            representation = representation_function(conformer)
            processed_object = construct_processed_object(representation, conformer)
            inside_routine(
                processed_object,
                input_object.nested_input_object,
                hyperparameters,
                input_work_array,
            )

    return calculate_representation


def cpuest_add_calculate_representation(
    processed_function_definition_list, inside_routine, compilation_parameters=None
):
    """The CPU-time estimator: hands down an object of the right TYPE and ATOM COUNT, nothing more.

    A cpuest routine counts -- calls and components -- and never reads a representation. `cpuest_resize_`
    returns 0.0 without looking at its argument; `cpuest_add_single` adds 1 per call; the summation
    levels iterate their components. So the estimate needs the number of atoms and the container type
    the nested chain was compiled against, and nothing else.

    Computing the real representation here would therefore be pure waste, and self-defeating at the
    sizes this level exists for: estimating the cost of an aSLATM would cost an aSLATM. The stand-in is
    a (natoms, 1) array of zeros, which the constructor turns into a correctly-typed per-atom container
    -- `ElementIDAugmentedClass` holds `float64[:]`, so the width is not part of the type and one column
    is as good as thousands.

    This level is still needed rather than left to `cpuest_add_simple`: that forwards the conformer
    unchanged, leaving the chain below typed against the wrong object, which is a hard TypingError the
    first time anything asks for an estimate -- not an inaccuracy.

    The representation's own cost is not counted. The estimate is used for relative balancing between
    molecules, and the nested count already scales with the atom count that dominates it.
    """
    # Only the constructor: the representation function is deliberately NOT called here.
    construct_processed_object = _required(
        compilation_parameters, PROCESSED_OBJECT_CONSTRUCTOR_KEY, calculate_representation_lvl
    )

    if is_grad_def(processed_function_definition_list):

        @jit_
        def cpuest(conformer, input_object, grad_object):
            placeholder = zeros_((conformer.nuclear_charges.shape[0], 1))
            processed_object = construct_processed_object(placeholder, conformer)
            return inside_routine(
                processed_object,
                input_object.nested_input_object,
                grad_object.nested_grad_object,
            )

    else:

        @jit_
        def cpuest(conformer, input_object):
            placeholder = zeros_((conformer.nuclear_charges.shape[0], 1))
            processed_object = construct_processed_object(placeholder, conformer)
            return inside_routine(processed_object, input_object.nested_input_object)

    return cpuest


function_level_additions = {calculate_representation_lvl: add_calculate_representation}
cpuest_level_additions = {calculate_representation_lvl: cpuest_add_calculate_representation}
