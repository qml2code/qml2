"""Constructors that run INSIDE numba: a representation array plus a conformer -> a processed object.

Every other module in this package converts a Python `Compound` into a processed object ahead of
time, outside jitted code. These do the same job at the other end of the pipeline, where the
representation was computed inside the parallel loop and the object the rest of `multilevel_sorf`
consumes has to be assembled there too -- the `calculate_representation` level's second step.

The split matters for a practical reason. A level that computes representations in-loop cannot
import a datatype without dragging numba type identity into `base_constructors`, and it should not
have to: it is handed a constructor as a COMPILE-TIME PARAMETER and calls it. So the knowledge of
which processed object a stack consumes lives here, next to the Python-side constructor for the same
object in `standard.py`, and the level stays ignorant of all of it. Adding support for another
processed-object shape means adding a factory here and nothing else.

Each factory bakes its configuration in and returns a jitted function. Two consequences:

* the returned function is a CLOSURE, so it must declare what it baked in via
  `set_compilation_identity` -- otherwise two models over different element lists would share one
  compiled routine. See `base_constructors/compilation_parameters.py`.
* it uses the CANONICAL datatype from `get_datatype`, not a fresh factory call, for the same reason
  `standard.py` uses `CONFORMER_CLASS`: two calls give two distinct numba types.
"""

from numba import typed

from ...jit_interfaces import array_, dint_, jit_, unique_
from ...utils import get_element_ids_from_sorted
from ..base_constructors import get_datatype
from ..base_constructors.compilation_parameters import set_compilation_identity
from .standard import el_aug_rep_def


def get_el_aug_rep_constructor(
    possible_nuclear_charges=array_([1, 6, 7, 8, 16], dtype=dint_),
):
    """In-numba analogue of `gen_element_augmented_local_rep_object`.

    Returns a jitted `(representation, conformer) -> typed.List[ElementIDAugmentedClass]`, one entry
    per atom, which is exactly what `component_sum` iterates over for an el-aug-rep stack.

    `possible_nuclear_charges` is sorted and baked in, mirroring
    `ElementAugmentedLocalRepresentationCalc.__init__`. It must therefore agree with the element list
    the rest of the model was built with -- the element-id table in `parameter_list` above all -- since
    the ids this produces index into it.
    """
    sorted_elements = unique_(possible_nuclear_charges)
    # [:-1] drops the "list" level: the class for ONE atom, whose instances fill the list.
    atom_class = get_datatype(el_aug_rep_def[:-1])
    atom_type = atom_class.class_type.instance_type

    @jit_
    def construct_el_aug_rep(representation, conformer):
        natoms = conformer.nuclear_charges.shape[0]
        if representation.shape[0] != natoms:
            raise ValueError(
                "representation has one row per atom; got a row count differing from nuclear_charges"
            )
        element_ids = get_element_ids_from_sorted(conformer.nuclear_charges, sorted_elements)
        processed_object = typed.List.empty_list(atom_type)
        for atom_id in range(natoms):
            processed_object.append(atom_class(representation[atom_id], element_ids[atom_id]))
        return processed_object

    return set_compilation_identity(construct_el_aug_rep, sorted_elements)
