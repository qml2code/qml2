from numba import float64, int64

from ...jit_interfaces import array_, copy_, dint_, empty_, jit_, reshape_
from ...representations.numba_calculators import CONFORMER_CLASS, CONFORMER_WLOCALREP_CLASS
from ..utils import jitclass_

# Naming convention
datatype_prefix = "dt"
input_object_prefix = "inp"
function_prefix = "f"


reducable_prefix = "red"
red_datatype_prefix = reducable_prefix + datatype_prefix
red_input_object_prefix = reducable_prefix + input_object_prefix

gradient_prefix = "g"
gradient_input_object_prefix = gradient_prefix + input_object_prefix
gradient_function_prefix = gradient_prefix + function_prefix

gradient_red_input_object_prefix = gradient_prefix + red_input_object_prefix
copy_prefix = "copy"
gradient_copy_prefix = gradient_prefix + copy_prefix
rcopy_prefix = "rcopy"
gradient_rcopy_prefix = gradient_prefix + rcopy_prefix
jcopy_prefix = "jcopy"
gradient_jcopy_prefix = gradient_prefix + jcopy_prefix

cpuest_prefix = "cpuest"
gradient_cpuest_prefix = gradient_prefix + "cpuest"


# Base datatype building blocks
default_float = float64
default_int = int64

array_1D_ = default_float[:]

array_2D_ = default_float[:, :]

array_3D_ = default_float[:, :, :]


def default_array(ndim):
    return empty_(tuple(1 for _ in range(ndim)))


default_class_instances = {}
for dim in range(1, 4):
    ending = str(dim) + "D_"
    # default array instances
    for pref in [datatype_prefix, red_datatype_prefix]:
        datatype_name = pref + "_array_" + ending
        globals()[datatype_name] = globals()["array_" + ending]
        default_class_instances[datatype_name] = default_array(dim)
    # default copying procedures
    for copy_pref in ["", "r", "j"]:
        globals()[copy_pref + "copy_array_" + ending] = copy_


# Conformer datatypes -- the LEAF a representation-calculating level starts from.
#
# Everything above is an array: a level receives feature vectors and returns feature vectors. A
# level that computes a representation inside the parallel loop instead starts from the geometry, so
# the machinery needs the numba conformer classes registered as datatypes in their own right.
#
# `Conformer` carries nuclear charges and coordinates. `ConformerWLocalRepresentation` adds a
# `representation` array that had to be computed OUTSIDE the jit -- tblite charges, for instance --
# so a level can combine a freshly computed representation with a pre-computed one without leaving
# the loop.
#
# Created eagerly here, unlike in numba_calculators where the factory exists because a jitclass is
# not picklable: this module's copy is a process-wide singleton that is never pickled (nothing
# pickles `default_class_instances`), and both classes build in under a millisecond. Holding one
# instance of each type also matters for correctness -- two calls to the factory produce two
# DISTINCT numba types, and a function compiled against one will not accept the other.
Conformer = CONFORMER_CLASS
ConformerWLocalRepresentation = CONFORMER_WLOCALREP_CLASS

conformer_ = Conformer.class_type.instance_type
conformer_wlocalrep_ = ConformerWLocalRepresentation.class_type.instance_type


def default_conformer():
    """Minimal one-atom instance, the counterpart of `default_array` for this datatype."""
    return Conformer(empty_((1,), dtype=dint_), empty_((1, 3)))


def default_conformer_wlocalrep(nfeatures=1):
    return ConformerWLocalRepresentation(
        empty_((1,), dtype=dint_), empty_((1, 3)), empty_((1, nfeatures))
    )


@jit_
def copy_conformer(conformer):
    """A conformer is a jitclass, so copying it means reconstructing it -- `copy_` works on the
    arrays above but not on an instance. Deep, like the array copies: the new object shares nothing
    with the old one."""
    return Conformer(copy_(conformer.nuclear_charges), copy_(conformer.coordinates))


@jit_
def copy_conformer_wlocalrep(conformer):
    return ConformerWLocalRepresentation(
        copy_(conformer.nuclear_charges),
        copy_(conformer.coordinates),
        copy_(conformer.representation),
    )


# The REDUCABLE twins, and the reason the `reddt_` half of a datatype exists at all.
#
# A jitclass instance cannot be pickled and has no useful `__dict__`. Both matter here: an object
# produced in a worker process has to cross a process boundary to reach the parent, and
# `recursive_class_dict` walks `__dict__` to turn a datatype back into plain data. So the `reddt_`
# form is a PLAIN PYTHON class holding the same fields, and `jcopy_` converts one into the real
# jitclass -- in the parent, after any pickling is done. `convert_dict_list` wraps a worker's dict in
# a generic `ConvertedDict` with exactly these attributes, so a constructor that reads attributes
# rather than keys serves both.
#
# For the array datatypes `dt_` and `reddt_` can point at the same thing, because a numpy array is
# already picklable and already plain. A jitclass is neither, which is why these need real twins.
class ReducableConformer:
    def __init__(self, nuclear_charges, coordinates):
        self.nuclear_charges = nuclear_charges
        self.coordinates = coordinates


class ReducableConformerWLocalRepresentation(ReducableConformer):
    def __init__(self, nuclear_charges, coordinates, representation):
        super().__init__(nuclear_charges, coordinates)
        self.representation = representation


def default_reducable_conformer():
    return ReducableConformer(empty_((1,), dtype=dint_), empty_((1, 3)))


def default_reducable_conformer_wlocalrep(nfeatures=1):
    return ReducableConformerWLocalRepresentation(
        empty_((1,), dtype=dint_), empty_((1, 3)), empty_((1, nfeatures))
    )


# jcopy_: reducable -> jit. NOT jitted, deliberately -- its argument is a plain Python object, often a
# `ConvertedDict` built straight from a worker's dictionary, which numba cannot type. The dtypes are
# coerced rather than trusted because whatever produced the dict is outside this module's control.
def jcopy_conformer(reducable):
    return Conformer(
        array_(reducable.nuclear_charges, dtype=dint_), array_(reducable.coordinates, dtype=float)
    )


def jcopy_conformer_wlocalrep(reducable):
    return ConformerWLocalRepresentation(
        array_(reducable.nuclear_charges, dtype=dint_),
        array_(reducable.coordinates, dtype=float),
        array_(reducable.representation, dtype=float),
    )


# rcopy_: jit -> reducable. Also un-jitted, and the direction `get_datatype2dict` needs.
def rcopy_conformer(conformer):
    return ReducableConformer(copy_(conformer.nuclear_charges), copy_(conformer.coordinates))


def rcopy_conformer_wlocalrep(conformer):
    return ReducableConformerWLocalRepresentation(
        copy_(conformer.nuclear_charges),
        copy_(conformer.coordinates),
        copy_(conformer.representation),
    )


# Registered under the same naming convention as the arrays, so a datatype level can name them and
# `get_precomputed_quant` can find them. Unlike the arrays, `dt_` and `reddt_` differ, and the three
# copy prefixes are three DIFFERENT procedures: `copy_` stays inside numba, `jcopy_` and `rcopy_`
# cross the boundary and so must not be jitted.
for (
    _dt_name,
    _dt_type,
    _dt_default,
    _red_default,
    _copy,
    _jcopy,
    _rcopy,
) in [
    (
        "conformer_",
        conformer_,
        default_conformer,
        default_reducable_conformer,
        copy_conformer,
        jcopy_conformer,
        rcopy_conformer,
    ),
    (
        "conformer_wlocalrep_",
        conformer_wlocalrep_,
        default_conformer_wlocalrep,
        default_reducable_conformer_wlocalrep,
        copy_conformer_wlocalrep,
        jcopy_conformer_wlocalrep,
        rcopy_conformer_wlocalrep,
    ),
]:
    _jit_name = datatype_prefix + "_" + _dt_name
    globals()[_jit_name] = _dt_type
    default_class_instances[_jit_name] = _dt_default()

    _red_name = red_datatype_prefix + "_" + _dt_name
    globals()[_red_name] = _red_default().__class__
    default_class_instances[_red_name] = _red_default()

    globals()["copy_" + _dt_name] = _copy
    globals()["jcopy_" + _dt_name] = _jcopy
    globals()["rcopy_" + _dt_name] = _rcopy


# Naming convention
def is_grad_def(processed_function_definition_list):
    return processed_function_definition_list[0] in [
        gradient_input_object_prefix,
        gradient_function_prefix,
        gradient_copy_prefix,
        gradient_red_input_object_prefix,
        gradient_rcopy_prefix,
        gradient_jcopy_prefix,
        gradient_cpuest_prefix,
    ]


def is_copy_def(processed_function_definition_list):
    if isinstance(processed_function_definition_list, tuple):
        return is_copy_def(processed_function_definition_list[0])
    return processed_function_definition_list[0] in [
        copy_prefix,
        gradient_copy_prefix,
        rcopy_prefix,
        gradient_rcopy_prefix,
        jcopy_prefix,
        gradient_jcopy_prefix,
    ]


def is_cpuest_def(processed_function_definition_list):
    if isinstance(processed_function_definition_list, tuple):
        return is_copy_def(processed_function_definition_list[0])
    return processed_function_definition_list[0] in [cpuest_prefix, gradient_cpuest_prefix]


def is_reducable_object(processed_function_definition_list):
    return processed_function_definition_list[0] in [
        red_datatype_prefix,
        red_input_object_prefix,
        gradient_red_input_object_prefix,
    ]


def is_red_vs_jit_copy_def(processed_function_definition_list):
    return processed_function_definition_list[0] in [
        rcopy_prefix,
        gradient_rcopy_prefix,
        jcopy_prefix,
        gradient_jcopy_prefix,
    ]


def is_red2jit_copy_def(processed_function_definition_list):
    return processed_function_definition_list[0] in [rcopy_prefix, gradient_rcopy_prefix]


def is_data_definition(processed_function_definition_list):
    for comp in ["array_1D", "array_2D", "rhos", "list"]:
        if comp in processed_function_definition_list:
            return True
    return False


def change_nested_component_name(nested_component_def, new_name):
    return [(new_name, nested_component_def[0][1])]


# Shorthands for extracting final results and final gradients.
def get_extract_final_result():
    @jit_
    def extract_final_result(input_object, input_work_array):
        return input_work_array[-input_object.output_size :]

    return extract_final_result


def get_extract_final_gradient():
    @jit_
    def extract_final_gradient(input_object, grad_object, grad_work_array):
        if grad_object.nhyperparameters != 0:
            return reshape_(
                grad_work_array[-grad_object.output_size :],
                (grad_object.nhyperparameters, input_object.output_size),
            )
        else:
            return empty_((0, input_object.output_size))

    return extract_final_gradient


def get_separate_input_work_arr():
    extract_final = get_extract_final_result()
    extract_nested = get_extract_final_result()

    @jit_
    def separate_input_work_arr(input_object, input_work_array):
        nested_input_work_array = input_work_array[: -input_object.output_size]
        final_result = extract_final(input_object, input_work_array)
        nested_final_result = extract_nested(
            input_object.nested_input_object, nested_input_work_array
        )
        return nested_input_work_array, final_result, nested_final_result

    return separate_input_work_arr


def get_separate_grad_work_arr():
    extract_final = get_extract_final_gradient()
    extract_nested = get_extract_final_gradient()

    @jit_
    def separate_grad_work_arr(input_object, grad_object, grad_work_array):
        nested_grad_work_array = grad_work_array[: -grad_object.output_size]
        final_gradient = extract_final(input_object, grad_object, grad_work_array)
        nested_final_gradient = extract_nested(
            input_object.nested_input_object,
            grad_object.nested_grad_object,
            nested_grad_work_array,
        )
        return nested_grad_work_array, final_gradient, nested_final_gradient

    return separate_grad_work_arr


basic_input_components = [("output_size", default_int), ("work_size", default_int)]
basic_grad_input_components = basic_input_components + [("nhyperparameters", default_int)]


# No extra items are added during nesting.
def inp_add_simple(processed_definition_list, nested_input_component, nested_default_input_object):
    skip_jit = is_reducable_object(processed_definition_list)
    if is_grad_def(processed_definition_list):

        @jitclass_(basic_grad_input_components + nested_input_component, skip=skip_jit)
        class simple:
            def __init__(self, nested_grad_object, output_size=0, nhyperparameters=0, work_size=0):
                self.nested_grad_object = nested_grad_object
                self.output_size = output_size
                self.nhyperparameters = nhyperparameters
                self.work_size = work_size

    else:

        @jitclass_(basic_input_components + nested_input_component, skip=skip_jit)
        class simple:
            def __init__(self, nested_input_object, output_size=0, work_size=0):
                self.nested_input_object = nested_input_object
                self.output_size = output_size
                self.work_size = work_size

    return simple, simple(nested_default_input_object)


def cadd_simple(processed_definition_list, inside_copy, object_class):
    skip_jit = is_red_vs_jit_copy_def(processed_definition_list)
    if is_grad_def(processed_definition_list):

        @jit_(skip=skip_jit)
        def get_copy(other_object):
            new_nested_object = inside_copy(other_object.nested_grad_object)
            return object_class(
                new_nested_object,
                other_object.output_size,
                other_object.nhyperparameters,
                other_object.work_size,
            )

    else:

        @jit_(skip=skip_jit)
        def get_copy(other_object):
            new_nested_object = inside_copy(other_object.nested_input_object)
            return object_class(
                new_nested_object, other_object.output_size, other_object.work_size
            )

    return get_copy


def cpuest_add_simple(processed_definition_list, routine):
    if is_grad_def(processed_definition_list):

        @jit_
        def cpuest(processed_object, input_object, grad_object):
            return routine(
                processed_object, input_object.nested_input_object, grad_object.nested_grad_object
            )

    else:

        @jit_
        def cpuest(processed_object, input_object):
            return routine(processed_object, input_object.nested_input_object)

    return cpuest


def cpuest_add_single(processed_definition_list, routine):
    if is_grad_def(processed_definition_list):

        @jit_
        def cpuest(processed_object, input_object, grad_object):
            return (
                routine(
                    processed_object,
                    input_object.nested_input_object,
                    grad_object.nested_grad_object,
                )
                + grad_object.nhyperparameters
                + 1
            )

    else:

        @jit_
        def cpuest(processed_object, input_object):
            return routine(processed_object, input_object.nested_input_object) + 1

    return cpuest
