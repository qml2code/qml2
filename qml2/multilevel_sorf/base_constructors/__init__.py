"""Naming convention for classes created here:
dt_* - (derived) datatype used as model input.
inp_* - input object containing all necessary model parameters (but for hyperparameters) and temporary arrays used to calculate the final result.
ginp_* - when gradients are calculated, an inp_* object should be accompanied with a ginp_* counterpart.
reddt_*, redinp_*, redginp_* - versions of all classes that are not JIT-compiled (and thus reducable, but not usable in Numba)
We also have functions named as:
f_* - generates SORF from arguments (dt_*, inp_*, hyperparameters).
gf_* - generates SORF and their gradients from arguments (dt_*, inp_*, ginp_*, hyperparameters).
For both classes and funtions the names are derived from a list of strings whose entries correspond to different levels of the SORF procedure.
The "processed" versions handled throughout the code and "input" versions defined by the user differ by appending a prefix in the beginning of the list.

For copying custom classes we also use functions with prefixes:
copy, gcopy - JIT-compiled copy function for copying JIT classes inside JIT functions (K.Karan.: TBH not sure how useful); gcopy is for gradient versions of same input.
rcopy, grcopy - copying reducable versions of classes into JIT-compiled.
jcopy, gjcopy - copying JIT-compiled versions into reducable

For roughly estimating relative cost of processing different objects we also use cpuest and gcpuest prefixes.
"""

from .base import get_extract_final_gradient, get_extract_final_result, input_object_prefix
from .common import (
    calc_grad_input_size_parameters,
    calc_hyperparameter_num,
    calc_input_size_parameters,
    create_input_object_from_def,
    get_class_from_def,
    get_copy_from_def,
    get_cpuest_from_def,
    get_datatype,
    get_datatype2dict,
    get_dict2datatype,
    get_routine_from_def,
    get_transform_list_dict2datatype,
)
from .rescalings import full_rescaling_lvl, power_rescaling_lvl, rescaling_lvl
from .sorf_related import (
    mixed_extensive_sorf_lvl,
    sign_invariant_sorf_lvl,
    sorf_lvl,
    unscaled_sign_invariant_sorf_lvl,
    unscaled_sorf_lvl,
)
from .special import (
    compression_lvl,
    concatenation_lvl,
    element_id_switch_lvl,
    normalization_lvl,
    resize_lvl,
)
from .summation import component_sum_lvl, weighted_component_sum_lvl
from .variance import (
    calc_variance_lvl,
    calc_variance_normalized_lvl,
    component_cycle_lvl,
    concatenation_variance_lvl,
    element_id_switch_variance_lvl,
    pass_variance_fork1_lvl,
    pass_variance_fork2_lvl,
    pass_variance_lvl,
    weighted_component_cycle_lvl,
)
