from itertools import chain

from numba import typed
from numba.experimental import jitclass

from ..jit_interfaces import concatenate_, ndarray_, sum_

# For quickly printing parameters input objects and checking whether they are the same.
# K.Karan: I am currently not aware of better ways to do it with Numba objects.
closing_brackets = {"<": ">", "(": ")", "[": "]"}


def first_position_or_None(remainder, symbol):
    if symbol in remainder:
        return remainder.index(symbol)
    else:
        return None


def cutoff_to_first_closing_bracket(remainder):
    comma_position = first_position_or_None(remainder, ",")

    first_bracket = None
    first_bracket_position = None
    for bracket in closing_brackets.keys():
        position = first_position_or_None(remainder, bracket)
        if position is None:
            continue
        if (first_bracket_position is None) or (position < first_bracket_position):
            first_bracket = bracket
            first_bracket_position = position
    c = (first_bracket is not None) and (first_bracket in ["<", "["])
    if comma_position is not None:
        c = c and (first_bracket_position < comma_position)
    if comma_position is None:
        return len(remainder), c
    if (first_bracket is None) or (comma_position < first_bracket_position):
        return comma_position, c
    closing_bracket = closing_brackets[first_bracket]
    nbracks = 1
    for i, s in enumerate(remainder[first_bracket_position + 1 :]):
        if s == closing_bracket:
            nbracks -= 1
        if s == first_bracket:
            nbracks += 1
        if nbracks == 0:
            return first_bracket_position + i + 2, c


def def_list(type):
    return type[-1] == "]"


def extract_contents(cont_str):
    inside_start = cont_str.index("<") + 1
    remainder = cont_str[inside_start:-1]
    inside_list = []
    while remainder:
        name_end = remainder.index(":")
        cur_name = remainder[:name_end]
        remainder = remainder[name_end + 1 :]
        end, c = cutoff_to_first_closing_bracket(remainder)
        cur_type = remainder[:end]
        inside_list.append((cur_name, cur_type, c))
        remainder = remainder[end + 1 :]
    return inside_list


def is_empty(attr_name):
    return False


#    return attr_name in ["final_result", "phases", "sin", "temp_phases", "temp_result"]


def jitclass_overview_str(obj, n=0, mute_empty=False):
    cl = str(vars(type(obj))["_numba_type_"])
    output_str = cl[: cl.index("<")] + "\n"
    for cur_name, cur_type, c in extract_contents(cl):
        comp = getattr(obj, cur_name)
        output_str += ">" * n + cur_name + " " + cur_type
        if c:
            output_str += ":\n"
            if def_list(cur_type):
                for subcomp in comp:
                    output_str += jitclass_overview_str(subcomp, n + 1) + "\n"
            else:
                output_str += jitclass_overview_str(comp, n + 1)
        else:
            if (not hasattr(comp, "shape")) or len(comp.shape) == 0:
                output_str += " " + str(comp)
            else:
                output_str += " " + str(comp.shape)
                if (not mute_empty) and (not is_empty(cur_name)):
                    output_str += " sum: " + str(sum_(comp))
        output_str += "\n"
    return output_str[:-1]


# For switching jitclass on an off.
def nojitclass(class_def):
    return class_def


def jitclass_(*args, skip=False):
    if skip:
        return nojitclass
    else:
        return jitclass(*args)


# Misc.
def optional_array_print_tuple(array_in):
    """
    For making `*array_in` part of a print statement without worrying it could be `None` or `array_(None)`.
    """
    if (array_in is None) or (len(array_in.shape) == 0):
        return (None,)
    else:
        return tuple(array_in)


def merge_or_replace(array_initial, array_added):
    if array_initial is None:
        return array_added
    assert isinstance(array_initial, type(array_added))
    if isinstance(array_initial, list):
        return array_initial + array_added
    if isinstance(array_initial, ndarray_):
        return concatenate_((array_initial, array_added))
    merged_array = typed.List()
    for element in chain(array_initial, array_added):
        merged_array.append(element)
    return merged_array
