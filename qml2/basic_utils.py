# Contains utils that do not require imports from .jit_interfaces.
import bz2
import os
import pickle
import subprocess
import time
import traceback
from copy import deepcopy
from datetime import datetime

from .data import NUCLEAR_CHARGE

rep_numba_parallel_env_var_name = "QML2_NUMBA_REP_PARALLELIZATION"


class OptionUnavailableError(Exception):
    pass


def now():
    return datetime.now()


def now_secs():
    return time.perf_counter()


def total_cpu_time():
    t = os.times()
    return t.user + t.system + t.children_user + t.children_system


# TODO: implement proper verbose mode displaying messages both from scipy and the rest of the code.
display_scipy_convergence = False  # True


# For building many-parent classes.
def filtering_kwargs(key_list, kwargs, choose_not_in_list=False):
    filtered_kwargs = {}
    for k, val in kwargs.items():
        add = k in key_list
        if choose_not_in_list:
            add = not add
        if add:
            filtered_kwargs[k] = val
    return filtered_kwargs


def divided_by_parents(
    obj, parent_init_order, subroutine_name, parent_kwarg_dict, input_kwargs={}
):
    list_of_others = None
    for parent in parent_init_order:
        attribute_subroutine = getattr(parent, subroutine_name)
        if parent in parent_kwarg_dict:
            relevant_kwargs = filtering_kwargs(parent_kwarg_dict[parent], input_kwargs)
        else:
            if list_of_others is None:
                list_of_others = []
                for v in parent_kwarg_dict.values():
                    list_of_others += v
            relevant_kwargs = filtering_kwargs(
                list_of_others, input_kwargs, choose_not_in_list=True
            )
        attribute_subroutine(obj, **relevant_kwargs)


# For checking environmental variables.
def checked_dict_entry(d: dict, key=None, default_answer=None):
    if key in d:
        return d[key]
    else:
        return default_answer


def checked_environ_val(
    environ_name: str, expected_answer=None, default_answer=None, var_class=int
):
    """
    Returns `os.environ[environ_name]` while checking for exceptions.

    Args:
        environ_name (str): name of the environmental variable
        exected_answer: if not None return this value (used in situations where environmental variable's contents should be overriden by an optional keyword argument if present).
        default_answer: if not None return `var_class(default_answer)` if environmental variable is not defined.
        var_class: function used to convert the `str` value of environmental variable.
    """
    if expected_answer is None:
        try:
            args = (os.environ[environ_name],)
        except LookupError:
            if default_answer is None:
                args = tuple()
            else:
                args = (default_answer,)
        return var_class(*args)
    else:
        return expected_answer


def checked_logical_environ_val(environ_name: str, default_answer=None):
    """
    Check whether an environmental variable defines a logical value. "0" and "1" correspond to "False" and "True", other values lead to raising an exception.

    Args:
        environ_name (str): name of the environmental variable.
        default_answer (bool or None): value returned if the environmental variable is not defined.
    """
    str_val = checked_environ_val(environ_name, default_answer=None, var_class=str)
    if not str_val:
        return default_answer
    match str_val:
        case "0":
            return False
        case "1":
            return True
        case _:
            raise Exception(
                f"Unknown value for environmental variable {environ_name}: found {str_val}, expecting '0' for `False` or '1' for `True`."
            )


# atom types
def canonical_atomtype(atomtype):
    return atomtype[0].upper() + atomtype[1:].lower()


def nuclear_charge(atomtype):
    return NUCLEAR_CHARGE[canonical_atomtype(atomtype)]


# dictionnary and list shorhands
def any_element_in_list(list_in, *els):
    for el in els:
        if el in list_in:
            return True
    return False


def repeated_dict(labels, repeated_el, copy_needed=False):
    output = {}
    for l in labels:
        if copy_needed:
            output[l] = deepcopy(repeated_el)
        else:
            output[l] = repeated_el
    return output


def all_None_dict(labels):
    return repeated_dict(labels, None)


def inverted_dictionary(forward_dictionnary):
    output = {}
    for k, val in forward_dictionnary.items():
        output[val] = k
    return output


def overwrite_when_possible(overwritten_dict, overwriting_dict):
    new_dict = deepcopy(overwritten_dict)
    for key, new_val in overwriting_dict.items():
        new_dict[key] = new_val
    return new_dict


ELEMENTS = None


def str_atom_corr(ncharge):
    global ELEMENTS
    if ELEMENTS is None:
        ELEMENTS = inverted_dictionary(NUCLEAR_CHARGE)
    return ELEMENTS[ncharge]


def package_root_dir():
    return os.path.dirname(__file__)


def package_data_position(filepath):
    return package_root_dir() + "/" + filepath


def fetch_package_data(filepath):
    return "".join(open(package_data_position(filepath), "r").readlines())


def run(*cmd_args, capture_output=False):
    return subprocess.run(list(cmd_args), capture_output=capture_output)


def copy_package_to(other_dir):
    run("cp", "-r", package_root_dir(), other_dir)


# def str_atom_corr(ncharge):
#    return canonical_atomtype(str_atom(ncharge))
compress_fileopener = {True: bz2.BZ2File, False: open}
pkl_compress_ending = {True: ".pkl.bz2", False: ".pkl"}


def dump2pkl(obj, filename: str, compress: bool = False):
    """
    Dump an object to a pickle file.
    obj : object to be saved
    filename : name of the output file
    compress : whether bz2 library is used for compressing the file.
    """
    # Written to a temporary name beside the target and moved into place rather than opened
    # at `filename` directly. os.replace is atomic within a filesystem, so a concurrent
    # reader sees either the old complete file or the new one and never the half-written
    # state in between. This matters because several SLURM jobs routinely share one savefile
    # folder and nothing here locks: without it, a reader arriving mid-write gets an
    # EOFError out of pickle, and two writers on one path can interleave into a file that
    # stays unreadable afterwards. Racing writers still race, but the loser's file is
    # replaced whole. The pid in the temporary name keeps two writers off each other's.
    directory = os.path.dirname(filename)
    temporary_filename = os.path.join(
        directory, "." + os.path.basename(filename) + ".tmp." + str(os.getpid())
    )
    try:
        output_file = compress_fileopener[compress](temporary_filename, "wb")
        pickle.dump(obj, output_file)
        output_file.close()
        os.replace(temporary_filename, filename)
    except BaseException:
        if os.path.isfile(temporary_filename):
            os.remove(temporary_filename)
        raise


def loadpkl(filename: str, compress: bool = False):
    """
    Load an object from a pickle file.
    filename : name of the imported file
    compress : whether bz2 compression was used in creating the loaded file.
    """
    input_file = compress_fileopener[compress](filename, "rb")
    obj = pickle.load(input_file)
    input_file.close()
    return obj


def ispklfile(filename: str):
    """
    Check whether filename is a pickle file.
    """
    return filename[-4:] == ".pkl"


def mkdir(dir_name):
    try:
        os.mkdir(dir_name)
    except FileExistsError:
        pass


# For going between nested dictionnary and classes (mainly appears in multilevel SORF routines)
def recursive_class_dict(obj):
    if isinstance(obj, list):
        return [recursive_class_dict(el) for el in obj]
    if not hasattr(obj, "__dict__"):
        return obj
    output = {}
    for k, val in obj.__dict__.items():
        output[k] = recursive_class_dict(val)
    return output


class ConvertedDict:
    def __init__(self, d: dict | list):
        for k, val in d.items():
            if type(val) in [dict, list]:
                added_val = convert_dict_list(val)
            else:
                added_val = val
            setattr(self, k, added_val)


def convert_dict_list(d: dict | list):
    if isinstance(d, list):
        return [convert_dict_list(el) for el in d]
    if isinstance(d, dict):
        return ConvertedDict(d)
    return d


class ExceptionRaisingFunc:
    def __init__(self, ex, returned_exception_type=None):
        self.exception_text = "\n".join(traceback.format_exception(ex))
        if returned_exception_type is None:
            returned_exception_type = type(ex)
        self.returned_exception_type = returned_exception_type

    def __call__(self, *args, **kwargs):
        raise self.returned_exception_type(self.exception_text)


def ExceptionRaisingClass(ex, returned_exception_type=None, add_attrs=None):
    internal_func = ExceptionRaisingFunc(ex, returned_exception_type=returned_exception_type)

    class OutputClass:
        def __init__(self, *args, **kwargs):
            internal_func(*args, **kwargs)

        def __call__(self, *args, **kwargs):
            internal_func(*args, **kwargs)

    if add_attrs is not None:
        for add_attr in add_attrs:
            setattr(OutputClass, add_attr, OutputClass.__call__)
    return OutputClass
