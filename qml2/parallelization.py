# Utils related to parallelization.
# K.Karan.: I don't think there is a good way to limit number of Numba threads once numba has been initialized somewhere.
# This is why limiting the number of threads in parallel execution is done by executing a script in a child environment where all
# variables controlling parallel execution of different libraries are set to one.
import sys
import tempfile
import warnings

from .basic_utils import checked_environ_val, dump2pkl, loadpkl, run
from .jit_interfaces import Pool_, multiprocessing_

num_procs_name = "QML2_NUM_PROCS"

# By default the code will try occupying all cores with one process
default_num_procs_val = multiprocessing_.cpu_count()
default_fixed_num_threads_val = 1


def set_default_num_procs(num_procs):
    global default_num_procs_val
    default_num_procs_val = num_procs


def set_fixed_num_threads(fixed_num_threads):
    global default_fixed_num_threads
    default_fixed_num_threads = fixed_num_threads


num_threads_var_names = [
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMBA_NUM_THREADS",
]


def default_num_procs(num_procs=None):
    return checked_environ_val(
        num_procs_name, expected_answer=num_procs, default_answer=default_num_procs_val
    )


def default_fixed_num_threads(fixed_num_threads=None):
    # TODO: would anyone one a separate variable for this?
    if fixed_num_threads is None:
        return default_fixed_num_threads_val
    else:
        return fixed_num_threads


class ParallelHelper:
    def __init__(self, func, other_args=(), other_kwargs={}):
        self.func = func
        self.other_args = other_args
        self.other_kwargs = other_kwargs

    def __call__(self, x):
        return self.func(x, *self.other_args, **self.other_kwargs)


def embarrassingly_parallel_no_thread_fix(
    func, array, other_args, other_kwargs={}, num_procs=None
):
    if num_procs == 1:
        return [func(element, *other_args, **other_kwargs) for element in array]
    else:
        # TODO: Make a global pool?
        p = Pool_(default_num_procs(num_procs))
        output = p.map(
            ParallelHelper(func, other_args=other_args, other_kwargs=other_kwargs),
            array,
        )
        p.close()
        return output


def embarrassingly_parallel(
    func, array, other_args, other_kwargs={}, num_procs=None, fixed_num_threads=None
):
    if type(other_args) is not tuple:
        other_args = (other_args,)
    if fixed_num_threads is not None:
        return create_run_func_exec(
            func,
            array,
            other_args,
            other_kwargs,
            num_procs=num_procs,
            fixed_num_threads=fixed_num_threads,
        )
    else:
        return embarrassingly_parallel_no_thread_fix(
            func, array, other_args, other_kwargs=other_kwargs, num_procs=num_procs
        )


def create_func_exec(dump_filename, exec_scr_name, num_procs=None, fixed_num_threads=None):
    # subprocess.run seems to have issues running commands of the type "QML2_NUM_PROCS=20 python bla.py"
    # we therefore modify the variables inside the run script
    contents = (
        """
import os
os.environ['"""
        + num_procs_name
        + """']='"""
        + str(default_num_procs(num_procs=num_procs))
        + "'\n"
    )
    for thread_var in num_threads_var_names:
        contents += (
            "os.environ['"
            + thread_var
            + "']='"
            + str(default_fixed_num_threads(fixed_num_threads=fixed_num_threads))
            + "'\n"
        )
    contents += (
        """
from qml2.basic_utils import loadpkl, dump2pkl
from qml2.parallelization import embarrassingly_parallel_no_thread_fix

dump_filename='"""
        + dump_filename
        + """'

executed=loadpkl(dump_filename)

output=embarrassingly_parallel_no_thread_fix(executed["func"], executed["array"], executed["args"], other_kwargs=executed["other_kwargs"])
dump2pkl(output, dump_filename)
"""
    )
    output = open(exec_scr_name, "w")
    output.write(contents)
    output.close()


def create_run_func_exec(
    func, array, other_args, other_kwargs, num_procs=None, fixed_num_threads=None
):
    tmpdir = tempfile.TemporaryDirectory(dir=".")

    dump_filename = tmpdir.name + "/temp_dump.pkl"

    exec_scr_name = tmpdir.name + "/temp_scr.py"

    dump2pkl(
        {"func": func, "array": array, "args": other_args, "other_kwargs": other_kwargs},
        dump_filename,
    )
    create_func_exec(
        dump_filename, exec_scr_name, num_procs=num_procs, fixed_num_threads=fixed_num_threads
    )
    run(sys.executable, exec_scr_name)
    output = loadpkl(dump_filename)
    tmpdir.cleanup()
    return output


# For parallelizing calculations over lists of instances of classes such as Compound class.
class ProcessedAttribute:
    def __init__(self, attribute_name, base_class):
        self.attribute = getattr(base_class, attribute_name)

    def __call__(self, obj, *args, **kwargs):
        self.attribute(obj, *args, **kwargs)
        return obj


class MultipleProcessedAttributes:
    def __init__(self, attribute_name, base_classes):
        self.attributes = {}
        self.base_classes = base_classes
        for base_class in self.base_classes:
            self.attributes[base_class] = getattr(base_class, attribute_name)

    def __call__(self, obj, *args, **kwargs):
        self.attributes[type(obj)](obj, *args, **kwargs)
        return obj


def get_ProcessedAttribute(attribute_name, base_class=None, base_classes=None):
    if base_class is not None:
        return ProcessedAttribute(attribute_name, base_class)
    if base_classes is not None:
        return MultipleProcessedAttributes(attribute_name, base_classes)
    raise Exception


class ParallelizedAttribute:
    def __init__(self, attribute_name, base_class=None, base_classes=None):
        self.processed_attribute = get_ProcessedAttribute(
            attribute_name, base_class=base_class, base_classes=base_classes
        )

    def __call__(
        self,
        list_obj,
        *args,
        num_procs=None,
        fixed_num_threads=default_fixed_num_threads_val,
        test_mode=False,
        serial=False,
        **kwargs
    ):
        if serial:
            for i, el in enumerate(list_obj):
                list_obj[i] = self.processed_attribute(el, *args, **kwargs)
            return
        if test_mode:
            dnum_procs = default_num_procs(num_procs)
            dfixed_num_threads = default_fixed_num_threads(fixed_num_threads)
            if dnum_procs == 1 and dfixed_num_threads == 1:
                warnings.warn(
                    "WARNING: cannot check whether Numba parallelization works correctly; check the number of CPUs available on your machine as well as values of parallelization-related environmental variables."
                )
            else:
                num_procs = default_num_procs(num_procs) // 2
                fixed_num_threads = default_fixed_num_threads(fixed_num_threads) * 2

        new_vals = embarrassingly_parallel(
            self.processed_attribute,
            list_obj,
            args,
            other_kwargs=kwargs,
            num_procs=num_procs,
            fixed_num_threads=fixed_num_threads,
        )
        for i, new_val in enumerate(new_vals):
            list_obj[i] = new_val


stored_parallelized_attributes = {}


def store_parallelize_attribute(attribute_name, base_class=None, base_classes=None):
    global stored_parallelized_attributes
    if base_class is None:
        assert base_classes is not None
        class_factor = tuple(base_classes)
    else:
        class_factor = base_class
    PA = ParallelizedAttribute(attribute_name, base_class=base_class, base_classes=base_classes)

    def attr(self, *args, **kwargs):
        PA(self, *args, **kwargs)

    label = (class_factor, attribute_name)
    global stored_parallelized_attributes
    stored_parallelized_attributes[label] = attr
    return label


class parallelized_inheritance_wrapper:
    def __init__(self, parallelized_attributes_labels):
        self.parallelized_attributes_labels = parallelized_attributes_labels

    def __call__(self, signature_or_function):
        for label in self.parallelized_attributes_labels:
            setattr(signature_or_function, label[1], stored_parallelized_attributes[label])
        return signature_or_function


def parallelized_inheritance(*attributes, base_class=None, base_classes=None):
    parallelized_attributes_labels = [
        store_parallelize_attribute(attribute, base_class=base_class, base_classes=base_classes)
        for attribute in attributes
    ]
    return parallelized_inheritance_wrapper(parallelized_attributes_labels)
