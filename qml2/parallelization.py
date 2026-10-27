# Utils related to parallelization.
# K.Karan.: I don't think there is a good way to limit number of Numba threads once numba has been initialized somewhere.
# This is why limiting the number of threads in parallel execution is done by executing a script in a child environment where all
# variables controlling parallel execution of different libraries are set to one.
import sys
import tempfile
import warnings

from .basic_utils import (
    checked_environ_val,
    dump2pkl,
    loadpkl,
    rep_numba_parallel_env_var_name,
    run,
    total_cpu_time,
)
from .jit_interfaces import get_default_jit, multiprocessing_, numba_flag

num_procs_name = "QML2_NUM_PROCS"

# procedures for generating one multiprocessing pool and reusing it.
global_mp_pool = None
global_mp_pool_nprocs = None
default_pool_start_method = "spawn"


def change_default_pool_start_method(new_default_pool_start_method):
    global default_pool_start_method
    default_pool_start_method = new_default_pool_start_method


def create_mp_pool(num_procs, pool_start_method=None):
    if pool_start_method is None:
        pool_start_method = default_pool_start_method
    used_num_procs = default_num_procs(num_procs)
    ctx = multiprocessing_.get_context(pool_start_method)
    return ctx.Pool(used_num_procs), used_num_procs


def get_global_mp_pool():
    return global_mp_pool


def check_get_global_mp_pool(num_procs, pool_start_method=None):
    mp_pool = get_global_mp_pool()
    if mp_pool is None:
        mp_pool, _ = create_mp_pool(num_procs, pool_start_method=pool_start_method)
    return mp_pool


def close_temp_mp_pool(pool):
    """
    If `pool` is not the module's global pool close it.
    """
    if pool is not global_mp_pool:
        pool.close()


def destroy_global_mp_pool():
    """
    Destroy current global MP pool.
    """
    global global_mp_pool, global_mp_pool_nprocs
    global_mp_pool.close()
    global_mp_pool = None
    global_mp_pool_nprocs = None


def create_global_mp_pool(num_procs=None, pool_start_method=None):
    """
    Initialize a global MP pool reused by QML2 in all parallelized calculations.

    Args:
        num_procs (int): if not None defines number of child processes, otherwise `default_num_procs_val` is used (which, unless changed by `set_default_num_procs`, equals `multiprocessing.cpu_count`).
    """
    global global_mp_pool, global_mp_pool_nprocs
    global_mp_pool, global_mp_pool_nprocs = create_mp_pool(
        num_procs, pool_start_method=pool_start_method
    )


def get_global_mp_pool_nprocs():
    return global_mp_pool_nprocs


# By default the code will try occupying all cores with one process
default_num_procs_val = multiprocessing_.cpu_count()
default_fixed_num_threads_val = 1


def set_default_num_procs(num_procs):
    global default_num_procs_val
    default_num_procs_val = num_procs


def set_fixed_num_threads(fixed_num_threads):
    global default_fixed_num_threads_val
    default_fixed_num_threads_val = fixed_num_threads


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
    def __init__(self, func, *other_args, **other_kwargs):
        self.func = func
        self.other_args = other_args
        self.other_kwargs = other_kwargs

    def __call__(self, x):
        return self.func(x, *self.other_args, **self.other_kwargs)


class TimedParallelHelper(ParallelHelper):
    """`ParallelHelper` that also reports the CPU time its call consumed.

    Measured in the worker, around the call and nothing else, which is the point: it
    excludes everything parallelizing costs rather than performs. With `fixed_num_threads`
    that is a child interpreter's startup and its `qml2` import and Numba compilation, one
    pool worker's worth of the same for every process, and the pickling of the arguments in
    and the results out -- measured on ensembles of tight-binding calculations, about 2.5 s
    for the child plus 1.2 s per worker, against 2 ms per element of pickling.

    So a caller wanting to know what the WORK cost, rather than what this run of it cost,
    cannot get there by subtracting the pickling: the pickling is the small part. Timing
    the call itself is both more accurate and simpler.
    """

    def __call__(self, x):
        start_cpu_time = total_cpu_time()
        result = self.func(x, *self.other_args, **self.other_kwargs)
        return result, total_cpu_time() - start_cpu_time


def embarrassingly_parallel_no_thread_fix(
    func,
    array,
    other_args,
    other_kwargs={},
    num_procs=None,
    pool_start_method=None,
    return_cpu_time=False,
):
    if num_procs == 1:
        if return_cpu_time:
            output = []
            func_cpu_time = 0.0
            for element in array:
                start_cpu_time = total_cpu_time()
                output.append(func(element, *other_args, **other_kwargs))
                func_cpu_time += total_cpu_time() - start_cpu_time
            return output, func_cpu_time
        return [func(element, *other_args, **other_kwargs) for element in array]
    else:
        if return_cpu_time:
            helper_class = TimedParallelHelper
        else:
            helper_class = ParallelHelper
        p = check_get_global_mp_pool(num_procs, pool_start_method=pool_start_method)
        output = p.map(
            helper_class(func, *other_args, **other_kwargs),
            array,
        )
        close_temp_mp_pool(p)
        if return_cpu_time:
            # Summed over workers: each entry was measured inside the worker that produced
            # it, so the total is CPU seconds of work regardless of how many processes ran.
            func_cpu_time = sum(element_cpu_time for _, element_cpu_time in output)
            return [result for result, _ in output], func_cpu_time
        return output


def embarrassingly_parallel(
    func,
    array,
    other_args,
    other_kwargs={},
    num_procs=None,
    fixed_num_threads=None,
    pool_start_method=None,
    return_cpu_time=False,
):
    """
    Find in parallel results of `[func(element, *other_args, **other_kwargs) for element in array]`. Additional keyword arguments are:
    - num_procs (int or None): if not None defines number of Python processes launched in parallel. If None the number is given by `multiprocessing.cpu_count()`.
    - fixed_num_threads (int or None): if not None the function launches the calculations in a child environment where OpenMP threads for all processes are set to fixed_num_threads.
    - return_cpu_time (bool): if True return `(results, cpu_seconds)` instead of `results`, `cpu_seconds` being the CPU time spent inside the `func` calls themselves, summed over workers. See `TimedParallelHelper` for what that deliberately leaves out; in short, it is the cost of the work rather than the cost of this particular way of running it, and it does not grow with `num_procs`.
    """
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
            pool_start_method=pool_start_method,
            return_cpu_time=return_cpu_time,
        )
    else:
        return embarrassingly_parallel_no_thread_fix(
            func,
            array,
            other_args,
            other_kwargs=other_kwargs,
            num_procs=num_procs,
            pool_start_method=pool_start_method,
            return_cpu_time=return_cpu_time,
        )


def create_func_exec(
    dump_filename,
    exec_scr_name,
    num_procs=None,
    fixed_num_threads=None,
    pool_start_method=None,
    return_cpu_time=False,
):
    # subprocess.run seems to have issues running commands of the type "QML2_NUM_PROCS=20 python bla.py"
    # we therefore modify the variables inside the run script
    if pool_start_method is None:
        pool_start_method = default_pool_start_method

    contents = (
        f"""# This script was set up as part of a call to qml2.parallelization.embarrassingly_parallel. If the script crashed, please consult the error log for where it had been called. If the script is frozen, to recover where it was called from consider killing the corresponding process. The latter is done in Linux with:
# pkill -9 {exec_scr_name}
if __name__ == '__main__':
\timport os
\tos.environ['"""
        + num_procs_name
        + """']='"""
        + str(default_num_procs(num_procs=num_procs))
        + "'\n"
    )
    for thread_var in num_threads_var_names:
        contents += (
            "\tos.environ['"
            + thread_var
            + "']='"
            + str(default_fixed_num_threads(fixed_num_threads=fixed_num_threads))
            + "'\n"
        )
    contents += (
        f"""
\tos.environ['{rep_numba_parallel_env_var_name}']='1'
\tfrom qml2.basic_utils import loadpkl, dump2pkl
\tfrom qml2.parallelization import embarrassingly_parallel_no_thread_fix

\tdump_filename='"""
        + dump_filename
        + """'

\texecuted=loadpkl(dump_filename)

\toutput=embarrassingly_parallel_no_thread_fix(executed["func"], executed["array"], executed["args"], other_kwargs=executed["other_kwargs"], pool_start_method='"""
        + pool_start_method
        + """', return_cpu_time="""
        + str(return_cpu_time)
        + """)
\tdump2pkl(output, dump_filename)
"""
    )
    output = open(exec_scr_name, "w")
    output.write(contents)
    output.close()


def create_run_func_exec(
    func,
    array,
    other_args,
    other_kwargs,
    num_procs=None,
    fixed_num_threads=None,
    pool_start_method=None,
    return_cpu_time=False,
):
    tmpdir = tempfile.TemporaryDirectory(dir=".", delete=False)

    dump_filename = tmpdir.name + "/temp_embarrassingly_parallel_dump.pkl"

    exec_scr_name = tmpdir.name + "/temp_embarrassingly_parallel_scr.py"

    dump2pkl(
        {"func": func, "array": array, "args": other_args, "other_kwargs": other_kwargs},
        dump_filename,
    )
    create_func_exec(
        dump_filename,
        exec_scr_name,
        num_procs=num_procs,
        fixed_num_threads=fixed_num_threads,
        pool_start_method=pool_start_method,
        return_cpu_time=return_cpu_time,
    )
    res = run(sys.executable, exec_scr_name, capture_output=True)
    assert res.returncode == 0, (
        f"Parallel execution failed, can be reproduced with: {tmpdir.name}\nFull error message:\n"
        + res.stderr.decode()
    )
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
        fixed_num_threads=None,
        test_mode=False,
        serial=False,
        **kwargs,
    ):
        if serial:
            for i, el in enumerate(list_obj):
                list_obj[i] = self.processed_attribute(el, *args, **kwargs)
            return
        if test_mode:
            if get_default_jit() == numba_flag:
                dnum_procs = default_num_procs(num_procs)
                dfixed_num_threads = default_fixed_num_threads(fixed_num_threads)
                if dfixed_num_threads is None:
                    dfixed_num_threads = 1
                if dnum_procs == 1 and dfixed_num_threads == 1:
                    warnings.warn(
                        "WARNING: cannot check whether Numba parallelization works correctly; check the number of CPUs available on your machine as well as values of parallelization-related environmental variables."
                    )
                else:
                    num_procs = default_num_procs(num_procs) // 2
                    fixed_num_threads = default_fixed_num_threads(fixed_num_threads) * 2
            else:
                warnings.warn("WARNING: multiprocessing not tested with Torch!")
                num_procs = 1
                fixed_num_threads = None

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
