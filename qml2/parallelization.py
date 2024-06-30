# Utils related to parallelization.
from threadpoolctl import threadpool_limits

from .basic_utils import checked_environ_val
from .jit_interfaces import Pool_

num_procs_name = "QML2_NUM_PROCS"

default_num_procs_val = 1


def set_default_num_procs(num_procs):
    global default_num_procs_val
    default_num_procs_val = num_procs


def default_num_procs(num_procs=None):
    return checked_environ_val(
        num_procs_name, expected_answer=num_procs, default_answer=default_num_procs_val
    )


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
        with threadpool_limits(limits=fixed_num_threads):
            return embarrassingly_parallel_no_thread_fix(
                func,
                array,
                other_args,
                other_kwargs=other_kwargs,
                num_procs=num_procs,
            )
    else:
        return embarrassingly_parallel_no_thread_fix(
            func, array, other_args, other_kwargs=other_kwargs, num_procs=num_procs
        )
