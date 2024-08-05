# KK: the entire conftest setup was largely inspired by qml1/qmllib's.
import glob
import itertools
import os
import random
import tarfile

from numba import njit

from qml2.basic_utils import mkdir
from qml2.compound import Compound
from qml2.jit_interfaces import abs_, array_, dint_, empty_, max_, prod_, sign_, sum_, zeros_
from qml2.utils import all_possible_indices, read_xyz_file, write_compound_to_xyz_file

# For creating "perturbed" QM7 files.
test_dir = os.path.dirname(__file__)
test_data_dir = test_dir + "/test_data"


def get_benchmark_filename(benchmark_name):
    return test_data_dir + "/" + benchmark_name + ".dat"


original_xyz_tar = test_data_dir + "/qm7.tar.gz"
perturbed_xyz_dir = test_dir + "/test_data/perturbed_qm7"
# The perturbation magnitude is probably not enough to disturb true heat of formation from the values in
# hof_qm7.txt, but enough to ensure that row ordering in QM7-like methods is determined not by floating-point error,
# but by RNG seed.
xyz_perturbation_magnitude = 1e-6


def str2rng(s):
    return random.Random(int.from_bytes(bytes(s, "utf-8")))


# Procedures analogous to Numpy that do not depend on Numpy/Torch random number generator.
def randint_arr(lbound, ubound, shape, rng):
    output = empty_(shape, dtype=dint_)
    for i in all_possible_indices(shape):
        output[i] = rng.randint(lbound, ubound)
    return output


def random_arr(shape, rng):
    output = empty_(shape)
    for i in all_possible_indices(shape):
        output[i] = rng.random()
    return output


def check_perturbed_coordinates_existence(perturbed_xyz_dir=perturbed_xyz_dir, seed=1):
    if os.path.isdir(perturbed_xyz_dir):
        return
    rng = random.Random(seed)
    mkdir(perturbed_xyz_dir)

    tar_input = tarfile.open(original_xyz_tar)
    for member, xyz_name in zip(tar_input.getmembers(), tar_input.getnames()):
        xyz = tar_input.extractfile(member)
        if xyz is None:
            continue
        cur_comp = Compound(xyz=xyz)
        short_xyz_name = os.path.basename(xyz_name)
        cur_comp.coordinates += xyz_perturbation_magnitude * (
            random_arr(cur_comp.coordinates.shape, rng) - 0.5
        )
        write_compound_to_xyz_file(cur_comp, perturbed_xyz_dir + "/" + short_xyz_name)


def path_to_perturbed_coordinates(seed=1):
    """
    The tests use perturbed QM7 coordinates for benchmarking due to ordering issues symmetric molecules can cause with CM.
    The subroutine therefore:
    1. Ensures that the perturbed xyz files exist.
    2. returns route to the perturbed xyz files.
    """
    check_perturbed_coordinates_existence(perturbed_xyz_dir=perturbed_xyz_dir, seed=seed)
    return perturbed_xyz_dir


def perturbed_xyz_list(seed=1):
    p = path_to_perturbed_coordinates(seed=seed)
    return sorted(glob.glob(p + "/*.xyz"))


def perturbed_xyz_examples(rng, nxyzs, seed=1):
    all_xyzs = perturbed_xyz_list(seed=seed)
    rng.shuffle(all_xyzs)
    return all_xyzs[:nxyzs]


def xyz_nhatoms(xyz):
    nuclear_charges, _, _, _ = read_xyz_file(xyz)
    return sum_(nuclear_charges != 1)


def perturbed_xyz_nhatoms_interval(min_nhatoms, max_nhatoms, seed=1):
    """
    Find perturbed xyzs with a small enough number of heavy atoms.
    """
    output = []
    all_xyzs = perturbed_xyz_list(seed=seed)
    for xyz in all_xyzs:
        nhatoms = xyz_nhatoms(xyz)
        if (nhatoms >= min_nhatoms) and (nhatoms <= max_nhatoms):
            output.append(xyz)
    return output


# KK: storing entire arrays can become very wasteful, hence subroutines for creating random checksums of arrays.
float_format = "{:.12E}"


# A function which is close to randomly changing sign of an entry,
# but is continuous w.r.t. RNG output.
@njit(fastmath=True)
def unsigned_polynom(r):
    return 16 * r**4 - 1


@njit(fastmath=True)
def random_phase(r_in):
    if r_in <= 0.5:
        return unsigned_polynom(r_in)
    else:
        return -unsigned_polynom(1.0 - r_in)


def randomly_assign_phase(val, rng):
    return val * random_phase(rng.random())


def get_stack_1dim_bounds(nstacks_1dim, arr_dim):
    output = empty_((nstacks_1dim + 1,), dtype=dint_)
    base_size = arr_dim // nstacks_1dim
    remainder = arr_dim % nstacks_1dim
    output[0] = 0
    for stack_id in range(nstacks_1dim):
        output[stack_id + 1] = output[stack_id] + base_size
        if stack_id < remainder:
            output[stack_id + 1] += 1
    return output


def get_stack_bounds(stacks, arr_shape):
    return [get_stack_1dim_bounds(s1d, d) for s1d, d in zip(stacks, arr_shape)]


def get_stack_indices(stack_bounds, stack_ids):
    id_iterators = []
    for sb, si in zip(stack_bounds, stack_ids):
        id_iterators.append(range(sb[si], sb[si + 1]))
    return itertools.product(*id_iterators)


def create_checksums(arr, rng, nstack_checksums=1, stacks=1):
    arr_shape = arr.shape
    arr_dim = len(arr_shape)
    if not isinstance(stacks, tuple):
        stacks = (stacks,)
    tot_checksum_num = prod_(stacks) * nstack_checksums
    checksums = zeros_((tot_checksum_num,), dtype=arr.dtype)

    if arr_dim > len(stacks):
        stacks = (*stacks, *[1 for _ in range(arr_dim - len(stacks))])
    stack_sizes = empty_(
        (arr_dim,),
    )
    stack_size_remainders = empty_(
        (arr_dim,),
    )
    for dim_id, st in enumerate(stacks):
        stack_sizes[dim_id] = arr_shape[dim_id] // st
        stack_size_remainders[dim_id] = arr_shape[dim_id] % st

    stack_counters = all_possible_indices(stacks)

    stack_bounds = get_stack_bounds(stacks, arr_shape)

    stack_checksums_lb = 0
    for stack_ids in stack_counters:
        stack_indices = get_stack_indices(stack_bounds, stack_ids)
        for poss_id in stack_indices:
            val = arr[poss_id]
            checksums[stack_checksums_lb] += val
            for i in range(1, nstack_checksums):
                checksums[stack_checksums_lb + i] += randomly_assign_phase(val, rng)
        stack_checksums_lb += nstack_checksums

    return checksums


def read_all_checksums(benchmark_name):
    benchmark_file = get_benchmark_filename(benchmark_name)
    if not os.path.isfile(benchmark_file):
        return None
    benchmark_input = open(benchmark_file, "r")
    arr_name = None
    output = {}
    for line in benchmark_input.read().splitlines():
        if ":" in line:
            lsplit = line.split(":")
            if lsplit[0] == "E":
                assert arr_name == lsplit[1]
                continue
            arr_name = lsplit[1]
            output[arr_name] = []
            continue
        assert arr_name is not None
        output[arr_name].append(float(line))
    benchmark_input.close()
    converted_output = {}
    for arr_name, arr in output.items():
        converted_output[arr_name] = array_(arr)
    return converted_output


def print_checksum(arr_name, checksums, benchmark_output):
    """
    Print random array to a file.
    """
    print("B:" + arr_name, file=benchmark_output)
    for val in checksums:
        print(float_format.format(val), file=benchmark_output)
    print("E:" + arr_name, file=benchmark_output)


def print_checksum_dict(benchmark_name, checksum_dictionnary):
    benchmark_file = get_benchmark_filename(benchmark_name)
    benchmark_output = open(benchmark_file, "w")
    for arr_name, checksums in checksum_dictionnary.items():
        print_checksum(arr_name, checksums, benchmark_output)
    benchmark_output.close()


def max_diff(arr1, arr2):
    return max_(abs_(arr1 - arr2))


def max_rel_diff(arr1, arr2):
    return max_(abs_((arr1 - arr2) / (arr1 + arr2) * 2.0))


def add_checksum_to_dict(checksum_dict, arr_name, arr, rng, nstack_checksums=1, stacks=1):
    checksum_dict[arr_name] = create_checksums(
        arr, rng, nstack_checksums=nstack_checksums, stacks=stacks
    )


def compare_or_create(
    checksums_dict,
    benchmark_name,
    max_difference=None,
    max_rel_difference=1.0e-10,
    jit_dependent=False,
    partial_comparison=False,
):
    # Methods that implicitly depend on whether Numpy or Torch random number generator is used
    # have different files for different jit versions.
    if jit_dependent:
        from qml2.jit_interfaces import used_jit_name

        full_benchmark_name = used_jit_name + "_" + benchmark_name
    else:
        full_benchmark_name = benchmark_name
    # check that the benchmark filename exists.
    benchmark_checksums = read_all_checksums(full_benchmark_name)
    if benchmark_checksums is None:
        exception_str_bench_name = benchmark_name
        if jit_dependent:
            exception_str_bench_name += " (" + used_jit_name + ")"
        if partial_comparison:
            raise Exception(
                "WARNING: a partially checked benchmark was absent: " + exception_str_bench_name
            )
        # The benchmark needs to be created and a warning about this needs to be raised.
        print_checksum_dict(full_benchmark_name, checksums_dict)
        raise Exception(
            "WARNING: a benchmark was absent but was created: " + exception_str_bench_name
        )

    all_passed = True
    for name, checksums in checksums_dict.items():
        assert name in benchmark_checksums
        bench_cs = benchmark_checksums[name]
        if max_difference is None:
            assert max_rel_difference is not None
            diff_measure = max_rel_diff(checksums, bench_cs)
            diff_bound = max_rel_difference
        else:
            diff_measure = max_diff(checksums, bench_cs)
            diff_bound = max_difference
        passed = diff_measure < diff_bound
        if passed:
            print("PASSED", name, "DIFF:", diff_measure)
        else:
            print("FAILED", name, "DIFF:", diff_measure, ">", diff_bound)
            all_passed = False
    assert all_passed


def fix_reductor_signs(reductor):
    reductor *= sign_(reductor[0])


def fix_reductors_signs(reductors):
    for red_id in range(reductors.shape[0]):
        fix_reductor_signs(reductors[red_id])
