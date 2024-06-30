from typing import Tuple

from ..data import nCartDim
from ..jit_interfaces import (
    dim0float_array_,
    dint_,
    dtype_,
    empty_,
    exp_,
    int_,
    jit_,
    ndarray_,
    prange_,
    save_,
    sum_,
)
from ..utils import check_allocation, get_atom_environment_ranges, l2_sq_norm
from .kernels import half_inv_sq_sigma

# KK: The module could be made more efficient, but TBH I think we should soon be switching to QMLightning anyway,
# so I focused on keeping it modular in case we decide to add more kernel expressions and more or less readable.


@jit_
def gaussian_kernel_function_wders(
    A_rep: ndarray_,
    B_rep: ndarray_,
    kernel_component_wders: ndarray_,
    inv_sq_half_sigma: dim0float_array_,
):
    """
    Creates inside kernel_component_wders array:
    0 - Gaussian kernel element
    1: - derivatives w.r.t. B_rep (which are also negative derivatives w.r.t. A_rep)
    """
    kernel_component_wders[1:] = B_rep[:] - A_rep[:]
    kcw_saved = save_(kernel_component_wders)
    kernel_component_wders[0] = exp_(-inv_sq_half_sigma * l2_sq_norm(kcw_saved[1:]))
    kernel_component_wders[1:] *= -inv_sq_half_sigma * 2 * kernel_component_wders[0]


@jit_
def gaussian_kernel_function_w2ders(
    A_rep: ndarray_,
    B_rep: ndarray_,
    kernel_component_wders: ndarray_,
    inv_sq_half_sigma: dim0float_array_,
):
    """
    Creates inside kernel_component_wders array:
    0,0 - Gaussian kernel element
    0,1: - derivatives w.r.t. B_rep
    1:,0 - derivatives w.r.t. A_rep
    1:,1: - derivatives w.r.t. A_rep and B_rep
    """
    kernel_component_wders[0, 1:] = B_rep[:] - A_rep[:]
    kcw_saved = save_(kernel_component_wders[0])
    kernel_component = exp_(-inv_sq_half_sigma * l2_sq_norm(kcw_saved[1:]))

    nreps = A_rep.shape[0]

    inv_sigma4 = 4 * inv_sq_half_sigma**2

    for iA in range(1, nreps + 1):
        kernel_component_wders[iA, 1:] = -kcw_saved[iA] * kcw_saved[1:] * inv_sigma4
        kernel_component_wders[iA, iA] += 2 * inv_sq_half_sigma
    kernel_component_wders[0, 1:] *= -kernel_component * 2 * inv_sq_half_sigma
    kernel_component_wders[1:, 1:] *= kernel_component
    kernel_component_wders[1:, 0] = -save_(kernel_component_wders)[0, 1:]
    kernel_component_wders[0, 0] = kernel_component


base_wders = {"gaussian": gaussian_kernel_function_wders}
base_w2ders = {"gaussian": gaussian_kernel_function_w2ders}


# Constructing force and Gaussian process kernels from base functions
@jit_
def add_force_kernel_contributions(
    B_drep: ndarray_,
    B_rel_neighbors: ndarray_,
    B_rel_neighbor_num: int_,
    kernel_component: ndarray_,
    temp_der_arr: ndarray_,
    nCartDim: int_ = nCartDim,
):
    tda = save_(temp_der_arr)
    nreps = B_drep.shape[0]
    for neighbor_id in range(int(B_rel_neighbor_num)):
        cur_neighbor = B_rel_neighbors[neighbor_id]
        lb = int(cur_neighbor * nCartDim)
        ub = lb + nCartDim
        for rep_comp_id in range(nreps):
            # Note the minus sign, because we are calculating force, not gradient.
            kernel_component[lb:ub] -= tda[rep_comp_id] * B_drep[rep_comp_id, neighbor_id, :]


def construct_add_oqml_kernel_function(base_kernel_function_wders):
    @jit_
    def add_oqml_kernel_function(
        A_rep: ndarray_,
        B_rep: ndarray_,
        B_drep: ndarray_,
        B_rel_neighbors: ndarray_,
        B_rel_neighbor_num: int_,
        kernel_component: ndarray_,
        temp_der_arr: ndarray_,
        sigma_param: dim0float_array_,
    ):
        base_kernel_function_wders(A_rep, B_rep, temp_der_arr, sigma_param)
        kernel_component[0] += temp_der_arr[0]
        add_force_kernel_contributions(
            B_drep, B_rel_neighbors, B_rel_neighbor_num, kernel_component[1:], temp_der_arr[1:]
        )

    return add_oqml_kernel_function


def construct_add_gp_kernel_function(base_kernel_function_w2ders):
    @jit_
    def add_gp_kernel_function(
        A_rep: ndarray_,
        B_rep: ndarray_,
        A_drep: ndarray_,
        B_drep: ndarray_,
        A_rel_neighbors: ndarray_,
        B_rel_neighbors: ndarray_,
        A_rel_neighbor_num: int_,
        B_rel_neighbor_num: int_,
        kernel_component: ndarray_,
        temp_der_arr: ndarray_,
        sigma_param: dim0float_array_,
        nCartDim: int_ = nCartDim,
    ):
        temp_der_arr = save_(temp_der_arr)
        base_kernel_function_w2ders(A_rep, B_rep, temp_der_arr, sigma_param)
        kernel_component[0, 0] += temp_der_arr[0, 0]
        add_force_kernel_contributions(
            B_drep,
            B_rel_neighbors,
            B_rel_neighbor_num,
            kernel_component[0, 1:],
            temp_der_arr[0, 1:],
        )
        add_force_kernel_contributions(
            A_drep,
            A_rel_neighbors,
            A_rel_neighbor_num,
            kernel_component[1:, 0],
            temp_der_arr[1:, 0],
        )
        nreps = A_rep.shape[0]
        for A_neighbor_id in range(A_rel_neighbor_num):
            cur_neighbor_A = A_rel_neighbors[A_neighbor_id]
            lbA = cur_neighbor_A * nCartDim + 1
            for B_neighbor_id in range(B_rel_neighbor_num):
                cur_neighbor_B = B_rel_neighbors[B_neighbor_id]
                lbB = cur_neighbor_B * nCartDim + 1
                ubB = lbB + nCartDim
                for iA in range(nCartDim):
                    for rep_id_A in range(nreps):
                        for rep_id_B in range(nreps):
                            kernel_component[lbA + iA, lbB:ubB] += (
                                A_drep[rep_id_A, A_neighbor_id, iA]
                                * B_drep[rep_id_B, B_neighbor_id, :]
                                * temp_der_arr[rep_id_A + 1, rep_id_B + 1]
                            )

    return add_gp_kernel_function


def construct_local_dn_oqml_kernel_function(base_kernel_function_wders):
    add_oqml_kernel_function_wders = construct_add_oqml_kernel_function(base_kernel_function_wders)

    @jit_
    def local_dn_oqml_kernel_function(
        A_rep: ndarray_,
        B_reps: ndarray_,
        B_dreps: ndarray_,
        A_ncharge: ndarray_,
        B_ncharges: ndarray_,
        B_rel_neighbors: ndarray_,
        B_rel_neighbor_nums: ndarray_,
        kernel_component: ndarray_,
        temp_der_arr: ndarray_,
        sigma_param: dim0float_array_,
    ):
        nB = B_reps.shape[0]

        kernel_component[:] = 0.0
        for iB in range(nB):
            if A_ncharge != B_ncharges[iB]:
                continue
            add_oqml_kernel_function_wders(
                A_rep,
                B_reps[iB],
                B_dreps[iB],
                B_rel_neighbors[iB],
                B_rel_neighbor_nums[iB],
                kernel_component,
                temp_der_arr,
                sigma_param,
            )

    return local_dn_oqml_kernel_function


def construct_local_dn_gp_kernel_function(base_kernel_function_w2ders):
    add_gp_kernel_function_wders = construct_add_gp_kernel_function(base_kernel_function_w2ders)

    @jit_
    def local_dn_gp_kernel_function(
        A_reps: ndarray_,
        B_reps: ndarray_,
        A_dreps: ndarray_,
        B_dreps: ndarray_,
        A_ncharges: ndarray_,
        B_ncharges: ndarray_,
        A_rel_neighbors: ndarray_,
        B_rel_neighbors: ndarray_,
        A_rel_neighbor_nums: ndarray_,
        B_rel_neighbor_nums: ndarray_,
        kernel_component: ndarray_,
        temp_der_arr: ndarray_,
        sigma_param: dim0float_array_,
    ):
        nA = A_reps.shape[0]
        nB = B_reps.shape[0]
        kernel_component[:, :] = 0.0
        for iA in range(nA):
            A_rep = A_reps[iA]
            A_drep = A_dreps[iA]
            cur_A_rel_neighbor_ids = A_rel_neighbors[iA]
            A_rel_neighbor_num = int(A_rel_neighbor_nums[iA])
            nA = A_ncharges[iA]
            for iB in range(nB):
                if nA != B_ncharges[iB]:
                    continue
                add_gp_kernel_function_wders(
                    A_rep,
                    B_reps[iB],
                    A_drep,
                    B_dreps[iB],
                    cur_A_rel_neighbor_ids,
                    B_rel_neighbors[iB],
                    A_rel_neighbor_num,
                    int(B_rel_neighbor_nums[iB]),
                    kernel_component,
                    temp_der_arr,
                    sigma_param,
                )

    return local_dn_gp_kernel_function


# Local force kernels.
@jit_
def prediction_vector_length(natoms: int_, nCartDim: int_ = nCartDim):
    return natoms * nCartDim + 1


@jit_
def get_energy_force_ranges(natoms: ndarray_, dint_: dtype_ = dint_):
    nmols = natoms.shape[0]
    ranges = empty_((nmols + 1,), dtype=dint_)
    ranges[0] = 0
    for i in range(nmols):
        ranges[i + 1] = ranges[i] + prediction_vector_length(int(natoms[i]))
    return ranges


# KK: Constructing symmetric version of this kernel looks like a lot of work with questionable payoff,
# especially given the switch to QML-Lightning.
def construct_local_dn_oqml_kernel_asymmetric(
    base_kernel_function_wders, sigma_to_param=half_inv_sq_sigma
):
    kernel_func = construct_local_dn_oqml_kernel_function(base_kernel_function_wders)

    @jit_
    def local_dn_oqml_kernel(
        A_reps,
        B_reps,
        B_dreps,
        nB,
        A_ncharges,
        B_ncharges,
        B_rel_neighbors,
        B_rel_neighbor_nums,
        sigma,
        output_kernel,
    ):
        sigma_param = sigma_to_param(sigma)
        nconfigs_A = A_reps.shape[0]
        nmols_B = nB.shape[0]
        nreps = A_reps.shape[1]
        ubound_arr_B = get_atom_environment_ranges(nB)
        B_en_force_ranges = get_energy_force_ranges(nB)
        for iA in prange_(nconfigs_A):
            temp_der_arr = empty_((nreps + 1,))
            for iB in range(nmols_B):
                lbB = ubound_arr_B[iB]
                ubB = ubound_arr_B[iB + 1]
                kernel_func(
                    A_reps[iA],
                    B_reps[lbB:ubB],
                    B_dreps[lbB:ubB],
                    A_ncharges[iA],
                    B_ncharges[lbB:ubB],
                    B_rel_neighbors[lbB:ubB],
                    B_rel_neighbor_nums[lbB:ubB],
                    output_kernel[iA, B_en_force_ranges[iB] : B_en_force_ranges[iB + 1]],
                    temp_der_arr,
                    sigma_param,
                )

    return local_dn_oqml_kernel


# Gaussian process kernels.
def construct_local_dn_gp_kernel_asymmetric(
    base_kernel_function_w2ders, sigma_to_param=half_inv_sq_sigma
):
    gp_kernel_func = construct_local_dn_gp_kernel_function(base_kernel_function_w2ders)

    @jit_
    def local_dn_gp_kernel_asymmetric(
        A_reps,
        B_reps,
        A_dreps,
        B_dreps,
        nA,
        nB,
        A_ncharges,
        B_ncharges,
        A_rel_neighbors,
        B_rel_neighbors,
        A_rel_nums,
        B_rel_nums,
        sigma,
        output_kernel,
    ):
        sigma_param = sigma_to_param(sigma)
        nmols_A = nA.shape[0]
        nmols_B = nB.shape[0]
        nreps = A_reps.shape[1]
        ubound_arr_A = get_atom_environment_ranges(nA)
        ubound_arr_B = get_atom_environment_ranges(nB)
        A_en_force_ranges = get_energy_force_ranges(nA)
        B_en_force_ranges = get_energy_force_ranges(nB)
        for iA in prange_(nmols_A):
            temp_der_arr = empty_((nreps + 1, nreps + 1))

            lbA = ubound_arr_A[iA]
            ubA = ubound_arr_A[iA + 1]
            lb_en_forceA = A_en_force_ranges[iA]
            ub_en_forceA = A_en_force_ranges[iA + 1]
            A_rep_subarray = A_reps[lbA:ubA]
            A_drep_subarray = A_dreps[lbA:ubA]
            A_ncharges_subarray = A_ncharges[lbA:ubA]
            A_rel_neighbors_subarray = A_rel_neighbors[lbA:ubA]
            A_rel_nums_subarray = A_rel_nums[lbA:ubA]
            for iB in range(nmols_B):
                lbB = ubound_arr_B[iB]
                ubB = ubound_arr_B[iB + 1]
                lb_en_forceB = B_en_force_ranges[iB]
                ub_en_forceB = B_en_force_ranges[iB + 1]
                gp_kernel_func(
                    A_rep_subarray,
                    B_reps[lbB:ubB],
                    A_drep_subarray,
                    B_dreps[lbB:ubB],
                    A_ncharges_subarray,
                    B_ncharges[lbB:ubB],
                    A_rel_neighbors_subarray,
                    B_rel_neighbors[lbB:ubB],
                    A_rel_nums_subarray,
                    B_rel_nums[lbB:ubB],
                    output_kernel[lb_en_forceA:ub_en_forceA, lb_en_forceB:ub_en_forceB],
                    temp_der_arr,
                    sigma_param,
                )

    return local_dn_gp_kernel_asymmetric


@jit_
def symmetrize_local_dn_gp_kernel(gp_kernel, en_force_ranges):
    copied_gp_kernel = save_(gp_kernel)
    nmols_A = en_force_ranges.shape[0] - 1
    for i1 in prange_(nmols_A):
        lb_en_force1 = en_force_ranges[i1]
        ub_en_force1 = en_force_ranges[i1 + 1]
        for i2 in range(i1):
            lb_en_force2 = en_force_ranges[i2]
            ub_en_force2 = en_force_ranges[i2 + 1]
            gp_kernel[lb_en_force2:ub_en_force2, lb_en_force1:ub_en_force1] = copied_gp_kernel[
                lb_en_force1:ub_en_force1, lb_en_force2:ub_en_force2
            ].T


def construct_local_dn_gp_kernel_symmetric(
    base_kernel_function_w2ders, sigma_to_param=half_inv_sq_sigma
):
    gp_kernel_func = construct_local_dn_gp_kernel_function(base_kernel_function_w2ders)

    @jit_
    def local_dn_gp_kernel_symmetric(
        A_reps, A_dreps, nA, A_ncharges, A_rel_neighbors, A_rel_nums, sigma, output_kernel
    ):
        sigma_param = sigma_to_param(sigma)
        nmols_A = nA.shape[0]
        nreps = A_reps.shape[1]
        ubound_arr_A = get_atom_environment_ranges(nA)
        A_en_force_ranges = get_energy_force_ranges(nA)
        for i1 in prange_(nmols_A):
            temp_der_arr = empty_((nreps + 1, nreps + 1))

            lb1 = ubound_arr_A[i1]
            ub1 = ubound_arr_A[i1 + 1]
            lb_en_force1 = A_en_force_ranges[i1]
            ub_en_force1 = A_en_force_ranges[i1 + 1]
            A_rep_subarray = A_reps[lb1:ub1]
            A_drep_subarray = A_dreps[lb1:ub1]
            A_ncharges_subarray = A_ncharges[lb1:ub1]
            A_rel_neighbors_subarray = A_rel_neighbors[lb1:ub1]
            A_rel_nums_subarray = A_rel_nums[lb1:ub1]
            for i2 in range(i1 + 1):
                lb2 = ubound_arr_A[i2]
                ub2 = ubound_arr_A[i2 + 1]
                lb_en_force2 = A_en_force_ranges[i2]
                ub_en_force2 = A_en_force_ranges[i2 + 1]
                gp_kernel_func(
                    A_rep_subarray,
                    A_reps[lb2:ub2],
                    A_drep_subarray,
                    A_dreps[lb2:ub2],
                    A_ncharges_subarray,
                    A_ncharges[lb2:ub2],
                    A_rel_neighbors_subarray,
                    A_rel_neighbors[lb2:ub2],
                    A_rel_nums_subarray,
                    A_rel_nums[lb2:ub2],
                    output_kernel[lb_en_force1:ub_en_force1, lb_en_force2:ub_en_force2],
                    temp_der_arr,
                    sigma_param,
                )
                if i2 != i1:
                    output_kernel[
                        lb_en_force2:ub_en_force2, lb_en_force1:ub_en_force1
                    ] = output_kernel[lb_en_force1:ub_en_force1, lb_en_force2:ub_en_force2].T
        symmetrize_local_dn_gp_kernel(output_kernel, A_en_force_ranges)

    return local_dn_gp_kernel_symmetric


precompiled_kernels = {}


def construct_derivative_kernel(symmetric=False, derivatives="oqml", type="gaussian"):
    match derivatives:
        case "oqml":
            base_dict = base_wders
            assert not symmetric
            constructor = construct_local_dn_oqml_kernel_asymmetric
        case "gaussian_process":
            base_dict = base_w2ders
            if symmetric:
                constructor = construct_local_dn_gp_kernel_symmetric
            else:
                constructor = construct_local_dn_gp_kernel_asymmetric
        case _:
            raise Exception
    base_func = base_dict[type]
    return constructor(base_func)


def get_derivative_kernel(symmetric=False, derivatives="oqml", type="gaussian"):
    assert type == "gaussian"
    search_tuple = (symmetric, derivatives, type)
    if search_tuple not in precompiled_kernels:
        precompiled_kernels[search_tuple] = construct_derivative_kernel(
            symmetric=symmetric, derivatives=derivatives, type=type
        )
    return precompiled_kernels[search_tuple]


def num_en_forces(atom_num_arr):
    return atom_num_arr.shape[0] + nCartDim * int(sum(atom_num_arr))


def local_dn_oqml_gaussian_kernel(
    A, B, B_dreps, nB, A_ncharges, B_ncharges, B_rel_neighbors, B_rel_neighbor_nums, sigma
):
    kernel_func = get_derivative_kernel(derivatives="oqml")
    output_kernel = empty_((A.shape[0], num_en_forces(nB)))
    kernel_func(
        A,
        B,
        B_dreps,
        nB,
        A_ncharges,
        B_ncharges,
        B_rel_neighbors,
        B_rel_neighbor_nums,
        sigma,
        output_kernel,
    )
    return output_kernel


def local_dn_gp_gaussian_kernel(
    A,
    B,
    A_dreps,
    B_dreps,
    nA,
    nB,
    A_ncharges,
    B_ncharges,
    A_rel_neighbors,
    B_rel_neighbors,
    A_rel_neighbor_nums,
    B_rel_neighbor_nums,
    sigma,
):
    symmetric = B is None
    kernel_func = get_derivative_kernel(symmetric=symmetric, derivatives="gaussian_process")
    output_kernel = empty_((num_en_forces(nA), num_en_forces(nB)))
    if symmetric:
        kernel_func(
            A, A_dreps, nA, A_ncharges, A_rel_neighbors, A_rel_neighbor_nums, sigma, output_kernel
        )
    else:
        kernel_func(
            A,
            B,
            A_dreps,
            B_dreps,
            nA,
            nB,
            A_ncharges,
            B_ncharges,
            A_rel_neighbors,
            B_rel_neighbors,
            A_rel_neighbor_nums,
            B_rel_neighbor_nums,
            sigma,
            output_kernel,
        )
    return output_kernel


def local_dn_oqml_gaussian_kernel_symmetric(
    A, A_dreps, nA, A_ncharges, A_rel_neighbors, A_rel_neighbor_nums, sigma
):
    return local_dn_oqml_gaussian_kernel(
        A, A, A_dreps, nA, nA, A_ncharges, A_ncharges, A_rel_neighbors, A_rel_neighbor_nums, sigma
    )


def local_dn_gp_gaussian_kernel_symmetric(
    A, A_dreps, nA, A_ncharges, A_rel_neighbors, A_rel_neighbor_nums, sigma
) -> ndarray_:
    return local_dn_gp_gaussian_kernel(
        A,
        None,
        A_dreps,
        None,
        nA,
        nA,
        A_ncharges,
        None,
        A_rel_neighbors,
        None,
        A_rel_neighbor_nums,
        None,
        sigma,
    )


# For conveniently getting kernel element and derivative indices from force and gp kernels.
@jit_
def energy_forces_ids(
    atom_nums: ndarray_, dint_: dtype_ = dint_, nCartDim: int_ = nCartDim
) -> Tuple[ndarray_, ndarray_]:
    nmols = atom_nums.shape[0]
    energy_force_ranges = get_energy_force_ranges(atom_nums)
    energy_ids = empty_((nmols,), dtype=dint_)
    tot_natoms = int(sum_(atom_nums))
    force_ids = empty_((tot_natoms, 2), dtype=dint_)
    i_atom = 0
    for i_mol in range(nmols):
        en_id = energy_force_ranges[i_mol]
        energy_ids[i_mol] = en_id
        cur_lb = en_id + 1
        for _ in range(int(atom_nums[i_mol])):
            force_ids[i_atom, 0] = cur_lb
            cur_lb += nCartDim
            force_ids[i_atom, 1] = cur_lb
            i_atom += 1
    return energy_ids, force_ids


@jit_
def nen_force_vals(natoms, nCartDim: int_ = nCartDim):
    return natoms.shape[0] + nCartDim * sum_(natoms)


def prediction_vector_to_forces_energies(
    prediction_vector, atom_nums, nmols, energy_output=None, forces_output=None
):
    used_atom_nums = atom_nums[:nmols]
    tot_natoms = sum_(used_atom_nums)
    energy_ids, force_ids = energy_forces_ids(used_atom_nums)
    energy_output = check_allocation((nmols,), output=energy_output)
    forces_output = check_allocation((tot_natoms, nCartDim), output=forces_output)
    for en_count, energy_id in enumerate(energy_ids):
        energy_output[en_count] = prediction_vector[energy_id]
        for atom_count, force_bounds in enumerate(force_ids):
            forces_output[atom_count, :] = prediction_vector[force_bounds[0] : force_bounds[1]]
    return energy_output, forces_output
