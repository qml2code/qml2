# Collection of routines for convenient handling of energies and forces during training of force models.
from typing import Tuple

from ..data import nCartDim
from ..jit_interfaces import (
    array_,
    dint_,
    dtype_,
    empty_,
    int_,
    jit_,
    ndarray_,
    prange_,
    sqrt_,
    sum_,
)
from ..kernels.gradient_kernels import get_energy_force_ranges
from ..utils import check_allocation


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


def combine_energy_forces_rhs(energies, forces_list):
    tot_natoms = sum(f.shape[0] for f in forces_list)
    nvals = len(energies) + tot_natoms * nCartDim
    output = empty_((nvals,))
    en_id = 0
    for en, f in zip(energies, forces_list):
        output[en_id] = en
        force_lb = en_id + 1
        force_ub = force_lb + f.shape[0] * nCartDim
        output[force_lb:force_ub] = f.flatten()
        en_id = force_ub
    return output


def get_importance_multipliers(atom_nums, energy_importance):
    energy_importance_multiplier = sqrt_(array_(energy_importance))
    all_importance_multipliers = empty_((atom_nums.shape[0] + nCartDim * sum_(atom_nums),))
    lb = 0
    for atom_num in atom_nums:
        all_importance_multipliers[lb] = energy_importance_multiplier
        lb += 1
        ub = lb + atom_num * nCartDim
        all_importance_multipliers[lb:ub] = 1 / sqrt_(array_(float(atom_num)))
        lb = ub

    return all_importance_multipliers


# For merging arrays associated with gradient calculations.
@jit_
def copy_grads_to_merged(
    destination_rep_grads,
    destination_rel_neighbors,
    copied_rep_grads,
    copied_rel_neighbors,
    rel_neighbor_nums,
):
    for i in prange_(rel_neighbor_nums.shape[0]):
        nneighbor_num = rel_neighbor_nums[i]
        destination_rep_grads[i, :, :nneighbor_num, :] = copied_rep_grads[i, :, :nneighbor_num, :]
        destination_rel_neighbors[i, :nneighbor_num] = copied_rel_neighbors[i, :nneighbor_num]


def merge_grad_lists(all_rep_grads_list, all_rel_atoms_list, all_rel_atom_nums_list):
    # First determine dimensionalities of the merged arrays.
    max_num_rel_neighbors = 0
    nconfigs = sum([len(grad) for grad in all_rep_grads_list])
    for i, rel_atom_nums in enumerate(all_rel_atom_nums_list):
        max_num_rel_neighbors = max(max_num_rel_neighbors, max(rel_atom_nums))

    rep_size = all_rep_grads_list[0].shape[1]
    all_rep_grads = empty_((nconfigs, rep_size, max_num_rel_neighbors, nCartDim))
    all_rel_atoms = empty_((nconfigs, max_num_rel_neighbors), dtype=dint_)
    all_rel_atom_nums = empty_((nconfigs,), dtype=dint_)

    cur_lb = 0
    for rep_grads, rel_atoms, rel_atom_nums in zip(
        all_rep_grads_list, all_rel_atoms_list, all_rel_atom_nums_list
    ):
        cur_ub = cur_lb + len(rep_grads)
        all_rel_atom_nums[cur_lb:cur_ub] = rel_atom_nums[:]
        copy_grads_to_merged(
            all_rep_grads[cur_lb:cur_ub],
            all_rel_atoms[cur_lb:cur_ub],
            rep_grads,
            rel_atoms,
            rel_atom_nums,
        )
        cur_lb = cur_ub
    return all_rep_grads, all_rel_atoms, all_rel_atom_nums
