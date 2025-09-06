"""
Routines for convenient post-processing of some custom data formats.
"""
from ...jit_interfaces import dint_, dot_, empty_, jit_, prange_, sqrt_
from ...utils import get_numba_list, l2_sq_norm
from ..base_constructors import get_datatype


@jit_
def extract_all_conformer_representations(ensemble_global_rep_list):
    num_conformers = 0
    rep_size = ensemble_global_rep_list[0][0].components[0].shape[0]
    for ensemble_global_rep_id in prange_(len(ensemble_global_rep_list)):
        cur_num_conformers = 0
        for generation in ensemble_global_rep_list[ensemble_global_rep_id]:
            cur_num_conformers += generation.components.shape[0]
        num_conformers += cur_num_conformers
    all_representations = empty_((num_conformers, rep_size))
    conformer_lb = 0
    for ensemble_global_rep in ensemble_global_rep_list:
        for generation in ensemble_global_rep:
            components = generation.components
            conformer_ub = conformer_lb + components.shape[0]
            all_representations[conformer_lb:conformer_ub] = components[:, :]
            conformer_lb = conformer_ub
    return all_representations


@jit_
def count_conformer_nums(ensemble_global_rep_list):
    num_ensembles = len(ensemble_global_rep_list)
    num_generations = len(ensemble_global_rep_list[0])
    conformer_nums = empty_((num_ensembles, num_generations), dtype=dint_)
    for ensemble_id in prange_(num_ensembles):
        for generation_id in range(num_generations):
            conformer_nums[ensemble_id, generation_id] = ensemble_global_rep_list[ensemble_id][
                generation_id
            ].components.shape[0]
    return conformer_nums


@jit_
def transform_all_conformer_representations_into_principal_component_versions(
    ensemble_global_rep_list,
    new_ensemble_global_rep_list,
    principal_component_matrix,
    add_remainder=False,
):
    npcas = principal_component_matrix.shape[0]
    for ensemble_id in prange_(len(ensemble_global_rep_list)):
        ensemble_global_rep = ensemble_global_rep_list[ensemble_id]
        num_generations = len(ensemble_global_rep)
        for generation_id in range(num_generations):
            conformers = ensemble_global_rep[generation_id]
            new_conformers = new_ensemble_global_rep_list[ensemble_id][generation_id]
            new_conformers.rhos[:] = conformers.rhos[:]
            new_components = new_conformers.components
            components = conformers.components
            new_components[:, :npcas] = dot_(components[:, :], principal_component_matrix.T)
            if add_remainder:
                for comp_id in range(new_components.shape[0]):
                    sq_remainder = l2_sq_norm(components[comp_id]) - l2_sq_norm(
                        new_components[comp_id, :npcas]
                    )
                    if sq_remainder < 0.0:
                        remainder = 0.0
                    else:
                        remainder = sqrt_(sq_remainder)
                    new_components[comp_id, -1] = remainder


def generate_blank_ensembles(conformer_nums, rep_size):
    conformers_datatype = get_datatype(["array_2D", "rhos"])
    blank_ensemble_list = get_numba_list()
    for generation_conformer_nums in conformer_nums:
        generation_list = get_numba_list()
        for conformer_num in generation_conformer_nums:
            conformer_reps = empty_((conformer_num, rep_size))
            rhos = empty_(conformer_num)
            generation_conformers = conformers_datatype(conformer_reps, rhos)
            generation_list.append(generation_conformers)
        blank_ensemble_list.append(generation_list)
    return blank_ensemble_list


def transform_all_conformer_representations_to_principal_components(
    ensemble_global_rep_list, principal_component_matrix, add_remainder=False
):
    conformer_nums = count_conformer_nums(ensemble_global_rep_list)
    rep_size = principal_component_matrix.shape[0]
    if add_remainder:
        rep_size += 1
    # allocate the space where it will be stored
    transformed_representations = generate_blank_ensembles(conformer_nums, rep_size)
    # transform from one to the other
    transform_all_conformer_representations_into_principal_component_versions(
        ensemble_global_rep_list,
        transformed_representations,
        principal_component_matrix,
        add_remainder=add_remainder,
    )
    return transformed_representations
