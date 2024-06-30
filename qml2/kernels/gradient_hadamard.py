# K.Karan.: I think the way memory is accessed during Z matrix calculation is not optimal,
# but it results in a grouping convenient enough for applying dot.
# The routines for calculating products of Z with alpha vector should be close to optimal though,
# which is the important part of doing model predictions.

from typing import Tuple

from ..data import nCartDim
from ..dimensionality_reduction import project_scale_local_representations
from ..jit_interfaces import (
    cos_,
    dim0float_array_,
    dot_,
    empty_,
    int_,
    jit_,
    max_,
    ndarray_,
    prange_,
    save_,
    sin_,
)
from ..utils import get_atom_environment_ranges, get_element_ids_from_sorted
from .gradient_kernels import add_force_kernel_contributions, get_energy_force_ranges
from .hadamard import (
    fast_walsh_hadamard,
    hadamard_norm_const,
    hadamard_rff_kernel_stack_phases,
    rff_vec_norm_const,
)


@jit_
def hadamard_rff_add_kernel_stack_wderivatives(
    red_scaled_reps,
    temp_red_reps,
    sorf_diags,
    biases,
    norm_const: dim0float_array_,
    rff_der_output,
) -> ndarray_:
    hadamard_rff_kernel_stack_phases(
        red_scaled_reps, temp_red_reps, sorf_diags, biases, norm_const
    )
    rff_der_output[:] = -sin_(save_(temp_red_reps))
    return cos_(save_(temp_red_reps))


@jit_
def hadamard_rff_add_kernel_func_forces(
    rep_grads,
    rel_neighbors,
    rel_neighbor_num: int_,
    sorf_diags,
    reductor,
    norm_const,
    temp_rff_ders,
    kernel_output,
):
    # First transform RFF derivatives into derivatives w.r.t. representation by doing operations in hadamard_rff_kernel_stack_phases
    # in reverse.
    ntransforms = sorf_diags.shape[0]
    for transform in range(ntransforms - 1, -1, -1):
        fast_walsh_hadamard(temp_rff_ders, norm_const)
        temp_rff_ders *= sorf_diags[transform]

    temp_rff_ders /= norm_const
    # Do backward of reduction.
    rff_rep_der = dot_(reductor, temp_rff_ders)
    # NOTE: rff_rep_der still needs to be rescaled by sigma; it is done globally.
    # add gradient components corresponding to rff_rep_der.
    add_force_kernel_contributions(
        rep_grads, rel_neighbors, rel_neighbor_num, kernel_output, rff_rep_der
    )


@jit_
def hadamard_rff_add_force_kernel_func(
    red_scaled_rep,
    rep_grads,
    rel_neighbors,
    rel_neighbor_num: int_,
    temp_red_reps,
    temp_rff_der,
    sorf_diags,
    biases,
    reductor,
    norm_const,
    kernel_products,
    kernel_product_derivatives,
    natoms: int_,
    nCartDim: int_ = nCartDim,
) -> None:
    # Add the kernel element.
    kernel_products[:] += hadamard_rff_add_kernel_stack_wderivatives(
        red_scaled_rep, temp_red_reps, sorf_diags, biases, norm_const, temp_rff_der
    )
    # For each RFF calculate and add corresponding component.
    nfeatures = sorf_diags.shape[-1]

    der_step = nCartDim * natoms
    der_lb = 0
    for feature in range(nfeatures):
        der_ub = der_lb + der_step
        temp_red_reps[:] = 0.0
        temp_red_reps[feature] = temp_rff_der[feature]
        hadamard_rff_add_kernel_func_forces(
            rep_grads,
            rel_neighbors,
            rel_neighbor_num,
            sorf_diags,
            reductor,
            norm_const,
            temp_red_reps,
            kernel_product_derivatives[der_lb:der_ub],
        )
        der_lb = der_ub


@jit_
def hadamard_rff_add_product_force_kernel_func(
    red_scaled_rep,
    rep_grads,
    rel_neighbors,
    rel_neighbor_num: int_,
    temp_red_reps,
    temp_rff_der,
    sorf_diags,
    biases,
    reductor,
    norm_const: dim0float_array_,
    kernel_component,
    alphas,
):
    kernel_component[0] += dot_(
        alphas,
        hadamard_rff_add_kernel_stack_wderivatives(
            red_scaled_rep, temp_red_reps, sorf_diags, biases, norm_const, temp_rff_der
        ),
    )
    temp_rff_der *= alphas
    hadamard_rff_add_kernel_func_forces(
        rep_grads,
        rel_neighbors,
        rel_neighbor_num,
        sorf_diags,
        reductor,
        norm_const,
        temp_rff_der,
        kernel_component[1:],
    )


@jit_
def local_hadamard_force_kernel_func(
    red_scaled_reps,
    rep_grads,
    rel_neighbors,
    rel_neighbor_nums,
    temp_red_reps,
    temp_rff_der,
    element_ids,
    sorf_diags,
    biases,
    reductors,
    norm_const,
    natoms: int_,
    kernel_products,
    kernel_product_derivatives,
):
    kernel_products[:] = 0.0
    kernel_product_derivatives[:] = 0.0
    for i_atom in range(natoms):
        el_id = int(element_ids[i_atom])
        hadamard_rff_add_force_kernel_func(
            red_scaled_reps[i_atom],
            rep_grads[i_atom],
            rel_neighbors[i_atom],
            int(rel_neighbor_nums[i_atom]),
            temp_red_reps,
            temp_rff_der,
            sorf_diags[el_id],
            biases[el_id],
            reductors[el_id],
            norm_const,
            kernel_products,
            kernel_product_derivatives,
            natoms,
        )


@jit_
def local_hadamard_product_force_kernel_func(
    red_scaled_reps,
    rep_grads,
    rel_neighbors,
    rel_neighbor_nums,
    temp_red_reps,
    temp_rff_der,
    element_ids,
    sorf_diags,
    biases,
    reductors,
    norm_const,
    natoms: int_,
    kernel_component,
    alphas,
):
    kernel_component[:] = 0.0
    for i_atom in range(natoms):
        el_id = int(element_ids[i_atom])
        hadamard_rff_add_product_force_kernel_func(
            red_scaled_reps[i_atom],
            rep_grads[i_atom],
            rel_neighbors[i_atom],
            int(rel_neighbor_nums[i_atom]),
            temp_red_reps,
            temp_rff_der,
            sorf_diags[el_id],
            biases[el_id],
            reductors[el_id],
            norm_const,
            kernel_component,
            alphas,
        )


@jit_
def rff_lower_upper_bounds(feature_stack: int_, npcas: int_) -> Tuple[int_, int_]:
    lb_rff = npcas * feature_stack
    ub_rff = lb_rff + npcas
    return lb_rff, ub_rff


# TODO: A more Python-esque way to rewrite avoiding copy-pasted between w. alphas - no alphas?
@jit_(numba_parallel=True)
def local_hadamard_force_kernel_processed_input(
    reduced_scaled_representations,
    representation_gradients,
    relevant_neighbors,
    relevant_neighbor_nums,
    element_ids,
    all_sorf_diags,
    all_biases,
    all_reductors,
    sigma,
    kernel,
    en_force_ranges_arr,
    mol_ubound_arr,
    nfeature_stacks: int_,
    npcas: int_,
    true_nfeatures: int_ = None,
    nCartDim: int_ = nCartDim,
):
    assert reduced_scaled_representations.shape[0] == element_ids.shape[0]
    assert all_sorf_diags.shape[0] == all_biases.shape[0]
    # KK: not %100 sure it's necessary
    assert max_(element_ids) <= all_sorf_diags.shape[0]

    assert npcas == reduced_scaled_representations.shape[1]

    norm_const = hadamard_norm_const(npcas)

    nmols = mol_ubound_arr.shape[0] - 1
    nfeatures = kernel.shape[1]
    assert nfeature_stacks * npcas == nfeatures

    for feature_stack in prange_(nfeature_stacks):
        temp_red_reps = empty_((npcas,))
        temp_rff_der = empty_((npcas,))
        temp_kernel_products = empty_((nmols, npcas))
        temp_kernel_product_derivatives = empty_((npcas * mol_ubound_arr[-1] * nCartDim))
        lb_rff, ub_rff = rff_lower_upper_bounds(feature_stack, npcas)

        product_derivative_lb = 0
        for mol_id in range(nmols):
            lb_mol_rep = mol_ubound_arr[mol_id]
            ub_mol_rep = mol_ubound_arr[mol_id + 1]
            natoms = ub_mol_rep - lb_mol_rep

            product_derivative_ub = product_derivative_lb + npcas * natoms * nCartDim
            local_hadamard_force_kernel_func(
                reduced_scaled_representations[lb_mol_rep:ub_mol_rep],
                representation_gradients[lb_mol_rep:ub_mol_rep],
                relevant_neighbors[lb_mol_rep:ub_mol_rep],
                relevant_neighbor_nums[lb_mol_rep:ub_mol_rep],
                temp_red_reps,
                temp_rff_der,
                element_ids[lb_mol_rep:ub_mol_rep],
                all_sorf_diags[feature_stack],
                all_biases[feature_stack],
                all_reductors,
                norm_const,
                natoms,
                temp_kernel_products[mol_id],
                temp_kernel_product_derivatives[product_derivative_lb:product_derivative_ub],
            )
            # Move address for copying forces
            product_derivative_lb = product_derivative_ub
        # Account for sigma rescaling.
        temp_kernel_product_derivatives /= sigma
        # Copy energies to kernel
        product_derivative_lb = 0
        for mol_id in range(nmols):
            kernel[en_force_ranges_arr[mol_id], lb_rff:ub_rff] = temp_kernel_products[mol_id, :]
            # Copy forces to kernel.
            prod_der_bound_step = nCartDim * (mol_ubound_arr[mol_id + 1] - mol_ubound_arr[mol_id])
            for rff_id in range(npcas):
                product_derivative_ub = product_derivative_lb + prod_der_bound_step
                kernel[
                    en_force_ranges_arr[mol_id] + 1 : en_force_ranges_arr[mol_id + 1],
                    rff_id + lb_rff,
                ] = temp_kernel_product_derivatives[product_derivative_lb:product_derivative_ub]
                product_derivative_lb = product_derivative_ub

    if true_nfeatures is None:
        true_nfeatures = nfeatures

    kernel[:, :] *= rff_vec_norm_const(true_nfeatures)  # normalization


@jit_
def local_hadamard_product_force_kernel_full_stack(
    reduced_scaled_representations,
    representation_gradients,
    relevant_neighbors,
    relevant_neighbor_nums,
    element_ids,
    stack_sorf_diags,
    stack_biases,
    all_reductors,
    norm_const,
    sigma,
    stack_alphas,
    mol_ubound_arr,
    en_force_ranges_arr,
):
    nred_reps = reduced_scaled_representations.shape[1]
    output = empty_((int(en_force_ranges_arr[-1]),))
    temp_red_reps = empty_((nred_reps,))
    temp_rff_der = empty_((nred_reps,))
    for mol_id in range(int(mol_ubound_arr.shape[0]) - 1):
        lb_mol = mol_ubound_arr[mol_id]
        ub_mol = mol_ubound_arr[mol_id + 1]
        lb_mol_en_forces = en_force_ranges_arr[mol_id]
        ub_mol_en_forces = en_force_ranges_arr[mol_id + 1]
        natoms = ub_mol - lb_mol
        local_hadamard_product_force_kernel_func(
            reduced_scaled_representations[lb_mol:ub_mol],
            representation_gradients[lb_mol:ub_mol],
            relevant_neighbors[lb_mol:ub_mol],
            relevant_neighbor_nums[lb_mol:ub_mol],
            temp_red_reps,
            temp_rff_der,
            element_ids[lb_mol:ub_mol],
            stack_sorf_diags,
            stack_biases,
            all_reductors,
            norm_const,
            natoms,
            output[lb_mol_en_forces:ub_mol_en_forces],
            stack_alphas,
        )
        # since sigma was not accounted for
        output[lb_mol_en_forces + 1 : ub_mol_en_forces] /= sigma
    return output


@jit_(numba_parallel=True)
def local_hadamard_product_force_kernel_processed_input(
    reduced_scaled_representations,
    representation_gradients,
    relevant_neighbors,
    relevant_neighbor_nums,
    element_ids,
    all_sorf_diags,
    all_biases,
    all_reductors,
    sigma,
    output_vector,
    alphas,
    en_force_ranges_arr,
    mol_ubound_arr,
    nfeature_stacks: int_,
    npcas: int_,
):
    assert reduced_scaled_representations.shape[0] == element_ids.shape[0]
    assert all_sorf_diags.shape[0] == all_biases.shape[0]
    # KK: not %100 sure it's necessary
    assert max_(element_ids) <= all_sorf_diags.shape[0]

    norm_const = hadamard_norm_const(npcas)

    output_vector[:] = 0.0

    for feature_stack in prange_(nfeature_stacks):
        rff_lb = feature_stack * npcas
        rff_ub = rff_lb + npcas
        temp_output_vector = local_hadamard_product_force_kernel_full_stack(
            reduced_scaled_representations,
            representation_gradients,
            relevant_neighbors,
            relevant_neighbor_nums,
            element_ids,
            all_sorf_diags[feature_stack],
            all_biases[feature_stack],
            all_reductors,
            norm_const,
            sigma,
            alphas[rff_lb:rff_ub],
            mol_ubound_arr,
            en_force_ranges_arr,
        )
        # reduce over parallel execution
        output_vector += temp_output_vector
    output_vector *= rff_vec_norm_const(npcas * nfeature_stacks)  # normalization


@jit_
def local_hadamard_force_kernel(
    representations,
    representation_gradients,
    ncharges,
    na,
    relevant_neighbors,
    relevant_neighbor_nums,
    reductors,
    sorted_elements,
    all_sorf_diags,
    all_biases,
    kernel,
    sigma: dim0float_array_,
    nfeature_stacks: int,
    npcas: int,
):
    ubound_arr = get_atom_environment_ranges(na)
    en_force_ranges = get_energy_force_ranges(na)
    all_element_ids = get_element_ids_from_sorted(ncharges, sorted_elements)
    reduced_scaled_representations = project_scale_local_representations(
        representations, all_element_ids, reductors, sigma
    )
    local_hadamard_force_kernel_processed_input(
        reduced_scaled_representations,
        representation_gradients,
        relevant_neighbors,
        relevant_neighbor_nums,
        all_element_ids,
        all_sorf_diags,
        all_biases,
        reductors,
        sigma,
        kernel,
        en_force_ranges,
        ubound_arr,
        nfeature_stacks,
        npcas,
    )


@jit_(numba_parallel=True)
def local_hadamard_product_force_kernel(
    representations,
    representation_gradients,
    ncharges,
    na,
    relevant_neighbors,
    relevant_neighbor_nums,
    reductors,
    sorted_elements,
    all_sorf_diags,
    all_biases,
    output_vector,
    alphas,
    sigma: dim0float_array_,
    nfeature_stacks: int,
    npcas: int,
):
    ubound_arr = get_atom_environment_ranges(na)
    en_force_ranges = get_energy_force_ranges(na)
    all_element_ids = get_element_ids_from_sorted(ncharges, sorted_elements)
    reduced_scaled_representations = project_scale_local_representations(
        representations, all_element_ids, reductors, sigma
    )
    local_hadamard_product_force_kernel_processed_input(
        reduced_scaled_representations,
        representation_gradients,
        relevant_neighbors,
        relevant_neighbor_nums,
        all_element_ids,
        all_sorf_diags,
        all_biases,
        reductors,
        sigma,
        output_vector,
        alphas,
        en_force_ranges,
        ubound_arr,
        nfeature_stacks,
        npcas,
    )
