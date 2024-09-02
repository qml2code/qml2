# K.Karan.: I think the way memory is accessed during Z matrix calculation is not optimal,
# but it results in a grouping convenient enough for applying dot.
# The routines for calculating products of Z with alpha vector should be close to optimal though,
# which is the important part of doing model predictions.

from typing import Tuple, Union

from ..data import nCartDim
from ..dimensionality_reduction import choose_reductor, project_scale_local_representations
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
from ..utils import check_allocation, get_atom_environment_ranges, get_element_ids_from_sorted
from .gradient_kernels import add_force_kernel_contributions, get_energy_force_ranges
from .sorf import (
    fast_walsh_hadamard,
    generate_sorf_stack_phases,
    hadamard_norm_const,
    rff_vec_norm_const,
)


@jit_
def sorf_stack_wderivatives(
    red_scaled_reps,
    temp_red_reps,
    sorf_diags,
    biases,
    norm_const: dim0float_array_,
    rff_der_output,
) -> ndarray_:
    generate_sorf_stack_phases(red_scaled_reps, temp_red_reps, sorf_diags, biases, norm_const)
    rff_der_output[:] = -sin_(save_(temp_red_reps))
    return cos_(save_(temp_red_reps))


@jit_
def add_sorf_func_forces(
    rep_grads,
    rel_neighbors,
    rel_neighbor_num: int_,
    sorf_diags,
    reductor: Union[None, ndarray_],
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
    if reductor is None:
        rff_rep_der = temp_rff_ders
    else:
        rff_rep_der = dot_(reductor, temp_rff_ders)
    # NOTE: rff_rep_der still needs to be rescaled by sigma; it is done globally.
    # add gradient components corresponding to rff_rep_der.
    add_force_kernel_contributions(
        rep_grads, rel_neighbors, rel_neighbor_num, kernel_output, rff_rep_der
    )


@jit_
def add_force_sorf_func(
    red_scaled_rep,
    rep_grads,
    rel_neighbors,
    rel_neighbor_num: int_,
    temp_red_reps,
    temp_rff_der,
    sorf_diags,
    biases,
    reductor: Union[None, ndarray_],
    norm_const,
    kernel_products,
    kernel_product_derivatives,
    natoms: int_,
    nCartDim: int_ = nCartDim,
) -> None:
    # Add the kernel element.
    kernel_products[:] += sorf_stack_wderivatives(
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
        add_sorf_func_forces(
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
def add_product_force_sorf_func(
    red_scaled_rep,
    rep_grads,
    rel_neighbors,
    rel_neighbor_num: int_,
    temp_red_reps,
    temp_rff_der,
    sorf_diags,
    biases,
    reductor: Union[None, ndarray_],
    norm_const: dim0float_array_,
    kernel_component,
    alphas,
):
    kernel_component[0] += dot_(
        alphas,
        sorf_stack_wderivatives(
            red_scaled_rep, temp_red_reps, sorf_diags, biases, norm_const, temp_rff_der
        ),
    )
    temp_rff_der *= alphas
    add_sorf_func_forces(
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
def generate_local_force_sorf_func(
    red_scaled_reps,
    rep_grads,
    rel_neighbors,
    rel_neighbor_nums,
    temp_red_reps,
    temp_rff_der,
    element_ids,
    sorf_diags,
    biases,
    reductors: Union[None, ndarray_],
    norm_const,
    natoms: int_,
    kernel_products,
    kernel_product_derivatives,
):
    kernel_products[:] = 0.0
    kernel_product_derivatives[:] = 0.0
    for i_atom in range(natoms):
        el_id = int(element_ids[i_atom])
        add_force_sorf_func(
            red_scaled_reps[i_atom],
            rep_grads[i_atom],
            rel_neighbors[i_atom],
            int(rel_neighbor_nums[i_atom]),
            temp_red_reps,
            temp_rff_der,
            sorf_diags[el_id],
            biases[el_id],
            choose_reductor(reductors, el_id),
            norm_const,
            kernel_products,
            kernel_product_derivatives,
            natoms,
        )


@jit_
def generate_local_force_sorf_product_func(
    red_scaled_reps,
    rep_grads,
    rel_neighbors,
    rel_neighbor_nums,
    temp_red_reps,
    temp_rff_der,
    element_ids,
    sorf_diags,
    biases,
    reductors: Union[None, ndarray_],
    norm_const,
    natoms: int_,
    kernel_component,
    alphas,
):
    kernel_component[:] = 0.0
    for i_atom in range(natoms):
        el_id = int(element_ids[i_atom])
        add_product_force_sorf_func(
            red_scaled_reps[i_atom],
            rep_grads[i_atom],
            rel_neighbors[i_atom],
            int(rel_neighbor_nums[i_atom]),
            temp_red_reps,
            temp_rff_der,
            sorf_diags[el_id],
            biases[el_id],
            choose_reductor(reductors, el_id),
            norm_const,
            kernel_component,
            alphas,
        )


@jit_
def rff_lower_upper_bounds(feature_stack: int_, init_size: int_) -> Tuple[int_, int_]:
    lb_rff = init_size * feature_stack
    ub_rff = lb_rff + init_size
    return lb_rff, ub_rff


# TODO: A more Python-esque way to rewrite avoiding copy-pasted between w. alphas - no alphas?
@jit_(numba_parallel=True)
def generate_local_force_sorf_processed_input(
    reduced_scaled_representations,
    representation_gradients,
    relevant_neighbors,
    relevant_neighbor_nums,
    element_ids,
    all_sorf_diags,
    all_biases,
    all_reductors: Union[None, ndarray_],
    sigma,
    kernel,
    en_force_ranges_arr,
    mol_ubound_arr,
    nfeature_stacks: int_,
    init_size: int_,
    true_nfeatures: Union[int_, None] = None,
    nCartDim: int_ = nCartDim,
):
    assert reduced_scaled_representations.shape[0] == element_ids.shape[0]
    assert all_sorf_diags.shape[0] == all_biases.shape[0]
    # KK: not %100 sure it's necessary
    assert max_(element_ids) <= all_sorf_diags.shape[0]

    assert reduced_scaled_representations.shape[1] <= init_size

    norm_const = hadamard_norm_const(init_size)

    nmols = mol_ubound_arr.shape[0] - 1
    nfeatures = kernel.shape[1]
    assert nfeature_stacks * init_size == nfeatures

    for feature_stack in prange_(nfeature_stacks):
        temp_red_reps = empty_((init_size,))
        temp_rff_der = empty_((init_size,))
        temp_kernel_products = empty_((nmols, init_size))
        temp_kernel_product_derivatives = empty_((init_size * mol_ubound_arr[-1] * nCartDim))
        lb_rff, ub_rff = rff_lower_upper_bounds(feature_stack, init_size)

        product_derivative_lb = 0
        for mol_id in range(nmols):
            lb_mol_rep = mol_ubound_arr[mol_id]
            ub_mol_rep = mol_ubound_arr[mol_id + 1]
            natoms = ub_mol_rep - lb_mol_rep

            product_derivative_ub = product_derivative_lb + init_size * natoms * nCartDim
            generate_local_force_sorf_func(
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
            for rff_id in range(init_size):
                product_derivative_ub = product_derivative_lb + prod_der_bound_step
                kernel[
                    en_force_ranges_arr[mol_id] + 1 : en_force_ranges_arr[mol_id + 1],
                    rff_id + lb_rff,
                ] = temp_kernel_product_derivatives[product_derivative_lb:product_derivative_ub]
                product_derivative_lb = product_derivative_ub

    if true_nfeatures is None:
        rff_nc = rff_vec_norm_const(nfeatures)
    else:
        rff_nc = rff_vec_norm_const(true_nfeatures)

    kernel[:, :] *= rff_nc  # normalization


@jit_
def generate_local_force_sorf_product_full_stack(
    reduced_scaled_representations,
    representation_gradients,
    relevant_neighbors,
    relevant_neighbor_nums,
    element_ids,
    stack_sorf_diags,
    stack_biases,
    all_reductors: Union[None, ndarray_],
    norm_const,
    sigma,
    stack_alphas,
    mol_ubound_arr,
    en_force_ranges_arr,
    init_size: int_,
):
    output = empty_((int(en_force_ranges_arr[-1]),))
    temp_red_reps = empty_((init_size,))
    temp_rff_der = empty_((init_size,))
    for mol_id in range(int(mol_ubound_arr.shape[0]) - 1):
        lb_mol = mol_ubound_arr[mol_id]
        ub_mol = mol_ubound_arr[mol_id + 1]
        lb_mol_en_forces = en_force_ranges_arr[mol_id]
        ub_mol_en_forces = en_force_ranges_arr[mol_id + 1]
        natoms = ub_mol - lb_mol
        generate_local_force_sorf_product_func(
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
def generate_local_force_sorf_product_processed_input(
    reduced_scaled_representations,
    representation_gradients,
    relevant_neighbors,
    relevant_neighbor_nums,
    element_ids,
    all_sorf_diags,
    all_biases,
    all_reductors: Union[None, ndarray_],
    sigma,
    output_vector,
    alphas,
    en_force_ranges_arr,
    mol_ubound_arr,
    nfeature_stacks: int_,
    init_size: int_,
):
    assert reduced_scaled_representations.shape[0] == element_ids.shape[0]
    assert all_sorf_diags.shape[0] == all_biases.shape[0]
    # KK: not %100 sure it's necessary
    assert max_(element_ids) <= all_sorf_diags.shape[0]

    norm_const = hadamard_norm_const(init_size)

    output_vector[:] = 0.0

    for feature_stack in prange_(nfeature_stacks):
        rff_lb = feature_stack * init_size
        rff_ub = rff_lb + init_size
        temp_output_vector = generate_local_force_sorf_product_full_stack(
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
            init_size,
        )
        # reduce over parallel execution
        output_vector += temp_output_vector
    output_vector *= rff_vec_norm_const(init_size * nfeature_stacks)  # normalization


@jit_
def generate_local_force_sorf(
    representations,
    representation_gradients,
    ncharges,
    na,
    relevant_neighbors,
    relevant_neighbor_nums,
    sorted_elements,
    all_sorf_diags,
    all_biases,
    sigma: dim0float_array_,
    nfeature_stacks: int,
    init_size: int,
    reductors: Union[None, ndarray_] = None,
    out: Union[None, ndarray_] = None,
):
    ubound_arr = get_atom_environment_ranges(na)
    en_force_ranges = get_energy_force_ranges(na)
    all_element_ids = get_element_ids_from_sorted(ncharges, sorted_elements)
    if reductors is None:
        assert representations.shape[1] <= init_size
    reduced_scaled_representations = project_scale_local_representations(
        representations, all_element_ids, reductors, sigma
    )
    out = check_allocation((en_force_ranges[-1], init_size * nfeature_stacks), output=out)
    generate_local_force_sorf_processed_input(
        reduced_scaled_representations,
        representation_gradients,
        relevant_neighbors,
        relevant_neighbor_nums,
        all_element_ids,
        all_sorf_diags,
        all_biases,
        reductors,
        sigma,
        out,
        en_force_ranges,
        ubound_arr,
        nfeature_stacks,
        init_size,
    )
    return out


@jit_
def generate_local_force_sorf_product(
    representations,
    representation_gradients,
    ncharges,
    natoms,
    relevant_neighbors,
    relevant_neighbor_nums,
    sorted_elements,
    all_sorf_diags,
    all_biases,
    alphas,
    sigma: dim0float_array_,
    nfeature_stacks: int,
    init_size: int,
    reductors: Union[None, ndarray_] = None,
    out: Union[None, ndarray_] = None,
):
    ubound_arr = get_atom_environment_ranges(natoms)
    en_force_ranges = get_energy_force_ranges(natoms)
    all_element_ids = get_element_ids_from_sorted(ncharges, sorted_elements)
    if reductors is None:
        assert representations.shape[1] <= init_size
    reduced_scaled_representations = project_scale_local_representations(
        representations, all_element_ids, reductors, sigma
    )
    out = check_allocation((en_force_ranges[-1],), output=out)
    generate_local_force_sorf_product_processed_input(
        reduced_scaled_representations,
        representation_gradients,
        relevant_neighbors,
        relevant_neighbor_nums,
        all_element_ids,
        all_sorf_diags,
        all_biases,
        reductors,
        sigma,
        out,
        alphas,
        en_force_ranges,
        ubound_arr,
        nfeature_stacks,
        init_size,
    )
    return out
