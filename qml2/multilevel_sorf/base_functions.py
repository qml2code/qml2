from ..jit_interfaces import OptionalGenerator_, cos_, dot_, int_, jit_, l2_norm_, sin_
from ..kernels.sorf import (
    create_sorf_matrices,
    generate_sorf_stack_phases,
    generate_sorf_stack_unbiased_phases,
)


# Standard SORF transform.
@jit_
def generate_sorf_unbiased_phases_serial(
    input_array, output_array, sorf_diags, norm_const, nfeature_stacks, init_size
):
    lb = 0
    for istack in range(nfeature_stacks):
        ub = lb + init_size
        generate_sorf_stack_unbiased_phases(
            input_array, output_array[lb:ub], sorf_diags[istack], norm_const
        )
        lb = ub


@jit_
def generate_sorf_stack_phases_serial(
    input_array, phases, sorf_diags, biases, norm_const, nfeature_stacks, init_size
):
    lb = 0
    for istack in range(nfeature_stacks):
        ub = lb + init_size
        generate_sorf_stack_phases(
            input_array, phases[lb:ub], sorf_diags[istack], biases[istack], norm_const
        )
        lb = ub


@jit_
def generate_sorf_serial(
    input_array,
    output_array,
    sorf_diags,
    biases,
    norm_const,
    rff_vec_norm_const,
    nfeature_stacks,
    init_size,
):
    """
    Serial version of qml2.kernels.hadamard.hadamard_kernel_processed_input
    """
    generate_sorf_stack_phases_serial(
        input_array, output_array, sorf_diags, biases, norm_const, nfeature_stacks, init_size
    )
    output_array[:] = cos_(output_array)[:] * rff_vec_norm_const


# For generating SORF with derivatives.
@jit_
def generate_sorf_wgrad(
    init_vector,
    init_gradient,
    final_result,
    final_gradient,
    sigma,
    phases_arr,
    sorf_diags,
    biases,
    norm_const,
    rff_vec_norm_const,
    nfeature_stacks,
    init_size,
):
    init_vector[:] /= sigma
    init_gradient[:, :] /= sigma
    generate_sorf_unbiased_phases_serial(
        init_vector,
        final_gradient[-1, :],
        sorf_diags,
        norm_const,
        nfeature_stacks,
        init_size,
    )
    lb = 0
    for feature_stack in range(nfeature_stacks):
        ub = lb + init_size
        phases_arr[lb:ub] = final_gradient[-1, lb:ub] + biases[feature_stack]
        lb = ub

    final_result[:] = cos_(phases_arr) * rff_vec_norm_const

    final_gradient[-1, :] /= -sigma

    nhyperparameters = final_gradient.shape[0]

    for hyperparam_id in range(nhyperparameters - 1):
        generate_sorf_unbiased_phases_serial(
            init_gradient[hyperparam_id],
            final_gradient[hyperparam_id],
            sorf_diags,
            norm_const,
            nfeature_stacks,
            init_size,
        )

    final_gradient[:, :] *= -sin_(phases_arr) * rff_vec_norm_const


# Sign-invariant SORF transform.
@jit_
def create_sign_invariant_sorf_matrices(
    nfeature_stacks: int_,
    ntransforms: int_,
    init_size: int_,
    rng: OptionalGenerator_ = None,
):
    biases, sorf_diags = create_sorf_matrices(nfeature_stacks, ntransforms, init_size, rng=rng)
    return cos_(biases), sorf_diags


@jit_
def multiply_by_stacks(arr, multiplier, nfeature_stacks, init_size):
    lb = 0
    for istack in range(nfeature_stacks):
        ub = lb + init_size
        arr[lb:ub] *= multiplier[istack]
        lb = ub


@jit_
def generate_sign_invariant_sorf_serial(
    input_array,
    output_array,
    sorf_diags,
    bias_cosines,
    norm_const,
    rff_vec_norm_const,
    nfeature_stacks,
    init_size,
):
    generate_sorf_unbiased_phases_serial(
        input_array, output_array, sorf_diags, norm_const, nfeature_stacks, init_size
    )
    output_array[:] = cos_(output_array) * rff_vec_norm_const
    multiply_by_stacks(output_array, bias_cosines, nfeature_stacks, init_size)


# For compression.
@jit_
def compress(inout_array, hyperparameters, compression_ratio):
    ub = inout_array.shape[0]
    output_id = ub - 1
    ub_hyp = hyperparameters.shape[0]
    while True:
        lb = ub - compression_ratio
        lb_hyp = ub_hyp - compression_ratio
        inout_array[output_id] = dot_(inout_array[lb:ub], hyperparameters[lb_hyp:ub_hyp])
        if lb == 0:
            return
        ub = lb
        if lb_hyp == 0:
            ub_hyp = hyperparameters.shape[0]
        else:
            ub_hyp = lb_hyp
        output_id -= 1


# For easy manipulation of 1D arrays that are actually 2D (appears for grad work arrays).
@jit_
def extract_row_from_1D(arr_1D, row_id, row_length, nrows):
    lb = arr_1D.shape[0] - (nrows - row_id) * row_length
    return arr_1D[lb : lb + row_length]


# Appearing in normalization and mixed_extensive_sorf layers.
@jit_
def inplace_normalization_wgrad_from_norm(vector, gradient, nhyperparameters, norm):
    vector /= norm
    gradient /= norm
    for ihyp in range(nhyperparameters):
        norm_der_comp = dot_(gradient[ihyp], vector)
        gradient[ihyp] -= norm_der_comp * vector


@jit_
def inplace_normalization_wgrad(vector, gradient, nhyperparameters):
    norm = l2_norm_(vector)
    inplace_normalization_wgrad_from_norm(vector, gradient, nhyperparameters, norm)
