from ...jit_interfaces import cos_, dot_, empty_, jit_, l2_norm_, sin_, sqrt_
from ..base_functions import (
    generate_sign_invariant_sorf_serial,
    generate_sorf_serial,
    generate_sorf_stack_phases_serial,
    generate_sorf_unbiased_phases_serial,
    generate_sorf_wgrad,
    multiply_by_stacks,
)
from ..utils import jitclass_
from .base import (
    array_2D_,
    array_3D_,
    basic_input_components,
    cadd_simple,
    cpuest_add_single,
    default_int,
    float64,
    get_separate_grad_work_arr,
    get_separate_input_work_arr,
    inp_add_simple,
    is_grad_def,
    is_red_vs_jit_copy_def,
    is_reducable_object,
)
from .rescalings import add_full_rescaling

sorf_phase_components = [
    ("sorf_diags", array_3D_),
    ("norm_const", float64),
    ("rff_vec_norm_const", float64),
    ("nfeature_stacks", default_int),
    ("init_size", default_int),
    ("ntransforms", default_int),
]


sorf_components = sorf_phase_components + [("biases", array_2D_)]


def get_separate_grad_work_arr_wphases():
    separate_grad_work_arr = get_separate_grad_work_arr()

    @jit_
    def separate_grad_work_arr_wphases(input_object, grad_object, grad_work_array):
        nested_grad_work_array, final_gradient, nested_final_gradient = separate_grad_work_arr(
            input_object, grad_object, grad_work_array
        )
        phases_ub = nested_grad_work_array.shape[0] - grad_object.nested_grad_object.output_size
        phases = nested_grad_work_array[phases_ub - input_object.output_size : phases_ub]
        return nested_grad_work_array, final_gradient, nested_final_gradient, phases

    return separate_grad_work_arr_wphases


# Unscaled SORF is SORF with sigma equalling one.
unscaled_sorf_lvl = "unscaled_sorf"


def add_unscaled_sorf(processed_function_definition_list, inside_routine):
    separate_input_work_arr = get_separate_input_work_arr()
    if is_grad_def(processed_function_definition_list):
        separate_grad_work_arr_wphases = get_separate_grad_work_arr_wphases()

        @jit_
        def unscaled_sorf_transform(
            processed_object,
            input_object,
            grad_object,
            hyperparameters,
            input_work_array,
            grad_work_array,
        ):
            nested_input_work_array, final_result, nested_final_result = separate_input_work_arr(
                input_object, input_work_array
            )
            (
                nested_grad_work_array,
                final_gradient,
                nested_final_gradient,
                phases,
            ) = separate_grad_work_arr_wphases(input_object, grad_object, grad_work_array)

            inside_routine(
                processed_object,
                input_object.nested_input_object,
                grad_object.nested_grad_object,
                hyperparameters,
                nested_input_work_array,
                nested_grad_work_array,
            )
            generate_sorf_stack_phases_serial(
                nested_final_result,
                phases,
                input_object.sorf_diags,
                input_object.biases,
                input_object.norm_const,
                input_object.nfeature_stacks,
                input_object.init_size,
            )
            final_result[:] = cos_(phases) * input_object.rff_vec_norm_const

            for hyperparam_id in range(hyperparameters.shape[0]):
                generate_sorf_unbiased_phases_serial(
                    nested_final_gradient[hyperparam_id],
                    final_gradient[hyperparam_id],
                    input_object.sorf_diags,
                    input_object.norm_const,
                    input_object.nfeature_stacks,
                    input_object.init_size,
                )

            final_gradient[:, :] *= -sin_(phases) * input_object.rff_vec_norm_const

    else:

        @jit_
        def unscaled_sorf_transform(
            processed_object, input_object, hyperparameters, input_work_array
        ):
            nested_input_work_array, final_result, nested_final_result = separate_input_work_arr(
                input_object, input_work_array
            )

            inside_routine(
                processed_object,
                input_object.nested_input_object,
                hyperparameters,
                nested_input_work_array,
            )
            generate_sorf_serial(
                nested_final_result,
                final_result,
                input_object.sorf_diags,
                input_object.biases,
                input_object.norm_const,
                input_object.rff_vec_norm_const,
                input_object.nfeature_stacks,
                input_object.init_size,
            )

    return unscaled_sorf_transform


def inp_add_unscaled_sorf(
    processed_definition_list, nested_input_component, nested_default_input_object
):
    if is_grad_def(processed_definition_list):
        return inp_add_simple(
            processed_definition_list, nested_input_component, nested_default_input_object
        )

    @jitclass_(
        basic_input_components + sorf_components + nested_input_component,
        skip=is_reducable_object(processed_definition_list),
    )
    class inp_unscaled_sorf:
        def __init__(
            self,
            nested_input_object,
            output_size=None,
            nfeature_stacks=0,
            ntransforms=0,
            init_size=0,
            work_size=0,
        ):
            if output_size is None:
                self.output_size = nfeature_stacks * init_size
            else:
                self.output_size = output_size
            self.nested_input_object = nested_input_object

            self.nfeature_stacks = nfeature_stacks
            self.ntransforms = ntransforms
            self.init_size = init_size

            self.norm_const = 0.0
            self.work_size = work_size

            self.biases = empty_((self.nfeature_stacks, self.init_size))
            self.sorf_diags = empty_((self.nfeature_stacks, self.ntransforms, self.init_size))

    return inp_unscaled_sorf, inp_unscaled_sorf(nested_default_input_object)


def cadd_unscaled_sorf(processed_definition_list, inside_copy, object_class):
    if is_grad_def(processed_definition_list):
        return cadd_simple(processed_definition_list, inside_copy, object_class)

    @jit_(skip=is_red_vs_jit_copy_def(processed_definition_list))
    def copy(input_object):
        new_nested_object = inside_copy(input_object.nested_input_object)
        nfeature_stacks = input_object.nfeature_stacks
        init_size = input_object.init_size
        ntransforms = input_object.ntransforms
        work_size = input_object.work_size
        new_object = object_class(
            new_nested_object,
            input_object.output_size,
            nfeature_stacks,
            ntransforms,
            init_size,
            work_size,
        )
        new_object.biases[:, :] = input_object.biases[:, :]
        new_object.sorf_diags[:, :, :] = input_object.sorf_diags[:, :, :]
        new_object.norm_const = input_object.norm_const
        new_object.rff_vec_norm_const = input_object.rff_vec_norm_const
        return new_object

    return copy


# K.Karan: add_sorf and add_unscaled_sorf share a lot in common, but had to be
# put separately due to different handling of gradient calculation.
sorf_lvl = "sorf"


def add_sorf(processed_function_definition_list, inside_routine):
    if not is_grad_def(processed_function_definition_list):
        new_inside_routine = add_full_rescaling(processed_function_definition_list, inside_routine)
        return add_unscaled_sorf(processed_function_definition_list, new_inside_routine)

    separate_input_work_arr = get_separate_input_work_arr()
    separate_grad_work_arr_wphases = get_separate_grad_work_arr_wphases()

    @jit_
    def sorf_transform(
        processed_object,
        input_object,
        grad_object,
        hyperparameters,
        input_work_array,
        grad_work_array,
    ):
        nested_input_work_array, final_result, nested_final_result = separate_input_work_arr(
            input_object, input_work_array
        )
        (
            nested_grad_work_array,
            final_gradient,
            nested_final_gradient,
            phases,
        ) = separate_grad_work_arr_wphases(input_object, grad_object, grad_work_array)

        inside_routine(
            processed_object,
            input_object.nested_input_object,
            grad_object.nested_grad_object,
            hyperparameters[:-1],
            nested_input_work_array,
            nested_grad_work_array,
        )

        generate_sorf_wgrad(
            nested_final_result,
            nested_final_gradient,
            final_result,
            final_gradient,
            hyperparameters[-1],
            phases,
            input_object.sorf_diags,
            input_object.biases,
            input_object.norm_const,
            input_object.rff_vec_norm_const,
            input_object.nfeature_stacks,
            input_object.init_size,
        )

    return sorf_transform


# For "mixed-extensive" version of SORF.
mixed_extensive_sorf_lvl = "mixed_extensive_sorf"


@jit_
def mixed_extensive_sorf_addition_coeff(x):
    return 1 / sqrt_(1 + 1 / x)


@jit_
def mixed_extensive_sorf_addition_coeff_wder(x):
    val = mixed_extensive_sorf_addition_coeff(x)
    return val, 0.5 * val**3 / x**2


def add_mixed_extensive_sorf(processed_function_definition_list, inside_routine):
    separate_input_work_arr = get_separate_input_work_arr()
    if is_grad_def(processed_function_definition_list):
        separate_grad_work_arr_wphases = get_separate_grad_work_arr_wphases()

        # K.Karan.: I was prioritizing readability over optimization here.
        @jit_
        def mixed_extensive_sorf_transform(
            processed_object,
            input_object,
            grad_object,
            hyperparameters,
            input_work_array,
            grad_work_array,
        ):
            prop_hyperparam = hyperparameters[-1]
            coeff_vec, coeff_vec_der = mixed_extensive_sorf_addition_coeff_wder(prop_hyperparam)
            inv_prop_hyperparam = 1 / prop_hyperparam
            coeff_sorf, coeff_sorf_der = mixed_extensive_sorf_addition_coeff_wder(
                inv_prop_hyperparam
            )
            coeff_sorf_der *= -(inv_prop_hyperparam**2)

            nested_input_work_array, final_result, nested_final_result = separate_input_work_arr(
                input_object, input_work_array
            )
            (
                nested_grad_work_array,
                final_gradient,
                nested_final_gradient,
                phases,
            ) = separate_grad_work_arr_wphases(input_object, grad_object, grad_work_array)
            nnested_hyperparameters = hyperparameters.shape[0] - 2
            # add space for derivatives of the vector's norm before the part allocated for phases.
            phases_end_id = (
                nested_grad_work_array.shape[0]
                - grad_object.nested_grad_object.output_size
                - input_object.output_size
            )
            vector_norm_derivatives = nested_grad_work_array[
                phases_end_id - nnested_hyperparameters : phases_end_id
            ]

            inside_routine(
                processed_object,
                input_object.nested_input_object,
                grad_object.nested_grad_object,
                hyperparameters[:-2],
                nested_input_work_array,
                nested_grad_work_array,
            )

            # norm of input vector and its derivatives
            # also transform nested_* results into normalized vector w. derivatives.
            vector_norm = l2_norm_(nested_final_result)
            nested_final_result /= vector_norm
            nested_final_gradient /= vector_norm
            for hyp_id in range(nnested_hyperparameters):
                vector_norm_derivatives[hyp_id] = dot_(
                    nested_final_result, nested_final_gradient[hyp_id]
                )
                nested_final_gradient[hyp_id] -= (
                    vector_norm_derivatives[hyp_id] * nested_final_result
                )
            vector_norm_derivatives *= vector_norm

            sigma = hyperparameters[-2]
            nfeature_stacks = input_object.nfeature_stacks
            # generate the SORF
            # note that generate_sorf_wgrad divided nested_final_* arrays by sigma
            generate_sorf_wgrad(
                nested_final_result,
                nested_final_gradient,
                final_result,
                final_gradient[:-1],
                sigma,
                phases,
                input_object.sorf_diags,
                input_object.biases,
                input_object.norm_const,
                input_object.rff_vec_norm_const,
                nfeature_stacks,
                input_object.init_size,
            )
            nfeature_stacks_add_norm = sigma / sqrt_(float(nfeature_stacks))
            nested_final_result *= nfeature_stacks_add_norm
            nested_final_gradient *= nfeature_stacks_add_norm

            prop_hyperparam_grad = final_gradient[-1]
            prop_hyperparam_grad[:] = final_result[:] * coeff_sorf_der
            final_result *= coeff_sorf
            final_gradient[:-1, :] *= coeff_sorf

            lb = 0
            for _ in range(input_object.nfeature_stacks):
                ub = lb + input_object.init_size
                final_result[lb:ub] += coeff_vec * nested_final_result
                prop_hyperparam_grad[lb:ub] += coeff_vec_der * nested_final_result
                lb = ub

            # combine derivatives of initial vector and SORF-transformed.
            # done separately for memory efficiency.
            nested_final_gradient *= coeff_vec
            for hyp_id in range(nnested_hyperparameters):
                lb = 0
                for _ in range(input_object.nfeature_stacks):
                    ub = lb + input_object.init_size
                    final_gradient[hyp_id, lb:ub] += nested_final_gradient[hyp_id, :]
                    lb = ub
            # in the gradient account for multiplication by vector_norm.
            final_gradient *= vector_norm
            for hyp_id in range(nnested_hyperparameters):
                final_gradient[hyp_id] += vector_norm_derivatives[hyp_id] * final_result
            final_result *= vector_norm

    else:

        @jit_
        def mixed_extensive_sorf_transform(
            processed_object, input_object, hyperparameters, input_work_array
        ):
            nested_input_work_array, final_result, nested_final_result = separate_input_work_arr(
                input_object, input_work_array
            )

            inside_routine(
                processed_object,
                input_object.nested_input_object,
                hyperparameters[:-2],
                nested_input_work_array,
            )
            prop_hyperparam = hyperparameters[-1]
            coeff_vec = mixed_extensive_sorf_addition_coeff(prop_hyperparam)
            coeff_sorf = mixed_extensive_sorf_addition_coeff(1 / prop_hyperparam)

            vec_norm = l2_norm_(nested_final_result)
            sigma = hyperparameters[-2]
            nested_final_result /= vec_norm * sigma

            nfeature_stacks = input_object.nfeature_stacks
            init_size = input_object.init_size
            generate_sorf_serial(
                nested_final_result,
                final_result,
                input_object.sorf_diags,
                input_object.biases,
                input_object.norm_const,
                input_object.rff_vec_norm_const,
                input_object.nfeature_stacks,
                init_size,
            )
            final_result *= coeff_sorf
            nested_final_result *= sigma * coeff_vec / sqrt_(float(nfeature_stacks))

            lb = 0
            for _ in range(nfeature_stacks):
                ub = lb + init_size
                final_result[lb:ub] += nested_final_result
                lb = ub

            final_result *= vec_norm

    return mixed_extensive_sorf_transform


# For sign-invariant unscaled SORF.
unscaled_sign_invariant_sorf_lvl = "unscaled_sign_invariant_sorf"

sign_invariant_sorf_components = sorf_phase_components + [("bias_cosines", array_2D_)]


def add_unscaled_sign_invariant_sorf(processed_function_definition_list, inside_routine):
    separate_input_work_arr = get_separate_input_work_arr()

    if is_grad_def(processed_function_definition_list):
        separate_grad_work_arr_wphases = get_separate_grad_work_arr_wphases()

        @jit_
        def unscaled_sign_invariant_sorf_transform(
            processed_object,
            input_object,
            grad_object,
            hyperparameters,
            input_work_array,
            grad_work_array,
        ):
            nested_input_work_array, final_result, nested_final_result = separate_input_work_arr(
                input_object, input_work_array
            )
            (
                nested_grad_work_array,
                final_gradient,
                nested_final_gradient,
                phases,
            ) = separate_grad_work_arr_wphases(input_object, grad_object, grad_work_array)

            inside_routine(
                processed_object,
                input_object.nested_input_object,
                grad_object.nested_grad_object,
                hyperparameters,
                nested_input_work_array,
                nested_grad_work_array,
            )
            generate_sorf_unbiased_phases_serial(
                nested_final_result,
                phases,
                input_object.sorf_diags,
                input_object.norm_const,
                input_object.nfeature_stacks,
                input_object.init_size,
            )

            final_result[:] = cos_(phases) * input_object.rff_vec_norm_const
            multiply_by_stacks(
                final_result,
                input_object.bias_cosines,
                input_object.nfeature_stacks,
                input_object.init_size,
            )
            for hyperparam_id in range(hyperparameters.shape[0]):
                generate_sorf_unbiased_phases_serial(
                    nested_final_gradient[hyperparam_id],
                    final_gradient[hyperparam_id],
                    input_object.sorf_diags,
                    input_object.norm_const,
                    input_object.nfeature_stacks,
                    input_object.init_size,
                )

            final_gradient[:, :] *= -sin_(phases) * input_object.rff_vec_norm_const
            for hyperparam_id in range(hyperparameters.shape[0]):
                multiply_by_stacks(
                    final_gradient[hyperparam_id],
                    input_object.bias_cosines,
                    input_object.nfeature_stacks,
                    input_object.init_size,
                )

    else:

        @jit_
        def unscaled_sign_invariant_sorf_transform(
            processed_object, input_object, hyperparameters, input_work_array
        ):
            nested_input_work_array, final_result, nested_final_result = separate_input_work_arr(
                input_object, input_work_array
            )
            inside_routine(
                processed_object,
                input_object.nested_input_object,
                hyperparameters,
                nested_input_work_array,
            )

            generate_sign_invariant_sorf_serial(
                nested_final_result,
                final_result,
                input_object.sorf_diags,
                input_object.bias_cosines,
                input_object.norm_const,
                input_object.rff_vec_norm_const,
                input_object.nfeature_stacks,
                input_object.init_size,
            )

    return unscaled_sign_invariant_sorf_transform


def inp_add_unscaled_sign_invariant_sorf(
    processed_definition_list, nested_input_component, nested_default_input_object
):
    if is_grad_def(processed_definition_list):
        return inp_add_unscaled_sorf(
            processed_definition_list, nested_input_component, nested_default_input_object
        )

    @jitclass_(
        basic_input_components + sign_invariant_sorf_components + nested_input_component,
        skip=is_reducable_object(processed_definition_list),
    )
    class inp_unscaled_sign_invariant_sorf:
        def __init__(
            self,
            nested_input_object,
            output_size=0,
            nfeature_stacks=0,
            ntransforms=0,
            init_size=0,
            work_size=0,
        ):
            self.nested_input_object = nested_input_object

            self.output_size = output_size
            self.work_size = work_size

            self.nfeature_stacks = nfeature_stacks
            self.ntransforms = ntransforms
            self.init_size = init_size

            self.norm_const = 0.0

            self.bias_cosines = empty_((self.nfeature_stacks, self.init_size))
            self.sorf_diags = empty_((self.nfeature_stacks, self.ntransforms, self.init_size))

    return inp_unscaled_sign_invariant_sorf, inp_unscaled_sign_invariant_sorf(
        nested_default_input_object
    )


def cadd_unscaled_sign_invariant_sorf(processed_definition_list, inside_copy, object_class):
    if is_grad_def(processed_definition_list):
        return cadd_unscaled_sorf(processed_definition_list, inside_copy, object_class)

    @jit_(skip=is_red_vs_jit_copy_def(processed_definition_list))
    def copy(input_object):
        new_nested_object = inside_copy(input_object.nested_input_object)
        nfeature_stacks = input_object.nfeature_stacks
        init_size = input_object.init_size
        ntransforms = input_object.ntransforms
        output_size = input_object.output_size
        work_size = input_object.work_size
        new_object = object_class(
            new_nested_object, output_size, nfeature_stacks, ntransforms, init_size, work_size
        )
        new_object.bias_cosines[:, :] = input_object.bias_cosines[:, :]
        new_object.sorf_diags[:, :, :] = input_object.sorf_diags[:, :, :]
        new_object.norm_const = input_object.norm_const
        new_object.rff_vec_norm_const = input_object.rff_vec_norm_const
        return new_object

    return copy


# (Scaled) sign-invariant SORF transform.
sign_invariant_sorf_lvl = "sign_invariant_sorf"


def add_sign_invariant_sorf(processed_function_definition_list, inside_routine):
    if not is_grad_def(processed_function_definition_list):
        new_inside_routine = add_full_rescaling(processed_function_definition_list, inside_routine)
        return add_unscaled_sign_invariant_sorf(
            processed_function_definition_list, new_inside_routine
        )

    separate_input_work_arr = get_separate_input_work_arr()
    separate_grad_work_arr_wphases = get_separate_grad_work_arr_wphases()

    @jit_
    def sign_invariant_sorf_transform(
        processed_object,
        input_object,
        grad_object,
        hyperparameters,
        input_work_array,
        grad_work_array,
    ):
        nested_input_work_array, final_result, nested_final_result = separate_input_work_arr(
            input_object, input_work_array
        )
        (
            nested_grad_work_array,
            final_gradient,
            nested_final_gradient,
            phases,
        ) = separate_grad_work_arr_wphases(input_object, grad_object, grad_work_array)

        inside_routine(
            processed_object,
            input_object.nested_input_object,
            grad_object.nested_grad_object,
            hyperparameters[:-1],
            nested_input_work_array,
            nested_grad_work_array,
        )
        current_sigma = hyperparameters[-1]

        nested_final_result[:] /= current_sigma
        nested_final_gradient[:, :] /= current_sigma
        generate_sorf_unbiased_phases_serial(
            nested_final_result,
            phases,
            input_object.sorf_diags,
            input_object.norm_const,
            input_object.nfeature_stacks,
            input_object.init_size,
        )
        final_result[:] = cos_(phases) * input_object.rff_vec_norm_const
        multiply_by_stacks(
            final_result,
            input_object.bias_cosines,
            input_object.nfeature_stacks,
            input_object.init_size,
        )

        final_gradient[-1, :] = -phases / current_sigma

        for hyperparam_id in range(hyperparameters.shape[0] - 1):
            generate_sorf_unbiased_phases_serial(
                nested_final_gradient[hyperparam_id],
                final_gradient[hyperparam_id],
                input_object.sorf_diags,
                input_object.norm_const,
                input_object.nfeature_stacks,
                input_object.init_size,
            )

        final_gradient[:, :] *= -sin_(phases) * input_object.rff_vec_norm_const
        for hyperparam_id in range(hyperparameters.shape[0]):
            multiply_by_stacks(
                final_gradient[hyperparam_id],
                input_object.bias_cosines,
                input_object.nfeature_stacks,
                input_object.init_size,
            )

    return sign_invariant_sorf_transform


function_level_additions = {
    unscaled_sorf_lvl: add_unscaled_sorf,
    sorf_lvl: add_sorf,
    mixed_extensive_sorf_lvl: add_mixed_extensive_sorf,
    unscaled_sign_invariant_sorf_lvl: add_unscaled_sign_invariant_sorf,
    sign_invariant_sorf_lvl: add_sign_invariant_sorf,
}
copy_level_additions = {
    unscaled_sorf_lvl: cadd_unscaled_sorf,
    sorf_lvl: cadd_unscaled_sorf,
    mixed_extensive_sorf_lvl: cadd_unscaled_sorf,
    unscaled_sign_invariant_sorf_lvl: cadd_unscaled_sign_invariant_sorf,
    sign_invariant_sorf_lvl: cadd_unscaled_sign_invariant_sorf,
}
class_level_additions = {
    unscaled_sorf_lvl: inp_add_unscaled_sorf,
    sorf_lvl: inp_add_unscaled_sorf,
    mixed_extensive_sorf_lvl: inp_add_unscaled_sorf,
    unscaled_sign_invariant_sorf_lvl: inp_add_unscaled_sign_invariant_sorf,
    sign_invariant_sorf_lvl: inp_add_unscaled_sign_invariant_sorf,
}
cpuest_level_additions = {
    unscaled_sorf_lvl: cpuest_add_single,
    sorf_lvl: cpuest_add_single,
    mixed_extensive_sorf_lvl: cpuest_add_single,
    unscaled_sign_invariant_sorf_lvl: cpuest_add_single,
    sign_invariant_sorf_lvl: cpuest_add_single,
}
