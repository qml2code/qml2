# Initialize multilevel SORF functions and use them to calculate Z-matrices and excluded point MAEs along with their derivatives w.r.t. hyperparameters.
from numba import get_num_threads, typed

from ..basic_utils import now
from ..jit_interfaces import array_, dot_, empty_, jit_, ndarray_, prange_, sqrt_, sum_, zeros_
from ..kernels.sorf import create_sorf_matrices, hadamard_norm_const, rff_vec_norm_const
from ..models.sorf_hyperparameter_optimization import leaveoneout_loss_der_wrt_single_feature
from ..utils import (
    concatenate_wNone_,
    get_assigned_jobs,
    get_thread_assignments,
    get_thread_assignments_cpu_loads,
    l2_sq_norm,
)
from .base_constructors import (
    calc_grad_input_size_parameters,
    calc_hyperparameter_num,
    calc_input_size_parameters,
    calc_variance_lvl,
    calc_variance_normalized_lvl,
    component_cycle_lvl,
    component_sum_lvl,
    compression_lvl,
    concatenation_lvl,
    concatenation_variance_lvl,
    create_input_object_from_def,
    element_id_switch_lvl,
    element_id_switch_variance_lvl,
    get_class_from_def,
    get_copy_from_def,
    get_cpuest_from_def,
    get_extract_final_gradient,
    get_extract_final_result,
    get_routine_from_def,
    gradient_input_object_prefix,
    input_object_prefix,
    mixed_extensive_sorf_lvl,
    normalization_lvl,
    pass_variance_fork1_lvl,
    pass_variance_fork2_lvl,
    pass_variance_lvl,
    power_rescaling_lvl,
    project_resize_lvl,
    rescaling_lvl,
    resize_lvl,
    sign_invariant_sorf_lvl,
    sorf_lvl,
    unscaled_sign_invariant_sorf_lvl,
    unscaled_sorf_lvl,
    weighted_component_cycle_lvl,
    weighted_component_sum_lvl,
)
from .base_functions import create_sign_invariant_sorf_matrices
from .pickle import Pickler

implemented_levels = [
    normalization_lvl,
    weighted_component_sum_lvl,
    component_sum_lvl,
    resize_lvl,
    project_resize_lvl,
    unscaled_sorf_lvl,
    sorf_lvl,
    mixed_extensive_sorf_lvl,
    unscaled_sign_invariant_sorf_lvl,
    sign_invariant_sorf_lvl,
    concatenation_lvl,
    rescaling_lvl,
    power_rescaling_lvl,
    compression_lvl,
    element_id_switch_lvl,
]

requires_hyperparameters = [
    sorf_lvl,
    sign_invariant_sorf_lvl,
    power_rescaling_lvl,
    rescaling_lvl,
    compression_lvl,
    mixed_extensive_sorf_lvl,
]


def get_Z_matrix_calculator(sorf_routine, disable_numba_parallelization=False):
    extract_final_result = get_extract_final_result()

    @jit_(numba_parallel=not disable_numba_parallelization)
    def calc_Z_matrix(
        processed_objects, Z_matrix, input_object, hyperparameters, thread_assignments=None
    ):
        nobjects = len(processed_objects)
        input_work_size = input_object.work_size
        for proc_id in prange_(get_num_threads()):
            input_work_array = empty_(input_work_size)
            for i in get_assigned_jobs(proc_id, nobjects, thread_assignments=thread_assignments):
                sorf_routine(processed_objects[i], input_object, hyperparameters, input_work_array)
                Z_matrix[i, :] = extract_final_result(input_object, input_work_array)

    return calc_Z_matrix


def get_Z_matrix_calculator_wgrad(sorf_routine, disable_numba_parallelization=False):
    extract_final_result = get_extract_final_result()
    extract_final_gradient = get_extract_final_gradient()

    @jit_(numba_parallel=not disable_numba_parallelization)
    def calc_Z_matrix_wgrad(
        processed_objects,
        Z_matrix,
        Z_matrix_grad,
        input_object,
        grad_input_object,
        hyperparameters,
        thread_assignments=None,
    ):
        nobjects = len(processed_objects)

        input_work_size = input_object.work_size
        grad_work_size = grad_input_object.work_size

        for proc_id in prange_(get_num_threads()):
            input_work_array = empty_(input_work_size)
            grad_work_array = empty_(grad_work_size)
            for i in get_assigned_jobs(proc_id, nobjects, thread_assignments=thread_assignments):
                sorf_routine(
                    processed_objects[i],
                    input_object,
                    grad_input_object,
                    hyperparameters,
                    input_work_array,
                    grad_work_array,
                )
                Z_matrix[i, :] = extract_final_result(input_object, input_work_array)
                Z_matrix_grad[i, :, :] = extract_final_gradient(
                    input_object, grad_input_object, grad_work_array
                )

    return calc_Z_matrix_wgrad


def get_loss_function_hyperparameter_grad_calculator(sorf_routine):
    extract_final_gradient = get_extract_final_gradient()

    @jit_(numba_parallel=True)
    def calc_loss_function_hyperparameter_grad(
        processed_objects,
        input_object,
        grad_input_object,
        hyperparameters,
        Z_U,
        Z_Vh,
        Z_singular_values,
        loss_error_ders,
        mult_transformed_alphas_rhs,
        reproduced_quantities,
        stat_factors,
        transformed_inv_K_Z,
        used_quantities,
        importance_multipliers,
        thread_assignments=None,
    ):
        npoints = len(processed_objects)
        nhyperparameters = hyperparameters.shape[0]

        input_work_size = input_object.work_size
        grad_work_size = grad_input_object.work_size

        all_loss_function_ders = empty_((npoints, nhyperparameters))

        for proc_id in prange_(get_num_threads()):
            input_work_array = empty_(input_work_size)
            grad_work_array = empty_(grad_work_size)
            for i in get_assigned_jobs(proc_id, npoints, thread_assignments=thread_assignments):
                sorf_routine(
                    processed_objects[i],
                    input_object,
                    grad_input_object,
                    hyperparameters,
                    input_work_array,
                    grad_work_array,
                )
                cur_gradient = extract_final_gradient(
                    input_object, grad_input_object, grad_work_array
                )
                if importance_multipliers is not None:
                    cur_gradient *= importance_multipliers[i]

                leaveoneout_loss_der_wrt_single_feature(
                    all_loss_function_ders[i],
                    cur_gradient,
                    i,
                    Z_U,
                    Z_Vh,
                    Z_singular_values,
                    loss_error_ders,
                    mult_transformed_alphas_rhs,
                    reproduced_quantities,
                    stat_factors,
                    transformed_inv_K_Z,
                    used_quantities,
                )
        return sum_(all_loss_function_ders, axis=0)

    return calc_loss_function_hyperparameter_grad


def get_list_cpu_estimator(cpu_estimator):
    @jit_(numba_parallel=True)
    def list_cpu_estimator(processed_objects, input_object):
        nobjects = len(processed_objects)
        output = empty_(nobjects)
        for i in prange_(nobjects):
            output[i] = cpu_estimator(processed_objects[i], input_object)
        return output

    return list_cpu_estimator


def get_list_gradient_cpu_estimator(cpu_estimator):
    @jit_(numba_parallel=True)
    def list_gradient_cpu_estimator(processed_objects, input_object, grad_object):
        nobjects = len(processed_objects)
        output = empty_(nobjects)
        for i in prange_(nobjects):
            output[i] = cpu_estimator(processed_objects[i], input_object, grad_object)
        return output

    return list_gradient_cpu_estimator


def get_features_calculator(sorf_routine):
    """
    For calculating features for one query.
    """
    extract_final_result = get_extract_final_result()

    @jit_
    def calc_features(processed_object, input_object, hyperparameters, input_work_array):
        sorf_routine(processed_object, input_object, hyperparameters, input_work_array)
        return extract_final_result(input_object, input_work_array)

    return calc_features


def get_prediction_calculator(features_calculator):
    @jit_
    def calc_prediction(processed_object, input_object, hyperparameters, alphas, input_work_array):
        features = features_calculator(
            processed_object, input_object, hyperparameters, input_work_array
        )
        return dot_(alphas, features)

    return calc_prediction


def get_predictions_calculator(features_calculator):
    """
    Get embarassingly parallelized version of output of `get_prediction_calculator`.
    """
    calc_prediction = get_prediction_calculator(features_calculator)

    @jit_(numba_parallel=True)
    def calc_predictions(processed_objects, input_object, hyperparameters, alphas):
        nobjects = len(processed_objects)
        output = empty_(nobjects)
        input_work_size = input_object.work_size
        for i in prange_(nobjects):
            input_work_array = empty_(input_work_size)
            output[i] = calc_prediction(
                processed_objects[i], input_work_array, hyperparameters, alphas, input_work_array
            )
        return output

    return calc_predictions


def get_mean_diagonal_element_calculator(sorf_routine):
    extract_final_result = get_extract_final_result()

    @jit_(numba_parallel=True)
    def calc_mean_diagonal_element(processed_objects, input_object, hyperparameters):
        nobjects = len(processed_objects)
        diagonal_sum = 0.0
        work_size = input_object.work_size
        for i in prange_(nobjects):
            input_work_array = empty_(work_size)
            sorf_routine(processed_objects[i], input_object, hyperparameters, input_work_array)
            final_result = extract_final_result(input_object, input_work_array)
            diagonal_sq_norm = l2_sq_norm(final_result)
            diagonal_sum += diagonal_sq_norm
        return diagonal_sum / nobjects

    return calc_mean_diagonal_element


def get_variance_input_creator_from_def(function_definition):
    if function_definition[-1] not in [
        component_cycle_lvl,
        weighted_component_cycle_lvl,
        calc_variance_lvl,
        pass_variance_lvl,
        pass_variance_fork1_lvl,
        pass_variance_fork2_lvl,
        element_id_switch_variance_lvl,
        concatenation_variance_lvl,
    ]:
        return get_copy_from_def(function_definition)

    variance_input_class = get_class_from_def(function_definition, input_object_prefix)
    if len(function_definition) == 1:

        @jit_
        def from_input(input_object):
            # a bit dirty that it uses output instead of input size, but probably not important.
            output_size = input_object.final_result.shape[0]
            return variance_input_class(output_size)

        return from_input

    nested_input_creator = get_variance_input_creator_from_def(function_definition[:-1])

    last_level = function_definition[-1]

    if last_level == "pass_variance_fork1":

        @jit_
        def from_input(input_object):
            nested_variance_obj = nested_input_creator(input_object.nested_input_object1)
            return variance_input_class(nested_variance_obj)

    elif last_level == "pass_variance_fork2":

        @jit_
        def from_input(input_object):
            nested_variance_obj = nested_input_creator(input_object.nested_input_object2)
            return variance_input_class(nested_variance_obj)

    elif last_level in [concatenation_variance_lvl, element_id_switch_variance_lvl]:

        @jit_
        def from_input(input_object):
            nested_variance_obj = nested_input_creator(input_object.nested_input_objects[0])
            return variance_input_class(nested_variance_obj)

    else:

        @jit_
        def from_input(input_object):
            nested_variance_obj = nested_input_creator(input_object.nested_input_object)
            return variance_input_class(nested_variance_obj)

    return from_input


@jit_(numba_parallel=True)
def guess_rescaling_hyperparameters(dispersions, resc_bounds):
    nsigmas = resc_bounds.shape[0]
    output = empty_(nsigmas)
    for i in prange_(nsigmas):
        lb, ub = resc_bounds[i][:]
        output[i] = sum_(dispersions[lb:ub])
    output *= output.shape[0]
    return sqrt_(output)


def guess_hyperparameters_from_dispersions(dispersions, level_type, level_parameters):
    match level_type:
        case "sorf" | "sign_invariant_sorf" | "mixed_extensive_sorf":
            input_size = dispersions.shape[0]
            resc_bounds = array_([[0, input_size]])
        case "rescaling":
            resc_bounds = level_parameters["resc_bounds"]
        case _:
            raise Exception("initial hyperparameter guess procedure not implemented:", level_type)
    sigmas = guess_rescaling_hyperparameters(dispersions, resc_bounds)
    if level_type == mixed_extensive_sorf_lvl:
        assert len(sigmas) == 1
        return array_([sigmas[0], 1.0])
    else:
        return sigmas


def get_sorf_variance_corrected_function_definition(function_definition_list, level_of_interest):
    special_level_replacements = {
        component_sum_lvl: component_cycle_lvl,
        weighted_component_sum_lvl: weighted_component_cycle_lvl,
        element_id_switch_lvl: element_id_switch_variance_lvl,
        concatenation_lvl: concatenation_variance_lvl,
    }

    # define function for calculating variance
    corrected_function_definition = function_definition_list[:level_of_interest]
    if function_definition_list[level_of_interest] == mixed_extensive_sorf_lvl:
        corrected_function_definition.append(calc_variance_normalized_lvl)
    else:
        corrected_function_definition.append(calc_variance_lvl)
    for level in function_definition_list[level_of_interest + 1 :]:
        if level in special_level_replacements:
            new_level = special_level_replacements[level]
        elif level in ["pass_variance_fork1", "pass_variance_fork2"]:
            new_level = level
        else:
            new_level = pass_variance_lvl
        corrected_function_definition.append(new_level)
    return corrected_function_definition


def get_sorf_variance_calculator(
    function_definition_list, level_of_interest, corrected_function_definition=None
):
    if corrected_function_definition is None:
        corrected_function_definition = get_sorf_variance_corrected_function_definition(
            function_definition_list, level_of_interest
        )
    # create variance function and the input object for it
    variance_routine = get_routine_from_def(corrected_function_definition)

    extract_final_result = get_extract_final_result()

    @jit_(numba_parallel=True)
    def calc_variance(processed_objects, input_object, hyperparameters):
        output_size = input_object.output_size
        true_output_size = (output_size - 1) // 2
        total_sum_array = zeros_(input_object.output_size)
        input_work_size = input_object.work_size
        for i in prange_(len(processed_objects)):
            input_work_array = empty_(input_work_size)
            final_result = extract_final_result(input_object, input_work_array)
            final_result[:] = 0.0
            variance_routine(processed_objects[i], input_object, hyperparameters, input_work_array)
            total_sum_array += final_result

        normalization = total_sum_array[-1]
        total_avs = total_sum_array[:true_output_size]
        total_avs2 = total_sum_array[true_output_size:-1]

        total_avs /= normalization
        total_avs2 /= normalization

        return total_avs2 - total_avs**2

    return calc_variance


def get_sorf_variance_input_creator(
    function_definition_list, level_of_interest, corrected_function_definition=None
):
    if corrected_function_definition is None:
        corrected_function_definition = get_sorf_variance_corrected_function_definition(
            function_definition_list, level_of_interest
        )
    return get_variance_input_creator_from_def(corrected_function_definition)


def checked_numba_list(input_list):
    if isinstance(input_list, typed.List) or isinstance(input_list, ndarray_):
        return input_list
    new_list = typed.List()
    for i in input_list:
        new_list.append(i)
    return new_list


def sum_of_last(list_tuple):
    return list_tuple[0][-1] + list_tuple[1][-1]


def fork_work_sizes(size_parameters, grad_size_parameters, final_gradient_size):
    nested_output_size2 = size_parameters[1][-1]["output_size"]
    nested_work_size2 = size_parameters[1][-1]["work_size"]
    nested_work_size1 = size_parameters[0][-1]["work_size"]

    input_work_size = max(nested_work_size2, nested_output_size2 + nested_work_size1)

    nested_grad_size2 = grad_size_parameters[1][-1]["output_size"]
    nested_grad_work_size2 = grad_size_parameters[1][-1]["work_size"]
    nested_grad_work_size1 = grad_size_parameters[0][-1]["work_size"]

    grad_work_size = max(
        [final_gradient_size, nested_grad_size2 + nested_grad_work_size1, nested_grad_work_size2]
    )
    return input_work_size, grad_work_size


def all_parameter_lists(function_definition_list, parameter_list):
    level_nhyperparameters = []
    output_sizes = []
    size_parameters = []
    grad_size_parameters = []
    if isinstance(function_definition_list[0], tuple):
        assert isinstance(parameter_list[0], tuple)
        for nested_func_def_list, nested_param_list in zip(
            function_definition_list[0], parameter_list[0]
        ):
            (
                nested_level_nhyperparameters,
                nested_output_sizes,
                nested_size_parameters,
                nested_grad_size_parameters,
            ) = all_parameter_lists(nested_func_def_list, nested_param_list)
            level_nhyperparameters.append(nested_level_nhyperparameters)
            output_sizes.append(nested_output_sizes)
            size_parameters.append(nested_size_parameters)
            grad_size_parameters.append(nested_grad_size_parameters)

        total_output_size = sum_of_last(output_sizes)
        nhyperparameters = sum_of_last(level_nhyperparameters)
        total_grad_size = total_output_size * nhyperparameters
        input_work_size, grad_work_size = fork_work_sizes(
            size_parameters, grad_size_parameters, total_grad_size
        )

        previous_input_size_parameters = {
            "output_size": total_output_size,
            "work_size": input_work_size,
        }

        size_parameters = [
            {
                "forked": tuple(size_parameters),
                "params": previous_input_size_parameters,
            }
        ]

        previous_grad_size_parameters = {
            "output_size": total_grad_size,
            "nhyperparameters": nhyperparameters,
            "work_size": grad_work_size,
        }

        grad_size_parameters = [
            {
                "forked": tuple(grad_size_parameters),
                "params": previous_grad_size_parameters,
            }
        ]

        level_nhyperparameters = [tuple(level_nhyperparameters)]
        output_sizes = [tuple(output_sizes)]

        remaining_function_definition_list = function_definition_list[1:]
        remaining_parameter_list = parameter_list[1:]
    else:
        nhyperparameters = 0
        previous_input_size_parameters = {}
        previous_grad_size_parameters = {}
        remaining_function_definition_list = function_definition_list
        remaining_parameter_list = parameter_list

    for level_type, level_parameters in zip(
        remaining_function_definition_list, remaining_parameter_list
    ):
        # update number of hyperparameters present.
        nhyperparameters += calc_hyperparameter_num(
            level_type,
            level_parameters,
            previous_input_size_parameters,
        )
        cur_size_parameters = calc_input_size_parameters(
            level_parameters,
            level_type,
            previous_input_size_parameters,
        )
        cur_grad_size_parameters = calc_grad_input_size_parameters(
            level_type,
            nhyperparameters,
            input_size_parameters=cur_size_parameters,
            previous_input_size_parameters=previous_input_size_parameters,
            previous_grad_input_size_parameters=previous_grad_size_parameters,
        )

        level_nhyperparameters.append(nhyperparameters)
        size_parameters.append(cur_size_parameters)
        grad_size_parameters.append(cur_grad_size_parameters)
        previous_input_size_parameters = cur_size_parameters
        previous_grad_size_parameters = cur_grad_size_parameters
        output_sizes.append(cur_size_parameters["output_size"])

    return level_nhyperparameters, output_sizes, size_parameters, grad_size_parameters


class MultilevelSORF:
    def __init__(
        self,
        function_definition_list,
        parameter_list,
        rng=None,
        disable_numba_parallelization=False,
    ):
        assert len(function_definition_list) == len(parameter_list)
        for level in function_definition_list:
            assert (isinstance(level, tuple)) or (level in implemented_levels), level
        self.function_definition_list = function_definition_list
        self.parameter_list = parameter_list
        # for test reproducability.
        self.rng = rng
        #
        self.disable_numba_parallelization = disable_numba_parallelization
        self.pickler = None
        self.init_size_parameter_list()
        self.create_input_objects()
        self.init_input_object()

        self.init_routines()

    def init_routines(self):
        self.copy_input_object = get_copy_from_def(self.function_definition_list)
        self.copy_grad_input_object = get_copy_from_def(
            self.function_definition_list, gradient=True
        )

        jit_attrs = [
            "copy_input_object",
            "copy_grad_input_object",
            "sorf_routine",
            "sorf_routine_wgrad",
            "Z_matrix_calculator",
            "Z_matrix_calculator_wgrad",
            "loss_function_calculator_wgrad",
            "cpu_estimator",
            "gradient_cpu_estimator",
            "list_cpu_estimator",
            "list_gradient_cpu_estimator",
            "features_calculator",
            "prediction_calculator",
            "preditions_calculator",
            "mean_diagonal_element_calculator",
            "variance_calculators",
            "variance_input_objects",
        ]
        for jit_attr in jit_attrs:
            setattr(self, jit_attr, None)

        # NOTE: "self.variance_input_objects" is not a routine, but is put here because it'll be re-created from input_object on restart.

        # for individual molecule predictions
        self.input_work_array = None

        complex_attr_definition_dict = dict(
            (name, [prefix, *self.function_definition_list])
            for name, prefix in [
                ("input_object", input_object_prefix),
                ("grad_input_object", gradient_input_object_prefix),
            ]
        )

        self.pickler = Pickler(
            deleted_attrs=jit_attrs, complex_attr_definition_dict=complex_attr_definition_dict
        )

    def get_input_work_array(self):
        if self.input_work_array is None:
            self.input_work_array = empty_(self.input_object.work_size)
        return self.input_work_array

    def get_sorf_routine(self, gradient=False):
        if gradient:
            if self.sorf_routine_wgrad is None:
                self.sorf_routine_wgrad = get_routine_from_def(
                    self.function_definition_list, gradient=True
                )
            return self.sorf_routine_wgrad
        else:
            if self.sorf_routine is None:
                self.sorf_routine = get_routine_from_def(self.function_definition_list)
            return self.sorf_routine

    def get_Z_matrix_calculator(self, gradient=False):
        if (gradient and (self.Z_matrix_calculator_wgrad is None)) or (
            (not gradient) and (self.Z_matrix_calculator is None)
        ):
            sorf_routine = self.get_sorf_routine(gradient=gradient)
            if gradient:
                self.Z_matrix_calculator_wgrad = get_Z_matrix_calculator_wgrad(
                    sorf_routine, disable_numba_parallelization=self.disable_numba_parallelization
                )
            else:
                self.Z_matrix_calculator = get_Z_matrix_calculator(
                    sorf_routine, disable_numba_parallelization=self.disable_numba_parallelization
                )

        if gradient:
            return self.Z_matrix_calculator_wgrad
        else:
            return self.Z_matrix_calculator

    def get_cpu_estimator(self, gradient=False):
        if gradient:
            if self.gradient_cpu_estimator is None:
                self.gradient_cpu_estimator = get_cpuest_from_def(
                    self.function_definition_list, gradient=True
                )
            return self.gradient_cpu_estimator
        else:
            if self.cpu_estimator is None:
                self.cpu_estimator = get_cpuest_from_def(self.function_definition_list)
            return self.cpu_estimator

    def get_list_cpu_estimator(self, gradient=False):
        if (gradient and (self.gradient_cpu_estimator is None)) or (
            (not gradient) and (self.cpu_estimator is None)
        ):
            cpu_estimator = self.get_cpu_estimator(gradient=gradient)
            if gradient:
                self.list_gradient_cpu_estimator = get_list_gradient_cpu_estimator(cpu_estimator)
            else:
                self.list_cpu_estimator = get_list_cpu_estimator(cpu_estimator)

        if gradient:
            return self.list_gradient_cpu_estimator
        else:
            return self.list_cpu_estimator

    def calc_Z_matrix(
        self,
        input_list,
        hyperparameters,
        gradient=False,
        temp_Z_matrix=None,
        thread_assignments=None,
    ):
        calculator = self.get_Z_matrix_calculator(gradient=gradient)
        input_numba_list = checked_numba_list(input_list)
        nitems = len(input_numba_list)
        nsorf = self.output_size()
        nhyperparameters = self.nhyperparameters()
        assert nhyperparameters == hyperparameters.shape[0]
        if temp_Z_matrix is None:
            Z_matrix = empty_((nitems, nsorf))
        else:
            Z_matrix = temp_Z_matrix
        if gradient:
            Z_matrix_grad = empty_((nitems, nhyperparameters, nsorf))
            calculator(
                input_numba_list,
                Z_matrix,
                Z_matrix_grad,
                self.input_object,
                self.grad_input_object,
                hyperparameters,
                thread_assignments=thread_assignments,
            )
            return Z_matrix, Z_matrix_grad
        calculator(
            input_numba_list,
            Z_matrix,
            self.input_object,
            hyperparameters,
            thread_assignments=thread_assignments,
        )
        return Z_matrix

    def calc_cpu_estimates(self, input_list, gradient=False):
        if gradient:
            inp_objects = (self.input_object, self.grad_input_object)
        else:
            inp_objects = (self.input_object,)
        return self.get_list_cpu_estimator(gradient=gradient)(input_list, *inp_objects)

    def calc_thread_assignments(self, input_list, gradient=False):
        cpu_loads = self.calc_cpu_estimates(input_list, gradient=gradient)
        return get_thread_assignments(cpu_loads)

    def calc_thread_assignments_cpu_loads(
        self, input_list, gradient=False, return_list_cpu_loads=False
    ):
        cpu_loads = self.calc_cpu_estimates(input_list, gradient=gradient)
        output = get_thread_assignments_cpu_loads(cpu_loads)
        if return_list_cpu_loads:
            return *output, cpu_loads
        else:
            return output

    def get_features_calculator(self):
        if self.features_calculator is None:
            self.features_calculator = get_features_calculator(self.sorf_routine)
        return self.features_calculator

    def get_prediction_calculator(self):
        if self.prediction_calculator is None:
            self.prediction_calculator = get_prediction_calculator(self.get_features_calculator())
        return self.prediction_calculator

    def get_predictions_calculator(self):
        if self.predictions_calculator is None:
            self.predictions_calculator = get_predictions_calculator(
                self.get_features_calculator()
            )

    def calc_prediction(self, processed_object, hyperparameters, alphas):
        return self.get_prediction_calculator()(
            processed_object, self.input_object, hyperparameters, alphas
        )

    def calc_features(self, query_object, hyperparameters):
        input_work_array = self.get_input_work_array()
        return self.get_features_calculator()(
            query_object, self.input_object, hyperparameters, input_work_array
        )

    def calc_predictions(self, processed_objects, hyperparameters, alphas):
        checked_processed_objects = checked_numba_list(processed_objects)
        return self.get_predictions_calculator()(
            checked_processed_objects, self.input_object, hyperparameters, alphas
        )

    def get_mean_diagonal_element_calculator(self):
        if self.mean_diagonal_element_calculator is None:
            self.mean_diagonal_element_calculator = get_mean_diagonal_element_calculator(
                self.get_sorf_routine(), self.copy_input_object
            )
        return self.mean_diagonal_element_calculator

    def calc_mean_diagonal_element(self, processed_objects, hyperparameters):
        checked_processed_objects = checked_numba_list(processed_objects)
        return self.get_mean_diagonal_element_calculator()(
            checked_processed_objects, self.input_object, hyperparameters
        )

    def get_loss_function_hyperparameter_grad_calculator(self):
        if self.loss_function_calculator_wgrad is None:
            self.loss_function_calculator_wgrad = get_loss_function_hyperparameter_grad_calculator(
                self.get_sorf_routine(gradient=True)
            )
        return self.loss_function_calculator_wgrad

    def calc_loss_function_hyperparameter_grad(
        self,
        processed_objects,
        hyperparameters,
        Z_U,
        Z_Vh,
        Z_singular_values,
        loss_error_ders,
        mult_transformed_alphas_rhs,
        reproduced_quantities,
        stat_factors,
        transformed_inv_K_Z,
        used_quantities,
        importance_multipliers,
        thread_assignments=None,
    ):
        return self.get_loss_function_hyperparameter_grad_calculator()(
            processed_objects,
            self.input_object,
            self.grad_input_object,
            hyperparameters,
            Z_U,
            Z_Vh,
            Z_singular_values,
            loss_error_ders,
            mult_transformed_alphas_rhs,
            reproduced_quantities,
            stat_factors,
            transformed_inv_K_Z,
            used_quantities,
            importance_multipliers,
            thread_assignments=thread_assignments,
        )

    def init_size_parameter_list(self):
        (
            self.level_nhyperparameters,
            self.output_sizes,
            self.size_parameters,
            self.grad_size_parameters,
        ) = all_parameter_lists(self.function_definition_list, self.parameter_list)

    def output_size(self):
        return self.output_sizes[-1]

    def nhyperparameters(self):
        return self.level_nhyperparameters[-1]

    def create_input_objects(self):
        self.input_object = create_input_object_from_def(
            self.function_definition_list, self.size_parameters, self.parameter_list
        )
        self.grad_input_object = create_input_object_from_def(
            self.function_definition_list,
            self.grad_size_parameters,
            self.parameter_list,
            gradient=True,
        )

    def init_sorf(self, input_object, size_parameters, level_parameters, sign_invariant=False):
        nfeature_stacks = level_parameters["nfeature_stacks"]
        ntransforms = level_parameters["ntransforms"]
        input_size = size_parameters["init_size"]
        output_size = nfeature_stacks * input_size
        if sign_invariant:
            cur_bias_cosines, cur_sorf_diags = create_sign_invariant_sorf_matrices(
                nfeature_stacks, ntransforms, input_size, rng=self.rng
            )
            input_object.bias_cosines[:, :] = cur_bias_cosines
        else:
            cur_biases, cur_sorf_diags = create_sorf_matrices(
                nfeature_stacks, ntransforms, input_size, rng=self.rng
            )
            input_object.biases[:, :] = cur_biases
        input_object.sorf_diags[:, :, :] = cur_sorf_diags
        input_object.norm_const = hadamard_norm_const(input_size)
        input_object.rff_vec_norm_const = rff_vec_norm_const(output_size)

    def init_power_rescaling(self, input_object, level_parameters):
        input_object.resc_powers[:, :] = level_parameters["resc_powers"][:, :]

    def init_rescaling(self, input_object, level_parameters):
        input_object.resc_bounds[:, :] = level_parameters["resc_bounds"][:, :]

    def init_input_object(
        self,
        input_object=None,
        function_definition_list=None,
        parameter_list=None,
        size_parameters=None,
        level_nhyperparameters=None,
    ):
        if input_object is None:
            input_object = self.input_object
            function_definition_list = self.function_definition_list
            parameter_list = self.parameter_list
            size_parameters = self.size_parameters
            level_nhyperparameters = self.level_nhyperparameters
        level_type = function_definition_list[-1]
        if isinstance(level_type, tuple):
            assert len(function_definition_list) == 1
            size_param_tuple = size_parameters[0]["forked"]
            for nest_id, nest_inp_obj in enumerate(
                (input_object.nested_input_object1, input_object.nested_input_object2)
            ):
                self.init_input_object(
                    nest_inp_obj,
                    function_definition_list=function_definition_list[0][nest_id],
                    parameter_list=parameter_list[0][nest_id],
                    size_parameters=size_param_tuple[nest_id],
                    level_nhyperparameters=level_nhyperparameters[0][nest_id],
                )
            input_object.result_separator = size_param_tuple[0][-1]["output_size"]
            input_object.hyperparameter_separator = level_nhyperparameters[0][0][-1]
            return
        if level_type in [
            sorf_lvl,
            unscaled_sorf_lvl,
            mixed_extensive_sorf_lvl,
            sign_invariant_sorf_lvl,
            unscaled_sign_invariant_sorf_lvl,
        ]:
            self.init_sorf(
                input_object,
                size_parameters[-1],
                parameter_list[-1],
                sign_invariant=(
                    level_type in [sign_invariant_sorf_lvl, unscaled_sign_invariant_sorf_lvl]
                ),
            )
        elif level_type == power_rescaling_lvl:
            self.init_power_rescaling(input_object, parameter_list[-1])
        elif level_type == rescaling_lvl:
            self.init_rescaling(input_object, parameter_list[-1])
        if len(function_definition_list) == 1:
            return
        nested_kwargs = {
            "function_definition_list": function_definition_list[:-1],
            "parameter_list": parameter_list[:-1],
            "size_parameters": size_parameters[:-1],
            "level_nhyperparameters": level_nhyperparameters[:-1],
        }
        if level_type == concatenation_lvl:
            for nested_input_object in input_object.nested_input_objects:
                self.init_input_object(input_object=nested_input_object, **nested_kwargs)
        elif level_type == element_id_switch_lvl:
            num_element_ids = len(input_object.nested_input_objects)
            for el_id in range(num_element_ids):
                self.init_input_object(
                    input_object=input_object.nested_input_objects[el_id], **nested_kwargs
                )
        else:
            self.init_input_object(input_object=input_object.nested_input_object, **nested_kwargs)

    def get_sorf_variance(
        self, training_objects, full_function_definition_list, ilevel, reasonable_hyperparameters
    ):
        if self.variance_calculators is None:
            self.variance_calculators = {}
        if ilevel not in self.variance_calculators:
            self.variance_calculators[ilevel] = get_sorf_variance_calculator(
                full_function_definition_list, ilevel
            )
        if self.variance_input_objects is None:
            self.variance_input_objects = {}
        if ilevel not in self.variance_input_objects:
            self.variance_input_objects[ilevel] = get_sorf_variance_input_creator(
                full_function_definition_list, ilevel
            )(self.input_object)
        return self.variance_calculators[ilevel](
            training_objects, self.variance_input_objects[ilevel], reasonable_hyperparameters
        )

    def hyperparameter_initial_guesses(
        self,
        training_objects,
        function_definition_list=None,
        topside_layers=None,
        parameter_list=None,
    ):
        if function_definition_list is None:
            function_definition_list = self.function_definition_list
            assert parameter_list is None
            parameter_list = self.parameter_list
        reasonable_hyperparameters = None
        begins_with_fork = isinstance(function_definition_list[0], tuple)
        if begins_with_fork:
            assert len(function_definition_list[0]) == 2
            fork_parameter_list_tuple = parameter_list[0]

            for fork_id, (fork_list, fork_parameter_list) in enumerate(
                zip(
                    function_definition_list[0],
                    fork_parameter_list_tuple,
                )
            ):
                # TODO: better way to handle passed_fork?
                fork_topside_layers = [
                    "pass_variance_fork" + str(fork_id + 1)
                ] + function_definition_list[1:]
                new_reasonable_hyperparameters = self.hyperparameter_initial_guesses(
                    training_objects,
                    function_definition_list=fork_list,
                    topside_layers=fork_topside_layers,
                    parameter_list=fork_parameter_list,
                )
                reasonable_hyperparameters = concatenate_wNone_(
                    reasonable_hyperparameters, new_reasonable_hyperparameters
                )

        for ilevel, (level_type, level_parameters) in enumerate(
            zip(function_definition_list, parameter_list)
        ):
            print("Guessing hyperparameters for:", level_type, "time:", now())
            if (ilevel == 0) and begins_with_fork:
                continue
            if level_type not in requires_hyperparameters:
                continue
            full_function_definition_list = function_definition_list
            if topside_layers is None:
                full_function_definition_list = function_definition_list
            else:
                full_function_definition_list = full_function_definition_list + topside_layers
            current_dispersions = self.get_sorf_variance(
                training_objects,
                full_function_definition_list,
                ilevel,
                reasonable_hyperparameters,
            )
            new_hyperparameters = guess_hyperparameters_from_dispersions(
                current_dispersions, level_type, level_parameters
            )
            reasonable_hyperparameters = concatenate_wNone_(
                reasonable_hyperparameters, new_hyperparameters
            )
        return reasonable_hyperparameters

    def get_pickler(self):
        if self.pickler is None:
            self.init_routines()
        return self.pickler

    def __getstate__(self):
        # NOTE : added to ease up dill checking
        return self.get_pickler().getstate(self)

    def __setstate__(self, d):
        pickler = d["pickler"]
        self.__dict__ = pickler.state_dict(d)
