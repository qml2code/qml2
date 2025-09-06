# For now hyperparameter optimization can be done with steepest descent (SD) or using BOSS.
# SD was put as default because BOSS optimization tends to be more expensive (especially if you include gradients).

# NOTE K.Karan: For derivatives w.r.t. hyperparameters for the loss function, it is possible to have an algorithm
# scaling as Nderivatives*Ntrain**2*Nfeatures or Nderivatives*Ntrain*Nfeatures**2. I chose the former due to training
# sets considered in the original paper being relatively small; however, should we revisit the procedure
# for larger datasets Nderivatives*Ntrain*Nfeatures**2 should be implemented to and put behind a keyword.

# NOTE K.Karan: Aalto-BOSS recommends iterpts=int(15*dim**1.5) (see "iterpts" https://cest-group.gitlab.io/boss/manual/keywords.html#main-options),
# but if gradient is used I set it to int(15*dim**0.5) because we have dim times more observables per point.
# I didn't thoroughly test this though, the user should be careful about this parameter anyway.

# TODO K.Karan: The hyperparameter optimization code underwent several last-minute changes and might require heavy revision.

# TODO K.Karan: I started implementing "training importance multipliers" for handling datasets with uneven quantity uncertainties, but due to time constraints I didn't thoroughly
# check whether they work correctly for uncertainty evaluation.

# TODO K.Karan: steepest descent's efficiency suffers a little from Z_matrix being calculated along with derivatives at the same time.
from scipy.optimize import minimize

from ..basic_utils import ExceptionRaisingClass, display_scipy_convergence, now
from ..jit_interfaces import (
    Module_,
    any_,
    array_,
    concatenate_,
    copy_,
    default_rng_,
    dot_,
    exp_,
    inf_,
    isinf_,
    isnan_,
    l2_norm_,
    log_,
    max_,
    mean_,
    min_,
    sqrt_,
    std_,
    tiny_,
    zeros_,
)
from ..math import regression_using_Z_SVD, svd_aligned
from ..models.learning_curves import learning_curve_from_predictions
from ..models.loss_functions import MAE, SquaredSelfConsistentLogCosh
from ..models.math import f_winf, ninv_f, possible_numerical_exceptions
from ..models.property_shifts import get_shift_coeffs
from ..models.sorf_hyperparameter_optimization import (
    get_stat_factors,
    leaveoneout_eigenvalue_multipliers,
    leaveoneout_errors,
    leaveoneout_errors_from_precalc,
    leaveoneout_loss_der_wrt_features,
    leaveoneout_loss_l2reg_der,
    mult_by_importance,
)
from ..optimizers import global_optimize_1D
from ..utils import (
    check_allocation,
    flatten_to_scalar,
    get_numba_list,
    get_sorted_elements,
    l2_sq_norm,
)
from .base_constructors import datatype_prefix
from .math import (
    error_matrices_for_shifts,
    get_model_uncertainty_fitted_ratios,
    get_training_plane_sq_distance_components,
    get_training_plane_sq_distances,
)
from .pickle import Pickler
from .sorf_calculation import MultilevelSORF
from .utils import merge_or_replace, optional_array_print_tuple

try:
    from boss.bo.bo_main import BOMain
except ModuleNotFoundError:
    BOMain = ExceptionRaisingClass(
        ModuleNotFoundError(
            "Procedure requires installation of aalto-boss via:\npip install aalto-boss"
        )
    )


implemented_hyperparameters_optimizers = ["steepest_descent", "boss"]


class Z_matrix_error(Exception):
    pass


class MultilevelSORFModel(Module_):
    def __init__(
        self,
        function_definition_list,
        parameter_list,
        intensive_quantity_shift=False,
        extensive_quantity_shift=False,
        quantity_shift=None,
        hyperparameters=None,
        l2reg_diag_ratio=None,
        shift_opt_method="L-BFGS-B",
        ln_l2reg_opt_method="SLSQP",
        grad_opt_tolerance=1e-9,
        hyperparameter_optimizer="steepest_descent",
        error_loss_constructor=SquaredSelfConsistentLogCosh,
        error_loss_kwargs={},
        ln_relative_l2reg_range=[-25.0, 0.0],
        ln_relative_l2reg_opt_iterpts=32,
        boss_kernel="rbf",
        boss_ln_sigma_bounds_range=3.0,
        boss_compromise_coeff=1.25e-1,
        boss_relative_loss_convergence=1.25e-1,
        boss_cycle_initpts=1,
        boss_cycle_iterpts=None,
        boss_cycle_iterpts_coeff=16,
        boss_assert_gpu=False,
        boss_num_printed_glmin=4,
        boss_use_grad=False,
        sd_ln_steps=[0.5, 0.25, 0.125],
        sd_compromise_coeffs=[0.5, 0.25, 0.125],
        sd_reuse_Z_derivatives=True,
        rng=None,
        object_definition_list=None,
        objects_definition_list=None,
        disable_numba_parallelization=False,
    ):
        """
        Multilevel SORF model containing automated routines for training, making predictions, and hyperparameter optimization.

        Note that all hyperparameters by default are None, but can be optimized with the corresponding routines.

        Args:
            function_definition_list (list): list of str definitions of model's levels.
            parameter_list (list): list of dicts of parameters of model's levels.
            intensive_quantity_shift (bool): if True quantities are shifted by a constant optimized as a hyperparameter.
            extensive_quantity_shift (bool): if True quantities are shifted by constants, optimized as a hyperparameter, which are multiplied by numbers of atoms of a given kind.
            quantity_shift (np.array): (hyperparameter) array of quantity shifts (intensive, extensive, or both).
            hyperparameters (np.array): (hyperparameter) array of sigma hyperparameters.
            l2reg_diag_ratio (np.array): (hyperparameter) ratio of l2-regularization coefficient (l2reg) and average diagonal element of the K matrix.
            shift_opt_method (str): method scipy.optimize uses to optimize shifts. Default is L-BFGS-B.
            ln_l2reg_opt_method (str): while log(l2reg) is pre-optimized globally by BOSS, the value is finalized with scipy.optimize and the method determined by this argument. Default is "SLSQP".
            grad_opt_tolerance (float): loss gradient tolerance during gradient optimization steps with scipy.optimize (default 1.e-9).
            processed_object_constructor: if the model's input needs extra processing place processing function here (see .processed_object_constructors section).
            hyperparameter_optimizer (str): optimize hyperparameters with steepest descent ("steepest_descent") or BOSS ("boss"). Default is "steepest_descent".
            error_loss_constructor: which homogeneous loss function is used. Default is SquaredSelfConsistentLogCosh.
            error_loss_kwargs (dict): additional keyword arguments of homogeneous loss constructor.
            ln_relative_l2reg_range (list): range of values in which BOSS pre-optimizes logarithm of the ratio of l2reg and average K diagonal value. Default is `[-25.0, 0.0]`.
            ln_relative_l2reg_opt_iterpts (int): number of loss function calls during BOSS pre-optimization of l2reg. Default is 32.
            boss_kernel (str): name of kernel used by BOSS. Default is "rbf".
            boss_ln_sigma_bounds_range (float): BOSS optimization of sigmas is performed over (boss_ln_sigma_bounds_range)-neighborhood of initial guesses. Default is 3.0.
            boss_compromise_coeff (float): compromise coefficient of the loss function during BOSS sigma optimization. Default is 1.25e-1.
            boss_relative_loss_convergence (float): stop readjusting search bounds of BOSS optimization if the relative loss gain compared to the previous optimization is worse than this value (default is 1.25e-1).
            boss_cycle_initpts (int): value of initpts parameter for BOSS sigmas optimization. Default is 1.
            boss_cycle_iterpts (int): value of iterpts parameter for BOSS sigmas optimization. Default is None, meaning iterpts is calculated from boss_cycle_iterpts_coeff.
            boss_cycle_iterpts_coeff (int): if boss_cycle_iterpts=None, the iterpts value used is defined as boss_cycle_iterpts_coeff*nsigmas**power, where power==0.5 is BOSS gradient optimization is used and power==1.5 otherwise. Default is 16.
            boss_assert_gpu (bool): double-check that BOSS can use GPU. Default is False.
            boss_num_printed_glmin (int): how many predicted minima BOSS prints in the end of sigmas optimization.
            boss_use_grad (bool): whether BOSS uses gradients during optimization. Default is False.
            sd_ln_steps (list): step magnitudes used during steepest descent optimization in space of logarithms of sigmas. Default is `[0.5, 0.25, 0.125]`.
            sd_compromise_coeffs (list): compromise coefficient values used during steepest descent optimization in space of logarithms of sigmas. Default is `[0.5, 0.25, 0.125]`.
            sd_reuse_Z_derivatives (bool): if False, decrease memory usage (at the expense of CPU time) by not storing derivatives of Z matrix w.r.t. sigmas. Default is True.
            rng (np.random RNG or None): if not None, defines RNG used in random component initialization (e.g. biases and diagonal elements in SORF).
            object_definition_list (list of None): definition list of the datatype the model acts on. Only needed for pickling.
            objects_definition_list (list of None): definition list of the datatype array the model acts on (in case it's not a list of types defined by `object_definition_list`). Only needed for pickling.
            disable_numba_parallelization (bool): if True calculating the feature matrix will not be parallelized over feature vectors (needed when principle components are used in a setting where np.dot is automatically parallelized). Default is False.
        """
        self.MultilevelSORF = MultilevelSORF(
            function_definition_list,
            parameter_list,
            rng=rng,
            disable_numba_parallelization=disable_numba_parallelization,
        )
        self.nhyperparameters = self.get_nhyperparameters()

        # for shifting quantities extensively or intensively
        self.intensive_quantity_shift = intensive_quantity_shift
        self.extensive_quantity_shift = extensive_quantity_shift

        # current hyperparameter values
        # concatenation of extensive and intensive shifts.
        self.quantity_shift = quantity_shift

        self.hyperparameters = hyperparameters
        self.l2reg_diag_ratio = l2reg_diag_ratio

        # Smooth error loss is initialized if gradient-based optimization is used.
        self.error_loss_constructor = error_loss_constructor
        self.error_loss_kwargs = error_loss_kwargs
        self.error_loss_function = None
        self.error_loss = None

        # Thorough gradient optimization appears for shift optimization
        self.shift_opt_method = shift_opt_method
        self.ln_l2reg_opt_method = ln_l2reg_opt_method
        self.grad_opt_tolerance = grad_opt_tolerance

        assert hyperparameter_optimizer in implemented_hyperparameters_optimizers
        self.hyperparameter_optimizer = hyperparameter_optimizer

        self.ln_relative_l2reg_range = ln_relative_l2reg_range
        self.ln_relative_l2reg_opt_iterpts = ln_relative_l2reg_opt_iterpts

        # hyperparameter optimization options with boss
        self.boss_kernel = boss_kernel
        self.boss_ln_sigma_bounds_range = boss_ln_sigma_bounds_range
        self.boss_compromise_coeff = boss_compromise_coeff
        self.boss_relative_loss_convergence = boss_relative_loss_convergence

        self.boss_cycle_initpts = boss_cycle_initpts
        self.boss_cycle_iterpts = boss_cycle_iterpts
        self.boss_cycle_iterpts_coeff = boss_cycle_iterpts_coeff
        self.boss_assert_gpu = boss_assert_gpu
        self.boss_num_printed_glmin = boss_num_printed_glmin
        self.boss_use_grad = boss_use_grad

        if boss_cycle_iterpts is None:
            if self.boss_use_grad:
                self.boss_cycle_iterpts = int(
                    self.boss_cycle_iterpts_coeff * sqrt_(float(self.nhyperparameters))
                )
            else:
                self.boss_cycle_iterpts = int(
                    self.boss_cycle_iterpts_coeff * self.nhyperparameters**1.5
                )
        else:
            self.boss_cycle_iterpts = boss_cycle_iterpts
        # hyperparameters optimization options with steepest descent
        self.sd_ln_steps = sd_ln_steps
        self.sd_compromise_coeffs = sd_compromise_coeffs
        self.sd_reuse_Z_derivatives = sd_reuse_Z_derivatives

        self.object_definition_list = object_definition_list
        self.objects_definition_list = objects_definition_list

        self.pickler = None

        self.clear_training_set()
        self.init_hyperparameter_optimization_temp_arrs()

    def clear_training_set_temp(self):
        self.hyperparameter_initial_guesses = None
        self.l2reg_diag_ratio_initial_guess = None
        self.quantity_shift_initial_guess = None
        self.error_loss_guess = None
        self.training_weighted_shift_coeffs = None
        self.training_weighted_quantities = None

        # for calculating shifts
        self.training_set_possible_nuclear_charges = None
        # for more-or-less optimally assigning different training objects to different processes.
        self.training_thread_assignments = None
        self.training_gradient_thread_assignments = None

    def clear_training_set(self):
        self.training_quantities = None
        self.training_objects = None
        self.training_nuclear_charges = None
        self.training_importance_multipliers = None
        self.training_set_size = None
        self.training_set_Z_matrix = None
        self.clear_training_set_temp()

    def mult_by_importance(self, arr_in):
        return mult_by_importance(arr_in, self.training_importance_multipliers)

    def init_hyperparameter_optimization_temp_arrs(self):
        self.temp_Z_matrix = None

    def nfeatures(self):
        return self.MultilevelSORF.output_size()

    def check_hyperparameter_optimization_temp_arrs(self):
        print("###checking allocation of temporary arrays:", now())
        nfeatures = self.nfeatures()
        self.temp_Z_matrix = check_allocation(
            (self.training_set_size, nfeatures), output=self.temp_Z_matrix
        )
        print("###finished:", now())

    def clear_hyperparameter_optimization_temp_arrs(self):
        del self.temp_Z_matrix
        self.init_hyperparameter_optimization_temp_arrs()

    def calc_cpu_estimates(self, input_list, gradient=False):
        return self.MultilevelSORF.calc_cpu_estimates(input_list, gradient=gradient)

    def calc_thread_assignments(self, input_list, gradient=False):
        return self.MultilevelSORF.calc_thread_assignments(input_list, gradient=gradient)

    def calc_thread_assignments_cpu_loads(
        self, input_list, gradient=False, return_list_cpu_loads=False
    ):
        return self.MultilevelSORF.calc_thread_assignments_cpu_loads(
            input_list, gradient=gradient, return_list_cpu_loads=return_list_cpu_loads
        )

    def init_training_thread_assignments(self):
        (
            self.training_thread_assignments,
            cpu_loads,
            list_cpu_loads,
        ) = self.calc_thread_assignments_cpu_loads(
            self.training_objects, return_list_cpu_loads=True
        )
        (
            self.training_gradient_thread_assignments,
            grad_cpu_loads,
            grad_list_cpu_loads,
        ) = self.calc_thread_assignments_cpu_loads(
            self.training_objects, gradient=True, return_list_cpu_loads=True
        )
        for str_descr, cur_cpu_loads, cur_list_cpu_loads in [
            ("", cpu_loads, list_cpu_loads),
            (" (grad)", grad_cpu_loads, grad_list_cpu_loads),
        ]:
            for other_str_descr, arr in [
                ("job CPU times", cur_list_cpu_loads),
                ("process CPU times", cur_cpu_loads),
            ]:
                print(
                    "###estimated "
                    + other_str_descr
                    + " times mean/min/max/std"
                    + str_descr
                    + ":",
                    mean_(arr),
                    "/",
                    min_(arr),
                    "/",
                    max_(arr),
                    "/",
                    std_(arr),
                )

    def assign_training_set(
        self,
        training_objects,
        training_quantities,
        training_nuclear_charges=None,
        training_importance_multipliers=None,
        init_thread_assignments=False,
    ):
        """
        Assign training set used in hyperparameter optimization.

        Args:
            training_objects (numba.List): Numba list of processed training set objects (i.e. objects generated with a routine in `.processed_object_constructors`).
            training_quantities (numpy.array): array of training set quantities of interest.
            training_nuclear_charges (list): list of numpy arrays of training set molecules' nuclear charges; used if extensive_quantity_shift is True. Default is None.
            training_importance_multipliers (numpy.array): weights for individual points during regression (not properly untested).
            init_thread_assignments (bool): if True generate estimated CPU times for feature calculations with and without gradients, which are then used to optimize parallelization performance.
        """
        self.clear_training_set_temp()
        self.training_objects = training_objects
        self.training_nuclear_charges = training_nuclear_charges
        self.training_importance_multipliers = training_importance_multipliers
        self.training_quantities = training_quantities
        self.training_set_size = len(self.training_objects)
        self.training_weighted_quantities = self.mult_by_importance(self.training_quantities)
        assert self.training_weighted_quantities.shape[0] == self.training_set_size
        if init_thread_assignments:
            self.init_training_thread_assignments()

    def calc_Z_matrix(
        self,
        query_objects,
        hyperparameters=None,
        temp_Z_matrix=None,
        gradient=False,
        thread_assignments=None,
    ):
        """
        Calculate features for query objects.

        Args:
            query_objects (numba.List): Numba list of processed objects (i.e. objects generated with a routine in `.processed_object_constructors`).
            hyperparameters (numpy.array): numpy.array of sigmas. If None (which is default) use self.hyperparameters.
            temp_Z_matrix (numpy.array or None): if not None, calculated `Z_matrix` will be stored in this array.
            gradient (bool): if True calculate derivatives of features w.r.t. hyperparameters. Default is False.
            thread_assignments (numpy.array or None): if not None (which is default), use this numpy.array of estimates of CPU times needed to calculate an object's features to optimize OpenMP parallelization

        Returns:
            if (gradient): tuple of `(Z_matrix, Z_matrix_derivatives)`, where `Z_matrix` is the `(len(query_objects),nfeatures)` feature matrix and `Z_matrix_derivatives` are derivatives of the feature matrix w.r.t. hyperparameters `(len(query_objects),len(hyperparameters),nfeatures)`.
            else: `Z_matrix`.
        """
        if hyperparameters is None:
            hyperparameters = self.hyperparameters
        return self.MultilevelSORF.calc_Z_matrix(
            query_objects,
            hyperparameters,
            gradient=gradient,
            temp_Z_matrix=temp_Z_matrix,
            thread_assignments=thread_assignments,
        )

    def calc_z_vector(self, query_object, hyperparameters=None):
        """
        Calculate features for a single query object.

        Args:
            query_object (custom Numba datatype): processed query object (i.e. generated with a routine in `.processed_object_constructors`)
            hyperparameters (numpy.array or None): numpy.array of sigmas. If None (which is default) use self.hyperparameters.

        Returns:
            `z_vector` (numpy.array): feature vector.
        """
        if hyperparameters is None:
            hyperparameters = self.hyperparameters
        return self.MultilevelSORF.calc_features(query_object, hyperparameters)

    def calc_training_Z_matrix(self, hyperparameters=None, gradient=False):
        if gradient:
            thread_assignments = self.training_gradient_thread_assignments
        else:
            thread_assignments = self.training_thread_assignments
        return self.calc_Z_matrix(
            self.training_objects,
            hyperparameters=hyperparameters,
            temp_Z_matrix=self.temp_Z_matrix,
            thread_assignments=thread_assignments,
            gradient=gradient,
        )

    def calc_training_weighted_Z_matrix(self, hyperparameters=None):
        return self.mult_by_importance(
            self.calc_training_Z_matrix(
                hyperparameters=hyperparameters,
            )
        )

    def get_shift_coeffs(self, npoints, all_nuclear_charges=None):
        if all_nuclear_charges is not None:
            all_nuclear_charges = get_numba_list(all_nuclear_charges)
        return get_shift_coeffs(
            npoints,
            intensive_shift=self.intensive_quantity_shift,
            extensive_shift=self.extensive_quantity_shift,
            possible_nuclear_charges=self.training_set_possible_nuclear_charges,
            all_nuclear_charges=all_nuclear_charges,
        )

    def init_training_set_possible_nuclear_charges(self):
        if (self.training_set_possible_nuclear_charges is not None) or (
            not self.extensive_quantity_shift
        ):
            return
        self.training_set_possible_nuclear_charges = get_sorted_elements(
            concatenate_(self.training_nuclear_charges)
        )
        print(
            "###Nuclear charges found in training set:", self.training_set_possible_nuclear_charges
        )

    def init_training_weighted_shift_coeffs(self):
        self.init_training_set_possible_nuclear_charges()
        if self.training_weighted_shift_coeffs is not None:
            return
        self.training_shift_coeffs = self.get_shift_coeffs(
            len(self.training_objects),
            all_nuclear_charges=self.training_nuclear_charges,
        )
        self.training_weighted_shift_coeffs = self.mult_by_importance(self.training_shift_coeffs)

    def get_shifted_quantities(
        self,
        quantities=None,
        nuclear_charges=None,
        quantity_shift_coeffs=None,
    ):
        if self.extensive_quantity_shift or self.intensive_quantity_shift:
            if quantity_shift_coeffs is None:
                quantity_shift_coeffs = self.get_shift_coeffs(
                    len(quantities), all_nuclear_charges=nuclear_charges
                )
            shifted_quantities = quantities + dot_(quantity_shift_coeffs, self.quantity_shift)
        else:
            shifted_quantities = copy_(quantities)
        return shifted_quantities

    def get_shifted_weighted_quantities(self, importance_multipliers=None, **kwargs):
        shifted_quantities = self.get_shifted_quantities(**kwargs)
        return mult_by_importance(shifted_quantities, importance_multipliers)

    def get_shifted_training_quantities(self, quantity_shift_coeffs=None):
        if (quantity_shift_coeffs is None) and (
            self.extensive_quantity_shift or self.intensive_quantity_shift
        ):
            quantity_shift_coeffs = self.training_shift_coeffs
        return self.get_shifted_quantities(
            quantities=self.training_quantities,
            nuclear_charges=self.training_nuclear_charges,
            quantity_shift_coeffs=quantity_shift_coeffs,
        )

    def get_shifted_weighted_training_quantities(self, quantity_shift_coeffs=None):
        shifted_quantities = self.get_shifted_training_quantities(
            quantity_shift_coeffs=quantity_shift_coeffs
        )
        return mult_by_importance(shifted_quantities, self.training_importance_multipliers)

    def get_initial_hyperparameter_guesses(self, training_objects=None):
        if training_objects is None:
            training_objects = self.training_objects
        return self.MultilevelSORF.hyperparameter_initial_guesses(training_objects)

    def get_init_ln_hyperparameters(self):
        return log_(self.hyperparameter_initial_guesses)

    def find_hyperparameter_guesses(self, training_objects=None):
        """
        Calculate initial guesses of hyperparameters (sigmas) based on input or the currently assigned training set and store them in self.hyperparameter_initial_guesses.

        Args:
            training_objects (numba.typed.List or None): objects based on which hyperparameter guesses are calculated; if `None` self.training_objects is used.

        Returns:
            hyperparameter_guesses (numpy.array): (also stored in self.hyperparameter_initial_guesses) a `nhyperparameters` array of sigma guesses.
        """
        if training_objects is None:
            training_objects = self.training_objects

        print("###Calculating hyperparameter guesses:", now())
        self.hyperparameter_initial_guesses = self.get_initial_hyperparameter_guesses(
            training_objects
        )
        print("###Ended calculating hyperparameter guesses", now())
        return self.hyperparameter_initial_guesses

    def copy_initial_guess_to_hyperparameters(self):
        self.l2reg = copy_(self.l2reg_initial_guess)
        self.l2reg_diag_ratio = copy_(self.l2reg_diag_ratio_initial_guess)
        self.hyperparameters = copy_(self.hyperparameter_initial_guesses)
        self.quantity_shift = copy_(self.quantity_shift_initial_guess)
        self.alphas = copy_(self.alphas_initial_guess)
        self.leaveoneout_errors = copy_(self.leaveoneout_errors_guess)
        self.error_loss = copy_(self.error_loss_guess)

    def copy_hyperparameters_to_initial_guess(self, new_error_loss_guess=None):
        self.l2reg_initial_guess = copy_(self.l2reg)
        self.l2reg_diag_ratio_initial_guess = copy_(self.l2reg_diag_ratio)
        self.hyperparameter_initial_guesses = copy_(self.hyperparameters)
        self.quantity_shift_initial_guess = copy_(self.quantity_shift)
        self.alphas_initial_guess = copy_(self.alphas)
        self.leaveoneout_errors_guess = copy_(self.leaveoneout_errors)
        if new_error_loss_guess is None:
            self.error_loss_guess = copy_(self.error_loss)
        else:
            self.error_loss_guess = new_error_loss_guess

    def init_default_error_loss(self, compromise_coefficient=None):
        if compromise_coefficient is None:
            if self.hyperparameter_optimizer == "boss":
                compromise_coefficient = self.boss_compromise_coeff
            else:
                compromise_coefficient = self.sd_compromise_coeffs[-1]
        self.error_loss_function = self.error_loss_constructor(compromise_coefficient)

    def find_initial_guesses(self, compromise_coefficient=None):
        assert self.training_objects is not None
        # sigmas
        self.find_hyperparameter_guesses()
        self.init_default_error_loss(compromise_coefficient=compromise_coefficient)
        self.error_loss = self.get_loss_function_ln_hyperparameters(
            self.get_init_ln_hyperparameters()
        )
        self.copy_hyperparameters_to_initial_guess()
        print("###Ended:", now())

    def optimize_quantity_shifts(
        self,
    ):
        """
        Optimize quantity shifts as part of loss function calculation for given sigma values.
        """
        if not (self.extensive_quantity_shift or self.intensive_quantity_shift):
            return
        print("###optimizing quantity shifts:", now())
        self.init_training_weighted_shift_coeffs()
        shift_A_mat, shift_b = error_matrices_for_shifts(
            self.training_weighted_shift_coeffs,
            self.training_weighted_quantities,
            self.Z_U,
            self.eigenvalue_multipliers,
            self.stat_factors,
            self.transformed_inv_K_Z,
        )
        self.quantity_shift = self.error_loss_function.find_minimum_linear_errors(
            shift_A_mat,
            shift_b,
            method=self.shift_opt_method,
            tol=self.grad_opt_tolerance,
            initial_guess=self.quantity_shift_initial_guess,
        )
        print("###finished:", now())

    def av_K_diag_element(self, Z_singular_values=None, Z_matrix=None, training_set_size=None):
        if Z_matrix is None:
            if Z_singular_values is None:
                magn_Z = self.Z_singular_values
            else:
                magn_Z = Z_singular_values
            if training_set_size is None:
                training_set_size = self.training_set_size
            Z_shape = (training_set_size, self.nfeatures())
        else:
            magn_Z = Z_matrix
            Z_shape = Z_matrix.shape
        return l2_sq_norm(magn_Z) / min(Z_shape)

    def get_loss_function_ln_l2reg(self, ln_l2reg, gradient=False, training_set_size=None):
        ln_l2reg = flatten_to_scalar(ln_l2reg)  # for BOSS usage
        self.l2reg = exp_(ln_l2reg)
        self.l2reg_diag_ratio = self.l2reg / self.av_K_diag_element(
            training_set_size=training_set_size
        )
        self.eigenvalue_multipliers = leaveoneout_eigenvalue_multipliers(
            self.Z_singular_values, self.l2reg
        )
        self.stat_factors, self.transformed_inv_K_Z = get_stat_factors(
            self.Z_U, self.eigenvalue_multipliers
        )
        self.optimize_quantity_shifts()
        self.shifted_training_weighted_quantities = self.get_shifted_weighted_training_quantities()
        self.transformed_alphas_rhs = dot_(self.shifted_training_weighted_quantities, self.Z_U)
        self.reproduced_quantities = dot_(self.transformed_inv_K_Z, self.transformed_alphas_rhs)
        self.alphas = dot_(
            self.Z_Vh.T,
            self.transformed_alphas_rhs * self.eigenvalue_multipliers / self.Z_singular_values,
        )

        self.leaveoneout_errors = leaveoneout_errors_from_precalc(
            self.reproduced_quantities,
            self.shifted_training_weighted_quantities,
            self.stat_factors,
        )

        if not gradient:
            self.error_loss = self.error_loss_function(self.leaveoneout_errors)
            print(
                "###ln_l2reg/val/shifts:",
                ln_l2reg,
                self.error_loss,
                "/",
                *optional_array_print_tuple(self.quantity_shift),
            )
            return self.error_loss

        self.error_loss, self.error_loss_error_ders = self.error_loss_function.calc_wders(
            self.leaveoneout_errors
        )

        loss_grad = leaveoneout_loss_l2reg_der(
            self.eigenvalue_multipliers,
            self.Z_singular_values,
            self.error_loss_error_ders,
            self.Z_U,
            self.reproduced_quantities,
            self.transformed_alphas_rhs,
            self.stat_factors,
            self.shifted_training_weighted_quantities,
        )
        # convert to ln_l2reg derivative
        loss_grad *= self.l2reg
        print(
            "###ln_l2reg/val/shifts/grad:",
            ln_l2reg,
            self.error_loss,
            "/",
            *optional_array_print_tuple(self.quantity_shift),
            "/",
            loss_grad,
        )
        return self.error_loss, loss_grad

    def get_loss_function_ln_l2reg_wder(self, ln_l2reg, training_set_size=None):
        return self.get_loss_function_ln_l2reg(
            ln_l2reg, gradient=True, training_set_size=training_set_size
        )

    def init_Z_matrix_decomposition(
        self, weighted_Z_matrix=None, Z_U=None, Z_singular_values=None, Z_Vh=None
    ):
        if (Z_U is not None) and (Z_singular_values is not None) and (Z_Vh is not None):
            self.Z_U = Z_U
            self.Z_singular_values = Z_singular_values
            self.Z_Vh = Z_Vh
            return
        # Find SVD.
        print("###finding SVD:", now())
        assert weighted_Z_matrix is not None
        self.Z_U, self.Z_singular_values, self.Z_Vh = svd_aligned(weighted_Z_matrix)

    def optimize_quantity_shifts_l2reg(
        self, weighted_Z_matrix=None, Z_U=None, Z_singular_values=None, Z_Vh=None
    ):
        """
        For a given Z_matrix and assigned training set optimize l2reg and quantity shifts.

        Args:
            weighted_Z_matrix (numpy.array): feature matrix multiplied by importance multipliers.
            Z_U, Z_singular_values, Z_Vh (numpy.array): output of svd_aligned(weighted_Z_matrix).
        """
        # Calculate Z-matrix.
        if (weighted_Z_matrix is None) and (
            (Z_U is None) or (Z_singular_values is None) or (Z_Vh is None)
        ):
            print("###calculating Z matrix:", now())
            weighted_Z_matrix = self.calc_training_weighted_Z_matrix(self.hyperparameters)
            if any_(isinf_(weighted_Z_matrix)) or any_(isnan_(weighted_Z_matrix)):
                raise Z_matrix_error
        self.init_Z_matrix_decomposition(
            weighted_Z_matrix=weighted_Z_matrix,
            Z_U=Z_U,
            Z_singular_values=Z_singular_values,
            Z_Vh=Z_Vh,
        )
        print("###optimizing l2reg", now())
        # use BOSS to rougly locate the minimum.
        if weighted_Z_matrix is None:
            default_training_set_size = None
        else:
            default_training_set_size = weighted_Z_matrix.shape[0]
        loss_function_ln_l2reg_kwargs = {"training_set_size": default_training_set_size}
        ln_l2reg_range = self.ln_relative_l2reg_range + log_(
            self.av_K_diag_element(training_set_size=default_training_set_size)
        )
        ln_l2reg_boss_min = global_optimize_1D(
            ninv_f(self.get_loss_function_ln_l2reg, **loss_function_ln_l2reg_kwargs),
            ln_l2reg_range,
            total_iterpts=self.ln_relative_l2reg_opt_iterpts,
            opt_name="l2reg_opt",
        )
        # Finalize minima location with gradient optimization.
        go = minimize(
            f_winf(
                self.get_loss_function_ln_l2reg_wder,
                gradient=True,
                **loss_function_ln_l2reg_kwargs,
            ),
            ln_l2reg_boss_min,
            method=self.ln_l2reg_opt_method,
            jac=True,
            options={"disp": display_scipy_convergence},
            tol=self.grad_opt_tolerance,
        )
        ln_l2reg_final_min = go.x[0]
        # finalize calculations
        try:
            self.get_loss_function_ln_l2reg(ln_l2reg_final_min, **loss_function_ln_l2reg_kwargs)
        except possible_numerical_exceptions:
            raise Z_matrix_error
        print("###optimal ln_l2reg:", ln_l2reg_final_min)
        print(
            "###optimal l2reg_diag_ratio (abs, log):",
            self.l2reg_diag_ratio,
            log_(self.l2reg_diag_ratio),
        )

    def init_hyperparams_from_ln(self, ln_hyperparameters):
        if len(ln_hyperparameters.shape) > 1:
            # optimization is done by BOSS
            ln_hyperparameters = ln_hyperparameters[0]
        assert ln_hyperparameters.shape == (self.nhyperparameters,)
        self.hyperparameters = exp_(ln_hyperparameters)

    def get_loss_function_ln_hyperparameters_invalid_value(
        self, gradient=False, return_Z_matrices=False
    ):
        if gradient:
            output = (inf_, zeros_(self.hyperparameters.shape))
        else:
            output = inf_
        if return_Z_matrices:
            if not gradient:
                output = (output,)
            return (*output, {})
        else:
            return output

    def get_loss_function_ln_hyperparameters(
        self,
        ln_hyperparameters,
        gradient=False,
        loss_upper_bound=None,
        return_Z_matrices=False,
        Z_matrix_derivatives=None,
        Z_U=None,
        Z_singular_values=None,
        Z_Vh=None,
    ):
        """
        Calculate loss function as a function of logarithms of hyperparameters.

        Args:
            ln_hyperparameters (numpy.array): array of logarithms of hyperparameters.
            gradient (bool): whether to calculate gradient. Default is False.
            loss_upper_bound (float of None): if not None (default), abort calculation if the loss function is larger than loss_upper_bound (saves some CPU time during steepest descent optimization).
            return_Z_matrices (bool): if True (default is False), additionally return intermediate data related to features.
            Z_matrix_derivatives (numpy.array or None): if not None - numpy.array of feature derivatives used in the calculation.
            Z_U, Z_singular_values, Z_Vh (np.array or None): output of `svd_aligned` called on the feature matrix.

        Returns:
            if (gradient): return `(loss, loss_gradient)`, where `loss_gradient` is `(nhyperparameters,)` numpy.array of derivatives of loss w.r.t. logarithms of hyperparameters.
            else: return loss.
            If `return_Z_matrices`, a dictionary is added in the end of the output tuple, containing the used `Z_U`, 'Z_singular_values', `Z_Vh`, and (if `self.sd_reuse_Z_derivatives`) `Z_matrix_derivatives`.
        """
        print("###Calculating loss function for:", ln_hyperparameters, now())
        if len(ln_hyperparameters.shape) == 2:
            ln_hyperparameters = ln_hyperparameters[0]
        self.init_hyperparams_from_ln(ln_hyperparameters)
        if any_(isinf_(self.hyperparameters)):
            return self.get_loss_function_ln_hyperparameters_invalid_value(
                gradient=gradient, return_Z_matrices=return_Z_matrices
            )
        try:
            self.optimize_quantity_shifts_l2reg(
                Z_U=Z_U, Z_singular_values=Z_singular_values, Z_Vh=Z_Vh
            )
        except Z_matrix_error:
            return self.get_loss_function_ln_hyperparameters_invalid_value(
                gradient=gradient, return_Z_matrices=return_Z_matrices
            )
        if not gradient:
            return self.error_loss
        if (loss_upper_bound is not None) and (loss_upper_bound < self.error_loss):
            # we do not care about calculating the gradient
            return self.get_loss_function_ln_hyperparameters_invalid_value(
                gradient=gradient, return_Z_matrices=return_Z_matrices
            )
        mult_transformed_alphas_rhs = self.transformed_alphas_rhs * self.eigenvalue_multipliers
        if return_Z_matrices and self.sd_reuse_Z_derivatives:
            if Z_matrix_derivatives is None:
                print("###calculating Z matrix with gradients:", now())
                _, Z_matrix_derivatives = self.calc_training_Z_matrix(gradient=True)
            print("###calling loss gradient w.r.t. hyperparameters calculator:", now())
            error_loss_hyperparameter_grad = leaveoneout_loss_der_wrt_features(
                Z_matrix_derivatives,
                self.Z_U,
                self.Z_Vh,
                self.Z_singular_values,
                self.error_loss_error_ders,
                mult_transformed_alphas_rhs,
                self.reproduced_quantities,
                self.stat_factors,
                self.transformed_inv_K_Z,
                self.shifted_training_weighted_quantities,
                self.training_importance_multipliers,
            )
        else:
            print("###calling loss gradient w.r.t. hyperparameters calculator:", now())
            error_loss_hyperparameter_grad = (
                self.MultilevelSORF.calc_loss_function_hyperparameter_grad(
                    self.training_objects,
                    self.hyperparameters,
                    self.Z_U,
                    self.Z_Vh,
                    self.Z_singular_values,
                    self.error_loss_error_ders,
                    mult_transformed_alphas_rhs,
                    self.reproduced_quantities,
                    self.stat_factors,
                    self.transformed_inv_K_Z,
                    self.shifted_training_weighted_quantities,
                    self.training_importance_multipliers,
                    thread_assignments=self.training_gradient_thread_assignments,
                )
            )
        # transform to derivatives w.r.t. ln
        error_loss_hyperparameter_grad *= self.hyperparameters

        print("###ended loss_function_wders calculation:", self.error_loss, now())
        print("###gradient:", *error_loss_hyperparameter_grad)
        output = (self.error_loss, error_loss_hyperparameter_grad)
        if return_Z_matrices:
            Z_data = {
                "Z_U": self.Z_U,
                "Z_singular_values": self.Z_singular_values,
                "Z_Vh": self.Z_Vh,
            }
            if self.sd_reuse_Z_derivatives:
                Z_data["Z_matrix_derivatives"] = Z_matrix_derivatives
            output = (*output, Z_data)

        return output

    def get_loss_function_ln_hyperparameters_wders(self, ln_hyperparameters, **kwargs):
        """
        Shorthand for self.get_loss_function_ln_hyperparameters(ln_hyperparameters, gradient=True, **kwargs).
        """
        return self.get_loss_function_ln_hyperparameters(
            ln_hyperparameters, gradient=True, **kwargs
        )

    def get_ninv_loss_function_wders(self, ln_hyperparameters):
        """
        Negative inverse of the loss function (-1/loss) with derivatives.
        """
        return ninv_f(self.get_loss_function_ln_hyperparameters_wders(ln_hyperparameters))

    def get_ninv_loss_function(self, ln_hyperparameters):
        """
        Negative inverse of the loss function (-1/loss).
        """
        return ninv_f(self.get_loss_function_ln_hyperparameters(ln_hyperparameters))

    def run_boss_hyperparameter_optimization_cycle(self):
        self.check_hyperparameter_optimization_temp_arrs()
        if self.boss_assert_gpu:
            # K.Karan: introduced because I found making sure that GPyTorch (and by extension BOSS)
            # uses GPU annoying.
            import torch as tc

            assert tc.cuda.is_available()
            tc.set_default_device("cuda")

        init_ln_hyperparameters = self.get_init_ln_hyperparameters()

        bounds = [
            [ln_hyp - self.boss_ln_sigma_bounds_range, ln_hyp + self.boss_ln_sigma_bounds_range]
            for ln_hyp in init_ln_hyperparameters
        ]

        if self.boss_num_printed_glmin == 0:
            minfreq = 0
        else:
            minfreq = self.boss_cycle_iterpts // self.boss_num_printed_glmin

        if self.boss_use_grad:
            optimized_loss = self.get_ninv_loss_function_wders
            extra_kwargs = {"model_name": "grad"}
        else:
            optimized_loss = self.get_ninv_loss_function
            extra_kwargs = {}

        print("###BOSS hyperparameter optimization:", now())
        bo = BOMain(
            optimized_loss,
            bounds,
            kernel=self.boss_kernel,
            initpts=self.boss_cycle_initpts,
            iterpts=self.boss_cycle_iterpts,
            minfreq=minfreq,
            **extra_kwargs,
        )
        res = bo.run()
        ln_hyperparameters_min = res.select("x_glmin", -1)

        print("###<<<global minima predictions by BOSS:")
        for i_step, glmin_pred in res.select("mu_glmin").items():
            print("###<<<", i_step, glmin_pred)

        print("###ended BOSS hyperparameter optimization:", now())
        print("###found hyperparameter logarithms at minimum:", ln_hyperparameters_min)
        self.init_hyperparams_from_ln(ln_hyperparameters_min)
        self.optimize_quantity_shifts_l2reg()
        self.clear_hyperparameter_optimization_temp_arrs()

    def hyperparameter_optimization_start(self, compromise_coefficient):
        self.find_initial_guesses(compromise_coefficient)
        print("###Starting reference loss:", self.error_loss_guess)
        print("###Starting l2reg_diag_ratio:", self.l2reg_diag_ratio_initial_guess)
        print("###Starting hyperparameters:", self.hyperparameter_initial_guesses)

    def hyperparameter_optimization_final_print(self, insert_str=None):
        if insert_str is None:
            ending_str = ":"
        else:
            ending_str = " (" + insert_str + "):"
        print("###Final l2reg_diag_ratio" + ending_str, self.l2reg_diag_ratio)
        print("###Final l2reg" + ending_str, self.l2reg)
        print("###Final hyperparameters" + ending_str, self.hyperparameters)
        print("###Final shifts" + ending_str, *optional_array_print_tuple(self.quantity_shift))
        print("###Final loss" + ending_str, self.error_loss)

    def optimize_hyperparameters_boss(self, initial_hyperparameter_guess=None):
        self.error_loss_function = self.error_loss_constructor(
            self.boss_compromise_coeff, **self.error_loss_kwargs
        )
        self.hyperparameter_optimization_start(self.boss_compromise_coeff)
        while True:
            self.run_boss_hyperparameter_optimization_cycle()
            # Use the current hyperparameters to calculate error loss function.
            print("###New loss:", self.error_loss)
            if self.error_loss >= self.error_loss_guess:
                print(
                    "###Optimization run produced larger loss function; something may've been wrong with BOSS. Reverting results"
                )
                self.copy_initial_guess_to_hyperparameters()
                break
            converged_wrt_boundaries = (
                (self.error_loss_guess - self.error_loss) / self.error_loss
            ) < self.boss_relative_loss_convergence
            if converged_wrt_boundaries:
                break
            self.copy_hyperparameters_to_initial_guess()

        self.hyperparameter_optimization_final_print()

    def run_sd_hyperparameter_optimization_cycle(self, starting_Z_quantities={}, ln_steps=None):
        """
        Run steepest descent hyperparameter optimization for the currently initialized error loss function. The optimized hyperparameters are stored in the corresponding attributes of self.

        Args:
            starting_Z_quantities (dict): intermediate quantities used in `self.get_loss_function_ln_hyperparameters` for the optimization's starting point.
            ln_steps (list or None): list of steps w.r.t. hyperparameter logarithms used during optimization.

        Returns:
            Tuple (loss, Z_quantities), same as output of self.get_loss_function_ln_hyperparameters with gradient=False and return_Z_matrices=True and ln_hyperparameters taken to be the optimization result.
        """
        if ln_steps is None:
            ln_steps = self.sd_ln_steps
        ln_hyperparameters = self.get_init_ln_hyperparameters()
        previous_loss = None
        # next line is redundant, introduced to avoid confusing autoformatter
        prev_Z_quantities = None
        prev_loss_grad = None
        ln_step_id = 0
        ln_step = ln_steps[ln_step_id]
        print("###ln_step:", ln_step)
        while True:
            loss, loss_grad, new_Z_quantities = self.get_loss_function_ln_hyperparameters_wders(
                ln_hyperparameters,
                loss_upper_bound=previous_loss,
                return_Z_matrices=True,
                **starting_Z_quantities,
            )
            starting_Z_quantities = {}
            if (previous_loss is not None) and (loss >= previous_loss):
                ln_step_id += 1
                if ln_step_id == len(ln_steps):
                    self.copy_initial_guess_to_hyperparameters()
                    return previous_loss, prev_Z_quantities
                ln_step = ln_steps[ln_step_id]
                print("###ln_step:", ln_step)
                # backtrack
                ln_hyperparameters += prev_loss_grad * (ln_steps[ln_step_id - 1] - ln_step)
                continue
            loss_grad_norm = l2_norm_(loss_grad)
            self.copy_hyperparameters_to_initial_guess()
            if loss_grad_norm <= tiny_:
                return loss, new_Z_quantities
            previous_loss = loss
            prev_Z_quantities = new_Z_quantities
            loss_grad /= loss_grad_norm
            ln_hyperparameters -= loss_grad * ln_step
            prev_loss_grad = loss_grad

    def optimize_hyperparameters_sd_single_perturbation(
        self, initial_guess_perturbation=None, initial_hyperparameter_guess=None
    ):
        print("###Steepest descent gradient optimization started.")
        self.hyperparameter_optimization_start(self.sd_compromise_coeffs[0])
        if initial_hyperparameter_guess is not None:
            self.hyperparameter_initial_guesses = array_(initial_hyperparameter_guess)
        if initial_guess_perturbation is not None:
            self.hyperparameter_initial_guesses *= exp_(initial_guess_perturbation)
        saved_Z_quantities = {}
        for compromise_coeff in self.sd_compromise_coeffs:
            self.error_loss_function = self.error_loss_constructor(compromise_coeff)
            print("###Compromise coefficient:", compromise_coeff)
            _, saved_Z_quantities = self.run_sd_hyperparameter_optimization_cycle(
                starting_Z_quantities=saved_Z_quantities
            )
        self.hyperparameter_optimization_final_print()
        print("###Final MAE:", MAE()(self.leaveoneout_errors))
        return self.error_loss

    def get_nhyperparameters(self):
        return self.MultilevelSORF.nhyperparameters()

    def optimize_hyperparameters_sd(
        self, initial_guess_perturbations=None, initial_hyperparameter_guess=None
    ):
        if initial_guess_perturbations is None:
            initial_guess_perturbations = zeros_((1, self.get_nhyperparameters()))

        min_loss = None
        for initial_guess_perturbation in initial_guess_perturbations:
            self.optimize_hyperparameters_sd_single_perturbation(
                initial_hyperparameter_guess=initial_hyperparameter_guess,
                initial_guess_perturbation=initial_guess_perturbation,
            )
            if (min_loss is not None) and (min_loss <= self.error_loss):
                continue
            min_loss = copy_(self.error_loss)
            min_hyperparameters = copy_(self.hyperparameters)
            min_l2reg_diag_ratio = copy_(self.l2reg_diag_ratio)
            min_l2reg = copy_(self.l2reg)
            min_quantity_shift = copy_(self.quantity_shift)
            min_alphas = copy_(self.alphas)
            min_leaveoneout_errors = copy_(self.leaveoneout_errors)
        self.error_loss = min_loss
        self.hyperparameters = min_hyperparameters
        self.l2reg_diag_ratio = min_l2reg_diag_ratio
        self.l2reg = min_l2reg
        self.quantity_shift = min_quantity_shift
        self.alphas = min_alphas
        self.leaveoneout_errors = min_leaveoneout_errors
        self.hyperparameter_optimization_final_print("all perturbations")

    def optimize_hyperparameters(
        self,
        hyperparameter_optimizer=None,
        initial_guess_perturbations=None,
        initial_hyperparameter_guess=None,
    ):
        """
        Optimize hyperpareters with the currently assigned training set.

        Args:
            hyperparameter_optimizer (str or None): whether use BOSS ("boss") or steepest descent ("steepest_descent") for hyperparameter optimization. If None (default), use `self.hyperparameter_optimizer` (defined during `__init__`).
            initial_guess_perturbations (numpy.array or None): if not None (default) and steepest descent is used, run `npert` optimization runs where the initial guess of hyperparameter logarithms is pertubed by `initial_guess_perturbations[i,:]` `(i=0,...,npert-1)`.
        """
        if hyperparameter_optimizer is None:
            hyperparameter_optimizer = self.hyperparameter_optimizer
        if hyperparameter_optimizer == "boss":
            return self.optimize_hyperparameters_boss(
                initial_hyperparameter_guess=initial_hyperparameter_guess
            )
        if hyperparameter_optimizer == "steepest_descent":
            return self.optimize_hyperparameters_sd(
                initial_hyperparameter_guess=initial_hyperparameter_guess,
                initial_guess_perturbations=initial_guess_perturbations,
            )
        raise Exception

    def get_leaveoneout_errors(self):
        """
        Get leaveoneout errors for currently assigned training sets. Mainly used for hyperparameter optimization verification.

        Returns:
            loss (float): leaveoneout error loss.
        """
        quantities = self.get_shifted_weighted_training_quantities()
        Z_matrix = self.calc_training_weighted_Z_matrix()
        return leaveoneout_errors(Z_matrix, quantities, self.l2reg)

    def initialize_uncertainty_components(
        self,
        Z_matrix,
        Z_U=None,
        Z_singular_values=None,
        Z_Vh=None,
        l2reg=None,
        training_quantities=None,
        alphas=None,
    ):
        if l2reg is None:
            l2reg = self.l2reg
        if alphas is None:
            alphas = self.alphas
        if Z_singular_values is None:
            Z_U, Z_singular_values, Z_Vh = svd_aligned(Z_matrix)
        if training_quantities is None:
            training_quantities = self.get_shifted_weighted_training_quantities()
        (
            self.train_kernel_eigenvalue_multipliers,
            self.train_kernel_eigenvectors,
        ) = get_training_plane_sq_distance_components(
            None, l2reg, Z_singular_values=Z_singular_values, Z_Vh=Z_Vh
        )

        eigenvalue_multipliers = leaveoneout_eigenvalue_multipliers(Z_singular_values, l2reg)
        stat_factors, _ = get_stat_factors(Z_U, eigenvalue_multipliers)
        uncertainty_ratios, uncertainty_sq_distances = get_model_uncertainty_fitted_ratios(
            Z_matrix,
            alphas,
            training_quantities,
            self.train_kernel_eigenvalue_multipliers,
            self.train_kernel_eigenvectors,
        )
        return uncertainty_ratios, uncertainty_sq_distances * stat_factors

    def fit(
        self,
        training_objects,
        training_quantities,
        training_nuclear_charges=None,
        training_importance_multipliers=None,
        hyperparameters=None,
        l2reg_diag_ratio=None,
        l2reg=None,
        initialize_training_plane_sq_distance=False,
        Z_matrix=None,
        optimize_shifts_l2reg=False,
        save_Z_matrix=False,
    ):
        """
        Fit the model using training data.

        Args:
            training_objects (numba.List): a numba list of processed training set objects.
            training_quantities (numpy.array): numpy array of training set quantities of interest.
            training_nuclear_charges (list or None): if not None (default) is used to calculate shifts if `self.extensive_quantity_shifts=True`.
            training_importance_multipliers (numpy.array or None): if not None (default) defines importance multipliers used during the fit.
            hyperparameters (numpy.array or None): if not None defines hyperparameters used. If is None (default) use currently initialized hyperparameters for the fit.
            l2reg_diag_ratio (float or None): ratio of l2 regularization coefficient and average diagonal element in the K matrix. If None (default) use the one currently initialized.
            l2reg (float or None): used l2 regularization. If None (default) calculate the l2reg used from l2reg_diag_ratio.
            initialize_training_plane_sq_distance (bool): If True, the fit includes initializing calculation of square distance to the training plane. Default is False.
            Z_matrix (numpy.array or None): if not None defines Z_matrix used in the fit.
            optimize_shifts_l2reg (bool): if True (by default False) optimize l2 regularization and quantity shifts for the fit (overwrites model attributes ).
            save_Z_matrix (bool): if True (by default False) save the calculated Z_matrix in `training_set_Z_matrix` attribute.
        """
        # calculate Z-matrix
        if hyperparameters is None:
            hyperparameters = self.hyperparameters

        if Z_matrix is None:
            Z_matrix = self.MultilevelSORF.calc_Z_matrix(training_objects, hyperparameters)
            if save_Z_matrix:
                self.training_set_Z_matrix = Z_matrix

        if save_Z_matrix or optimize_shifts_l2reg:
            self.assign_training_set(
                training_objects,
                training_quantities,
                training_nuclear_charges=training_nuclear_charges,
                training_importance_multipliers=training_importance_multipliers,
            )

        used_Z_matrix = mult_by_importance(Z_matrix, training_importance_multipliers)
        if optimize_shifts_l2reg:
            self.optimize_quantity_shifts_l2reg(weighted_Z_matrix=used_Z_matrix)
        else:
            # find regularization used.
            if l2reg is None:
                if l2reg_diag_ratio is None:
                    l2reg_diag_ratio = self.l2reg_diag_ratio
                l2reg = l2reg_diag_ratio * self.av_K_diag_element(Z_matrix=used_Z_matrix)

            shifted_weighted_training_quantities = self.get_shifted_weighted_quantities(
                quantities=training_quantities,
                nuclear_charges=training_nuclear_charges,
                importance_multipliers=training_importance_multipliers,
            )
            self.init_Z_matrix_decomposition(used_Z_matrix)
            self.alphas = regression_using_Z_SVD(
                None,
                l2reg,
                shifted_weighted_training_quantities,
                Z_U=self.Z_U,
                Z_singular_values=self.Z_singular_values,
                Z_Vh=self.Z_Vh,
            )
        if initialize_training_plane_sq_distance:
            return self.initialize_uncertainty_components(
                used_Z_matrix,
                l2reg=l2reg,
                training_quantities=shifted_weighted_training_quantities,
                Z_U=self.Z_U,
                Z_singular_values=self.Z_singular_values,
                Z_Vh=self.Z_Vh,
            )

    def refit(
        self,
        added_training_objects,
        added_training_quantities,
        added_training_nuclear_charges=None,
        added_training_importance_multipliers=None,
        optimize_shifts_l2reg=False,
    ):
        """
        Expand the currently assigned training set and refit the model accordingly.
        """
        # assert to avoid a hard-to-track Numba crash.
        assert (
            len(added_training_objects) != 0
        ), "Number of added training points should be non-zero!"
        self.training_objects = merge_or_replace(self.training_objects, added_training_objects)
        self.training_quantities = merge_or_replace(
            self.training_quantities, added_training_quantities
        )
        self.training_nuclear_charges = merge_or_replace(
            self.training_nuclear_charges, added_training_nuclear_charges
        )
        self.training_importance_multipliers = merge_or_replace(
            self.training_importance_multipliers, added_training_importance_multipliers
        )

        if self.training_set_Z_matrix is None:
            self.training_set_Z_matrix = self.calc_Z_matrix(self.training_objects)
        else:
            added_Z_matrix = self.calc_Z_matrix(added_training_objects)
            self.training_set_Z_matrix = concatenate_((self.training_set_Z_matrix, added_Z_matrix))

        self.training_set_size = len(self.training_objects)
        assert self.training_set_size == len(self.training_quantities)
        assert self.training_set_size == self.training_set_Z_matrix.shape[0]

        self.fit(
            self.training_objects,
            self.training_quantities,
            training_nuclear_charges=self.training_nuclear_charges,
            training_importance_multipliers=self.training_importance_multipliers,
            Z_matrix=self.training_set_Z_matrix,
            optimize_shifts_l2reg=optimize_shifts_l2reg,
        )

    def get_all_predictions(
        self,
        query_objects,
        all_query_nuclear_charges=None,
        Z_matrix=None,
        return_tpsd=False,
        thread_assignments=None,
    ):
        """
        Get model predictions for molecules of interest.

        Args:
            query_objects (numba.List): processed representations of query molecules.
            query_nuclear_charges (list or None): if not None define list of query molecule's nuclear charges, which are used is `self.extensive_quantity_shift==True`. Default is None.
            Z_matrix (numpy.array or None): if not None defines the feature vector matrix used in the calculation. Default is None, meaning features are calculated.
            return_tpsd (bool): calculate and return training plane square distance. Default is False.
            thread_assignments (numpy.array or None): if not None, defines numpy.array of estimated CPU times of calculating features of query objects, used to optimize OpenMP performance. Default is None

        Returns:
            if return_tpsd: tuple of Numpy arrays `(predictions, training_plane_sq_distances)` of model predictions and square distances to the training plane.
            else: returns numpy.array of model predictions
        """
        if Z_matrix is None:
            Z_matrix = self.calc_Z_matrix(query_objects, thread_assignments=thread_assignments)
        predictions = dot_(Z_matrix, self.alphas)
        if self.extensive_quantity_shift or self.intensive_quantity_shift:
            shift_coeffs = self.get_shift_coeffs(
                len(query_objects), all_nuclear_charges=all_query_nuclear_charges
            )
            predictions -= dot_(shift_coeffs, self.quantity_shift)
        if not return_tpsd:
            return predictions
        training_plane_sq_distances = get_training_plane_sq_distances(
            Z_matrix, self.train_kernel_eigenvalue_multipliers, self.train_kernel_eigenvectors
        )
        return predictions, training_plane_sq_distances

    def get_prediction(self, query_object, query_nuclear_charges=None):
        if query_nuclear_charges is None:
            all_query_nuclear_charges = None
        else:
            all_query_nuclear_charges = [query_nuclear_charges]
        return self.get_all_predictions(
            get_numba_list([query_object]), all_query_nuclear_charges=all_query_nuclear_charges
        )[0]

    def learning_curve_predictions(
        self,
        training_objects,
        training_quantities,
        test_objects,
        training_set_sizes,
        max_subset_num=8,
        training_nuclear_charges=None,
        test_nuclear_charges=None,
        training_importance_multipliers=None,
        rng=None,
        hyperparameter_reoptimization=False,
        hyperparameter_optimization_kwargs={},
        init_thread_assignments=False,
    ):
        """
        For current hyperparameter values calculate all predictions necessary to build a learning curve.

        Args:
            training_objects (numba.list): processed objects representing training set molecules.
            training_quantities (numpy.array): quantities of interest for training set molecules.
            test_objects (numba.list): processed objects representing test set molecules.
            training_set_sizes (list): list of sizes of the subsets of the training set used to build the learning curve.
            max_subset_num (int): for a given training set subset size, how many subsets are used at most. Default is 8.
            training_nuclear_charges (list): if not None, contains list of training molecules' nuclear charges. Used if `self.extensive_quantity_shift==True`. Default is None.
            test_nuclear_charges (list): if not None, contains list of test molecules' nuclear charges. Used if `self.extensive_quantity_shift==True`. Default is None.
            training_importance_multipliers (numpy.array or None): if not None, defines importance multipliers used during training. Default is None.
            rng (numpy.random RNG or None): RNG used during shuffling of training set. If None (default), an RNG is generated by the code.
            hyperparameter_reoptimization (bool): if True perform a separate hyperparameter optimization for each training set subset. If False (default), hyperparameters currently initialized in the model are used.
            hyperparameter_optimization_kwargs (dict): if `hyperparameter_reoptimization==True`, perform each hyperparameter optimization with these keyword arguments.
            init_thread_assignments (bool): If True, generate CPU estimates of different objects' features calculation and use them to improve OpenMP performance. Default is False.

        Returns:
            For each training set subset numpy.array of predictions is generated, which is combined into lists by subset size. A list of these lists is returned.
        """

        tot_ntrain = len(training_objects)
        if rng is None:
            rng = default_rng_()
        all_training_indices = list(range(tot_ntrain))

        all_predictions = []

        if init_thread_assignments:
            test_thread_assignments = self.calc_thread_assignments(test_objects)
        else:
            test_thread_assignments = None

        if hyperparameter_reoptimization:
            test_Z_matrix = None
        else:
            test_Z_matrix = self.MultilevelSORF.calc_Z_matrix(
                test_objects,
                self.hyperparameters,
                thread_assignments=test_thread_assignments,
            )

        for training_set_size in training_set_sizes:
            rng.shuffle(all_training_indices)
            nsubsets = min(tot_ntrain // training_set_size, max_subset_num)
            assert nsubsets != 0
            subset_predictions = []
            for isubset in range(nsubsets):
                training_indices = array_(
                    all_training_indices[
                        isubset * training_set_size : (isubset + 1) * training_set_size
                    ]
                )
                if training_nuclear_charges is None:
                    subset_nuclear_charges = None
                else:
                    subset_nuclear_charges = [
                        training_nuclear_charges[ti] for ti in training_indices
                    ]
                subset_training_objects = get_numba_list(
                    [training_objects[ti] for ti in training_indices]
                )
                if training_importance_multipliers is None:
                    subset_training_importance_multipliers = None
                else:
                    subset_training_importance_multipliers = training_importance_multipliers[
                        training_indices
                    ]
                subset_training_quantities = training_quantities[training_indices]
                if hyperparameter_reoptimization:
                    self.assign_training_set(
                        subset_training_objects,
                        subset_training_quantities,
                        training_nuclear_charges=subset_nuclear_charges,
                        training_importance_multipliers=subset_training_importance_multipliers,
                        init_thread_assignments=init_thread_assignments,
                    )
                    self.optimize_hyperparameters(**hyperparameter_optimization_kwargs)
                else:
                    self.fit(
                        subset_training_objects,
                        subset_training_quantities,
                        training_nuclear_charges=subset_nuclear_charges,
                        training_importance_multipliers=subset_training_importance_multipliers,
                    )
                predictions = self.get_all_predictions(
                    test_objects,
                    all_query_nuclear_charges=test_nuclear_charges,
                    Z_matrix=test_Z_matrix,
                    thread_assignments=test_thread_assignments,
                )
                subset_predictions.append(predictions)
            all_predictions.append(subset_predictions)
        return all_predictions

    def learning_curve(
        self,
        training_objects,
        training_quantities,
        test_objects,
        test_quantities,
        training_set_sizes,
        max_subset_num=8,
        training_nuclear_charges=None,
        test_nuclear_charges=None,
        training_importance_multipliers=None,
        test_importance_multipliers=None,
        rng=None,
        lc_error_loss_function=MAE(),
        hyperparameter_reoptimization=False,
        hyperparameter_optimization_kwargs={},
        init_thread_assignments=False,
    ):
        """
        Make a learning curve with errors.

        Args:
            training_objects (numba.list): processed objects representing training set molecules.
            training_quantities (numpy.array): quantities of interest for training set molecules.
            test_objects (numba.list): processed objects representing test set molecules.
            test_quantities (numpy.array): quantities of interest for test set molecules.
            training_set_sizes (list): list of sizes of the subsets of the training set used to build the learning curve.
            max_subset_num (int): for a given training set subset size, how many subsets are used at most. Default is 8.
            training_nuclear_charges (list): if not None, contains list of training molecules' nuclear charges. Used if `self.extensive_quantity_shift==True`. Default is None.
            test_nuclear_charges (list): if not None, contains list of test molecules' nuclear charges. Used if `self.extensive_quantity_shift==True`. Default is None.
            training_importance_multipliers (numpy.array): if not None, defines importance multipliers used during training. Default is None.
            test_importance_multipliers (numpy.array or None): if not None, defines importance multipliers for comparing test set predictions and quantities.
            rng (numpy.random RNG or None): RNG used during shuffling of training set. If None (default), an RNG is generated by the code.
            lc_error_loss_function (function): loss function of prediction errors used in the learning curve. Default is Mean Absolute Error.
            hyperparameter_reoptimization (bool): if True perform a separate hyperparameter optimization for each training set subset. If False (default), hyperparameters currently initialized in the model are used.
            hyperparameter_optimization_kwargs (dict): if `hyperparameter_reoptimization==True`, perform each hyperparameter optimization with these keyword arguments.
            init_thread_assignments (bool): If True, generate CPU estimates of different objects' features calculation and use them to improve OpenMP performance. Default is False.

        Returns:
            Tuple of `(means, stds)`, both members Numpy arrays, `means` containing mean values of loss function for a given training set size, `stds` containing standard deviations of the loss function among subsets of the same size.
        """
        all_predictions = self.learning_curve_predictions(
            training_objects,
            training_quantities,
            test_objects,
            training_set_sizes,
            max_subset_num=max_subset_num,
            training_nuclear_charges=training_nuclear_charges,
            test_nuclear_charges=test_nuclear_charges,
            training_importance_multipliers=training_importance_multipliers,
            rng=rng,
            hyperparameter_reoptimization=hyperparameter_reoptimization,
            hyperparameter_optimization_kwargs=hyperparameter_optimization_kwargs,
            init_thread_assignments=init_thread_assignments,
        )
        return learning_curve_from_predictions(
            all_predictions,
            test_quantities,
            error_loss_function=lc_error_loss_function,
            test_importance_multipliers=test_importance_multipliers,
        )

    def get_query_objects_full_definition_list(self):
        if self.objects_definition_list is None:
            if self.object_definition_list is None:
                return None
            else:
                objects_definition_list = self.object_definition_list + ["list"]
        else:
            objects_definition_list = self.objects_definition_list
        return [datatype_prefix, *objects_definition_list]

    def get_pickler(self):
        if self.pickler is None:
            complex_attr_definition_list = {}
            objects_full_definition_list = self.get_query_objects_full_definition_list()
            if objects_full_definition_list is not None:
                complex_attr_definition_list = {"training_objects": objects_full_definition_list}
            self.pickler = Pickler(complex_attr_definition_dict=complex_attr_definition_list)
        return self.pickler

    def __getstate__(self):
        return self.get_pickler().getstate(self)

    def __setstate__(self, d):
        pickler = d["pickler"]
        self.__dict__ = pickler.state_dict(d)
