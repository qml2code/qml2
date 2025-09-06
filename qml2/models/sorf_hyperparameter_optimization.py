from ..basic_utils import now
from ..dimensionality_reduction import get_reductors_diff_species
from ..jit_interfaces import (
    LinAlgError_,
    abs_,
    allow_numba_numpy_parallelization,
    array_,
    concatenate_,
    copy_,
    diag_indices_from_,
    dint_,
    dot_,
    empty_,
    exp_,
    jit_,
    lstsq_,
    max_,
    mean_,
    prange_,
    zeros_,
)
from ..kernels.gradient_common import atom_force_dim
from ..kernels.gradient_kernels import prediction_vector_length
from ..kernels.gradient_sorf import generate_local_force_sorf
from ..kernels.sorf import create_sorf_matrices_diff_species, generate_local_sorf
from ..math import cho_solve, svd_aligned
from ..utils import get_sorted_elements, multiply_transposed
from .forces_utils import combine_energy_forces_rhs, get_importance_multipliers
from .hyperparameter_optimization import (
    KFoldsMultipleObservables,
    KRRLeaveOneOutL2regLoss,
    KRRLeaveOneOutL2regOpt,
    callable_ninv_MAE_local_dn,
)
from .loss_functions import MAE


# For BO in space of lambda and sigma for local_dn SORF and SORF with forces
# using kfolds analogously to ``qml2.models.hyperparameter_optimization`.
class callable_ninv_MAE_local_dn_SORF(callable_ninv_MAE_local_dn):
    def __init__(
        self,
        training_representations_list,
        training_nuclear_charges_list,
        training_quantities,
        nfeature_stacks,
        init_size,
        ntransforms=3,
        use_reductors=False,
        pca_num_samples=1024,
        use_lstsq=False,
        lstsq_rcond=0.0,
        **kfold_kwargs,
    ):
        self.use_lstsq = use_lstsq
        self.lstsq_rcond = lstsq_rcond
        self.init_training_set(
            training_representations_list, training_nuclear_charges_list, training_quantities
        )
        self.init_sorf(
            nfeature_stacks,
            init_size,
            ntransforms=ntransforms,
            use_reductors=use_reductors,
            pca_num_samples=pca_num_samples,
        )
        self.init_temp_arrays()
        self.init_kfolds(**kfold_kwargs)

    def init_training_set(
        self, training_representations_list, training_nuclear_charges_list, training_quantities
    ):
        super().init_training_set(
            training_representations_list, training_nuclear_charges_list, training_quantities
        )
        self.sorted_elements = get_sorted_elements(self.training_nuclear_charges)
        self.nspecies = self.sorted_elements.shape[0]

    def init_sorf(
        self, nfeature_stacks, init_size, ntransforms=3, use_reductors=False, pca_num_samples=1024
    ):
        self.ntransforms = ntransforms
        self.nfeature_stacks = nfeature_stacks
        self.init_size = init_size
        self.nfeatures = self.nfeature_stacks * self.init_size
        self.all_biases, self.all_sorf_diags = create_sorf_matrices_diff_species(
            self.nfeature_stacks, self.nspecies, self.ntransforms, self.init_size
        )
        self.use_reductors = use_reductors
        if self.use_reductors:
            self.pca_num_samples = pca_num_samples
            self.all_reductors = get_reductors_diff_species(
                self.training_representations,
                self.training_nuclear_charges,
                self.init_size,
                num_samples=self.pca_num_samples,
                sorted_elements=self.sorted_elements,
            )
        else:
            self.all_reductors = None
            self.pca_num_samples = None

    def init_temp_arrays(self):
        self.temp_Z_matrix = empty_((self.num_observables, self.nfeatures))
        if not self.use_lstsq:
            self.temp_K_matrix = empty_((self.nfeatures, self.nfeatures))

    def get_sorf(self, sigma):
        return generate_local_sorf(
            self.training_representations,
            self.training_nuclear_charges,
            self.training_natoms,
            self.sorted_elements,
            self.all_sorf_diags,
            self.all_biases,
            sigma,
            self.nfeature_stacks,
            self.init_size,
            out=self.temp_Z_matrix,
            reductors=self.all_reductors,
        )

    def kfold_train_test_sorf_matrices(self, full_sorf_matrix, kfold_id):
        train_indices, test_indices = self.kfolds.train_test_indices(kfold_id)
        return full_sorf_matrix[train_indices], full_sorf_matrix[test_indices]

    def kfold_train_test_quantities(self, kfold_id):
        train_indices, test_indices = self.kfolds.train_test_indices(kfold_id)
        return self.training_quantities[train_indices], self.training_quantities[test_indices]

    def __call__(self, ln_parameters):
        print("started MAE calculation for:", ln_parameters, now())
        parameters = exp_(ln_parameters[0])
        sigma = parameters[-1]
        if not self.use_lstsq:
            normalized_lambda = parameters[0]
        full_sorf_matrix = self.get_sorf(sigma)
        tot_MAE = 0.0
        for kfold_id in range(self.nkfolds):
            train_sorf_matrix, test_sorf_matrix = self.kfold_train_test_sorf_matrices(
                full_sorf_matrix, kfold_id
            )
            train_quantities, test_quantities = self.kfold_train_test_quantities(kfold_id)

            if self.use_lstsq:
                alphas = lstsq_(train_sorf_matrix, train_quantities, rcond=self.lstsq_rcond)[0]
            else:
                K = dot_(train_sorf_matrix.T, train_sorf_matrix, out=self.temp_K_matrix)
                diag_ids = diag_indices_from_(K)
                lambda_val = mean_(K[diag_ids]) * normalized_lambda

                alpha_rhs = dot_(train_quantities, train_sorf_matrix)
                try:
                    alphas = cho_solve(K, alpha_rhs, l2reg=lambda_val)
                except LinAlgError_:
                    return 0.0
            predictions = dot_(test_sorf_matrix, alphas)
            tot_MAE += mean_(abs_(predictions - test_quantities))
        av_MAE = tot_MAE / self.nkfolds
        print("finished calculations:", av_MAE, now())
        return -1 / av_MAE


class callable_ninv_MAE_local_dn_forces_SORF(callable_ninv_MAE_local_dn_SORF):
    def __init__(
        self,
        training_representations_list,
        training_nuclear_charges_list,
        training_grad_representations_list,
        training_relevant_atom_ids_list,
        training_relevant_atom_nums_list,
        training_energies,
        training_forces_list,
        nfeature_stacks,
        init_size,
        energy_importance=0.0,
        **callable_ninv_MAE_local_dn_SORF_kwargs,
    ):
        self.energy_importance = energy_importance
        self.init_num_en_force_observables(training_forces_list)
        super().__init__(
            training_representations_list,
            training_nuclear_charges_list,
            None,
            nfeature_stacks,
            init_size,
            **callable_ninv_MAE_local_dn_SORF_kwargs,
        )
        self.init_training_energy_forces(training_energies, training_forces_list)
        self.init_training_set_grad_reps(
            training_grad_representations_list,
            training_relevant_atom_ids_list,
            training_relevant_atom_nums_list,
        )

    def init_num_en_force_observables(self, training_forces_list):
        self.num_observables = len(training_forces_list) + atom_force_dim * sum(
            [force.shape[0] for force in training_forces_list]
        )

    def init_prediction_vector_lengths(self):
        self.prediction_vector_lengths = array_(
            [prediction_vector_length(na) for na in self.training_natoms]
        )

    def init_kfolds(self, nkfolds=8, training_set_ratio=0.5, mol_kfolds=None):
        self.init_prediction_vector_lengths()
        self.kfolds = KFoldsMultipleObservables(
            self.prediction_vector_lengths,
            nkfolds=nkfolds,
            training_set_ratio=training_set_ratio,
            mol_kfolds=mol_kfolds,
        )
        self.nkfolds = self.kfolds.nkfolds

    def init_training_energy_forces(self, training_energies, training_forces_list):
        self.training_quantities = combine_energy_forces_rhs(
            training_energies, training_forces_list
        )
        self.importance_multipliers = get_importance_multipliers(
            self.training_natoms, self.energy_importance
        )
        self.training_quantities *= self.importance_multipliers

    def get_sorf(self, sigma):
        sorf_matrix = generate_local_force_sorf(
            self.training_representations,
            self.training_grad_representations,
            self.training_nuclear_charges,
            self.training_natoms,
            self.training_relevant_atom_ids,
            self.training_relevant_atom_nums,
            self.sorted_elements,
            self.all_sorf_diags,
            self.all_biases,
            sigma,
            self.nfeature_stacks,
            self.init_size,
            reductors=self.all_reductors,
            out=self.temp_Z_matrix,
        )
        multiply_transposed(sorf_matrix, self.importance_multipliers)
        return sorf_matrix

    def init_training_set_grad_reps(
        self,
        training_grad_representations_list,
        training_relevant_atom_ids_list,
        training_relevant_atom_nums_list,
    ):
        self.training_relevant_atom_nums = concatenate_(training_relevant_atom_nums_list)
        self.max_relevant_atom_num = max_(self.training_relevant_atom_nums)
        self.training_grad_representations = empty_(
            (*self.training_representations.shape, self.max_relevant_atom_num, atom_force_dim)
        )
        self.training_relevant_atom_ids = empty_(
            (self.training_representations.shape[0], self.max_relevant_atom_num), dtype=dint_
        )

        lbound = 0
        for grad_rep, rel_atom_ids in zip(
            training_grad_representations_list, training_relevant_atom_ids_list
        ):
            ubound = lbound + grad_rep.shape[0]
            cur_max_rel_atom_num = max_(self.training_relevant_atom_nums[lbound:ubound])
            self.training_grad_representations[
                lbound:ubound, :, :cur_max_rel_atom_num, :
            ] = grad_rep[:, :, :cur_max_rel_atom_num, :]
            self.training_relevant_atom_ids[lbound:ubound, :cur_max_rel_atom_num] = rel_atom_ids[
                :, :cur_max_rel_atom_num
            ]
            lbound = ubound


# Formula for calculating the "leave-one-out errors" without gradients.
# Main building block for optimization in qml2.multilevel_sorf
def mult_by_importance(arr, importance_multipliers=None):
    if importance_multipliers is None:
        return arr
    if arr is None:
        return None
    out = copy_(arr)
    multiply_transposed(out, importance_multipliers)
    return out


@jit_(numba_parallel=allow_numba_numpy_parallelization())
def get_stat_factors(Z_U, eigenvalue_multipliers):
    nmols = Z_U.shape[0]
    output = empty_((nmols,))
    transformed_inv_K_Z = eigenvalue_multipliers * Z_U
    for i in prange_(nmols):
        output[i] = 1 / (1 - dot_(Z_U[i], transformed_inv_K_Z[i]))
    return output, transformed_inv_K_Z


@jit_
def leaveoneout_errors_from_precalc(reproduced_quantities, quantities, stat_factors):
    return (reproduced_quantities - quantities) * stat_factors


@jit_
def leaveoneout_eigenvalue_multipliers(Z_singular_values, l2reg):
    sq_Z_singular_values = Z_singular_values**2
    return sq_Z_singular_values / (sq_Z_singular_values + l2reg)


def leaveoneout_errors(
    Z_matrix,
    quantities,
    l2reg,
    importance_multipliers=None,
    return_intermediates=False,
    Z_U=None,
    Z_singular_values=None,
    Z_Vh=None,
):
    used_quantities = mult_by_importance(quantities, importance_multipliers=importance_multipliers)

    if (Z_singular_values is None) or (Z_Vh is None) or (Z_U is None):
        used_Z_matrix = mult_by_importance(Z_matrix, importance_multipliers=importance_multipliers)
        Z_U, Z_singular_values, Z_Vh = svd_aligned(used_Z_matrix)

    eigenvalue_multipliers = leaveoneout_eigenvalue_multipliers(Z_singular_values, l2reg)

    stat_factors, transformed_inv_K_Z = get_stat_factors(Z_U, eigenvalue_multipliers)

    transformed_alphas_rhs = dot_(used_quantities, Z_U)

    reproduced_quantities = dot_(transformed_inv_K_Z, transformed_alphas_rhs)

    errors = leaveoneout_errors_from_precalc(reproduced_quantities, used_quantities, stat_factors)

    if return_intermediates:
        return (
            errors,
            Z_U,
            Z_Vh,
            Z_singular_values,
            eigenvalue_multipliers,
            reproduced_quantities,
            stat_factors,
            transformed_alphas_rhs,
            transformed_inv_K_Z,
            used_quantities,
        )
    else:
        return errors


@jit_(numba_parallel=True)
def leaveoneout_loss_l2reg_der(
    eigenvalue_multipliers,
    Z_singular_values,
    loss_error_ders,
    Z_U,
    reproduced_quantities,
    transformed_alphas_rhs,
    stat_factors,
    used_quantities,
):
    eigenvalue_multiplier_derivatives = -((eigenvalue_multipliers / Z_singular_values) ** 2)
    mult_transformed_alphas_rhs_ders = transformed_alphas_rhs * eigenvalue_multiplier_derivatives
    reproduced_quantities_ders = dot_(Z_U, mult_transformed_alphas_rhs_ders)
    npoints = Z_singular_values.shape[0]
    stat_factor_ders = empty_(npoints)
    for i in prange_(npoints):
        stat_factor_ders[i] = stat_factors[i] ** 2 * dot_(
            Z_U[i] * eigenvalue_multiplier_derivatives, Z_U[i]
        )
    error_ders = (
        stat_factor_ders * (reproduced_quantities - used_quantities)
        + reproduced_quantities_ders * stat_factors
    )
    return dot_(
        error_ders,
        loss_error_ders,
    )


@jit_
def leaveoneout_errors_quantity_ders(
    quantity_derivatives, Z_U, eigenvalue_multipliers, stat_factors, importance_multipliers
):
    if importance_multipliers is None:
        weighted_quantity_derivatives = quantity_derivatives
    else:
        weighted_quantity_derivatives = copy_(quantity_derivatives)
        multiply_transposed(weighted_quantity_derivatives, importance_multipliers)
    error_ders = (
        dot_(
            Z_U,
            (dot_(weighted_quantity_derivatives.T, Z_U) * eigenvalue_multipliers).T,
        )
        - weighted_quantity_derivatives
    )
    multiply_transposed(error_ders, stat_factors)
    return error_ders


@jit_
def leaveoneout_loss_quantity_ders(
    quantity_derivatives,
    loss_error_ders,
    Z_U,
    eigenvalue_multipliers,
    stat_factors,
    importance_multipliers,
):
    return dot_(
        loss_error_ders,
        leaveoneout_errors_quantity_ders(
            quantity_derivatives, Z_U, eigenvalue_multipliers, stat_factors, importance_multipliers
        ),
    )


@jit_
def leaveoneout_loss_der_wrt_single_feature(
    output_arr,
    feature_ders,
    feature_id,
    Z_U,
    Z_Vh,
    Z_singular_values,
    loss_error_ders,
    mult_transformed_alphas_rhs,
    reproduced_quantities,
    stat_factors,
    transformed_inv_K_Z,
    used_quantities,
):
    # transform derivatives
    transformed_feature_ders = dot_(feature_ders, Z_Vh.T) / Z_singular_values
    # resulting change of alphas RHS
    npoints = used_quantities.shape[0]
    # remaining derivative w.r.t. feature_vector in feature_vector*alphas.
    stat_der_mult = loss_error_ders[feature_id] * stat_factors[feature_id]
    output_arr[:] = stat_der_mult * dot_(transformed_feature_ders, mult_transformed_alphas_rhs)
    # extra from stat_factors in error corresponding to feature id
    output_arr += (
        2
        * (reproduced_quantities[feature_id] - used_quantities[feature_id])
        * stat_factors[feature_id]
        * stat_der_mult
        * dot_(transformed_feature_ders, transformed_inv_K_Z[feature_id])
    )
    # remaining derivative w.r.t. feature_vector in stat_factors[feature_id]
    # derivatives caused by change of alphas RHS and stat factors.
    for i in range(npoints):
        stat_der_mult = loss_error_ders[i] * stat_factors[i]

        inv_K_Z_der_Z = dot_(transformed_feature_ders, transformed_inv_K_Z[i])
        inv_K_Z_Z = dot_(Z_U[feature_id], transformed_inv_K_Z[i])
        # derivatives due to stat factors
        repr_err = reproduced_quantities[i] - used_quantities[i]
        output_arr -= 2 * stat_der_mult * repr_err * stat_factors[i] * inv_K_Z_der_Z * inv_K_Z_Z
        # derivatives due to change of alphas RHS
        output_arr += used_quantities[feature_id] * inv_K_Z_der_Z * stat_der_mult
        # derivatives due to K^{-1} multiplying alphas RHS
        output_arr -= (
            stat_der_mult * inv_K_Z_der_Z * dot_(mult_transformed_alphas_rhs, Z_U[feature_id])
        )
        output_arr -= (
            stat_der_mult * inv_K_Z_Z * dot_(transformed_feature_ders, mult_transformed_alphas_rhs)
        )


@jit_(numba_parallel=allow_numba_numpy_parallelization())
def leaveoneout_loss_der_wrt_features(
    Z_matrix_derivatives,
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
):
    npoints = Z_matrix_derivatives.shape[0]
    nders = Z_matrix_derivatives.shape[1]

    output = zeros_((nders,))
    for i in prange_(npoints):
        temp_der = empty_((nders,))
        leaveoneout_loss_der_wrt_single_feature(
            temp_der,
            Z_matrix_derivatives[i],
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
        if importance_multipliers is not None:
            temp_der *= importance_multipliers[i]
        output += temp_der
    return output


# Expression written in a way best for only storing SORF derivatives for one entry at a time during summation.
def leaveoneout_error_loss(
    Z_matrix,
    quantities,
    l2reg,
    error_loss=MAE(),
    Z_matrix_derivatives=None,
    quantity_derivatives=None,
    gradient=False,
    importance_multipliers=None,
    return_errors=False,
    Z_U=None,
    Z_singular_values=None,
    Z_Vh=None,
):
    if not gradient:
        loo_errors = leaveoneout_errors(
            Z_matrix,
            quantities,
            l2reg,
            importance_multipliers=importance_multipliers,
            Z_U=Z_U,
            Z_singular_values=Z_singular_values,
            Z_Vh=Z_Vh,
        )
        loss = error_loss(loo_errors)
        if return_errors:
            return loss, loo_errors
        else:
            return loss
    (
        loo_errors,
        Z_U,
        Z_Vh,
        Z_singular_values,
        eigenvalue_multipliers,
        reproduced_quantities,
        stat_factors,
        transformed_alphas_rhs,
        transformed_inv_K_Z,
        used_quantities,
    ) = leaveoneout_errors(
        Z_matrix,
        quantities,
        l2reg,
        importance_multipliers=importance_multipliers,
        return_intermediates=True,
        Z_U=Z_U,
        Z_singular_values=Z_singular_values,
        Z_Vh=Z_Vh,
    )
    loss, loss_error_ders = error_loss.calc_wders(loo_errors)

    nders = 1
    if Z_matrix_derivatives is not None:
        nZ_ders = Z_matrix_derivatives.shape[1]
        nders += nZ_ders
    if quantity_derivatives is not None:
        nquant_ders = quantity_derivatives.shape[1]
        nders += nquant_ders
    loss_ders = empty_(nders)
    # l2reg derivatives
    loss_ders[0] = leaveoneout_loss_l2reg_der(
        eigenvalue_multipliers,
        Z_singular_values,
        loss_error_ders,
        Z_U,
        reproduced_quantities,
        transformed_alphas_rhs,
        stat_factors,
        used_quantities,
    )
    if Z_matrix_derivatives is not None:
        mult_transformed_alphas_rhs = transformed_alphas_rhs * eigenvalue_multipliers
        loss_ders[1 : nZ_ders + 1] = leaveoneout_loss_der_wrt_features(
            Z_matrix_derivatives,
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
        )
    if quantity_derivatives is not None:
        loss_ders[-nquant_ders:] = leaveoneout_loss_quantity_ders(
            quantity_derivatives,
            loss_error_ders,
            Z_U,
            eigenvalue_multipliers,
            stat_factors,
            importance_multipliers,
        )
    if return_errors:
        return loss, loss_ders, loo_errors
    return loss, loss_ders


class SORFLeaveOneOutL2regLoss(KRRLeaveOneOutL2regLoss):
    def __init__(
        self,
        train_quantities,
        train_Z=None,
        train_Z_U=None,
        train_Z_singular_values=None,
        train_Z_Vh=None,
        loss_function=MAE(),
        overwrite_train_kernel=False,
    ):
        """
        Auxiliary class for convenient l2reg optimization in qml2.models.sorf.SORFModel
        """
        self.train_quantities = train_quantities
        if train_Z_U is None or train_Z_singular_values is None or train_Z_Vh is None:
            assert train_Z is not None
            train_Z_U, train_Z_singular_values, train_Z_Vh = svd_aligned(
                train_Z, overwrite_a=overwrite_train_kernel
            )
        self.train_Z_U = train_Z_U
        self.train_Z_singular_values = train_Z_singular_values
        self.train_Z_Vh = train_Z_Vh
        self.loss_function = loss_function
        self.mean_diag_element = mean_(self.train_Z_singular_values**2)

    def calculate_for_l2reg(self, l2reg):
        return leaveoneout_error_loss(
            None,
            self.train_quantities,
            l2reg,
            error_loss=self.loss_function,
            Z_U=self.train_Z_U,
            Z_singular_values=self.train_Z_singular_values,
            Z_Vh=self.train_Z_Vh,
        )


class SORFLeaveOneOutL2regOpt(KRRLeaveOneOutL2regOpt):
    def __init__(self, train_feature_vectors_generator, train_quantities, **kwargs):
        self.train_feature_vectors_generator = train_feature_vectors_generator
        self.train_quantities = train_quantities
        self.basic_init(**kwargs)

    def get_l2reg_optimizer_from_parameters(self, parameters):
        train_feature_vectors = self.train_feature_vectors_generator(parameters)
        return SORFLeaveOneOutL2regLoss(
            self.train_quantities,
            train_Z=train_feature_vectors,
            overwrite_train_kernel=True,
            loss_function=self.loss_function,
        )
