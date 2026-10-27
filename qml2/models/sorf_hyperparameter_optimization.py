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
    min_,
    prange_,
    unique_,
    zeros_,
)
from ..kernels.gradient_common import atom_force_dim
from ..kernels.gradient_kernels import prediction_vector_length
from ..kernels.gradient_sorf import generate_local_force_sorf
from ..kernels.sorf import create_sorf_matrices_diff_species, generate_local_sorf
from ..math import cho_solve_from_matrix, svd_aligned
from ..utils import multiply_transposed
from .forces_utils import combine_energy_forces_rhs, get_importance_multipliers
from .hyperparameter_optimization import (
    KfoldL2regLossMixin,
    KRRKfoldL2regOpt,
    KRRLeaveOneOutL2regLoss,
    KRRLeaveOneOutL2regOpt,
)
from .kfolds import GroupKFoldsMultipleObservables, KFolds, build_kfolds, conformer_folds
from .loss_functions import MAE


# For BO in space of lambda and sigma for local_dn SORF and SORF with forces
# using kfolds analogously to ``qml2.models.hyperparameter_optimization`.
class callable_ninv_MAE_local_dn_SORF:
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
        test_mode=False,
        **kfold_kwargs,
    ):
        self.use_lstsq = use_lstsq
        self.lstsq_rcond = lstsq_rcond
        self.test_mode = test_mode
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
        """Flatten the per-molecule lists into the concatenated arrays the SORF calls want.

        Inlined from the callable_ninv_MAE / _local / _local_dn chain that used to live in
        qml2.models.kfolds. Only these few lines of it were ever reached on the SORF path --
        the rest was kernel machinery this class never touches, since it builds SORF feature
        matrices instead -- so inheriting the chain kept three obsolete classes alive for
        two attributes.

        ``training_quantities`` is None for the forces subclass, which sets num_observables
        itself from the energy-plus-forces count before calling this; the guard is what
        stops that being overwritten.
        """
        self.training_representations = concatenate_(training_representations_list)
        self.training_natoms = array_([len(rep) for rep in training_representations_list])
        self.training_set_size = self.training_natoms.shape[0]
        if training_quantities is not None:
            self.training_quantities = training_quantities
            self.num_observables = len(self.training_quantities)
        self.training_nuclear_charges = concatenate_(training_nuclear_charges_list)
        self.sorted_elements = unique_(self.training_nuclear_charges)
        self.nspecies = self.sorted_elements.shape[0]

    def init_kfolds(self, nkfolds=8, training_set_ratio=0.5, rng=None):
        """Plain random k-folds over molecules; the forces subclass overrides this with
        conformer-grouped folds.

        ``rng=None`` draws from the global numpy random state, so the split differs between
        constructions; pass a Generator to make it reproducible.
        """
        self.nkfolds = nkfolds
        self.kfolds = KFolds(
            len(self.training_quantities),
            self.nkfolds,
            training_set_ratio=training_set_ratio,
            rng=rng,
        )

    def init_sorf(
        self, nfeature_stacks, init_size, ntransforms=3, use_reductors=False, pca_num_samples=1024
    ):
        self.ntransforms = ntransforms
        self.nfeature_stacks = nfeature_stacks
        self.init_size = init_size
        self.nfeatures = self.nfeature_stacks * self.init_size
        self.all_biases, self.all_sorf_diags = create_sorf_matrices_diff_species(
            self.nfeature_stacks,
            self.nspecies,
            self.ntransforms,
            self.init_size,
            test_mode=self.test_mode,
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
                    alphas = cho_solve_from_matrix(K, alpha_rhs, l2reg=lambda_val)
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

    def init_kfolds(
        self, nkfolds=8, training_set_ratio=0.5, mol_kfolds=None, conformer_group_ids=None
    ):
        """Folds whose validation blocks are whole conformers.

        A conformer owns one energy and three force components per atom, and those
        observables have to move between train and validation together -- otherwise the
        model is fitted to some of a conformer's forces and validated on the rest of the
        same geometry, which measures nothing.

        ``conformer_group_ids`` groups the conformers themselves (by scaffold, say, so
        every conformer of a molecule moves as one) and defaults to one group per
        conformer.

        ``training_set_ratio`` is kept for signature compatibility and IGNORED here: group
        folds partition the set, so the training fraction is 1 - 1/nkfolds by
        construction. Pass ``mol_kfolds`` explicitly to use a different split.
        """
        self.init_prediction_vector_lengths()
        if mol_kfolds is None:
            mol_kfolds = conformer_folds(
                self.training_set_size, nkfolds, conformer_group_ids=conformer_group_ids
            )
        self.mol_kfolds = mol_kfolds
        self.kfolds = GroupKFoldsMultipleObservables(self.prediction_vector_lengths, mol_kfolds)
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


# ---------------------------------------------------------------------------
# Leave-one-GROUP-out.
#
# Everything above holds a single POINT out. These hold out a whole GROUP -- a scaffold,
# in practice -- and are otherwise the same expressions with a block where there was a
# scalar. With every group a singleton they reduce to the leave-one-out functions exactly,
# which is the first thing the sanity check verifies.
#
# The output convention is deliberately identical to leave-one-out: ONE error per data
# point, the error on that point when its entire group was held out. So a loss written for
# leave-one-out errors works unchanged, and so does the shift optimization built on it.
#
# Writing Z = U diag(s) Vh and m_i = s_i^2 / (s_i^2 + l2reg), the smoother is
# P = U diag(m) U^T and, for a validation block V,
#
#     e_V = (I - P_VV)^{-1} ( (P y)_V - y_V ) .
#
# The derivatives all follow from differentiating that. With A_V = I - P_VV,
# h_V = A_V^{-1} (dLoss/de_V), and w_V = y + (e_V padded to full length),
#
#     dLoss = sum_V h_V^T (dP w_V)_V ,
#
# which is the one identity the three derivative routines below specialize.
# ---------------------------------------------------------------------------


def fold_test_index_list(folds):
    """Validation index array per fold.

    Accepts either a KFolds-like object or, directly, the list of index arrays naming each
    group's members -- which is what a caller holding scaffold groups already has, and what
    MultilevelSORFModel passes as ``training_group_ids``.
    """
    if hasattr(folds, "train_test_indices"):
        return [folds.train_test_indices(fold_id)[1] for fold_id in range(folds.nkfolds)]
    return [array_(list(indices), dtype=dint_) for indices in folds]


def group_stat_matrices(Z_U, eigenvalue_multipliers, test_index_list):
    """``I - P_VV`` for every validation block, in fold order.

    Positive definite for l2reg > 0, so each is used through a Cholesky solve.
    """
    stat_matrices = []
    for test_indices in test_index_list:
        test_basis_vectors = Z_U[test_indices]
        stat_matrix = -dot_(test_basis_vectors * eigenvalue_multipliers, test_basis_vectors.T)
        diag_ids = diag_indices_from_(stat_matrix)
        stat_matrix[diag_ids] += 1.0
        stat_matrices.append(stat_matrix)
    return stat_matrices


def group_errors_from_precalc(
    reproduced_quantities, used_quantities, stat_matrices, test_index_list
):
    """Leave-one-group-out errors from the pieces a caller already has.

    The block analogue of :func:`leaveoneout_errors_from_precalc`: there the stat factor is
    a scalar and this is a multiplication, here it is a matrix and this is a solve.
    """
    errors = empty_((used_quantities.shape[0],))
    for stat_matrix, test_indices in zip(stat_matrices, test_index_list):
        errors[test_indices] = cho_solve_from_matrix(
            stat_matrix,
            reproduced_quantities[test_indices] - used_quantities[test_indices],
            overwrite_a=False,
        )
    return errors


def group_errors_quantity_ders(
    quantity_derivatives,
    Z_U,
    eigenvalue_multipliers,
    stat_matrices,
    test_index_list,
    importance_multipliers,
):
    """d(error_i)/d(quantity parameter_j) as a full (npoints, nders) matrix.

    The block analogue of :func:`leaveoneout_errors_quantity_ders`, and needed -- rather
    than the contracted :func:`group_loss_quantity_ders` -- wherever the errors themselves
    are differentiated, as in the quantity-shift optimization, where the shift is found by
    solving a linear system in these derivatives.

    This costs a Cholesky solve per block on an (nblock, nders) right-hand side, where the
    LOSS derivative needs only one solve per block on a vector. Both forms exist for that
    reason; a test pins that they agree.
    """
    if importance_multipliers is None:
        weighted_quantity_derivatives = quantity_derivatives
    else:
        weighted_quantity_derivatives = copy_(quantity_derivatives)
        multiply_transposed(weighted_quantity_derivatives, importance_multipliers)
    residual_derivatives = (
        dot_(Z_U, (dot_(weighted_quantity_derivatives.T, Z_U) * eigenvalue_multipliers).T)
        - weighted_quantity_derivatives
    )
    error_derivatives = empty_(residual_derivatives.shape)
    for stat_matrix, test_indices in zip(stat_matrices, test_index_list):
        error_derivatives[test_indices] = cho_solve_from_matrix(
            stat_matrix, residual_derivatives[test_indices], overwrite_a=False
        )
    return error_derivatives


def apply_smoother(Z_U, eigenvalue_multipliers, vector):
    """``P x`` -- the fitted values a full-set fit reproduces from ``x``."""
    return dot_(Z_U, eigenvalue_multipliers * dot_(vector, Z_U))


def apply_residual_projector(Z_U, eigenvalue_multipliers, vector):
    """``(I - P) x``."""
    return vector - apply_smoother(Z_U, eigenvalue_multipliers, vector)


def apply_regression_operator(Z_U, Z_Vh, Z_singular_values, eigenvalue_multipliers, vector):
    """``G x`` with ``G = (Z^T Z + l2reg)^{-1} Z^T``, i.e. the alphas fitted to ``x``.

    In SVD terms ``G = Vh^T diag(m/s) U^T``, so this never forms ``Z^T Z``.
    """
    return dot_(
        (eigenvalue_multipliers / Z_singular_values) * dot_(vector, Z_U),
        Z_Vh,
    )


def group_errors(
    Z_matrix,
    quantities,
    l2reg,
    folds,
    importance_multipliers=None,
    return_intermediates=False,
    Z_U=None,
    Z_singular_values=None,
    Z_Vh=None,
):
    """Leave-one-group-out errors: one per point, its own group held out.

    ``folds`` is any KFolds-like object whose validation blocks PARTITION the set --
    :class:`qml2.models.kfolds.GroupKFolds` or
    :func:`qml2.models.kfolds.leave_one_group_out_folds`. The partition property is what
    makes "one error per point" well defined, and it is asserted rather than assumed.
    """
    used_quantities = mult_by_importance(quantities, importance_multipliers=importance_multipliers)

    if (Z_singular_values is None) or (Z_Vh is None) or (Z_U is None):
        used_Z_matrix = mult_by_importance(Z_matrix, importance_multipliers=importance_multipliers)
        Z_U, Z_singular_values, Z_Vh = svd_aligned(used_Z_matrix)

    eigenvalue_multipliers = leaveoneout_eigenvalue_multipliers(Z_singular_values, l2reg)

    transformed_alphas_rhs = dot_(used_quantities, Z_U)
    reproduced_quantities = dot_(Z_U, eigenvalue_multipliers * transformed_alphas_rhs)

    test_index_list = fold_test_index_list(folds)
    npoints = used_quantities.shape[0]
    covered = zeros_((npoints,), dtype=dint_)
    for test_indices in test_index_list:
        covered[test_indices] += 1
    assert max_(covered) == 1 and min_(covered) == 1, (
        "leave-one-group-out needs the validation blocks to partition the set; "
        "got a point covered %d times" % int(max_(covered))
    )

    stat_matrices = group_stat_matrices(Z_U, eigenvalue_multipliers, test_index_list)
    errors = group_errors_from_precalc(
        reproduced_quantities, used_quantities, stat_matrices, test_index_list
    )

    if return_intermediates:
        return (
            errors,
            Z_U,
            Z_Vh,
            Z_singular_values,
            eigenvalue_multipliers,
            reproduced_quantities,
            stat_matrices,
            test_index_list,
            transformed_alphas_rhs,
            used_quantities,
        )
    return errors


def group_loss_multipliers(stat_matrices, test_index_list, loss_error_ders, npoints):
    """``h`` with ``h_V = A_V^{-1} (dLoss/de_V)`` -- the vector every derivative needs."""
    multipliers = empty_((npoints,))
    for stat_matrix, test_indices in zip(stat_matrices, test_index_list):
        multipliers[test_indices] = cho_solve_from_matrix(
            stat_matrix, loss_error_ders[test_indices], overwrite_a=False
        )
    return multipliers


def group_loss_l2reg_der(
    eigenvalue_multipliers,
    Z_singular_values,
    Z_U,
    errors,
    loss_multipliers,
    test_index_list,
    transformed_alphas_rhs,
):
    """d(loss)/d(l2reg).

    Only the multipliers depend on l2reg, through dm/dl2reg = -(m/s)^2, so
    dP = U diag(dm) U^T and the master identity becomes a sum of small dot products.
    """
    multiplier_derivatives = -((eigenvalue_multipliers / Z_singular_values) ** 2)
    total = 0.0
    for test_indices in test_index_list:
        test_basis_vectors = Z_U[test_indices]
        # U^T w_V, with w_V = y + (errors restricted to this block).
        transformed_w = transformed_alphas_rhs + dot_(errors[test_indices], test_basis_vectors)
        total += dot_(
            dot_(loss_multipliers[test_indices], test_basis_vectors),
            multiplier_derivatives * transformed_w,
        )
    return total


def group_loss_quantity_ders(
    quantity_derivatives,
    eigenvalue_multipliers,
    Z_U,
    loss_multipliers,
    test_index_list,
    importance_multipliers,
):
    """d(loss)/d(quantity parameters), e.g. the property shifts.

    ``A_V`` does not depend on the quantities, so only the residual moves:
    de_V = A_V^{-1} ( (P dy)_V - dy_V ).
    """
    if importance_multipliers is None:
        weighted_quantity_derivatives = quantity_derivatives
    else:
        weighted_quantity_derivatives = copy_(quantity_derivatives)
        multiply_transposed(weighted_quantity_derivatives, importance_multipliers)
    # P applied to every derivative direction at once.
    smoothed = dot_(
        Z_U,
        (dot_(weighted_quantity_derivatives.T, Z_U) * eigenvalue_multipliers).T,
    )
    error_ders = smoothed - weighted_quantity_derivatives
    total = zeros_((quantity_derivatives.shape[1],))
    for test_indices in test_index_list:
        total += dot_(loss_multipliers[test_indices], error_ders[test_indices])
    return total


def group_feature_der_factors(
    Z_U,
    Z_Vh,
    Z_singular_values,
    eigenvalue_multipliers,
    errors,
    loss_multipliers,
    test_index_list,
    used_quantities,
):
    """The two factor matrices whose row-wise outer products give d(loss)/dZ.

    d(loss)/dZ is a sum of rank-one terms, because

        dP = (I - P) dZ G + G^T dZ^T (I - P)

    and the master identity splits into a part that sums over folds into ONE global term
    and a part that stays per-fold:

        dLoss = h^T dP y + sum_V h_V^T (dP e_V)_V .

    Returning the factors rather than the (npoints x nfeatures) matrix they generate is
    what keeps this affordable: the caller contracts them against one point's feature
    derivatives at a time, exactly as the leave-one-out routine does.
    """
    npoints = used_quantities.shape[0]
    ncolumns = 2 + 2 * len(test_index_list)
    left = zeros_((npoints, ncolumns))
    right = zeros_((ncolumns, Z_Vh.shape[1]))

    # Global pair: h^T dP y.
    left[:, 0] = apply_residual_projector(Z_U, eigenvalue_multipliers, loss_multipliers)
    right[0] = apply_regression_operator(
        Z_U, Z_Vh, Z_singular_values, eigenvalue_multipliers, used_quantities
    )
    left[:, 1] = apply_residual_projector(Z_U, eigenvalue_multipliers, used_quantities)
    right[1] = apply_regression_operator(
        Z_U, Z_Vh, Z_singular_values, eigenvalue_multipliers, loss_multipliers
    )

    # Per-fold pairs: h_V^T dP e_V, with both vectors supported on the block alone.
    for fold_id, test_indices in enumerate(test_index_list):
        padded_multipliers = zeros_((npoints,))
        padded_multipliers[test_indices] = loss_multipliers[test_indices]
        padded_errors = zeros_((npoints,))
        padded_errors[test_indices] = errors[test_indices]

        column = 2 + 2 * fold_id
        left[:, column] = apply_residual_projector(Z_U, eigenvalue_multipliers, padded_multipliers)
        right[column] = apply_regression_operator(
            Z_U, Z_Vh, Z_singular_values, eigenvalue_multipliers, padded_errors
        )
        left[:, column + 1] = apply_residual_projector(Z_U, eigenvalue_multipliers, padded_errors)
        right[column + 1] = apply_regression_operator(
            Z_U, Z_Vh, Z_singular_values, eigenvalue_multipliers, padded_multipliers
        )
    return left, right


def group_loss_der_wrt_features(
    Z_matrix_derivatives,
    Z_U,
    Z_Vh,
    Z_singular_values,
    eigenvalue_multipliers,
    errors,
    loss_multipliers,
    test_index_list,
    used_quantities,
    importance_multipliers,
):
    """d(loss)/d(hyperparameters) given dZ/d(hyperparameters), one point at a time."""
    left, right = group_feature_der_factors(
        Z_U,
        Z_Vh,
        Z_singular_values,
        eigenvalue_multipliers,
        errors,
        loss_multipliers,
        test_index_list,
        used_quantities,
    )
    npoints = Z_matrix_derivatives.shape[0]
    nders = Z_matrix_derivatives.shape[1]
    output = zeros_((nders,))
    for point_id in range(npoints):
        # (ncolumns x nders): every factor pair contracted with this point's derivatives.
        contracted = dot_(right, Z_matrix_derivatives[point_id].T)
        point_der = dot_(left[point_id], contracted)
        if importance_multipliers is not None:
            point_der = point_der * importance_multipliers[point_id]
        output += point_der
    return output


def group_error_loss(
    Z_matrix,
    quantities,
    l2reg,
    folds,
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
    """Leave-one-group-out analogue of :func:`leaveoneout_error_loss`.

    Same signature bar ``folds``, same return shapes, same ordering of ``loss_ders``
    (l2reg first, then feature derivatives, then quantity derivatives).
    """
    if not gradient:
        errors = group_errors(
            Z_matrix,
            quantities,
            l2reg,
            folds,
            importance_multipliers=importance_multipliers,
            Z_U=Z_U,
            Z_singular_values=Z_singular_values,
            Z_Vh=Z_Vh,
        )
        loss = error_loss(errors)
        if return_errors:
            return loss, errors
        return loss

    (
        errors,
        Z_U,
        Z_Vh,
        Z_singular_values,
        eigenvalue_multipliers,
        reproduced_quantities,
        stat_matrices,
        test_index_list,
        transformed_alphas_rhs,
        used_quantities,
    ) = group_errors(
        Z_matrix,
        quantities,
        l2reg,
        folds,
        importance_multipliers=importance_multipliers,
        return_intermediates=True,
        Z_U=Z_U,
        Z_singular_values=Z_singular_values,
        Z_Vh=Z_Vh,
    )
    loss, loss_error_ders = error_loss.calc_wders(errors)
    loss_multipliers = group_loss_multipliers(
        stat_matrices, test_index_list, loss_error_ders, used_quantities.shape[0]
    )

    nders = 1
    if Z_matrix_derivatives is not None:
        nZ_ders = Z_matrix_derivatives.shape[1]
        nders += nZ_ders
    if quantity_derivatives is not None:
        nquant_ders = quantity_derivatives.shape[1]
        nders += nquant_ders
    loss_ders = empty_(nders)

    loss_ders[0] = group_loss_l2reg_der(
        eigenvalue_multipliers,
        Z_singular_values,
        Z_U,
        errors,
        loss_multipliers,
        test_index_list,
        transformed_alphas_rhs,
    )
    if Z_matrix_derivatives is not None:
        loss_ders[1 : nZ_ders + 1] = group_loss_der_wrt_features(
            Z_matrix_derivatives,
            Z_U,
            Z_Vh,
            Z_singular_values,
            eigenvalue_multipliers,
            errors,
            loss_multipliers,
            test_index_list,
            used_quantities,
            importance_multipliers,
        )
    if quantity_derivatives is not None:
        loss_ders[-nquant_ders:] = group_loss_quantity_ders(
            quantity_derivatives,
            eigenvalue_multipliers,
            Z_U,
            loss_multipliers,
            test_index_list,
            importance_multipliers,
        )
    if return_errors:
        return loss, loss_ders, errors
    return loss, loss_ders


class SORFKfoldL2regLoss(KfoldL2regLossMixin, SORFLeaveOneOutL2regLoss):
    def __init__(
        self,
        train_quantities,
        kfolds=None,
        nkfolds=8,
        training_set_ratio=0.5,
        training_set_size=None,
        group_ids=None,
        leave_one_group_out=False,
        kfold_rng=None,
        mode="micro",
        point_loss=None,
        across_group_loss=None,
        aggregation_weights=None,
        **kwargs,
    ):
        """K-fold analogue of SORFLeaveOneOutL2regLoss, specialized to random features.

        The smoother is P = U diag( s^2 / (s^2 + l2reg) ) U^T for Z = U diag(s) Vh, so only
        the left singular vectors and the singular values are needed and Vh is never
        touched. The null space of Z^T Z drops out of the k-fold expression, which is why
        the same formula serves both nfeatures < npoints and nfeatures > npoints.

        Pass ``group_ids`` to hold out whole scaffolds instead of random blocks.

        NOTE: ``train_Z`` here is the feature matrix of the FULL cross-validation set; each
        fold trains on the complement of its validation block.
        """
        super().__init__(train_quantities, **kwargs)
        self.init_kfolds(
            train_quantities.shape[0],
            kfolds=kfolds,
            nkfolds=nkfolds,
            training_set_ratio=training_set_ratio,
            training_set_size=training_set_size,
            group_ids=group_ids,
            leave_one_group_out=leave_one_group_out,
            kfold_rng=kfold_rng,
            mode=mode,
            point_loss=point_loss,
            across_group_loss=across_group_loss,
            aggregation_weights=aggregation_weights,
        )

    def smoother_precalculations(self, l2reg):
        eigenvalue_multipliers = leaveoneout_eigenvalue_multipliers(
            self.train_Z_singular_values, l2reg
        )
        transformed_alphas_rhs = dot_(self.train_quantities, self.train_Z_U)
        reproduced_quantities = dot_(
            eigenvalue_multipliers * self.train_Z_U, transformed_alphas_rhs
        )
        return eigenvalue_multipliers, self.train_Z_U, reproduced_quantities


class SORFKfoldL2regOpt(KRRKfoldL2regOpt):
    def __init__(
        self,
        train_feature_vectors_generator,
        train_quantities,
        kfolds=None,
        nkfolds=8,
        training_set_ratio=0.5,
        training_set_size=None,
        group_ids=None,
        leave_one_group_out=False,
        kfold_rng=None,
        mode="micro",
        point_loss=None,
        across_group_loss=None,
        aggregation_weights=None,
        **kwargs,
    ):
        """K-fold analogue of SORFLeaveOneOutL2regOpt.

        As in KRRKfoldL2regOpt the folds are generated once and reused for every parameter
        evaluation, so the optimized quantity is a deterministic function of the
        parameters.
        """
        self.train_feature_vectors_generator = train_feature_vectors_generator
        self.train_quantities = train_quantities
        self.kfolds = build_kfolds(
            train_quantities.shape[0],
            kfolds=kfolds,
            nkfolds=nkfolds,
            training_set_ratio=training_set_ratio,
            training_set_size=training_set_size,
            group_ids=group_ids,
            rng=kfold_rng,
            leave_one_group_out=leave_one_group_out,
        )
        self.grouped_cv_mode = mode
        self.point_loss = point_loss
        self.across_group_loss = across_group_loss
        self.aggregation_weights = aggregation_weights
        self.basic_init(**kwargs)

    def get_l2reg_optimizer_from_parameters(self, parameters):
        train_feature_vectors = self.train_feature_vectors_generator(parameters)
        return SORFKfoldL2regLoss(
            self.train_quantities,
            kfolds=self.kfolds,
            mode=self.grouped_cv_mode,
            point_loss=self.point_loss,
            across_group_loss=self.across_group_loss,
            aggregation_weights=self.aggregation_weights,
            train_Z=train_feature_vectors,
            overwrite_train_kernel=True,
            loss_function=self.loss_function,
        )
