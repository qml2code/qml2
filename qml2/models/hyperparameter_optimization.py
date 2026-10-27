# Collection of routines for hyperparameter optimization specific for KRR.
#
# The k-fold SPLITTING machinery, and the older loss classes that refit the model once per
# fold, live in .kfolds; what is here evaluates a hyperparameter from a single
# decomposition of the full training matrix instead.
from ..jit_interfaces import (
    LinAlgError_,
    allow_numba_numpy_parallelization,
    any_,
    cho_factor_,
    cho_solve_,
    concatenate_,
    diag_indices_from_,
    dot_,
    empty_,
    exp_,
    inf_,
    jit_,
    log_,
    mean_,
    pi_,
    prange_,
    sum_,
)
from ..math import cho_solve_from_matrix, get_factorization_w_l2reg, gram_eigh
from ..optimizers import global_optimize_1D
from ..utils import flatten_to_scalar
from .kfolds import build_kfolds
from .loss_functions import MAE, grouped_cv_loss
from .math import ninv_f


def GPR_MLL_single_quantity(cholesky_factorization, train_quantities):
    Ntrain = train_quantities.shape[0]
    quant_inv_kern_product = cho_solve_(cholesky_factorization, train_quantities)
    MLL = -(dot_(train_quantities, quant_inv_kern_product) + Ntrain * log_(2 * pi_)) / 2
    # add -log(det(K))/2
    # using log instead of multiplication to avoid potential overflow error.
    L = cholesky_factorization[0]
    diag_ids = diag_indices_from_(L)
    MLL -= sum_(log_(L[diag_ids]))
    return MLL


# TODO: K.Karan: Someone needs to double-check I did the generalization to several quantities correctly.
def GPR_Marginal_Likelihood_Logarithm(train_kernel, train_quantities):
    """
    Calculate GPR's marginal likelihood logarithm (expression taken from: https://doi.org/10.1021/acs.chemrev.1c00022, Eq. 49).
    Maximizing it gives an estimate of optimal hyperparameters.

    train_kernel : ndarray (Ntrain * Ntrain)
        (regularized) kernel matrix for the training set.
    train_quantities : ndarray (Ntrain or nquats * Ntrain)
        array of training set's quantities of interest. If there is only one quantity can be 1D.
    """
    try:
        cholesky_factorization = cho_factor_(train_kernel)
    except LinAlgError_:
        return -inf_
    if len(train_quantities.shape) == 1:
        return GPR_MLL_single_quantity(cholesky_factorization, train_quantities)
    nquants = train_quantities.shape[0]
    MLL = 0.0
    for quant_id in range(nquants):
        MLL += GPR_MLL_single_quantity(cholesky_factorization, train_quantities[quant_id])


# For calculating leave-one-out errors using DOI:10.1016/S0031-3203(03)00136-5
def get_leaveoneout_stat_factors(train_kernel, cholesky_factorization):
    inverse_multiplied_kernel = cho_solve_(cholesky_factorization, train_kernel)
    inv_mult_kern_diag_indices = diag_indices_from_(inverse_multiplied_kernel)
    inv_mult_kern_diag_elements = inverse_multiplied_kernel[inv_mult_kern_diag_indices]
    return 1 / (1 - inv_mult_kern_diag_elements)


def KRR_leaveoneout_errors(train_kernel, train_quantities, l2reg=None):
    """
    KRR leave-one-out errors calculated with the formula from DOI:10.1016/S0031-3203(03)00136-5.
    """
    # get_factorization_w_l2reg adds l2reg to the diagonal, factorizes, and restores the
    # diagonal afterwards -- which is what the hand-rolled version here used to do, except
    # that it copied the whole kernel to avoid touching it. Sharing the helper keeps the
    # torch-backend detachment and the l2reg_diag_ratio option in one place.
    cholesky_factorization = get_factorization_w_l2reg(
        train_kernel, cho_factor_, l2reg=l2reg, overwrite_a=False
    )
    alphas = cho_solve_(cholesky_factorization, train_quantities)
    repr_quantities = dot_(train_kernel, alphas)
    stat_factors = get_leaveoneout_stat_factors(train_kernel, cholesky_factorization)
    return (repr_quantities - train_quantities) * stat_factors


def KRR_leaveoneout_loss(train_kernel, train_quantities, l2reg=None, loss_function=MAE()):
    """
    Loss (by default MAE) KRR from leave-one-out errors calculated with the formula from DOI:10.1016/S0031-3203(03)00136-5.
    """
    leaveoneout_errors = KRR_leaveoneout_errors(train_kernel, train_quantities, l2reg=l2reg)
    return loss_function(leaveoneout_errors)


# Procedures for optimizing l2reg w.r.t. leave-one-out loss using a pre-calculated
# eigendecomposition (qml2.math.gram_eigh).
@jit_(numba_parallel=allow_numba_numpy_parallelization())
def get_leaveoneout_stat_factors_from_eigh(eigenval_multipliers, train_kernel_eigenvectors):
    npoints = train_kernel_eigenvectors.shape[0]
    dot_products = empty_(eigenval_multipliers.shape)
    for point_id in prange_(npoints):
        dot_products[point_id] = dot_(
            eigenval_multipliers, train_kernel_eigenvectors[point_id] ** 2
        )
    return 1 / (1 - dot_products)


class KRRLeaveOneOutL2regLoss:
    def __init__(
        self,
        train_quantities,
        train_kernel=None,
        train_kernel_eigenvalues=None,
        train_kernel_eigenvectors=None,
        loss_function=MAE(),
        overwrite_train_kernel=False,
    ):
        """
        For convenient optimization of l2 regularization w.r.t. leave-one-out errors.

        NOTE: The multitude of attributes related to different versions of l2reg are because:
            - l2reg is the quantity actually added to the training kernel's diagonal matrix;
            - l2reg_rel_diag is the ratio of l2reg and average diagonal element in the training kernel; works better e.g. if you re-train on larger molecules with local or local-dn kernels;
            - ln_l2reg_rel_diag is the most convenient variable for optimization since it is connected to l2reg_rel_diag and enforces the latter's positivity.
        """
        if train_kernel_eigenvalues is None or train_kernel_eigenvectors is None:
            assert train_kernel is not None
            # gram_eigh rather than scipy.linalg.eigh: it picks the LAPACK backend by size,
            # which matters in both directions. Below ~46341 it uses driver="evr", whose
            # workspace is O(N) instead of numpy's ~2*N^2; above it, scipy's LP64 LAPACK
            # overflows outright and gram_eigh falls back to an ILP64 backend.
            train_kernel_eigenvalues, train_kernel_eigenvectors = gram_eigh(
                train_kernel, overwrite_a=overwrite_train_kernel
            )
        self.train_kernel_eigenvalues = train_kernel_eigenvalues
        self.train_kernel_eigenvectors = train_kernel_eigenvectors
        self.train_quantities = train_quantities
        self.mean_diag_element = mean_(self.train_kernel_eigenvalues)
        self.loss_function = loss_function
        self.train_quantities = train_quantities
        self.eigenvalue_rhs = dot_(self.train_kernel_eigenvectors.T, train_quantities)

    def calculate_for_l2reg(self, l2reg):
        if any_(self.train_kernel_eigenvalues <= -l2reg):
            raise LinAlgError_
        eigenval_multipliers = self.train_kernel_eigenvalues / (
            self.train_kernel_eigenvalues + l2reg
        )
        reproduced_quantities = dot_(
            self.train_kernel_eigenvectors, eigenval_multipliers * self.eigenvalue_rhs
        )
        stat_factors = get_leaveoneout_stat_factors_from_eigh(
            eigenval_multipliers, self.train_kernel_eigenvectors
        )
        leaveoneout_errors = (reproduced_quantities - self.train_quantities) * stat_factors
        return self.loss_function(leaveoneout_errors)

    def l2reg_from_relative(self, l2reg_rel_diag):
        return l2reg_rel_diag * self.mean_diag_element

    def l2reg_from_relative_ln(self, ln_l2reg_rel_diag):
        return self.l2reg_from_relative(exp_(ln_l2reg_rel_diag))

    def calculate_for_l2reg_rel_diag(self, l2reg_rel_diag):
        l2reg = self.l2reg_from_relative(l2reg_rel_diag)
        return self.calculate_for_l2reg(l2reg)

    def calculate_for_ln_l2reg_rel_diag(self, ln_l2reg_rel_diag):
        l2reg = self.l2reg_from_relative_ln(ln_l2reg_rel_diag)
        return self.calculate_for_l2reg(l2reg)

    def __call__(self, ln_l2reg_rel_diag):
        # for optimizer calls
        ln_l2reg_rel_diag = flatten_to_scalar(ln_l2reg_rel_diag)
        return self.calculate_for_ln_l2reg_rel_diag(ln_l2reg_rel_diag)


class KRRLeaveOneOutL2regOpt:
    def __init__(self, train_kernel_generator, train_quantities, **kwargs):
        """
        Class that for a given sigma uses BOSS to optimize l2reg w.r.t. leave-one-out error loss, and then calculates the corresponding minimal loss
        """
        self.train_kernel_generator = train_kernel_generator
        self.train_quantities = train_quantities
        self.basic_init(**kwargs)

    def basic_init(
        self,
        loss_function=MAE(),
        ln_l2reg_diag_ratio_bounds=[-25.0, 0.0],
        total_iterpts=32,
        opt_method="boss",
    ):
        self.loss_function = loss_function
        self.ln_l2reg_diag_ratio_bounds = ln_l2reg_diag_ratio_bounds
        self.total_iterpts = total_iterpts
        self.opt_method = opt_method

    def get_l2reg_optimizer_from_parameters(self, parameters):
        train_kernel = self.train_kernel_generator(parameters)
        return KRRLeaveOneOutL2regLoss(
            self.train_quantities,
            train_kernel=train_kernel,
            overwrite_train_kernel=True,
            loss_function=self.loss_function,
        )

    def __call__(self, parameters, min_output=False):
        internal_l2reg_loss = self.get_l2reg_optimizer_from_parameters(parameters)
        final_minimized_l2reg_loss = ninv_f(internal_l2reg_loss)
        min_ln_l2reg_rel_diag = global_optimize_1D(
            final_minimized_l2reg_loss,
            self.ln_l2reg_diag_ratio_bounds,
            opt_name="l2reg_opt",
            total_iterpts=self.total_iterpts,
            opt_method=self.opt_method,
        )

        if min_output:
            return min_ln_l2reg_rel_diag, internal_l2reg_loss

        final_loss = final_minimized_l2reg_loss(min_ln_l2reg_rel_diag)
        print("Final loss:", final_loss)
        return final_loss


# K-fold generalization of the leave-one-out expressions above.
#
# Both KRR and random-feature ridge are linear smoothers: the fit on the full set S
# reproduces the training quantities as y_hat = P y, with
#
#     KRR:             P = Q diag( w_i / (w_i + l2reg) ) Q^T   ,  K = Q diag(w) Q^T
#     random features: P = U diag( s_i^2 / (s_i^2 + l2reg) ) U^T,  Z = U diag(s) Vh
#
# For a validation block V, with the model trained on S \ V, the validation errors follow
# from P alone and WITHOUT refitting:
#
#     e_V = (I - P_VV)^{-1} ( (P y)_V - y_V )
#
# This is exact rather than an approximation, and for |V| = 1 it collapses to the familiar
# leave-one-out expression with stat_factors = 1 / (1 - P_ii) that
# get_leaveoneout_stat_factors_from_eigh implements. It is the Schur-complement plus
# eigendecomposition scheme of arXiv:2605.15927 (Sec. II.2-II.3) written in terms of the
# smoother matrix: with M = K + l2reg*I decomposed once, (I - P_VV) = l2reg * H, where
# H = Q_V diag(1/(w+l2reg)) Q_V^T is the validation-validation block of M^{-1}.
#
# Cost per (kernel, l2reg) pair is nkfolds * (Ntest^2 * Ntotal + Ntest^3 / 3), so the
# scheme pays off when Ntest is a modest fraction of Ntotal. The one-off eigh/svd is
# shared across every l2reg value AND every fold, which is where the saving comes from.
#
# The folds may equally be GROUP folds -- pass a GroupKFolds, or group_ids, and the
# validation blocks become whole scaffolds. Nothing below changes: the expression only
# needs each fold's train/validation sets to partition the set, which group folds do.


def get_kfold_stat_matrix(eigenval_multipliers, test_basis_vectors):
    """``I - P_VV`` for one validation block.

    The k-fold analogue of the reciprocal of ``get_leaveoneout_stat_factors_from_eigh``:
    there the block is one point and this is a scalar.

    eigenval_multipliers : ndarray (nbasis,)
        w_i / (w_i + l2reg) for KRR, s_i^2 / (s_i^2 + l2reg) for random features.
    test_basis_vectors : ndarray (ntest, nbasis)
        rows of the kernel's eigenvector matrix (or of Z's left singular vectors)
        belonging to the validation block.
    """
    stat_matrix = -dot_(test_basis_vectors * eigenval_multipliers, test_basis_vectors.T)
    diag_ids = diag_indices_from_(stat_matrix)
    stat_matrix[diag_ids] += 1.0
    return stat_matrix


def kfold_errors_from_smoother(
    eigenval_multipliers,
    test_basis_vectors,
    reproduced_test_quantities,
    test_quantities,
):
    """Errors on one validation block for a model trained on its complement.

    ``I - P_VV`` is positive definite for l2reg > 0, hence the Cholesky solve.
    """
    stat_matrix = get_kfold_stat_matrix(eigenval_multipliers, test_basis_vectors)
    return cho_solve_from_matrix(stat_matrix, reproduced_test_quantities - test_quantities)


class KfoldL2regLossMixin:
    """Shared k-fold machinery for the KRR and random-feature l2reg losses.

    Subclasses supply ``smoother_precalculations(l2reg)`` returning
    ``(eigenval_multipliers, basis_vectors, reproduced_quantities)``, and are expected to
    already carry ``train_quantities`` and ``loss_function``.
    """

    def init_kfolds(
        self,
        total_set_size,
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
    ):
        """
        leave_one_group_out : bool
            True gives one fold per group -- the leave-one-GROUP-out criterion -- instead of
            ``nkfolds`` group-respecting folds. Requires ``group_ids``, and makes ``nkfolds``
            and ``kfold_rng`` immaterial: the fold count IS the group count, and with one
            group per fold an rng could only permute the order of folds, which nothing
            aggregating over them depends on.

            Without it, ``group_ids`` alone gives a group-respecting k-fold split at the
            DEFAULT ``nkfolds=8`` -- a different criterion, and with more groups than folds a
            silently different one. That is why this is a flag rather than leaving callers to
            pass ``nkfolds=len(group_index_lists(group_ids))`` themselves.

        kfold_rng : numpy Generator or None
            Seeds the fold construction. Without it the plain KFolds path draws from the
            GLOBAL numpy random state, so two loss objects built from identical arguments
            hold DIFFERENT folds -- which makes a comparison between them meaningless and
            the run irreproducible. None keeps that historical behaviour; pass a Generator
            for a split that can be reproduced.

            Named kfold_rng rather than rng because these classes may later need their own
            generator for the SORF feature matrices, and one ambiguous ``rng`` shared
            between the two would be worse than two explicit names.

        group_ids : ndarray or None
            Per-point group labels deciding how the DATA is split. When given, the
            validation blocks become whole groups (see qml2.models.kfolds and
            qml2.cheminfo.scaffolds) and the criterion becomes leave-one-GROUP-out rather
            than leave-one-point-out.

        mode, point_loss, across_group_loss, aggregation_weights
            Passed to qml2.models.loss_functions.grouped_cv_loss, which turns the per-fold
            residuals into the scalar this class returns. See its docstring; note that the
            "group" it averages over here is the FOLD, not the scaffold group that
            ``group_ids`` builds the folds from. The distinction matters because a plain
            KFolds produces overlapping validation blocks, so a point can contribute a
            residual to several folds -- labelling by fold is well defined either way,
            labelling by point would not be.

            ``point_loss=None`` falls back to ``self.loss_function``, itself MAE() by
            default, so the default objective is MAE over the pooled residuals.

        Replaces the earlier ``average_over_kfolds`` flag, exactly:
        ``average_over_kfolds=False`` is ``mode="micro"`` (pool every residual, one loss
        call) and ``average_over_kfolds=True`` is ``mode="macro"`` with uniform weights
        (mean of the per-fold losses). NOTE THE DEFAULT MOVED: it was the averaging form,
        and is now "micro", which weights every residual equally rather than every fold.
        That is the form the paper uses, and the one that does not silently up-weight the
        points in small folds.
        """
        self.kfolds = build_kfolds(
            total_set_size,
            kfolds=kfolds,
            nkfolds=nkfolds,
            training_set_ratio=training_set_ratio,
            training_set_size=training_set_size,
            group_ids=group_ids,
            rng=kfold_rng,
            leave_one_group_out=leave_one_group_out,
        )
        self.grouped_cv_mode = mode
        if point_loss is None:
            # loss_function is NOT defined on this mixin, and looking at the mixin alone
            # makes this line seem pointless. It resolves through the MRO SIBLING: the
            # concrete classes are (KfoldL2regLossMixin, <something>LeaveOneOutL2regLoss),
            # and that second base's __init__ -- which runs before init_kfolds -- sets
            # self.loss_function, itself defaulting to MAE(). So passing loss_function= and
            # leaving point_loss=None uses the loss that was passed.
            #
            # getattr rather than self.loss_function so the mixin still satisfies its
            # documented contract ("None means MAE()") if it is ever combined with a base
            # that does not define one.
            point_loss = getattr(self, "loss_function", MAE())
        self.point_loss = point_loss
        self.across_group_loss = across_group_loss
        self.aggregation_weights = aggregation_weights

    def smoother_precalculations(self, l2reg):
        raise NotImplementedError

    def kfold_errors(self, l2reg):
        """Validation error arrays, one per fold."""
        (
            eigenval_multipliers,
            basis_vectors,
            reproduced_quantities,
        ) = self.smoother_precalculations(l2reg)
        all_errors = []
        for kfold_id in range(self.kfolds.nkfolds):
            _, test_indices = self.kfolds.train_test_indices(kfold_id)
            all_errors.append(
                kfold_errors_from_smoother(
                    eigenval_multipliers,
                    basis_vectors[test_indices],
                    reproduced_quantities[test_indices],
                    self.train_quantities[test_indices],
                )
            )
        return all_errors

    def fold_labels(self, all_errors):
        """One label per residual, naming the fold it came out of.

        The residuals are indexed by (point, fold) rather than by point, because with a
        plain KFolds the validation blocks overlap and a point can appear in several. Fold
        labels are well defined in both the overlapping and the partitioning case.
        """
        return [
            fold_id for fold_id, errors in enumerate(all_errors) for _ in range(errors.shape[0])
        ]

    def calculate_for_l2reg(self, l2reg):
        all_errors = self.kfold_errors(l2reg)
        return grouped_cv_loss(
            concatenate_(all_errors),
            self.fold_labels(all_errors),
            self.point_loss,
            mode=self.grouped_cv_mode,
            across_group_loss=self.across_group_loss,
            aggregation_weights=self.aggregation_weights,
        )


class KRRKfoldL2regLoss(KfoldL2regLossMixin, KRRLeaveOneOutL2regLoss):
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
        """K-fold analogue of KRRLeaveOneOutL2regLoss, via arXiv:2605.15927 Sec. II.2-II.3.

        NOTE: ``train_kernel`` (or its eigendecomposition) here is the kernel of the FULL
        cross-validation set, training and validation points together; each fold's
        training set is the complement of its validation block. The name is kept for
        interface compatibility with the leave-one-out classes.

        Folds can be passed as an object via ``kfolds``, described by ``nkfolds`` /
        ``training_set_ratio`` / ``training_set_size``, or made group-aware with
        ``group_ids``.
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
        if any_(self.train_kernel_eigenvalues <= -l2reg):
            raise LinAlgError_
        eigenval_multipliers = self.train_kernel_eigenvalues / (
            self.train_kernel_eigenvalues + l2reg
        )
        reproduced_quantities = dot_(
            self.train_kernel_eigenvectors, eigenval_multipliers * self.eigenvalue_rhs
        )
        return eigenval_multipliers, self.train_kernel_eigenvectors, reproduced_quantities


class KRRKfoldL2regOpt(KRRLeaveOneOutL2regOpt):
    def __init__(
        self,
        train_kernel_generator,
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
        """K-fold analogue of KRRLeaveOneOutL2regOpt.

        For given kernel parameters, optimizes l2reg against the k-fold loss and returns
        the minimal loss.

        NOTE: the folds are generated ONCE and reused for every parameter evaluation, so
        the optimized quantity is a deterministic function of the parameters. Regenerating
        them per evaluation would add noise the optimizer would chase.
        """
        self.train_kernel_generator = train_kernel_generator
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
        train_kernel = self.train_kernel_generator(parameters)
        return KRRKfoldL2regLoss(
            self.train_quantities,
            kfolds=self.kfolds,
            mode=self.grouped_cv_mode,
            point_loss=self.point_loss,
            across_group_loss=self.across_group_loss,
            aggregation_weights=self.aggregation_weights,
            train_kernel=train_kernel,
            overwrite_train_kernel=True,
            loss_function=self.loss_function,
        )
