# Collection of routines for hyperparameter optimization specific for KRR.
import numpy as np
from numpy import ndarray
from scipy.linalg import cho_factor, cho_solve, svd
from scipy.optimize import bisect

from ..basic_utils import now
from ..math import cho_solve as cho_solve_full
from ..math import solve_from_svd
from ..utils import get_atom_environment_ranges, l2_norm_sq_dist


# Separating sets into training and test.
# training_test_atomic_separator - to separate individual atoms in a way that atoms from same molecule are all in one set.
def training_test_mols(total_set_size, training_set_size, nkfolds):
    temp_id_arr = np.array(range(total_set_size), dtype=int)
    shuffled_mols = np.empty((nkfolds, total_set_size), dtype=int)
    set_separators = np.repeat(training_set_size, nkfolds)
    for kfold_id in range(nkfolds):
        np.random.shuffle(temp_id_arr)
        shuffled_mols[kfold_id, :] = temp_id_arr[:]
    return shuffled_mols, set_separators


def training_test_mol_observables(
    mol_observable_numbers: ndarray,
    mol_set_size=None,
    mol_training_set_size=None,
    mol_kfolds=None,
    nkfolds=None,
):
    if mol_kfolds is None:
        shuffled_mols, mol_set_separators = training_test_mols(
            mol_set_size, mol_training_set_size, nkfolds
        )
    else:
        shuffled_mols = mol_kfolds.all_shuffled_indices
        mol_set_separators = mol_kfolds.all_training_test_set_separators
    total_set_size = sum(mol_observable_numbers)
    if nkfolds is None:
        nkfolds = shuffled_mols.shape[0]

    shuffled_observables = np.empty(
        (
            nkfolds,
            total_set_size,
        ),
        dtype=int,
    )
    ubound_arr = get_atom_environment_ranges(mol_observable_numbers)
    shuffled_observables_separators = np.repeat(-1, nkfolds)
    for kfold_id in range(nkfolds):
        observable_counter = 0
        for set_id, mol_id in enumerate(shuffled_mols[kfold_id]):
            if set_id == mol_set_separators[kfold_id]:
                shuffled_observables_separators[kfold_id] = observable_counter
            for observable_id in range(ubound_arr[mol_id], ubound_arr[mol_id + 1]):
                shuffled_observables[kfold_id, observable_counter] = observable_id
                observable_counter += 1
    assert np.all(shuffled_observables_separators >= 0)
    assert observable_counter == total_set_size
    return shuffled_observables, shuffled_observables_separators


# Creating k-folds.
# KK: Is there a standardised k-fold generator that allows training_test_atomic_separator usage?
class KFolds:
    def __init__(
        self,
        total_set_size,
        nkfolds,
        training_set_ratio=0.5,
        training_set_size=None,
    ):
        """
        Stores indices of KFolds for hyperparameter optimization.
        """
        self.total_set_size = total_set_size
        self.nkfolds = nkfolds
        self.training_set_size = training_set_size
        if (self.training_set_size is None) and (training_set_ratio is not None):
            self.training_set_size = int(training_set_ratio * total_set_size)
        self.all_shuffled_indices = None
        self.all_training_test_set_separators = None
        self.all_shuffled_indices, self.all_training_test_set_separators = training_test_mols(
            self.total_set_size, self.training_set_size, nkfolds
        )

    def train_test_indices(self, kfold_id):
        shuffled_indices = self.all_shuffled_indices[kfold_id]
        separator = self.all_training_test_set_separators[kfold_id]
        return shuffled_indices[:separator], shuffled_indices[separator:]


class KFoldsMultipleObservables(KFolds):
    def __init__(self, mol_observable_numbers, mol_kfolds=None, nkfolds=None, **kwargs):
        """
        Stores indices of KFolds for case when many observables are available for one molecule.
        """
        self.total_set_size = np.sum(mol_observable_numbers)
        self.mol_set_size = len(mol_observable_numbers)
        if mol_kfolds is None:
            mol_kfolds = KFolds(self.mol_set_size, nkfolds, **kwargs)
        (
            self.all_shuffled_indices,
            self.all_training_test_set_separators,
        ) = training_test_mol_observables(mol_observable_numbers, mol_kfolds=mol_kfolds)
        self.nkfolds = mol_kfolds.nkfolds


# For optimizing lambda in KRR without recalculating the kernel.
# Useful when kernel is expensive (i.e. FJK).
# @njit(fastmath=True)
def MAE_lambda_derivative(train_kernel, train_values, test_kernel, test_values, with_ders=True):
    """
    MAE and logarithmic derivative of MAE w.r.t. lambda.
    """
    try:
        U_mat, singular_values, Vh_mat = svd(train_kernel)
    except np.linalg.LinAlgError:
        # For some reason SVD did not converge.
        print("WARNING: SVD did not converge")
        singular_values = np.array([-1.0])
    if np.any(singular_values < 0.0):
        if with_ders:
            return np.inf, 0.0, False
        else:
            return np.inf, False
    alphas = solve_from_svd(U_mat, singular_values, Vh_mat, train_values, 0.0)
    predictions = np.matmul(test_kernel, alphas)
    errors = predictions - test_values
    MAE = np.mean(np.abs(errors))
    if not with_ders:
        return MAE, True
    alpha_ders = -solve_from_svd(U_mat, singular_values, Vh_mat, alphas, 0.0)
    prediction_ders = np.matmul(test_kernel, alpha_ders)
    MAE_der = np.mean(np.sign(errors) * prediction_ders)
    return MAE, MAE_der, True


# @njit(fastmath=True)
def KFolds_MAE_lambda_derivative_mod_kernel(
    full_kernel, all_values, all_shuffled_indices, all_separators, nkfolds
):
    av_MAE = 0.0
    av_MAE_der = 0.0
    for kfold_id in range(nkfolds):
        cur_indices = all_shuffled_indices[kfold_id]
        cur_separator = all_separators[kfold_id]
        train_indices = cur_indices[:cur_separator]
        test_indices = cur_indices[cur_separator:]
        train_kernel = full_kernel[train_indices][:, train_indices]
        test_kernel = full_kernel[test_indices][:, train_indices]
        cur_MAE, cur_MAE_der, valid = MAE_lambda_derivative(
            train_kernel, all_values[train_indices], test_kernel, all_values[test_indices]
        )
        if not valid:
            return 0.0, 0.0, False
        av_MAE += cur_MAE
        av_MAE_der += cur_MAE_der
    return av_MAE / nkfolds, av_MAE_der / nkfolds, True


# @njit(fastmath=True)
def KFolds_MAE_log_lambda_derivative(
    full_kernel,
    all_values,
    all_shuffled_indices,
    all_separators,
    nkfolds,
    saved_diag_vals,
    num_points,
    log_lambda_val,
):
    saved_diag_vals = np.empty((num_points,))
    lambda_val = np.exp(log_lambda_val)
    for diag_id in range(num_points):
        saved_diag_vals[diag_id] = full_kernel[diag_id, diag_id]
        full_kernel[diag_id, diag_id] += lambda_val
    av_MAE, av_MAE_der, valid = KFolds_MAE_lambda_derivative_mod_kernel(
        full_kernel, all_values, all_shuffled_indices, all_separators, nkfolds
    )
    for diag_id in range(num_points):
        full_kernel[diag_id, diag_id] = saved_diag_vals[diag_id]
    return av_MAE, av_MAE_der / lambda_val, valid


def KFolds_MAE_optimized_lambda(
    full_kernel: ndarray,
    all_values: ndarray,
    kfold: KFolds,
    lambda_log_precision=0.1,
    lambda_init_guess=1.0e-6,
    lambda_log_stride=np.log(4.0),
    max_lambda_log=0.0,
    min_lambda_log=-36.0,
):
    """
    Use bisection to determine optimal lambda.
    """
    num_points = all_values.shape[0]
    saved_diag_vals = np.copy(full_kernel[np.diag_indices(num_points)])

    # Convenient shorthands
    class MAE_wders:
        def __init__(self):
            self.call_counter = 0

        def __call__(self, log_lambda_val):
            self.call_counter += 1
            return KFolds_MAE_log_lambda_derivative(
                full_kernel,
                all_values,
                kfold.all_shuffled_indices,
                kfold.all_training_test_set_separators,
                kfold.nkfolds,
                saved_diag_vals,
                num_points,
                log_lambda_val,
            )

    MAE_wders_func = MAE_wders()

    def MAE_der(log_lambda_val):
        _, MAE_der, converged = MAE_wders_func(log_lambda_val)
        if converged:
            return MAE_der
        else:
            return (
                -1.0
            )  # as lambda increases kernel becomes more stable, going from infinite error to finite.

    # Determine derivative sign and MAE at initial point:
    init_log_lambda = np.log(lambda_init_guess)
    start_MAE, start_der, converged = MAE_wders_func(init_log_lambda)
    if not converged:
        start_MAE = None
        start_der = -1.0
    interval_step = -lambda_log_stride * np.sign(start_der)
    l1 = init_log_lambda
    l2 = l1 + interval_step
    der1 = start_der
    MAE1 = start_MAE
    MAE2, der2, _ = MAE_wders_func(l2)
    while (
        (max(l1, l2) < max_lambda_log) and (min(l1, l2) > min_lambda_log) and (der1 * der2 > 0.0)
    ):
        l1 = l2
        MAE1 = MAE2
        l2 += interval_step
        MAE2, der2, _ = MAE_wders_func(l2)
    if der1 * der2 < 0.0:
        min_lambda_log = bisect(MAE_der, l1, l2, xtol=lambda_log_precision, full_output=False)
        finish_MAE, _, _ = MAE_wders_func(min_lambda_log)
        if (start_MAE is not None) and (finish_MAE > start_MAE):
            finish_MAE = start_MAE
            min_lambda_log = init_log_lambda
        return finish_MAE, np.exp(min_lambda_log), MAE_wders_func.call_counter
    else:
        return MAE1, np.exp(l1), MAE_wders_func.call_counter


class callable_MAE_w_opt_lambda:
    def __init__(
        self,
        train_kernel_generator,
        train_vals,
        lambda_init=1.0e-6,
        nkfolds=8,
        training_set_ratio=0.5,
    ):
        """
        For optimization of sigma in KRR in such a way that for each kernel
        matrix calculation lambda value is found via bisection. Useful for
        methods where kernel calculations are expensive.
        """
        self.last_used_lambda = lambda_init
        self.train_kernel_generator = train_kernel_generator
        self.train_vals = train_vals
        self.kfold = KFolds(len(self.train_vals), nkfolds, training_set_ratio=training_set_ratio)
        self.MAE_lambda_log = []

    def lambda_init_guess(self, params):
        if len(self.MAE_lambda_log) == 0:
            return self.last_used_lambda
        min_tuple = min(self.MAE_lambda_log, key=lambda x: l2_norm_sq_dist(x[1], params))
        return min_tuple[2]

    def __call__(self, parameters):
        params = parameters[0]
        sym_matrix = self.train_kernel_generator(params)
        MAE, self.last_used_lambda, num_evals = KFolds_MAE_optimized_lambda(
            sym_matrix,
            self.train_vals,
            self.kfold,
            lambda_init_guess=self.lambda_init_guess(params),
        )
        self.MAE_lambda_log.append(
            (MAE, np.copy(params), np.copy(self.last_used_lambda), num_evals)
        )  # saving minimal lambda through BOSS would be problematic.
        print("Latest evaluation:", *self.MAE_lambda_log[-1])
        return MAE

    def smallest_encountered_MAE(self):
        return min(self.MAE_lambda_log, key=lambda x: x[0])[:3]


# For BO in space of both lambda and sigma.
class callable_ninv_MAE:
    def __init__(
        self,
        symmetric_kernel_function,
        training_representations,
        training_quantities,
        kernel_function_kwargs={},
        nkfolds=8,
        training_set_ratio=0.5,
    ):
        """
        For optimization of sigma and lambda.
        """
        self.init_training_set(training_representations, training_quantities)
        self.init_kernel(symmetric_kernel_function, kernel_function_kwargs)
        self.init_kfolds(nkfolds=nkfolds, training_set_ratio=training_set_ratio)

    def init_kernel(self, symmetric_kernel_function, kernel_function_kwargs):
        self.symmetric_kernel_function = symmetric_kernel_function
        self.kernel_function_kwargs = kernel_function_kwargs

    def init_training_set(self, training_representations, training_quantities):
        self.training_representations = training_representations
        self.training_quantities = training_quantities
        self.training_set_size = len(self.training_representations)

    def init_kfolds(self, nkfolds=8, training_set_ratio=0.5):
        self.nkfolds = nkfolds
        self.kfolds = KFolds(
            len(self.training_quantities), self.nkfolds, training_set_ratio=training_set_ratio
        )

    def get_full_kernel_matrix(self, sigma):
        return self.symmetric_kernel_function(
            self.training_representations, sigma, **self.kernel_function_kwargs
        )

    def __call__(self, ln_parameters):
        """
        Calculate mean MAE over all kfolds and returns the negative inverse of it. If for at least one MAE
        the training matrix is non-invertible returns zero.

        input:
        ln_parameters : array of shape (2,), parameters[0] is logarithm of the ratio of lambda and mean diagonal
                        element of the kernel matrix calculated for all training representations; parameters[1]
                        is logarithm of sigma.
        output:
        ninv_MAE : negative inverse of MAE averaged over all kfolds. If at least one MAE cannot be evaluated
                    due to training matrix being non-invertible returns 0.
        """
        print("started MAE calculation for:", ln_parameters, now())
        normalized_lambda, sigma = np.exp(ln_parameters[0])

        full_kernel_matrix = self.get_full_kernel_matrix(sigma)
        lambda_val = normalized_lambda * np.mean(np.diagonal(full_kernel_matrix))
        full_kernel_matrix[np.diag_indices_from(full_kernel_matrix)] += lambda_val

        tot_MAE = 0.0
        for kfold_id in range(self.nkfolds):
            train_indices, test_indices = self.kfolds.train_test_indices(kfold_id)
            kfold_train_kernel = full_kernel_matrix[train_indices][:, train_indices]
            kfold_test_kernel = full_kernel_matrix[test_indices][:, train_indices]
            kfold_train_quantities = self.training_quantities[train_indices]
            try:
                kfold_alphas = cho_solve_full(kfold_train_kernel, kfold_train_quantities)
            except np.linalg.LinAlgError:
                return 0.0
            kfold_predictions = np.dot(kfold_test_kernel, kfold_alphas)
            tot_MAE += np.mean(np.abs(kfold_predictions - self.training_quantities[test_indices]))
        av_MAE = tot_MAE / self.nkfolds
        print("finished calculations:", av_MAE, now())
        return -1 / av_MAE


class callable_ninv_MAE_local(callable_ninv_MAE):
    def __init__(
        self,
        symmetric_kernel_function,
        training_representations_list,
        training_quantities,
        kernel_function_kwargs={},
        **kfolds_kwargs
    ):
        """
        For optimization of sigma and lambda.
        """
        self.init_training_set(training_representations_list, training_quantities)
        self.init_kernel(symmetric_kernel_function, kernel_function_kwargs)
        self.init_kfolds(**kfolds_kwargs)

    def init_training_set(self, training_representations_list, training_quantities):
        self.training_representations = np.concatenate(training_representations_list)
        self.training_natoms = np.array([len(rep) for rep in training_representations_list])
        self.training_set_size = self.training_natoms.shape[0]
        if training_quantities is not None:
            self.training_quantities = training_quantities
            self.num_observables = len(self.training_quantities)

    def get_full_kernel_matrix(self, sigma):
        return self.symmetric_kernel_function(
            self.training_representations,
            self.training_natoms,
            sigma,
            **self.kernel_function_kwargs
        )


class callable_ninv_MAE_local_dn(callable_ninv_MAE_local):
    def __init__(
        self,
        symmetric_kernel_function,
        training_representations_list,
        training_nuclear_charges_list,
        training_quantities,
        kernel_function_kwargs={},
        **kfolds_kwargs
    ):
        """
        For optimization of sigma and lambda.
        """
        self.init_training_set(
            training_representations_list, training_nuclear_charges_list, training_quantities
        )
        self.init_kfolds(**kfolds_kwargs)
        self.init_kernel(symmetric_kernel_function, kernel_function_kwargs)

    def init_training_set(
        self, training_representations_list, training_nuclear_charges_list, training_quantities
    ):
        super().init_training_set(training_representations_list, training_quantities)
        self.training_nuclear_charges = np.concatenate(training_nuclear_charges_list)

    def get_full_kernel_matrix(self, sigma):
        return self.symmetric_kernel_function(
            self.training_representations,
            self.training_natoms,
            self.training_nuclear_charges,
            sigma,
            **self.kernel_function_kwargs
        )


def GPR_MLL_single_quantity(cholesky_factorization, train_quantities):
    Ntrain = train_quantities.shape[0]
    quant_inv_kern_product = cho_solve(cholesky_factorization, train_quantities)
    MLL = -(np.dot(train_quantities, quant_inv_kern_product) + Ntrain * np.log(2 * np.pi)) / 2
    # add -log(det(K))/2
    # using log instead of multiplication to avoid potential overflow error.
    L = cholesky_factorization[0]
    diag_ids = np.diag_indices_from(L)
    MLL -= np.sum(np.log(L[diag_ids]))
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
        cholesky_factorization = cho_factor(train_kernel)
    except np.linalg.LinAlgError:
        return -np.inf
    if len(train_quantities.shape) == 1:
        return GPR_MLL_single_quantity(cholesky_factorization, train_quantities)
    nquants = train_quantities.shape[0]
    MLL = 0.0
    for quant_id in range(nquants):
        MLL += GPR_MLL_single_quantity(cholesky_factorization, train_quantities[quant_id])
