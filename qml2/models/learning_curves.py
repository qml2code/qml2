# Standardized learning curve building.
import random
from typing import List

import numpy as np
from numpy import ndarray

from ..jit_interfaces import mean_, std_
from .hyperparameter_optimization import MAE_lambda_derivative
from .loss_functions import MAE


def learning_curve_KRR(
    K_train: ndarray,
    y_train: ndarray,
    K_test: ndarray,
    y_test: ndarray,
    training_set_sizes: List[int],
    max_subset_num: int = 8,
):
    """
    Generate a MAE learning curve for KRR based on training and test kernel matrices.
    """
    ntrain = len(y_train)
    ntest = len(y_test)

    assert K_train.shape == (ntrain, ntrain)
    assert K_test.shape == (ntest, ntrain)

    all_ids = list(range(ntrain))
    random.shuffle(all_ids)

    MAEs = []

    MAEs_avs_std = []

    for training_set_size in training_set_sizes:
        MAEs_line = []
        num_subsets = min(max_subset_num, ntrain // training_set_size)

        all_subset_ids = np.array(random.sample(all_ids, training_set_size * num_subsets))
        lb = 0
        ub = training_set_size
        for _ in range(num_subsets):
            subset_ids = all_subset_ids[lb:ub]
            cur_K_train = K_train[subset_ids][:, subset_ids]
            cur_y_train = y_train[subset_ids]
            cur_K_test = K_test[:, subset_ids]

            MAE, invertible = MAE_lambda_derivative(
                cur_K_train, cur_y_train, cur_K_test, y_test, with_ders=False
            )

            if not invertible:
                raise Exception(
                    "Encountered non-invertible kernel during learning curve building."
                )

            MAEs_line.append(MAE)

            lb += training_set_size
            ub += training_set_size
        MAEs.append(MAEs_line)
        av = np.mean(MAEs_line)
        stddev = np.std(MAEs_line)
        MAEs_avs_std.append((av, stddev))
    return MAEs, MAEs_avs_std


def learning_curve_from_predictions(
    all_predictions, test_quantities, error_loss_function=MAE(), test_importance_multipliers=None
):
    """
    Learning curves from predictions (used in different model classes). Includes calculating both the mean loss and the standard deviation (e.g. over several k-folds).
    """
    loss_means = []
    loss_stds = []
    for subset_predictions in all_predictions:
        subset_losses = []
        for predictions in subset_predictions:
            errors = predictions - test_quantities
            if test_importance_multipliers is not None:
                errors *= test_importance_multipliers
            subset_losses.append(error_loss_function(errors))
        MAE_mean = mean_(subset_losses)
        MAE_std = std_(subset_losses)
        loss_means.append(MAE_mean)
        loss_stds.append(MAE_std)
    return loss_means, loss_stds
