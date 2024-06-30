# Standardized learning curve building.
import random
from typing import List

import numpy as np
from numpy import ndarray

from .hyperparameter_optimization import MAE_lambda_derivative


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
