# Subroutines for going from full sets of hyperparameters to reduced sets (with often decreased dimensionality)
# where optimization goes easier.
# KK: I originally looked into the topic because hyperparameter optimization for FJK was very painful and ReducedHyperparamFunc
# was designed as a base class for an FJK-friendly version. If we switch more towards
# standard libraries this section might prove redundant.
from ..jit_interfaces import exp_, log_


class ReducedHyperparamFunc:
    def __init__(self):
        """
        For going between the full hyperparameter set (lambda, global sigma, and other sigmas)
        and a reduced hyperparameter set. The default class uses no reduction, just rescaling to logarithms.
        """
        self.num_full_params = None
        self.num_reduced_params = None

    def initiate_param_nums(self, param_dim_arr):
        if (self.num_full_params is not None) and (self.num_reduced_params is not None):
            return
        if isinstance(param_dim_arr, int):
            self.num_full_params = param_dim_arr
        else:
            self.num_full_params = len(param_dim_arr)
        self.num_reduced_params = self.num_full_params

    def reduced_params_to_full(self, reduced_parameters):
        return exp_(reduced_parameters)

    def full_params_to_reduced(self, full_parameters):
        self.initiate_param_nums(full_parameters)
        return log_(full_parameters)

    def str_output_dict(self, global_name, output_dict=None):
        str_output = global_name + "\n"
        if output_dict is not None:
            str_output = global_name + "\n"
            for str_id in output_dict:
                str_output += str_id + ": " + str(output_dict[str_id]) + "\n"
        return str_output[:-1]

    def __str__(self):
        return self.str_output_dict("Default Reduced_hyperparam_func")

    def __repr__(self):
        return str(self)
