# NOTE K.Karan: A quick attempt of mine to predict model uncertainty in a way inspired by https://link.springer.com/chapter/10.1007/11736790_5
# The code capitalizes on some (IMHO) neat formulas, but the R2 and Spearman's correlation for errors vs. uncertainties are lackluster.
# If you are interested enough in the topic to potentially collaborate over it feel free to e-mail me at kvkarandashev@gmail.com.
from copy import deepcopy

from ..basic_utils import now, overwrite_when_possible
from ..jit_interfaces import exp_, log_, where_
from .models import MultilevelSORFModel
from .sorf_calculation import normalization_lvl

abs_uncertainty = "abs"
log_uncertainty = "log"

uncertainty_expressions = [abs_uncertainty, log_uncertainty]


class MultilevelSORFModelUncertainty(MultilevelSORFModel):
    def __init__(
        self,
        function_definition_list,
        parameter_list,
        intensive_quantity_shift=False,
        extensive_quantity_shift=False,
        extensive_features=False,
        error_ln_relative_l2reg_range=[-20.0, 2.0],
        error_ln_relative_l2reg_opt_iterpts=64,
        uncertainty_expression=log_uncertainty,
        error_model_kwargs={},
        **other_kwargs
    ):
        """
        Combined class for calculating both multilevel SORF model result and the corresponding uncertainty.
        """
        # construct model for result.
        self.msorf_model = MultilevelSORFModel(
            function_definition_list,
            parameter_list,
            intensive_quantity_shift=intensive_quantity_shift,
            extensive_quantity_shift=extensive_quantity_shift,
            **other_kwargs,
        )
        # construct model for result uncertainty.
        uncertainty_function_definition_list = deepcopy(function_definition_list)
        uncertainty_parameter_list = deepcopy(parameter_list)
        if extensive_features:
            uncertainty_function_definition_list += [normalization_lvl]
            uncertainty_parameter_list += [{}]
        used_uncertainty_model_kwargs = overwrite_when_possible(other_kwargs, error_model_kwargs)
        used_uncertainty_model_kwargs["ln_relative_l2reg_range"] = error_ln_relative_l2reg_range
        used_uncertainty_model_kwargs[
            "ln_relative_l2reg_opt_iterpts"
        ] = error_ln_relative_l2reg_opt_iterpts
        self.msorf_uncertainty_model = MultilevelSORFModel(
            uncertainty_function_definition_list,
            uncertainty_parameter_list,
            intensive_quantity_shift=True,
            extensive_quantity_shift=False,
            **used_uncertainty_model_kwargs,
        )
        self.clear_training_set()
        self.uncertainty_expression = uncertainty_expression
        assert self.uncertainty_expression in uncertainty_expressions

        # for final
        self.uncertainty_rhs = None
        self.uncertainty_importance_multipliers = None

    def init_uncertainty_rhs(self, uncertainty_ratios, training_plane_sq_distances):
        if self.uncertainty_expression == log_uncertainty:
            self.uncertainty_rhs = log_(uncertainty_ratios)
            self.uncertainty_importance_multipliers = None
            return
        if self.uncertainty_expression == abs_uncertainty:
            self.uncertainty_rhs = uncertainty_ratios
            self.uncertainty_importance_multipliers = training_plane_sq_distances
            return

        raise Exception

    def optimize_hyperparameters(self, **kwargs):
        # optimize hyperparameters for the value model
        self.msorf_model.assign_training_set(
            self.training_objects,
            self.training_quantities,
            training_nuclear_charges=self.training_nuclear_charges,
            training_importance_multipliers=self.training_importance_multipliers,
        )
        print("###Optimizing model hyperparameters:", now())
        self.msorf_model.optimize_hyperparameters(**kwargs)
        # initialize calculations of training plane distance
        print("###Generating RHS for uncertainty model:", now())
        weighted_Z_matrix = self.msorf_model.calc_training_weighted_Z_matrix()
        (
            uncertainty_ratios,
            training_plane_sq_distances,
        ) = self.msorf_model.initialize_uncertainty_components(
            Z_matrix=weighted_Z_matrix,
            training_quantities=self.msorf_model.get_shifted_weighted_training_quantities(),
        )
        self.init_uncertainty_rhs(uncertainty_ratios, training_plane_sq_distances)

        # optimize hyperparameters for the uncertainty model
        self.msorf_uncertainty_model.assign_training_set(
            self.training_objects,
            self.uncertainty_rhs,
            training_importance_multipliers=self.uncertainty_importance_multipliers,
        )
        print("###Optimizing uncertainty model hyperparameters:", now())
        self.msorf_uncertainty_model.optimize_hyperparameters(**kwargs)

    def fit(
        self,
        training_objects,
        training_quantities,
        training_nuclear_charges=None,
        training_importance_multipliers=None,
    ):
        (
            uncertainty_ratios,
            training_plane_sq_distances,
        ) = self.msorf_model.fit(
            training_objects,
            training_quantities,
            training_nuclear_charges=training_nuclear_charges,
            training_importance_multipliers=training_importance_multipliers,
            initialize_training_plane_sq_distance=True,
        )

        self.init_uncertainty_rhs(uncertainty_ratios, training_plane_sq_distances)
        self.msorf_uncertainty_model.fit(
            training_objects,
            self.uncertainty_rhs,
            training_importance_multipliers=self.uncertainty_importance_multipliers,
        )

    def get_leaveoneout_errors(self):
        return self.msorf_model.get_leaveoneout_errors()

    def get_uncertainty_prefacs(self, query_objects):
        prefacs = self.msorf_uncertainty_model.get_all_predictions(query_objects)
        if self.uncertainty_expression == log_uncertainty:
            return exp_(prefacs)
        if self.uncertainty_expression == abs_uncertainty:
            prefacs[where_(prefacs < 0.0)] = 0.0
            return prefacs
        raise Exception

    def get_all_predictions_with_uncertainties(self, query_objects, query_nuclear_charges=None):
        predictions, training_plane_sq_distances = self.msorf_model.get_all_predictions(
            query_objects, query_nuclear_charges=query_nuclear_charges, return_tpsd=True
        )

        uncertainty_prefacs = self.get_uncertainty_prefacs(query_objects)

        return predictions, uncertainty_prefacs * training_plane_sq_distances

    def get_nhyperparameters(self):
        return self.msorf_model.get_nhyperparameters()

    def learning_curve(self, *args, **kwargs):
        return self.msorf_model.learning_curve(*args, **kwargs)
