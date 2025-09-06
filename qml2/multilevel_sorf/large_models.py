"""
Routines for conveniently applying SORF for datasets too large to be stored in a computer's CPU in their entirety.

The future plan is to implementing fine-tuning here.
"""
from ..jit_interfaces import diag_indices_from_, dot_, mean_, zeros_
from ..math import cho_solve
from ..utils import check_allocation
from .base_constructors import get_dict2class
from .math import inplace_add_dot_product
from .models import MultilevelSORFModel


class LargeMSORFModel:
    def __init__(
        self,
        function_definition_list,
        parameter_list,
        use_inplace_dot_addition=False,
        full_hyperparameter_optimization=False,
        **kwargs,
    ):
        """
        A class for training MSORF models without storing all the training data in RAM at once.
        """
        self.base_model = MultilevelSORFModel(function_definition_list, parameter_list, **kwargs)
        self.nfeatures = self.base_model.nfeatures()
        self.objects_processor = None
        self.use_inplace_dot_addition = use_inplace_dot_addition
        self.full_hyperparameter_optimization = full_hyperparameter_optimization
        self.clear_arrays()

    def init_object_processing(self):
        if self.objects_processor is not None:
            return
        objects_full_definition_list = self.base_model.get_query_objects_full_definition_list()
        assert (
            objects_full_definition_list is not None
        ), "objects_definition_list should have been defined!"
        self.objects_processor = get_dict2class(objects_full_definition_list)

    def start_training(self):
        self.training_kernel = zeros_((self.nfeatures, self.nfeatures))
        self.training_rhs = zeros_(self.nfeatures)

    def optimize_hyperparameters_full(self, training_objects, training_quantities):
        self.base_model.assign_training_set(training_objects, training_quantities)
        self.base_model.optimize_hyperparameters()
        self.base_model.clear_training_set()

    def optimize_hyperparameters_short(self, training_objects, training_quantities):
        self.base_model.hyperparameters = self.base_model.find_hyperparameter_guesses(
            training_objects
        )

    def optimize_hyperparameters(self, training_objects, training_quantities):
        if self.full_hyperparameter_optimization:
            self.optimize_hyperparameters_full(training_objects, training_quantities)
        else:
            self.optimize_hyperparameters_short(training_objects, training_quantities)

    def check_hyperparameters(self, training_objects, training_quantities):
        if self.base_model.hyperparameters is not None:
            return
        print(
            "WARNING: hyperparameters not initialized, estimating from the current training batch"
        )
        self.optimize_hyperparameters(training_objects, training_quantities)

    def calculate_training_features(self, training_objects):
        nmols = len(training_objects)
        self.temp_feature_vectors = check_allocation(
            (nmols, self.nfeatures), output=self.temp_feature_vectors
        )
        return self.base_model.calc_Z_matrix(
            training_objects, temp_Z_matrix=self.temp_feature_vectors[:nmols]
        )

    def get_processed_objects(self, objects):
        return self.objects_processor(objects)

    def add_processed_training_points(self, training_objects, training_quantities):
        self.check_hyperparameters(training_objects, training_quantities)
        feature_vectors = self.calculate_training_features(training_objects)
        if self.training_kernel is None or self.training_rhs is None:
            self.start_training()
        self.temp_training_rhs_addition = dot_(
            feature_vectors.T, training_quantities, out=self.temp_training_rhs_addition
        )
        self.training_rhs += self.temp_training_rhs_addition
        if self.use_inplace_dot_addition:
            inplace_add_dot_product(self.training_kernel, feature_vectors)
        else:
            self.temp_kernel_addition = dot_(
                feature_vectors.T, feature_vectors, out=self.temp_kernel_addition
            )
            self.training_kernel += self.temp_kernel_addition
        self.ntrain_vals += len(training_quantities)

    def add_training_points(
        self, training_objects, training_quantities, unprocessed=False, **kwargs
    ):
        if unprocessed:
            self.init_object_processing()
            training_objects = self.get_processed_objects(training_objects)
        self.add_processed_training_points(training_objects, training_quantities, **kwargs)

    def get_l2reg_diag_ratio(self):
        return self.base_model.l2reg_diag_ratio

    def assign_alphas(self, alphas):
        self.base_model.alphas = alphas

    def finalize_training(self, solver=cho_solve, overwrite_temp=True):
        l2reg_diag_ratio = self.get_l2reg_diag_ratio()
        if l2reg_diag_ratio is None:
            l2reg = None
        else:
            diag_indices = diag_indices_from_(self.training_kernel)
            l2reg = l2reg_diag_ratio * mean_(self.training_kernel[diag_indices])
        alphas = solver(
            self.training_kernel,
            self.training_rhs,
            l2reg=l2reg,
            overwrite_a=overwrite_temp,
            overwrite_b=overwrite_temp,
        )
        self.assign_alphas(alphas)

    def clear_arrays(self):
        self.training_kernel = None
        self.training_rhs = None
        self.ntrain_vals = 0

        self.temp_feature_vectors = None
        self.temp_training_rhs_addition = None
        if not self.use_inplace_dot_addition:
            self.temp_kernel_addition = None

    def get_all_predictions(self, *args, **kwargs):
        return self.base_model.get_all_predictions(*args, **kwargs)

    def get_prediction(self, *args, **kwargs):
        return self.base_model.get_prediction(*args, **kwargs)
