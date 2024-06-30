from ..basic_utils import checked_dict_entry
from ..jit_interfaces import (
    LinAlgError_,
    Module_,
    abs_,
    array_,
    concatenate_,
    copy_,
    diag_indices_from_,
    dint_,
    dot_,
    empty_,
    inf_,
    matmul_,
    max_,
    mean_,
)
from ..kernels.kernels import construct_gaussian_kernel
from ..math import cho_solve
from ..parallelization import embarrassingly_parallel
from ..representations import generate_coulomb_matrix, generate_fchl19
from ..utils import check_allocation
from .hyperparameter_init_guesses import vector_std


class KRRModel(Module_):
    def __init__(
        self,
        representation_function=generate_coulomb_matrix,
        sigma=None,
        kernel_constructor=construct_gaussian_kernel,
        rep_kwargs={},
        l2reg_diag_ratio=array_(1.0e-6),
        l2reg=None,
        max_l2reg_diag_ratio=array_(1.0),
        num_consistency_check=array_(1.0e-6),
        additional_kernel_constructor_kwargs={},
        training_reps_suppress_openmp=False,
        apply_shift=False,
    ):
        """
        Models using global kernels
        """
        Module_.__init__(self)
        # hyperparameters
        self.sigma = sigma
        self.init_stability_checks(
            l2_reg_diag_ratio=l2reg_diag_ratio,
            l2reg=l2reg,
            num_consistency_check=num_consistency_check,
            max_l2reg_diag_ratio=max_l2reg_diag_ratio,
        )
        # representation parameters
        self.representation_function = representation_function
        self.rep_kwargs = rep_kwargs
        # kernel
        self.init_kernel_functions(kernel_constructor, **additional_kernel_constructor_kwargs)
        self.init_basic_arrays()
        # For checking model's numerical stability.
        self.apply_shift = apply_shift
        self.training_reps_suppress_openmp = training_reps_suppress_openmp

    def init_stability_checks(
        self,
        l2_reg_diag_ratio=array_(1.0e-6),
        l2reg=None,
        num_consistency_check=array_(1.0e-6),
        max_l2reg_diag_ratio=array_(1.0),
    ):
        self.l2reg = l2reg
        self.l2reg_diag_ratio = l2_reg_diag_ratio
        self.num_consistency_check = num_consistency_check
        self.max_l2reg_diag_ratio = max_l2reg_diag_ratio

    def init_kernel_functions(self, kernel_constructor, **additional_kernel_constructor_kwargs):
        self.kernel_function_asym = kernel_constructor(
            symmetric=False, **additional_kernel_constructor_kwargs
        )
        self.kernel_function_sym = kernel_constructor(
            symmetric=True, **additional_kernel_constructor_kwargs
        )

    def init_basic_arrays(self):
        # temporary arrays
        # fitted model parameters
        self.alphas = None
        self.temp_kernel = None
        # will be useful if (hopefully) we start doing representation padding.
        self.temp_reps = None
        # For shifting predicted quantities.
        self.val_shift = None

    def get_rep(self, nuclear_charges, coords):
        assert nuclear_charges.shape[0] == coords.shape[0]
        assert coords.shape[1] == 3
        return self.representation_function(nuclear_charges, coords, **self.rep_kwargs)

    def get_rep_tuple(self, ncharges_coords_tuple):
        return self.get_rep(*ncharges_coords_tuple)

    def optimize_hyperparameters(self, all_representations, *other_args, **other_kwargs):
        """
        For now just initial guess.
        TODO: make better
        """
        if self.sigma is not None:
            return
        print("Optimizing hyperparameters")
        # KK: The 1.96 appears from 95% confidence interval formula.
        # Bing (IIRC) defined optimal sigma value in terms of maximum distance between two
        # members of the training set. I was aiming for a definition that does not depend on outliers.
        self.sigma = vector_std(all_representations) * 1.96 * 4
        print("Done")

    def reference_diag_ids(self):
        """
        If we estimate l2 regularization magnitude based on average kernel element, for KRR
        we use all diagonal elements.
        (This is not so for GPR forces.)
        """
        r = array_(list(range(self.ntrain)))
        return (r, r)

    def add_l2reg(self, train_kernel):
        if self.l2reg is None:
            l2reg_addition = self.l2reg_diag_ratio * mean_(train_kernel[self.reference_diag_ids()])
        else:
            l2reg_addition = self.l2reg
        train_kernel[diag_indices_from_(train_kernel)] += l2reg_addition

    def increase_l2reg(self):
        if self.l2reg is None:
            self.l2reg_diag_ratio *= 2.0
            print("l2reg_diag_ratio:", self.l2reg_diag_ratio)
        else:
            self.l2reg *= 2.0
            print("l2reg:", self.l2reg)

    def get_alphas_w_lambda(self, train_kernel, rhs, solver=cho_solve, **solver_kwargs):
        diag_indices = diag_indices_from_(train_kernel)
        diag_el_backup = copy_(train_kernel[diag_indices])
        reasonable_accuracy = False
        while not reasonable_accuracy:
            self.add_l2reg(train_kernel)
            try:
                self.alphas = solver(train_kernel, rhs, **solver_kwargs)
                rhs_check = matmul_(self.alphas, train_kernel)
                consistency_measure = max_(abs_(rhs_check - rhs))
                reasonable_accuracy = consistency_measure < self.num_consistency_check
            except LinAlgError_:
                reasonable_accuracy = False
                consistency_measure = inf_
            train_kernel[diag_indices] = copy_(diag_el_backup)
            if not reasonable_accuracy:
                print(
                    "Bad numerical consistency check:", consistency_measure, "increasing lambda:"
                )
                self.increase_l2reg()
                if self.l2reg_diag_ratio > self.max_l2reg_diag_ratio:
                    raise Exception

    def init_apply_shift(self, vals):
        # TODO: Do we need this?
        if self.apply_shift:
            self.val_shift = mean_(vals)
        else:
            self.val_shift = 0.0
        return vals - self.val_shift

    def fit(self, all_representations, training_set_values):
        train_kernel = empty_((self.ntrain, self.ntrain))
        self.training_set_representations = all_representations
        self.kernel_function_sym(self.training_set_representations, self.sigma, train_kernel)
        shifted_training_set_values = self.init_apply_shift(training_set_values)
        self.get_alphas_w_lambda(train_kernel, shifted_training_set_values)

    def check_rep_kwargs_valid(self, all_coordinates):
        if self.representation_function is generate_coulomb_matrix:
            observed_nmax = max([len(coords) for coords in all_coordinates])
            nmax = checked_dict_entry(self.rep_kwargs, "size", -1)
            if nmax < observed_nmax:
                self.rep_kwargs["size"] = observed_nmax

    def combine_representations(self, representations_list):
        return array_(representations_list)

    def get_all_representations(self, all_nuclear_charges, all_coords, suppress_openmp=False):
        if suppress_openmp:
            fixed_num_threads = 1
        else:
            fixed_num_threads = None
        all_representations_list = embarrassingly_parallel(
            self.get_rep_tuple,
            zip(all_nuclear_charges, all_coords),
            (),
            fixed_num_threads=fixed_num_threads,
        )

        return self.combine_representations(all_representations_list)

    def train(
        self,
        training_set_nuclear_charges,
        training_set_coords,
        training_set_values,
    ):
        self.check_rep_kwargs_valid(training_set_coords)
        print("Calculating representations.")
        all_representations = self.get_all_representations(
            training_set_nuclear_charges,
            training_set_coords,
            suppress_openmp=self.training_reps_suppress_openmp,
        )
        print("Done")
        self.ntrain = len(training_set_nuclear_charges)
        assert self.ntrain == len(training_set_values)
        self.optimize_hyperparameters(all_representations)
        self.fit(all_representations, training_set_values)
        self.temp_kernel = empty_((self.ntrain, 1))

    def predict_from_kernel(self, nmols=1):
        return dot_(self.alphas, self.temp_kernel[:, :nmols]) + self.val_shift

    def predict_from_representations(self, representations):
        nmols = representations.shape[0]
        self.temp_kernel = check_allocation((self.ntrain, nmols), output=self.temp_kernel)
        self.kernel_function_asym(
            self.training_set_representations, representations, self.sigma, self.temp_kernel
        )
        return self.predict_from_kernel(nmols=nmols)

    def forward(self, nuclear_charges, coords):
        self.temp_reps = check_allocation(
            (1, self.training_set_representations.shape[1]), output=self.temp_reps
        )
        self.temp_reps[0, :] = self.get_rep(nuclear_charges, coords)
        return self.predict_from_representations(self.temp_reps)[0]

    def save(self, filename):
        from ..utils import dump2pkl

        """
        Save the trained model to a file.
        filename: string
        """

        dump2pkl(self, filename)


class KRRLocalModel(KRRModel):
    def __init__(self, representation_function=generate_fchl19, local_dn=True, **other_kwargs):
        """
        Models using local/local(dn) kernels
        """
        add_kernel_constr_kwargs = checked_dict_entry(
            other_kwargs, "additional_kernel_constructor_kwargs", {}
        )
        other_kwargs["additional_kernel_constructor_kwargs"] = {
            **add_kernel_constr_kwargs,
            "local": True,
            "local_dn": local_dn,
        }
        self.local_dn = True
        super().__init__(**other_kwargs, representation_function=representation_function)

    def init_basic_arrays(self):
        super().init_basic_arrays()
        self.default_na_arr = empty_((1,), dtype=dint_)

    def fit(self, all_representations, all_nuclear_charges, training_set_values, atom_nums):
        train_kernel = empty_((self.ntrain, self.ntrain))
        self.training_set_representations = all_representations
        self.training_set_nuclear_charges = all_nuclear_charges
        self.training_set_natoms = atom_nums
        if self.local_dn:
            self.kernel_function_sym(
                self.training_set_representations,
                self.training_set_natoms,
                self.training_set_nuclear_charges,
                self.sigma,
                train_kernel,
            )
        else:
            self.kernel_function_sym(
                self.training_set_representations,
                self.training_set_natoms,
                self.sigma,
                train_kernel,
            )
        self.training_set_representations = all_representations
        shifted_training_set_values = self.init_apply_shift(training_set_values)
        self.get_alphas_w_lambda(train_kernel, shifted_training_set_values)

    def combine_representations(self, representations_list):
        return concatenate_(tuple(representations_list))

    def train(
        self,
        training_set_nuclear_charges,
        training_set_coords,
        training_set_values,
    ):
        print("Calculating representations.")
        all_representations = self.get_all_representations(
            training_set_nuclear_charges, training_set_coords
        )
        print("Done")
        all_nuclear_charges = concatenate_(training_set_nuclear_charges)
        self.ntrain = len(training_set_nuclear_charges)
        assert self.ntrain == training_set_values.shape[0]
        atom_nums = empty_((self.ntrain,), dtype=dint_)
        for i, nuclear_charges in enumerate(training_set_nuclear_charges):
            atom_nums[i] = nuclear_charges.shape[0]
        self.optimize_hyperparameters(all_representations)
        self.fit(all_representations, all_nuclear_charges, training_set_values, atom_nums)

    def predict_from_representations(self, representations, nuclear_charges, atom_nums=None):
        if atom_nums is None:
            # we are doing calculations for just one molecule
            self.default_na_arr[0] = representations.shape[0]
            atom_nums = self.default_na_arr
            nmols = 1
        else:
            nmols = atom_nums.shape[0]
        self.temp_kernel = check_allocation(
            (
                self.ntrain,
                atom_nums.shape[0],
            ),
            output=self.temp_kernel,
        )
        if self.local_dn:
            self.kernel_function_asym(
                self.training_set_representations,
                representations,
                self.training_set_natoms,
                atom_nums,
                self.training_set_nuclear_charges,
                nuclear_charges,
                self.sigma,
                self.temp_kernel,
            )
        else:
            self.kernel_function_asym(
                self.training_set_representations,
                representations,
                self.training_set_natoms,
                atom_nums,
                self.sigma,
                self.temp_kernel,
            )
        return self.predict_from_kernel(nmols=nmols)

    def forward(self, nuclear_charges, coords):
        self.temp_reps = self.get_rep(nuclear_charges, coords)
        return self.predict_from_representations(self.temp_reps, nuclear_charges)[0]
