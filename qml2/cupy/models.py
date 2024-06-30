import cupy as cp
import numpy as np

from ..kernels.kernels import construct_gaussian_kernel
from ..models.krr import KRRLocalModel, KRRModel
from .kernels import gaussian_kernel, local_dn_gaussian_kernel


class CuPyKRRModel(KRRModel):
    def __init__(self, blocks_per_grid=(1,), threads_per_block=(1,), **other_kwargs):
        """
        First value in blocks and threads number tuples controls parallelization
        over training set molecules, second value - over representation of test molecules (in predict_from_representations).
        """
        self.base_init(**other_kwargs)
        self.blocks_per_grid = blocks_per_grid
        self.threads_per_block = threads_per_block
        self.nfeatures = None

    def base_init(self, **kwargs):
        KRRModel.__init__(self, **kwargs)

    def init_kernel_functions(self, kernel_constructor, **other_kwargs):
        assert kernel_constructor is construct_gaussian_kernel
        # Use NumPy kernel function for training.
        self.kernel_function_sym = kernel_constructor(symmetric=True, **other_kwargs)

    def transfer_model_param_to_gpu(self, model_param_name):
        # NOTE: Both gaussian_kernel and local_dn_gaussian_kernel are written to assume that
        # all arrays involved are contiguous. I am not %100 sure simple cupy.asarray guarantees that,
        # hence the care.
        old_param = getattr(self, model_param_name)
        if (isinstance(old_param, float)) or (len(old_param.shape) == 0):
            setattr(self, model_param_name, cp.float64(old_param))
            return
        if old_param.dtype == np.int64:
            dtype = cp.int64
        else:
            dtype = cp.float64
        new_param = cp.empty(old_param.shape, dtype=dtype)
        new_param[:] = cp.asarray(old_param)[:]
        setattr(self, model_param_name, new_param)

    def relevant_model_parameters(self):
        return ["sigma", "alphas", "training_set_representations", "val_shift", "temp_kernel"]

    def transfer_model_params_to_gpu(self):
        for rp in self.relevant_model_parameters():
            self.transfer_model_param_to_gpu(rp)

    def init_temp_reps(self):
        self.nfeatures = self.training_set_representations.shape[1]
        self.temp_reps = cp.empty((1, self.nfeatures))

    def base_train(self, *args, **kwargs):
        KRRModel.train(self, *args, **kwargs)

    def train(self, *args, **kwargs):
        self.base_train(*args, **kwargs)
        self.transfer_model_params_to_gpu()
        self.init_temp_reps()

    def predict_from_kernel(self, nmols=1):
        return cp.dot(self.alphas, self.temp_kernel[:, :nmols]) + self.val_shift

    def predict_from_representations(self, representations_gpu):
        assert self.nfeatures == representations_gpu.shape[1]
        nmols = representations_gpu.shape[0]
        if (self.temp_kernel is None) or (nmols > self.temp_kernel.shape[1]):
            self.temp_kernel = cp.empty((self.ntrain, nmols))
        gaussian_kernel(
            self.blocks_per_grid,
            self.threads_per_block,
            (
                self.training_set_representations,
                representations_gpu,
                self.sigma,
                self.ntrain,
                nmols,
                self.nfeatures,
                self.temp_kernel,
            ),
        )
        return self.predict_from_kernel(nmols=nmols)

    def forward(self, nuclear_charges, coords):
        self.temp_reps[:] = cp.asarray(self.get_rep(nuclear_charges, coords))[:]
        result_gpu = self.predict_from_representations(self.temp_reps)
        return cp.asnumpy(result_gpu)[0]


class CuPyKRRLocalModel(CuPyKRRModel, KRRLocalModel):
    def base_init(self, **kwargs):
        KRRLocalModel.__init__(self, **kwargs)

    def relevant_model_parameters(self):
        return [
            "sigma",
            "alphas",
            "training_set_representations",
            "training_set_nuclear_charges",
            "training_set_natoms",
            "default_na_arr",
            "val_shift",
        ]

    def init_temp_reps(self):
        CuPyKRRModel.init_temp_reps(self)
        self.temp_nuclear_charges = cp.empty((1,), dtype=int)

    def base_train(self, *args, **kwargs):
        KRRLocalModel.train(self, *args, **kwargs)

    def predict_from_representations(self, representations, nuclear_charges, atom_nums=None):
        if atom_nums is None:
            # we are doing calculations for just one molecule
            self.default_na_arr[0] = representations.shape[0]
            atom_nums = self.default_na_arr
            nmols = 1
        else:
            nmols = atom_nums.shape[0]
        if (self.temp_kernel is None) or (nmols > self.temp_kernel.shape[1]):
            self.temp_kernel = cp.empty((self.ntrain, nmols))

        assert self.local_dn
        local_dn_gaussian_kernel(
            self.blocks_per_grid,
            self.threads_per_block,
            (
                self.training_set_representations,
                self.temp_reps,
                self.training_set_natoms,
                atom_nums,
                self.training_set_nuclear_charges,
                nuclear_charges,
                self.sigma,
                self.ntrain,
                nmols,
                self.nfeatures,
                self.temp_kernel,
            ),
        )
        return CuPyKRRModel.predict_from_kernel(self, nmols=nmols)

    def forward(self, nuclear_charges, coords):
        natoms = nuclear_charges.shape[0]
        if natoms > self.temp_reps.shape[0]:
            self.temp_reps = cp.empty((natoms, self.nfeatures))
            self.temp_nuclear_charges = cp.empty((natoms,), dtype=int)
        self.temp_reps[:natoms] = cp.asarray(self.get_rep(nuclear_charges, coords))[:]
        self.temp_nuclear_charges[:natoms] = cp.asarray(nuclear_charges)[:]
        self.default_na_arr[0] = natoms
        result_gpu = self.predict_from_representations(
            self.temp_reps, self.temp_nuclear_charges, atom_nums=self.default_na_arr
        )
        return cp.asnumpy(result_gpu)[0]
