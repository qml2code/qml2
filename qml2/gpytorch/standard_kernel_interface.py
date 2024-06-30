# KK: kept consulting https://github.com/leojklarner/gauche for this one.
import torch as tc
from gpytorch.kernels import Kernel

from ..jit_interfaces import empty_
from ..kernels.kernels import get_kernel


class GlobalKernel(Kernel):
    """
    Creates a GPyTorch-compliant kernel from a global kernel in qml2.kernels
    """

    is_stationary = True
    has_lengthscale = True

    def __init__(self, global_kernel, **kwargs):
        super(GlobalKernel, self).__init__(**kwargs)
        self.global_kernel_func = global_kernel

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False):
        assert not last_dim_is_batch  # KK: temporary
        assert not diag
        output = empty_((x1.shape[-2], x2.shape[-2]))
        self.global_kernel_func(x1, x2, self.lengthscale[0, 0], output)
        if diag:
            assert x1.size() == x2.size() and tc.equal(x1, x2)
            return tc.ones(*x1.shape[:-2], x1.shape[-2], dtype=x1.dtype, device=x1.device)
        else:
            return output


def construct_GlobalKernel(**get_kernel_kwargs):
    global_kernel_function = get_kernel({}, symmetric=False, local=False, **get_kernel_kwargs)
    return GlobalKernel(global_kernel_function)
