# For dealing with several Cholesky decompositions at once.
from .jit_interfaces import (
    array_,
    cho_factor_,
    cho_solve_,
    copy_detached_,
    diag_indices_from_,
    dot_,
    flip_,
    float_,
    jit_,
    logical_not_,
    lu_factor_,
    lu_solve_,
    searchsorted_,
    sum_,
    svd_,
    zeros_,
)
from .utils import where2slice


# KK: Based on https://stackoverflow.com/questions/59292279/solving-linear-systems-of-equations-with-svd-decomposition
@jit_(numba_parallel=True)
def solve_from_svd(U_mat, singular_values, Vh_mat, rhs, rcond: float_):
    cutoff_s = rcond * singular_values[0]
    cutoff_id = rhs.shape[0] - searchsorted_(flip_(singular_values), cutoff_s)
    rescaled_U_mat_vec_prod = dot_(U_mat.T[:cutoff_id], rhs)
    rescaled_U_mat_vec_prod /= singular_values[:cutoff_id]
    # TODO KK: can this be optimized w.r.t. memory access?
    return dot_(rescaled_U_mat_vec_prod, Vh_mat[:cutoff_id])


def svd_solve(mat, rhs, rcond=0.0):
    """
    Solve "mat*x=rhs" using SVD decomposition.
    """
    U_mat, singular_values, Vh_mat = svd_(mat)
    return solve_from_svd(U_mat, singular_values, Vh_mat, rhs, rcond)


def solve_w_reg_factor(mat, rhs, factorization, solver, l2reg=None):
    if l2reg is not None:
        diag_indices = diag_indices_from_(mat)
        mat_diag_backup = copy_detached_(mat[diag_indices])
        mat[diag_indices] += l2reg
    fact = factorization(mat)
    if l2reg is not None:
        mat[diag_indices] = mat_diag_backup[:]
    return solver(fact, rhs)


def cho_solve(mat, rhs, l2reg=None):
    """
    Solve "mat*x=rhs" via Cholesky decomposition with either scipy.linalg or torch.linalg.
    """
    return solve_w_reg_factor(mat, rhs, cho_factor_, cho_solve_, l2reg=l2reg)


def lu_solve(mat, rhs, l2reg=None):
    """
    Solve "mat*x=rhs" via LU decomposition.
    """
    return solve_w_reg_factor(mat, rhs, lu_factor_, lu_solve_, l2reg=l2reg)


# KK: Accelerates training if you train on several properties that can be sorted in such a way that each property is available for each molecule for which the
# next property is available (but not necessarily vice-versa). In this case re-indexing allows usage of a single Cholesky decomposition for all properties.
# Originally developed for doing G2S for molecules of different sizes. Might require heavy revision to be usable again.
def nullify_ignored(arr, indices_to_ignore):
    if indices_to_ignore is not None:
        for row_id, cur_ignore_indices in enumerate(indices_to_ignore):
            arr[row_id][where2slice(logical_not_(cur_ignore_indices))] = 0.0


class Cho_multi_factors:
    def __init__(self, train_kernel, indices_to_ignore=None, ignored_orderable=False):
        self.indices_to_ignore = indices_to_ignore
        self.ignored_orderable = ignored_orderable
        self.single_cho_decomp = indices_to_ignore is None
        if not self.single_cho_decomp:
            self.single_cho_decomp = not self.indices_to_ignore.any()
        if self.single_cho_decomp:
            self.cho_factors = [cho_factor_(train_kernel)]
        else:
            if self.ignored_orderable:
                ignored_nums = []
                for i, cur_ignored in enumerate(self.indices_to_ignore):
                    ignored_nums.append((i, sum_(cur_ignored)))
                ignored_nums.sort(key=lambda x: x[1])
                self.availability_order = array_([i[0] for i in ignored_nums])
                self.avail_quant_nums = [
                    self.indices_to_ignore.shape[0] - sum_(cur_ignored)
                    for cur_ignored in self.indices_to_ignore.T
                ]
                self.cho_factors = [
                    cho_factor_(
                        train_kernel[self.availability_order, :][:, self.availability_order]
                    )
                ]
            else:
                self.cho_factors = []
                for cur_ignore_ids in self.indices_to_ignore.T:
                    s = where2slice(cur_ignore_ids)
                    self.cho_factors.append(cho_factor_(train_kernel[s, :][:, s]))

    def solve_with(self, rhs):
        if len(rhs.shape) == 1:
            assert self.single_cho_decomp
            cycled_rhs = array_([rhs])
        else:
            if not self.single_cho_decomp:
                if self.ignored_orderable:
                    assert len(self.avail_quant_nums) == rhs.shape[1]
                else:
                    assert len(self.cho_factors) == rhs.shape[1]
            cycled_rhs = rhs.T
        output = zeros_(cycled_rhs.shape)
        for rhs_id, rhs_component in enumerate(cycled_rhs):
            if self.indices_to_ignore is None:
                included_indices = array_(range(len(rhs_component)))
            else:
                if self.ignored_orderable:
                    included_indices = self.availability_order[: self.avail_quant_nums[rhs_id]]
                else:
                    included_indices = where2slice(self.indices_to_ignore[:, rhs_id])
            if self.single_cho_decomp:
                cur_decomp = self.cho_factors[0]
            else:
                if self.ignored_orderable:
                    cur_decomp = (
                        self.cho_factors[0][0][: self.avail_quant_nums[rhs_id], :][
                            :, : self.avail_quant_nums[rhs_id]
                        ],
                        self.cho_factors[0][1],
                    )
                else:
                    cur_decomp = self.cho_factors[rhs_id]
            output[rhs_id, included_indices] = cho_solve(
                cur_decomp, rhs_component[included_indices]
            )
        if len(rhs.shape) == 1:
            return output[0]
        else:
            return output
