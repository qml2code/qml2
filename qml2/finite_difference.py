# Short utility for easy finite difference test setups.

from .jit_interfaces import copy_, empty_, int_, is_scalar_, jit_, ndarray_, zeros_
from .utils import all_possible_indices, expanded_shape


@jit_
def finite_difference_coefficients(point_positions, derivative_order: int_):
    """
    Generate finite difference coefficients for arbitrary position of interpolation points.
    Taken from Fornberg Math. Comput. v51 n184 p699-706 Oct 1988.
    """
    n_max = point_positions.shape[0]

    coeffs_all = zeros_((int(derivative_order + 2), n_max, n_max))
    coeffs_all[1, 0, 0] = 1.0

    c1 = 1.0

    for n in range(1, n_max):  # cycle over number of points used in the finite difference
        c2 = 1.0
        for j in range(n):  # cycle over all points but for the last one
            c3 = point_positions[n] - point_positions[j]
            c2 *= c3
            for der in range(
                1, derivative_order + 2
            ):  # cycle over coefficients for each considered derivative order
                coeffs_all[der, n, j] = (
                    point_positions[n] * coeffs_all[der, n - 1, j]
                    - (der - 1) * coeffs_all[der - 1, n - 1, j]
                ) / c3
        for der in range(1, derivative_order + 2):
            coeffs_all[der, n, n] = (
                c1
                / c2
                * (
                    (der - 1) * coeffs_all[der - 1, n - 1, n - 1]
                    - point_positions[n - 1] * coeffs_all[der, n - 1, n - 1]
                )
            )
        c1 = c2

    return copy_(coeffs_all[derivative_order + 1, n_max - 1, :])


@jit_
def grid_finite_difference_coefficients_1var(nfd_steps, derivative_order, dx):
    npoints = int(2 * nfd_steps + 1)

    fd_grid_pos = zeros_((npoints,))
    for i in range(npoints):
        new_pos = (i + 1) // 2
        if i % 2 == 0:
            new_pos *= -1
        fd_grid_pos[i] = new_pos

    coeffs = finite_difference_coefficients(fd_grid_pos, derivative_order) * dx ** (
        -derivative_order
    )

    if derivative_order % 2 == 0:
        return coeffs, fd_grid_pos
    else:  # central point cancels out exactly
        return coeffs[1:], fd_grid_pos[1:]


def single_finite_difference(f, x0, dx, differenciated=(), derivative_order=1, nfd_steps=2):
    der_f = None
    fd_coeffs, grid_positions = grid_finite_difference_coefficients_1var(
        nfd_steps, derivative_order, dx
    )
    for fd_coeff, grid_pos in zip(fd_coeffs, grid_positions):
        x = copy_(x0)
        x[differenciated] += dx * grid_pos
        add_f_val = f(x) * fd_coeff
        if der_f is None:
            der_f = add_f_val
        else:
            der_f += add_f_val
    return der_f


def all_finite_differences(f, x0, dx, **fd_kwargs):
    output = None
    f_scalar = None
    f_shape = None
    x0_scalar, x0_shape = expanded_shape(x0)

    for el_tuple in all_possible_indices(x0_shape):
        if is_scalar_(dx):
            cur_dx = dx
        else:
            cur_dx = dx[el_tuple]
        output_component = single_finite_difference(
            f, x0, cur_dx, differenciated=el_tuple, **fd_kwargs
        )
        if f_scalar is None:
            f_scalar, f_shape = expanded_shape(output_component)
        if output is None:
            if not (f_scalar and x0_scalar):
                output = empty_((*f_shape, *x0_shape))
            else:  # both are scalars, there was just one component.
                return output_component
        for output_tuple in all_possible_indices(f_shape):
            output[(*output_tuple, *el_tuple)] = output_component[output_tuple]
    return output


def finite_difference(
    f, x0: ndarray_, dx: ndarray_, differenciated: tuple | None = None, **fd_kwargs
):
    """
    For function f(x) returning y (float or ndarray_) and starting point x0 (float or ndarray_) evaluate
    via finite difference tensor of derivatives with shape (*y.shape, *x0.shape).
    differenciated
    f : differentiated function
    x0 : point at wich derivatives are evaluated
    dx : if float - finite difference step; if ndarray_ - finite difference steps for
        each separate component of x
    differenciated = None : if not None sets the code to return array of derivatives of y w.r.t.
    x[differenciated].
    derivative_order = 1 : order of derivative to be calculated.
    nfd_steps = 2 : number of finite difference steps taken forward and backward (e.g. nfd_steps=2
        corresponds to 5-point finite difference scheme).
    """
    if differenciated is None:
        return all_finite_differences(f, x0, dx, **fd_kwargs)
    else:
        return single_finite_difference(f, x0, dx, **fd_kwargs)
