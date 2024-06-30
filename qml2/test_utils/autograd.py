import torch as tc

from ..utils import all_possible_indices


def concatenated_shapes(*x_args):
    output = []
    for x in x_args:
        output += list(x.shapes)
    return tuple(output)


def single_autograd(val, x, retain_graph=False, create_graph=False, allow_unused=False):
    all_ders = tc.empty((*val.shape, *x.shape))
    for t in all_possible_indices(val.shape):
        der_val = tc.autograd.grad(
            val[t],
            x,
            retain_graph=retain_graph,
            create_graph=create_graph,
            allow_unused=allow_unused,
        )[0]
        if der_val is None:
            all_ders[t] = 0.0
        else:
            all_ders[t] = der_val
    return all_ders


def full_autograd(val, *x_args, allow_unused=False):
    assert isinstance(val, tc.Tensor)
    last_diff = val
    for i, x in enumerate(x_args):
        assert isinstance(x, tc.Tensor)
        create_graph = i != (len(x_args) - 1)
        last_diff = single_autograd(
            last_diff, x, retain_graph=True, create_graph=create_graph, allow_unused=allow_unused
        )
    return last_diff
