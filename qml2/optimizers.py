"""
Interface with procedures for global optimization that pop up in hyperparameter optimization.

For now only contains interface with BOSS which switches to simple grid optimization during testing.
"""
from numpy import linspace

from .basic_utils import ExceptionRaisingClass

try:
    from boss.bo.bo_main import BOMain
except ModuleNotFoundError:
    BOMain = ExceptionRaisingClass(
        ModuleNotFoundError(
            "Procedure requires installation of aalto-boss via:\npip install aalto-boss"
        )
    )


def grid_optimize_1D(function, bounds, total_iterpts=32):
    """
    Simple grid optimization. Currently only used for tests.
    """
    min_val = None
    min_pos = None
    for pos in linspace(bounds[0], bounds[1], total_iterpts):
        val = function(pos)
        if val is None:
            continue
        if min_val is None or val < min_val:
            min_val = val
            min_pos = pos
    assert min_val is not None, "Optimization failed to find a single valid value!"
    return min_pos


def global_optimize_1D(
    function,
    bounds,
    total_iterpts=32,
    iterpts=None,
    initpts=None,
    opt_method="boss",
    opt_name="opt",
    seed=None,
):
    """1D global optimization, by BOSS ("boss") or an even grid ("grid").

    seed: forwarded to BOSS as its own `seed` keyword, which BOSS implements as
        `np.random.seed(seed)` at Settings construction. Without it, successive BOSS runs
        continue the global RNG stream, so a repeated call can return a slightly different
        optimum -- fine for a single optimization, not fine for anything comparing two
        supposedly identical evaluations. Ignored by the deterministic "grid" method.

        NOTE: BOSS's own comment on this keyword is "TODO: Propagate this seed to GPy to
        eliminate all randomness", so the seed reaches numpy's global RNG but is not handed
        to GPy directly. Treat it as strongly reducing run-to-run variation rather than
        guaranteeing bit-identical results.
    """
    assert len(bounds) == 2
    if opt_method == "grid":
        return grid_optimize_1D(function, bounds, total_iterpts=total_iterpts)

    if iterpts is None and initpts is None:
        iterpts = total_iterpts // 2
        initpts = total_iterpts // 2
    else:
        assert not (
            iterpts is None or initpts is None
        ), "Either define `total_iterpts`, or both `iterpts` and `initpts`!"

    bo = BOMain(
        function,
        bounds=[bounds],
        minfreq=0,
        outfile=f"{opt_name}.out",
        rstfile=f"{opt_name}.rst",
        iterpts=iterpts,
        initpts=initpts,
        **({} if seed is None else {"seed": seed}),
    )
    res = bo.run()
    return res.select("x_glmin", -1)[0]
