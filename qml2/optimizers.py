"""
Interface with procedures for global optimization that pop up in hyperparameter optimization.

For now only contains interface with BOSS which switches to simple grid optimization during testing.
"""
from numpy import linspace


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


def global_optimize_1D(function, bounds, total_iterpts=32, test_mode=False, opt_name="opt"):
    assert len(bounds) == 2
    if test_mode:
        return grid_optimize_1D(function, bounds, total_iterpts=total_iterpts)
    try:
        from boss.bo.bo_main import BOMain
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "Procedure requires installation of aalto-boss via:\npip install aalto-boss"
        )

    bo = BOMain(
        function,
        bounds=[bounds],
        minfreq=0,
        outfile=f"{opt_name}.out",
        rstfile=f"{opt_name}.rst",
        iterpts=total_iterpts // 2,
        initpts=total_iterpts // 2,
    )
    res = bo.run()
    return res.select("x_glmin", -1)[0]
