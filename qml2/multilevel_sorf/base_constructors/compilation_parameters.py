"""Per-level parameters that must be baked in at COMPILE time, not read at run time.

Two kinds of parameter reach a level today, and neither can do this job:

* ``parameter_list`` entries shape the INPUT OBJECTS -- sizes, counts, element-id tables. They are
  read when the jitclass instances are built (``create_input_object``), and the routines never see
  them.
* hyperparameters arrive as an array at call time.

A third kind is needed for anything that must be a compile-time constant inside the generated numba
function: a *callable closed over by the jitted body*. The motivating case is a representation level
-- one that takes a numba conformer (nuclear charges + coordinates, and optionally a pre-calculated
local representation), computes a representation inside the parallel loop, wraps it in the object the
rest of ``multilevel_sorf`` consumes, and passes that down. Its representation function and kwargs
have to be baked in for numba to inline them; passed at run time they would be untypeable Python
objects. ``qml2.representations.numba_calculators`` provides such functions
(``aSLATMJITCalculator(...).numba_calculator`` and friends).

Swapping a level's activation function is the same shape of problem and works through the same
mechanism.

This module carries only the MECHANISM.

How it is spelled
-----------------
A compile-time parameter goes in the level's ordinary ``parameter_list`` entry, under the reserved
key ``"compilation_parameters"``::

    function_definition_list = ["resize", "sorf", "component_sum"]
    parameter_list = [
        {"output_size": 1024},
        {"nfeature_stacks": 4, "ntransforms": 3,
         "compilation_parameters": {"representation_function": aslatm_calc.numba_calculator}},
        {},
    ]

Living in ``parameter_list`` rather than a third parallel list means the length invariant
``len(function_definition_list) == len(parameter_list)`` is untouched, the parameters sit next to the
level they configure, and every existing caller keeps working by simply not using the key.

The cache key, which is the part that can go silently wrong
----------------------------------------------------------
Routines are memoised in ``common.py``'s module globals under a name derived from the definition list
alone (``get_name_from_definition_list``). Two models differing ONLY in a compile-time parameter
would therefore share one compiled routine, and the second would silently run the first's baked-in
function. So a level carrying compilation parameters must contribute a token to that name --
:func:`compilation_token` -- and a level carrying none must produce the name it always did, so
existing caches, pickles and behaviour are bit-for-bit unchanged.

Closures, and a TODO
--------------------
A callable is identified by ``module.qualname``, which is stable across processes -- the property that
keeps a restart from recompiling under a fresh name. It is also BLIND to what a closure baked in:
every function a factory returns shares one qualname, so two constructors built by one factory over
different values token identically and the second silently reuses the first's compiled routine. A
factory must therefore declare its baked-in values with :func:`set_compilation_identity`;
``processed_object_constructors.numba_constructors`` does so with its ``sorted_elements``.

That declaration is opt-in, so an undeclared closure still collides. The risk is bounded: the cache is
``common.py``'s module ``globals()``, so a collision needs BOTH models in ONE process -- rebuilding a
``MultilevelSORF`` does not clear it, separate processes never share it -- and the assumed convention
is one model type per process.

**TODO (design, deferred).** The root cause is the ``globals()`` cache itself: intermediate
constructed functions have process lifetime rather than belonging to the model or an explicit
registry. Fixing that would retire this hole and the process-lifetime surprise together, instead of
relying on every future factory to declare. A guard raising on an undeclared ``<locals>`` closure was
considered and deliberately not added, the convention already excluding the case. See
``multilevel_sorf/AGENTS.md``.
"""

from hashlib import blake2b

from ...jit_interfaces import ndarray_

COMPILATION_PARAMETERS_KEY = "compilation_parameters"

# Attribute through which a function DECLARES what it baked in; see set_compilation_identity.
COMPILATION_IDENTITY_ATTR = "_msorf_compilation_identity"


def _canonical(value) -> str:
    """A stable textual form of a compilation-parameter value, for hashing.

    ``repr`` will not do. The values that matter most here are FUNCTIONS, and ``repr`` of a function
    embeds its ``id()``, which changes between processes -- so the token would differ between a run
    and its own restart, and the routine would be recompiled under a new name every time. Identify a
    callable by ``module.qualname`` instead: stable across processes, and exactly the notion of
    identity meant by "which function was baked in".

    A closure needs more than that, and this is the subtle case. Every function a factory returns
    shares one ``module.qualname``, so two constructors built by the same factory over DIFFERENT
    baked-in values are indistinguishable by it -- and since the token is a routine's cache key, the
    second model would silently be handed the routine compiled for the first. What a closure's
    identity actually IS is the values it closed over, so a factory declares them through
    ``set_compilation_identity`` and they are preferred here when present.
    """
    if callable(value):
        module = getattr(value, "__module__", "?")
        qualname = getattr(value, "__qualname__", getattr(value, "__name__", repr(value)))
        identity = getattr(value, COMPILATION_IDENTITY_ATTR, None)
        if identity is not None:
            return f"<callable {module}.{qualname} {_canonical(identity)}>"
        return f"<callable {module}.{qualname}>"
    if isinstance(value, ndarray_):
        # NOT repr: numpy elides the middle of a long array, so two arrays differing only there
        # would canonicalise -- and therefore compile -- identically.
        return f"<array {value.dtype}{value.shape}{_canonical(value.tolist())}>"
    if isinstance(value, dict):
        return "{" + ",".join(f"{k!r}:{_canonical(v)}" for k, v in sorted(value.items())) + "}"
    if isinstance(value, (list, tuple)):
        brackets = "[]" if isinstance(value, list) else "()"
        return brackets[0] + ",".join(_canonical(v) for v in value) + brackets[1]
    return repr(value)


def compilation_token(compilation_parameters) -> str:
    """Short identifier-safe token for a level's compilation parameters, ``""`` when there are none.

    Empty in, empty out, and that is load-bearing: it is what keeps the generated names for every
    existing model byte-identical to what they were before this mechanism existed.
    """
    if not compilation_parameters:
        return ""
    canonical = _canonical(dict(compilation_parameters))
    return blake2b(canonical.encode(), digest_size=5).hexdigest()


def set_compilation_identity(func, *values):
    """Declare the values ``func`` baked in, so its compilation token depends on them.

    A factory that closes over compile-time data must call this on what it returns. Without it two
    constructors from one factory share a cache key and the second silently reuses the first's
    compiled routine -- for an element list, that is wrong element ids rather than an error.

    Returns ``func``, so a factory can ``return set_compilation_identity(inner, a, b)``.
    """
    setattr(func, COMPILATION_IDENTITY_ATTR, values)
    return func


def level_cache_name(level_name: str, compilation_parameters=None) -> str:
    """The level's contribution to a routine's cache key.

    ``"sorf"`` with no compilation parameters, ``"sorf_c1f3a09b2e"`` with some. The token is appended
    rather than replacing anything, so a name stays readable in a traceback and remains a valid
    Python identifier (it is stored as a key in module globals).
    """
    token = compilation_token(compilation_parameters)
    return level_name if not token else f"{level_name}_c{token}"


def split_compilation_parameters(parameter_list):
    """Split ``parameter_list`` into ``(clean_parameter_list, compilation_parameter_list)``.

    The clean list has the reserved key removed, so everything downstream that reads a level's
    parameters sees exactly what it saw before. The compilation list is parallel in SHAPE --
    including fork entries, which are tuples of per-branch lists -- and holds ``{}`` for levels that
    declared nothing, so it can be sliced alongside a definition list without special cases.
    """
    clean, compilation = [], []
    for entry in parameter_list:
        if isinstance(entry, tuple):  # a fork: a tuple of per-branch parameter lists
            nested = [split_compilation_parameters(branch) for branch in entry]
            clean.append(tuple(branch_clean for branch_clean, _ in nested))
            compilation.append(tuple(branch_comp for _, branch_comp in nested))
            continue
        if not isinstance(entry, dict):
            clean.append(entry)
            compilation.append({})
            continue
        level_compilation = entry.get(COMPILATION_PARAMETERS_KEY) or {}
        if not isinstance(level_compilation, dict):
            raise TypeError(
                f"{COMPILATION_PARAMETERS_KEY} must be a dict of compile-time parameters, got "
                f"{type(level_compilation).__name__}"
            )
        clean.append({k: v for k, v in entry.items() if k != COMPILATION_PARAMETERS_KEY})
        compilation.append(dict(level_compilation))
    return clean, compilation


def compilation_parameters_for_level(compilation_parameter_list, position=-1):
    """The compilation parameters belonging to one level of a processed definition list.

    A processed list is ``[prefix, lvl_0, ..., lvl_N]`` while a compilation list is
    ``[params_0, ..., params_N]`` -- offset by the prefix. Callers index the level they are building,
    which is the last one, so the default is what nearly everything wants. ``None`` (no compilation
    list threaded at all) yields ``{}``, which is how every pre-existing call path behaves.
    """
    if not compilation_parameter_list:
        return {}
    entry = compilation_parameter_list[position]
    return entry if isinstance(entry, dict) else {}


def call_level_addition(
    level_addition, processed_definition_list, nested_args, compilation_params
):
    """Invoke a level constructor, handing it compilation parameters only if it can take them.

    A level constructor's established signature is ``(processed_definition_list, *nested_args)``, and
    there are about twenty of them. Rather than widen all of those, the rule is: **a constructor that
    wants compilation parameters declares a ``compilation_parameters`` argument**, and one that does
    not is called exactly as before. The declaration is the registration -- no second registry to
    keep in step with the first.

    Supplying parameters to a level that cannot accept them RAISES. Silently dropping them is the one
    behaviour to avoid: the caller would get a working model built with the default activation
    function while believing it had swapped one in, and no output would look wrong.
    """
    import inspect

    if not compilation_params:
        return level_addition(processed_definition_list, *nested_args)

    try:
        accepts = "compilation_parameters" in inspect.signature(level_addition).parameters
    except (TypeError, ValueError):  # builtins / C callables expose no signature
        accepts = False
    if not accepts:
        level_name = processed_definition_list[-1]
        raise TypeError(
            f"level {level_name!r} was given {COMPILATION_PARAMETERS_KEY}="
            f"{sorted(compilation_params)} but its constructor "
            f"{getattr(level_addition, '__name__', level_addition)} does not declare a "
            "'compilation_parameters' argument, so they would be silently ignored. Either the level "
            "does not support compile-time parameters, or its constructor needs to declare that "
            "argument."
        )
    return level_addition(
        processed_definition_list, *nested_args, compilation_parameters=compilation_params
    )
