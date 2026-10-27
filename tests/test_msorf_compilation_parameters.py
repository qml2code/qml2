"""Compile-time (baked-in) parameters for multilevel_sorf levels.

The mechanism, not any level that uses it -- no level does yet, deliberately. What is pinned here:

* a model that declares no compilation parameters produces byte-identical cache names, so nothing
  that worked before can change;
* two models differing ONLY in a compilation parameter get DIFFERENT compiled routines. This is the
  one that would fail silently: routines are memoised in base_constructors/common.py's module
  globals under a name derived from the definition list, so without a token in that name the second
  model would quietly reuse the first's baked-in function and every number it produced would look
  plausible;
* parameters handed to a level that cannot accept them raise rather than being dropped, for the
  same reason.
"""
import numpy as np
import pytest

from qml2.multilevel_sorf.base_constructors import common
from qml2.multilevel_sorf.base_constructors.base import function_prefix
from qml2.multilevel_sorf.base_constructors.compilation_parameters import (
    COMPILATION_PARAMETERS_KEY,
    call_level_addition,
    compilation_token,
    level_cache_name,
    split_compilation_parameters,
)
from qml2.multilevel_sorf.sorf_calculation import MultilevelSORF

# --------------------------------------------------------------------------- splitting


def test_split_extracts_and_cleans():
    params = [
        {"output_size": 64},
        {"nfeature_stacks": 2, COMPILATION_PARAMETERS_KEY: {"activation_function": np.cos}},
        {},
    ]
    clean, compilation = split_compilation_parameters(params)
    assert clean == [{"output_size": 64}, {"nfeature_stacks": 2}, {}]
    assert compilation == [{}, {"activation_function": np.cos}, {}]
    # the caller's dicts must not be mutated -- they may be reused for a second model
    assert COMPILATION_PARAMETERS_KEY in params[1]


def test_split_pads_every_level_so_it_can_be_sliced_alongside_a_definition_list():
    clean, compilation = split_compilation_parameters([{"a": 1}, {}, {"b": 2}])
    assert compilation == [{}, {}, {}]
    assert len(compilation) == len(clean) == 3


def test_split_handles_forks():
    branch_a = [{"output_size": 8, COMPILATION_PARAMETERS_KEY: {"f": abs}}]
    branch_b = [{"output_size": 16}]
    clean, compilation = split_compilation_parameters([(branch_a, branch_b), {}])
    assert clean[0] == ([{"output_size": 8}], [{"output_size": 16}])
    assert compilation[0] == ([{"f": abs}], [{}])
    assert compilation[1] == {}


def test_split_rejects_a_non_dict():
    with pytest.raises(TypeError, match=COMPILATION_PARAMETERS_KEY):
        split_compilation_parameters([{COMPILATION_PARAMETERS_KEY: ["not", "a", "dict"]}])


# --------------------------------------------------------------------------- the token


def test_token_is_empty_without_parameters():
    """Load-bearing: it is what keeps every pre-existing cache name unchanged."""
    assert compilation_token({}) == ""
    assert compilation_token(None) == ""
    assert level_cache_name("sorf") == "sorf"
    assert level_cache_name("sorf", {}) == "sorf"


def test_token_is_stable_sensitive_and_order_independent():
    assert compilation_token({"f": np.cos}) == compilation_token({"f": np.cos})
    assert compilation_token({"f": np.cos}) != compilation_token({"f": np.sin})
    assert compilation_token({"a": 1, "b": 2}) == compilation_token({"b": 2, "a": 1})
    assert level_cache_name("sorf", {"f": np.cos}).startswith("sorf_c")
    assert level_cache_name("sorf", {"f": np.cos}).isidentifier(), "used as a key in globals()"


def test_token_identifies_a_callable_by_qualname_not_by_id():
    """repr() of a function embeds its id(), which differs between a run and its own restart -- the
    routine would then be recompiled under a new name every process. Two distinct function objects
    with the same module and qualname must therefore agree."""

    def make():
        def activation(x):
            return x

        return activation

    f, g = make(), make()
    assert f is not g
    assert compilation_token({"a": f}) == compilation_token({"a": g})


# --------------------------------------------------------------------------- cache keys


def test_names_are_unchanged_when_nothing_declares_parameters():
    d = [function_prefix, "resize", "sorf", "component_sum"]
    assert common.get_name_from_definition_list(d) == common.get_name_from_definition_list(
        d, [{}, {}, {}]
    )


def test_a_declared_parameter_changes_only_its_own_level_in_the_name():
    d = [function_prefix, "resize", "sorf", "component_sum"]
    plain = common.get_name_from_definition_list(d)
    annotated = common.get_name_from_definition_list(d, [{}, {"f": np.cos}, {}])
    assert annotated != plain
    assert annotated.startswith(function_prefix + "_resize_sorf_c")
    assert annotated.endswith("_component_sum_")


def test_differing_parameters_give_differing_names():
    """The correctness-critical case: same levels, different baked-in function, different key."""
    d = [function_prefix, "resize", "sorf", "component_sum"]
    a = common.get_name_from_definition_list(d, [{}, {"f": np.cos}, {}])
    b = common.get_name_from_definition_list(d, [{}, {"f": np.sin}, {}])
    assert a != b


# --------------------------------------------------------------------------- delivery


def _recording_level(record):
    def level_addition(processed_definition_list, nested_routine, compilation_parameters):
        record["seen"] = compilation_parameters
        record["levels"] = list(processed_definition_list)
        return nested_routine

    return level_addition


def test_level_constructor_receives_its_parameters(monkeypatch):
    """End to end through get_routine, with a level registered only for this test -- the mechanism is
    exercised without committing to any real level's specification."""
    record = {}
    monkeypatch.setitem(
        common.function_level_additions, "recording_test_lvl", _recording_level(record)
    )
    baked = {"representation_function": np.cos, "rep_kwargs": {"rcut": 4.8}}
    common.get_routine([function_prefix, "resize", "recording_test_lvl"], [{}, baked])

    assert record["seen"] == baked, "the level did not receive its compilation parameters"
    assert record["levels"][-1] == "recording_test_lvl"


def test_a_level_that_cannot_accept_parameters_raises_rather_than_dropping_them(monkeypatch):
    """Silently ignoring them is the failure to avoid: the caller would get a working model built
    with the default function while believing its own had been baked in."""

    def without_support(processed_definition_list, nested_routine):
        return nested_routine

    monkeypatch.setitem(common.function_level_additions, "unsupported_test_lvl", without_support)
    with pytest.raises(TypeError, match="does not declare a 'compilation_parameters'"):
        common.get_routine(
            [function_prefix, "resize", "unsupported_test_lvl"], [{}, {"f": np.cos}]
        )


def test_call_level_addition_leaves_unsupporting_levels_alone():
    """With no parameters, a constructor lacking the argument must still be called exactly as before
    -- that is what keeps the twenty existing level constructors untouched."""

    def legacy(processed_definition_list, nested_routine):
        return ("called", nested_routine)

    assert call_level_addition(legacy, [function_prefix, "resize"], ("nested",), {}) == (
        "called",
        "nested",
    )


# --------------------------------------------------------------------------- model integration


def test_model_strips_the_reserved_key_before_anything_else_sees_it():
    """Input-object construction and size bookkeeping must see the dict they always saw."""
    model = MultilevelSORF(
        ["resize", "sorf", "component_sum"],
        [
            {"output_size": 64},
            {"nfeature_stacks": 2, "ntransforms": 3},
            {},
        ],
    )
    assert model.compilation_parameter_list == [{}, {}, {}]
    assert all(COMPILATION_PARAMETERS_KEY not in p for p in model.parameter_list)
    assert model.parameter_list[1] == {"nfeature_stacks": 2, "ntransforms": 3}


def test_model_without_parameters_is_unaffected_end_to_end():
    """The regression guard for every existing model: it still builds and still produces features."""
    model = MultilevelSORF(
        ["resize", "sorf", "component_sum"],
        [{"output_size": 32}, {"nfeature_stacks": 2, "ntransforms": 3}, {}],
    )
    assert model.get_sorf_routine() is not None
    assert model.get_cpu_estimator() is not None
