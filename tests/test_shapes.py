"""The helpers for dynamic shapes, and the --dynamic override."""

import pytest
import torch

import downshift
from downshift.core.shapes import (
    alternative_sizes,
    apply_dynamic_override,
    dim_bounds,
    lower_axis_max,
    parse_dynamic_spec,
    safe_capture_inputs,
)
from tests.models import clean_mlp


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("x:0,edge_index:1", {"x": [0], "edge_index": [1]}),
        ("x:0:1", {"x": [0, 1]}),
        (" x:0 , edge_index:1 ,", {"x": [0], "edge_index": [1]}),
        ("x:0,x:1", {"x": [0, 1]}),
    ],
    ids=["two-names", "two-axes", "whitespace-and-trailing-comma", "repeated-name"],
)
def test_parse_dynamic_spec(spec, expected):
    assert parse_dynamic_spec(spec) == expected


@pytest.mark.parametrize(
    "spec", ["x", "x:", ":0", "x:zero"], ids=["no-axis", "empty-axis", "no-name", "non-int"]
)
def test_parse_dynamic_spec_rejects_bad_entries(spec):
    with pytest.raises(ValueError):
        parse_dynamic_spec(spec)


def test_apply_dynamic_override_builds_one_dim_per_axis():
    inputs = (torch.randn(6, 8), torch.randint(0, 6, (2, 10)))

    shapes = apply_dynamic_override(("x", "edge_index"), inputs, {"edge_index": [1]})

    assert shapes[0] is None
    assert list(shapes[1]) == [1]
    assert isinstance(shapes[1][1], type(torch.export.Dim("probe")))


def test_apply_dynamic_override_rejects_unknown_name():
    inputs = (torch.randn(6, 8),)
    with pytest.raises(ValueError, match="edge_index"):
        apply_dynamic_override(("x",), inputs, {"edge_index": [1]})


def test_alternative_sizes_excludes_base_and_covers_edges():
    sizes = alternative_sizes(6)
    assert 6 not in sizes
    assert {1, 2, 3, 7, 12} <= set(sizes)
    assert sizes == sorted(sizes)


def test_alternative_sizes_clamps_to_bounds():
    assert alternative_sizes(2, lo=1, hi=3) == [1, 3]


def test_dim_bounds_reads_min_and_max_off_the_dim():
    dim = torch.export.Dim("n", min=1, max=64)
    assert dim_bounds({0: dim}, 0) == (1, 64)


def test_dim_bounds_falls_back_when_axis_isnt_dynamic():
    assert dim_bounds(None, 0) == (1, 1 << 16)
    assert dim_bounds({}, 0) == (1, 1 << 16)


def test_safe_capture_inputs_doubles_only_size_one_dynamic_axes():
    dim = torch.export.Dim("d", min=1, max=64)
    x = torch.randn(1, 16)
    mask = torch.ones(1, 16)
    idx = torch.arange(3)

    safe = safe_capture_inputs((x, mask, idx), ({0: dim}, None, {0: dim}))

    assert tuple(safe[0].shape) == (2, 16)
    assert torch.equal(safe[0][0], x[0]) and torch.equal(safe[0][1], x[0])
    assert safe[1] is mask  # not dynamic, unchanged
    assert safe[2] is idx  # dynamic, but already > 1, unchanged


def test_check_with_dynamic_override_is_still_clean():
    verdict = downshift.check(clean_mlp.make_model(), clean_mlp.make_inputs(), dynamic={"x": [0]})
    assert verdict.status == "CLEAN", verdict.reason
    assert verdict.dynamic_dims == {"x": [0]}


def test_lower_axis_max_keeps_name_and_min_and_shares_dims():
    batch = torch.export.Dim("batch", min=2, max=64)
    seq = torch.export.Dim("seq", min=1, max=128)
    spec = {0: batch, 1: seq}

    shapes = lower_axis_max((spec, spec, None), {"seq": 32})

    assert shapes[2] is None
    assert shapes[0][1] is shapes[1][1]
    assert dim_bounds(shapes[0], 1) == (1, 32)
    assert shapes[0][1].__name__ == "seq"
    assert shapes[0][0] is batch
    assert lower_axis_max((spec,), None) == (spec,)


def test_lower_axis_max_rejects_unknown_name_listing_the_axes():
    spec = {0: torch.export.Dim("batch", min=1, max=64)}
    with pytest.raises(ValueError, match=r"\['nodes'\].*\['batch'\]"):
        lower_axis_max((spec,), {"nodes": 5})


def test_lower_axis_max_rejects_a_value_above_the_ceiling():
    spec = {0: torch.export.Dim("seq", min=1, max=64)}
    with pytest.raises(ValueError, match="seq=65 is above the limit of 64"):
        lower_axis_max((spec,), {"seq": 65})


def test_generic_axis_max_lowers_dim0_and_pins_sample_one():
    verdict = downshift.check(
        clean_mlp.make_model(), clean_mlp.make_inputs(), axis_max={"dim0": 20}, k=3
    )

    assert verdict.status == "CLEAN", verdict.reason
    (fact,) = verdict.axes
    assert fact.served_max == 20
    assert [shapes[0][0] for shapes in verdict.numerics.sample_shapes][1] == 20


def test_axis_max_applies_to_dynamic_override_axes():
    verdict = downshift.check(
        clean_mlp.make_model(),
        clean_mlp.make_inputs(),
        dynamic={"x": [0]},
        axis_max={"x_0": 12},
        k=3,
    )

    assert verdict.status == "CLEAN", verdict.reason
    assert verdict.axes[0].name == "x_0"
    assert verdict.axes[0].served_max == 12
    assert verdict.numerics.sample_shapes[1][0][0] == 12


def test_axis_max_unknown_name_is_a_value_error():
    with pytest.raises(ValueError, match="dim0"):
        downshift.check(clean_mlp.make_model(), clean_mlp.make_inputs(), axis_max={"seq": 4})
