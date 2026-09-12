"""Dynamic-shape helpers and the --dynamic override."""

import pytest
import torch

import downshift
from downshift.export.shapes import (
    alternative_sizes,
    apply_dynamic_override,
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


def test_safe_capture_inputs_doubles_only_size_one_dynamic_axes():
    dim = torch.export.Dim("d", min=1, max=64)
    x = torch.randn(1, 16)
    mask = torch.ones(1, 16)
    idx = torch.arange(3)

    safe = safe_capture_inputs((x, mask, idx), ({0: dim}, None, {0: dim}))

    assert tuple(safe[0].shape) == (2, 16)
    assert torch.equal(safe[0][0], x[0]) and torch.equal(safe[0][1], x[0])
    assert safe[1] is mask  # not dynamic, untouched
    assert safe[2] is idx  # dynamic but already > 1, untouched


def test_check_with_dynamic_override_is_still_clean():
    verdict = downshift.check(clean_mlp.make_model(), clean_mlp.make_inputs(), dynamic={"x": [0]})
    assert verdict.status == "CLEAN", verdict.reason
    assert verdict.dynamic_dims == {"x": [0]}
