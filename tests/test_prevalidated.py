"""intake(): a .onnx someone else produced, with and without a reference model."""

from pathlib import Path

import pytest
import torch

import downshift
from downshift.export.prevalidated import intake
from tests.models import clean_mlp


@pytest.fixture(scope="module")
def onnx_file(tmp_path_factory) -> tuple[Path, torch.nn.Module]:
    model = clean_mlp.make_model()
    out = tmp_path_factory.mktemp("prevalidated") / "m.onnx"
    verdict = downshift.export(model, out, clean_mlp.make_inputs())
    assert verdict.status == "CLEAN", verdict.reason
    return out, model


def test_intake_without_reference_is_unverified(onnx_file):
    path, _ = onnx_file

    verdict = intake(path)

    assert verdict.status == "UNVERIFIED"
    assert verdict.exit_code == 3
    assert verdict.recommended_backend == "onnxruntime"
    assert verdict.model_family == "onnx"
    assert verdict.input_names == ("x",)
    assert isinstance(verdict.opset, int)
    assert verdict.op_types
    assert verdict.numerics is None
    assert verdict.onnx_path == path


def test_intake_with_matching_reference_is_clean(onnx_file):
    path, model = onnx_file

    verdict = intake(path, reference=model, example_inputs=clean_mlp.make_inputs())

    assert verdict.status == "CLEAN", verdict.reason
    assert verdict.exit_code == 0
    assert verdict.recommended_backend == "onnxruntime"
    assert verdict.model_family == "generic-torch"
    assert verdict.input_names == ("x",)
    assert verdict.numerics is not None and verdict.numerics.passed


def test_intake_with_different_reference_is_degraded(onnx_file):
    path, _ = onnx_file
    other = clean_mlp.make_model()  # fresh random weights: the graph no longer matches

    verdict = intake(path, reference=other, example_inputs=clean_mlp.make_inputs())

    assert verdict.status == "DEGRADED"
    assert verdict.exit_code == 2
    assert verdict.recommended_backend == "torch"
    assert verdict.numerics is not None
    assert verdict.numerics.failures > 0
    assert verdict.numerics.max_abs_err > verdict.numerics.tolerance_abs
