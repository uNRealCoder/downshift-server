"""intake(): a .onnx someone else produced, with and without a reference model."""

from downshift.core import verify as verify_mod
from downshift.core.prevalidated import intake
from tests.models import clean_mlp


def test_intake_without_reference_is_unverified(exported_mlp):
    path, _, _ = exported_mlp

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


def test_intake_with_matching_reference_is_clean(exported_mlp):
    path, model, _ = exported_mlp

    verdict = intake(path, reference=model, example_inputs=clean_mlp.make_inputs())

    assert verdict.status == "CLEAN", verdict.reason
    assert verdict.exit_code == 0
    assert verdict.recommended_backend == "onnxruntime"
    assert verdict.model_family == "generic-torch"
    assert verdict.input_names == ("x",)
    assert verdict.numerics is not None and verdict.numerics.passed


def test_intake_with_reference_fails_via_onnx_runtime_not_a_crash(exported_mlp, monkeypatch):
    path, model, _ = exported_mlp

    def boom(onnx_model):
        raise RuntimeError("NOT_IMPLEMENTED : Could not find an implementation for Gemm(13)")

    monkeypatch.setattr(verify_mod, "_to_session", boom)

    verdict = intake(path, reference=model, example_inputs=clean_mlp.make_inputs())

    assert verdict.status == "FAILED"
    assert verdict.exit_code == 1
    assert verdict.recommended_backend == "torch"
    assert "ONNX Runtime" in verdict.reason
    assert verdict.numerics is None


def test_intake_with_different_reference_is_degraded(exported_mlp):
    path, _, _ = exported_mlp
    other = clean_mlp.make_model()  # fresh random weights: the graph no longer matches

    verdict = intake(path, reference=other, example_inputs=clean_mlp.make_inputs())

    assert verdict.status == "DEGRADED"
    assert verdict.exit_code == 2
    assert verdict.recommended_backend == "torch"
    assert verdict.numerics is not None
    assert verdict.numerics.failures > 0
    assert verdict.numerics.max_abs_err > verdict.numerics.tolerance_abs
