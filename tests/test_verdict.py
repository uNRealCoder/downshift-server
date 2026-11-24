"""core/verdict.py: prepare_model, and intake() of a .onnx file with and without a reference."""

import pytest
import torch

import downshift
from downshift.core import verify as verify_mod
from downshift.core.verdict import intake, prepare_model
from downshift.sources import ONNX_FILE
from tests.models import clean_mlp, scatter_include_self_false


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
    assert verdict.model_family == "generic"
    assert verdict.input_names == ("x",)
    assert verdict.numerics is not None and verdict.numerics.passed


def test_intake_of_an_fp16_graph_uses_the_float16_tolerances(tmp_path):
    model = clean_mlp.make_model()
    out = tmp_path / "half.onnx"
    downshift.export(model, out, clean_mlp.make_inputs(), fp16=True)

    verdict = intake(out, reference=model, example_inputs=clean_mlp.make_inputs())

    assert verdict.status == "CLEAN", verdict.reason
    assert verdict.numerics is not None
    assert verdict.numerics.tolerance_dtype == "float16"


def test_serving_check_of_an_fp16_file_without_a_reference_uses_float16_tolerances(
    tmp_path, exported_mlp
):
    from downshift.serve.engine import ServeOptions, _serving_tolerances

    out = tmp_path / "half.onnx"
    downshift.export(clean_mlp.make_model(), out, clean_mlp.make_inputs(), fp16=True)

    half = intake(out)
    full = intake(exported_mlp[0])

    assert _serving_tolerances(half, ServeOptions()) == (1e-2, 1e-2)
    assert _serving_tolerances(full, ServeOptions()) == (1e-4, 1e-3)


def test_intake_with_reference_fails_via_onnx_runtime_not_a_crash(exported_mlp, monkeypatch):
    path, model, _ = exported_mlp

    def boom(onnx_model):
        raise RuntimeError("NOT_IMPLEMENTED : Could not find an implementation for Gemm(13)")

    monkeypatch.setattr(verify_mod, "new_session", boom)

    verdict = intake(path, reference=model, example_inputs=clean_mlp.make_inputs())

    assert verdict.status == "FAILED"
    assert verdict.exit_code == 1
    assert verdict.recommended_backend == "torch"
    assert "ONNX Runtime" in verdict.reason
    assert verdict.numerics is None


def test_intake_with_different_reference_is_degraded(exported_mlp):
    path, _, _ = exported_mlp
    other = clean_mlp.make_model()  # new random weights: the graph no longer matches

    verdict = intake(path, reference=other, example_inputs=clean_mlp.make_inputs())

    assert verdict.status == "DEGRADED"
    assert verdict.exit_code == 2
    assert verdict.recommended_backend == "torch"
    assert verdict.numerics is not None
    assert verdict.numerics.failures > 0
    assert verdict.numerics.max_abs_err > verdict.numerics.tolerance_abs


def test_intake_serves_an_onnx_with_an_external_data_file(tmp_path, monkeypatch):
    from downshift.core import capture as capture_mod
    from downshift.entities.cache import ExportEntry
    from downshift.entities.options import ServeOptions
    from downshift.serve.engine import state_from_entry

    monkeypatch.setattr(capture_mod, "EXTERNAL_DATA_THRESHOLD", 0)
    path = tmp_path / "ext.onnx"
    exported = downshift.export(clean_mlp.make_model(), path, clean_mlp.make_inputs())
    assert exported.status == "CLEAN", exported.reason
    assert (tmp_path / "ext.onnx.data").is_file()

    verdict = intake(path)

    assert verdict.status == "UNVERIFIED"
    assert verdict.onnx_path == path
    assert verdict.op_types["Gemm"] == 2
    entry = ExportEntry(verdict.to_dict(), list(verdict.input_names), {}, onnx_path=path)
    opts = ServeOptions(samples=1, device="cpu", warmup=1)
    state = state_from_entry(entry, None, opts, None, source=str(path), kind=ONNX_FILE)
    assert state.backend.name == "onnxruntime"


def test_prepare_model_raises_when_nothing_can_guess_inputs():
    model = scatter_include_self_false.make_model()
    with pytest.raises(ValueError, match="Cannot find example inputs"):
        prepare_model(model, None, "generic")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
def test_a_model_on_a_cuda_device_is_verified():
    model = torch.nn.Linear(8, 4).to("cuda").eval()

    verdict = downshift.check(model, (torch.randn(2, 8, device="cuda"),), samples=2, cache=False)

    assert verdict.status == "CLEAN", verdict.reason
