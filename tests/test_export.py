"""Integration tests for the export core: the adapter prep, the capture, the verification and the
verdict, run on each hazard fixture.

The expected statuses are what torch 2.14 does. They are not what the fixtures were written to
cause. Two cases are important. custom_autograd is CLEAN, because torch.export traces directly
through the forward of the Function (clamp and multiply can both be traced).
scatter_include_self_false is DEGRADED and not FAILED, because it exports under strict=False
and returns wrong numbers without a message. Verification finds the second case.
"""

import onnx
import pytest
import torch

import downshift
from downshift.core import verdict as verdict_mod
from downshift.core import verify as verify_mod
from downshift.entities.verdict import CaptureResult
from tests.models import (
    bf16_weights,
    clean_mlp,
    custom_autograd,
    data_dependent_branch,
    dict_input,
    dropout_model,
    dynamic_batch_cnn,
    scatter_include_self_false,
    tied_weights,
)

EXPECTED = [
    (clean_mlp, "CLEAN"),
    (dynamic_batch_cnn, "CLEAN"),
    (data_dependent_branch, "FAILED"),
    (custom_autograd, "CLEAN"),
    (tied_weights, "CLEAN"),
    (dict_input, "CLEAN"),
    (dropout_model, "CLEAN"),
    (scatter_include_self_false, "DEGRADED"),
]


@pytest.mark.parametrize(
    ("module", "expected_status"),
    EXPECTED,
    ids=lambda v: getattr(v, "__name__", v).rsplit(".", 1)[-1],
)
def test_fixture_verdict(module, expected_status: str) -> None:
    model = module.make_model()
    inputs = module.make_inputs()

    verdict = downshift.check(model, inputs, samples=8)

    assert verdict.status == expected_status, verdict.reason


def test_clean_mlp_verdict_fields() -> None:
    """Spot-check the full ExportVerdict shape on the simplest fixture."""
    model = clean_mlp.make_model()
    inputs = clean_mlp.make_inputs()

    verdict = downshift.check(model, inputs, samples=8)

    assert verdict.capture_strategy == "strict=False"
    assert verdict.opset is not None
    assert verdict.op_types  # not empty: Gemm, Relu and Gemm for this model
    assert verdict.numerics is not None
    assert verdict.numerics.passed
    assert verdict.recommended_backend == "onnxruntime"
    assert verdict.onnx_program is not None


def test_check_switches_a_training_mode_model_to_eval_with_a_warning() -> None:
    model = clean_mlp.make_model().train()

    verdict = downshift.check(model, clean_mlp.make_inputs(), samples=2)

    assert any("training mode" in w for w in verdict.warnings)
    assert model.training is False


def test_bf16_weights_fixture_fails_via_onnx_runtime_not_a_crash() -> None:
    """The CPU EP of ORT has no bf16 Gemm kernel. This must become a FAILED verdict. It must not
    be an exception that check() raises."""
    model = bf16_weights.make_model()
    inputs = bf16_weights.make_inputs()

    verdict = downshift.check(model, inputs, samples=2)

    assert verdict.status == "FAILED"
    assert verdict.recommended_backend == "torch"
    assert "ONNX Runtime" in verdict.reason
    assert verdict.capture_strategy is not None  # torch.export and torch.onnx.export both worked
    assert verdict.opset is not None
    assert verdict.op_types


def _io_and_weight_types(model_bytes: bytes) -> tuple[set[int], set[int]]:
    proto = onnx.load_from_string(model_bytes)
    io = {v.type.tensor_type.elem_type for v in [*proto.graph.input, *proto.graph.output]}
    return io, {init.data_type for init in proto.graph.initializer}


def test_check_with_fp16_converts_the_graph_and_keeps_the_model_in_fp32() -> None:
    model = clean_mlp.make_model()

    verdict = downshift.check(model, clean_mlp.make_inputs(), samples=2, fp16=True)

    assert verdict.status == "CLEAN", verdict.reason
    assert next(model.parameters()).dtype == torch.float32
    assert verdict.prepared is not None
    assert next(verdict.prepared.model.parameters()).dtype == torch.float32
    io, weights = _io_and_weight_types(verdict.onnx_bytes)
    assert io == {onnx.TensorProto.FLOAT}
    assert weights == {onnx.TensorProto.FLOAT16}
    assert verdict.numerics is not None
    assert verdict.numerics.tolerance_dtype == "float16"
    assert verdict.op_types["Cast"] >= 1


def test_export_with_fp16_and_external_data_writes_the_converted_graph(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr("downshift.core.capture.EXTERNAL_DATA_THRESHOLD", 0)
    out = tmp_path / "half.onnx"

    verdict = downshift.export(clean_mlp.make_model(), out, clean_mlp.make_inputs(), fp16=True)

    assert verdict.status == "CLEAN", verdict.reason
    assert (tmp_path / "half.onnx.data").is_file()
    proto = onnx.load(str(out))
    assert {init.data_type for init in proto.graph.initializer} == {onnx.TensorProto.FLOAT16}


def test_fp16_graph_that_onnx_runtime_cannot_run_points_to_cuda(monkeypatch) -> None:
    def boom(onnx_model):
        raise RuntimeError("Could not find an implementation for Foo(1)")

    monkeypatch.setattr(verify_mod, "new_session", boom)

    verdict = downshift.check(clean_mlp.make_model(), clean_mlp.make_inputs(), fp16=True)

    assert verdict.status == "FAILED"
    assert "Foo(1)" in verdict.reason
    assert "without --fp16" in verdict.reason


def test_build_verdict_uses_first_failure_and_mines_unsupported_ops_from_all(monkeypatch) -> None:
    first = RuntimeError("aten.scatter_reduce.two not supported")
    second = RuntimeError("generic export failure mentioning aten.index_put too")

    def fake_capture(model, inputs, dynamic_shapes=None, external_data_threshold=None):
        return CaptureResult(
            success=False,
            capture_strategy=None,
            exception=first,
            exceptions=[("strict=False", first), ("strict=True", second)],
        )

    monkeypatch.setattr(verdict_mod, "capture", fake_capture)
    prepared = verdict_mod.prepare_model(clean_mlp.make_model(), clean_mlp.make_inputs())

    verdict = verdict_mod.build_verdict(prepared)

    assert verdict.status == "FAILED"
    assert "aten.scatter_reduce" in verdict.reason  # the first failure and not the second
    assert verdict.unsupported_ops == ["index_put", "scatter_reduce"]  # taken from both
    assert verdict.capture_exceptions == [("strict=False", first), ("strict=True", second)]


def test_scatter_fixture_numerics_actually_diverge() -> None:
    """DEGRADED has to mean real numeric divergence, not just a non-empty failure reason."""
    model = scatter_include_self_false.make_model()
    inputs = scatter_include_self_false.make_inputs()

    verdict = downshift.check(model, inputs, samples=8)

    assert verdict.status == "DEGRADED"
    assert verdict.numerics is not None
    assert verdict.numerics.failures > 0
    assert verdict.numerics.max_abs_err > verdict.numerics.tolerance_abs
    assert verdict.recommended_backend == "torch"


@pytest.mark.parametrize(
    ("module", "status"),
    [
        (clean_mlp, "CLEAN"),
        (scatter_include_self_false, "DEGRADED"),
        (data_dependent_branch, "FAILED"),
    ],
    ids=["clean", "degraded", "failed"],
)
def test_verdict_round_trips_through_dict(module, status: str) -> None:
    """from_dict(v.to_dict()) must reproduce to_dict() exactly. A `serve --workers N` worker
    gets its verdict in this way. It does not do the capture and the verification again."""
    verdict = downshift.check(module.make_model(), module.make_inputs(), samples=4)
    assert verdict.status == status, verdict.reason

    rebuilt = verdict_mod.ExportVerdict.from_dict(verdict.to_dict())

    assert rebuilt.to_dict() == verdict.to_dict()


def test_axes_survive_the_dict_round_trip() -> None:
    verdict = downshift.check(clean_mlp.make_model(), clean_mlp.make_inputs(), samples=4)
    assert verdict.axes

    rebuilt = verdict_mod.ExportVerdict.from_dict(verdict.to_dict())

    assert rebuilt.axes == verdict.axes


def test_export_of_a_failed_verdict_writes_nothing(tmp_path) -> None:
    out = tmp_path / "bf16.onnx"

    verdict = downshift.export(
        bf16_weights.make_model(), out, bf16_weights.make_inputs(), samples=2, cache=False
    )

    assert verdict.status == "FAILED"
    assert verdict.onnx_path is None
    assert list(tmp_path.iterdir()) == []
