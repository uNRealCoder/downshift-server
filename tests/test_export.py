"""Export-core integration tests: adapter prep, capture, verification, verdict, run over
every hazard fixture.

Expected statuses are what torch 2.14 actually does, not what the fixtures were written
to provoke. Two worth knowing about: custom_autograd is CLEAN because torch.export traces
straight through the Function's forward (clamp and multiply, both traceable), and
scatter_include_self_false is DEGRADED rather than FAILED because it exports under
strict=False and silently returns wrong numbers. Verification is what catches the second.
"""

import pytest
import torch

import downshift
from downshift.core import verdict as verdict_mod
from downshift.core.capture import CaptureResult
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

    verdict = downshift.check(model, inputs, k=8)

    assert verdict.status == expected_status, verdict.reason


def test_clean_mlp_verdict_fields() -> None:
    """Spot-check the full ExportVerdict shape on the simplest fixture."""
    model = clean_mlp.make_model()
    inputs = clean_mlp.make_inputs()

    verdict = downshift.check(model, inputs, k=8)

    assert verdict.capture_strategy == "strict=False"
    assert verdict.opset is not None
    assert verdict.op_types  # non-empty: Gemm/Relu/Gemm for this model
    assert verdict.numerics is not None
    assert verdict.numerics.passed
    assert verdict.recommended_backend == "onnxruntime"
    assert verdict.onnx_program is not None


def test_check_switches_a_training_mode_model_to_eval_with_a_warning() -> None:
    model = clean_mlp.make_model().train()

    verdict = downshift.check(model, clean_mlp.make_inputs(), k=2)

    assert any("training mode" in w for w in verdict.warnings)
    assert model.training is False


def test_bf16_weights_fixture_fails_via_onnx_runtime_not_a_crash() -> None:
    """ORT's CPU EP has no bf16 Gemm kernel; that must become a FAILED verdict, not a
    raised exception out of check()."""
    model = bf16_weights.make_model()
    inputs = bf16_weights.make_inputs()

    verdict = downshift.check(model, inputs, k=2)

    assert verdict.status == "FAILED"
    assert verdict.recommended_backend == "torch"
    assert "ONNX Runtime" in verdict.reason
    assert verdict.capture_strategy is not None  # torch.export/torch.onnx.export both worked
    assert verdict.opset is not None
    assert verdict.op_types


def test_check_with_fp16_does_not_mutate_the_callers_model() -> None:
    model = clean_mlp.make_model()
    original_dtype = next(model.parameters()).dtype

    verdict = downshift.check(model, clean_mlp.make_inputs(), k=2, fp16=True)

    assert next(model.parameters()).dtype == original_dtype
    assert verdict.prepared is not None
    assert next(verdict.prepared.model.parameters()).dtype == torch.float16


def test_build_verdict_uses_first_failure_and_mines_unsupported_ops_from_all(monkeypatch) -> None:
    first = RuntimeError("aten.scatter_reduce.two not supported")
    second = RuntimeError("generic export failure mentioning aten.index_put too")

    def fake_capture(model, inputs, dynamic_shapes=None):
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
    assert "aten.scatter_reduce" in verdict.reason  # the first failure, not the second's
    assert verdict.unsupported_ops == ["index_put", "scatter_reduce"]  # mined from both
    assert verdict.capture_exceptions == [("strict=False", first), ("strict=True", second)]


def test_scatter_fixture_numerics_actually_diverge() -> None:
    """DEGRADED has to mean real numeric divergence, not just a non-empty failure reason."""
    model = scatter_include_self_false.make_model()
    inputs = scatter_include_self_false.make_inputs()

    verdict = downshift.check(model, inputs, k=8)

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
    """from_dict(v.to_dict()) must reproduce to_dict() exactly, since this is how a
    `serve --workers N` worker gets its verdict without redoing capture/verify itself."""
    verdict = downshift.check(module.make_model(), module.make_inputs(), k=4)
    assert verdict.status == status, verdict.reason

    rebuilt = verdict_mod.ExportVerdict.from_dict(verdict.to_dict())

    assert rebuilt.to_dict() == verdict.to_dict()
