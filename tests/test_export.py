"""Export-core integration tests: adapter prep, capture, verification, verdict, run over
every hazard fixture.

Expected statuses are what torch 2.14 actually does, not what the fixtures were written
to provoke. Two worth knowing about: custom_autograd is CLEAN because torch.export traces
straight through the Function's forward (clamp and multiply, both traceable), and
scatter_include_self_false is DEGRADED rather than FAILED because it exports under
strict=False and silently returns wrong numbers. Verification is what catches the second.
"""

import pytest

import downshift
from tests.models import (
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
