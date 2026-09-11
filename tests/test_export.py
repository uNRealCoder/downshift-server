"""Export-core integration tests (sprint plan H4-H10): the full pipeline — adapter prep,
capture, mandatory numerical verification, verdict — run against every hazard fixture.

Expected statuses reflect *observed* behavior on torch 2.14, verified by actually running
the pipeline, not the sprint plan's predictions:
  - custom_autograd: the plan guessed FAILED ("no symbolic trace by default"); in practice
    torch.export traces straight through the autograd.Function's forward body, since it's
    just clamp + multiply — both already-traceable ops. Verdict is CLEAN.
  - scatter_include_self_false: the plan guessed a loud FAILED; in practice it exports
    "successfully" under strict=False and silently produces wrong numbers — exactly the
    dangerous case IMPLEMENTATION_PLAN.md §5.5 warns about. Mandatory verification catches
    it and the verdict is DEGRADED, not FAILED.
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
    ("module", "expected_status"), EXPECTED, ids=lambda v: getattr(v, "__name__", v).rsplit(".", 1)[-1]
)
def test_fixture_verdict(module, expected_status: str) -> None:
    model = module.make_model()
    inputs = module.make_inputs()

    verdict = downshift.export(model, inputs, k=8)

    assert verdict.status == expected_status, verdict.reason


def test_clean_mlp_verdict_fields() -> None:
    """Spot-check the full ExportVerdict shape on the simplest fixture."""
    model = clean_mlp.make_model()
    inputs = clean_mlp.make_inputs()

    verdict = downshift.export(model, inputs, k=8)

    assert verdict.capture_strategy == "strict=False"
    assert verdict.opset is not None
    assert verdict.op_types  # non-empty: Gemm/Relu/Gemm for this model
    assert verdict.numerics is not None
    assert verdict.numerics.passed
    assert verdict.recommended_backend == "onnxruntime"
    assert verdict.onnx_program is not None


def test_scatter_fixture_numerics_actually_diverge() -> None:
    """The DEGRADED verdict must be backed by a real, non-trivial numeric divergence —
    not just any failure reason."""
    model = scatter_include_self_false.make_model()
    inputs = scatter_include_self_false.make_inputs()

    verdict = downshift.export(model, inputs, k=8)

    assert verdict.status == "DEGRADED"
    assert verdict.numerics is not None
    assert verdict.numerics.failures > 0
    assert verdict.numerics.max_abs_err > verdict.numerics.tolerance_abs
    assert verdict.recommended_backend == "torch"
