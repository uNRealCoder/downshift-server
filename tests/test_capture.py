"""capture(): the ONNX-translation failure paths, which never happen on the happy fixtures."""

from downshift.core import capture as capture_mod
from downshift.core.capture import capture
from tests.models import clean_mlp, data_dependent_branch


def test_no_strategy_worked_records_every_attempt_and_quotes_the_first():
    result = capture(data_dependent_branch.make_model(), data_dependent_branch.make_inputs())

    assert result.success is False
    assert result.capture_strategy is None
    assert [name for name, _ in result.exceptions] == ["strict=False", "strict=True"]
    assert result.exception is result.exceptions[0][1]


def test_op_types_is_a_count_descending_histogram():
    result = capture(clean_mlp.make_model(), clean_mlp.make_inputs())

    assert result.success is True
    assert result.op_types == {"Gemm": 2, "Relu": 1}
    counts = list(result.op_types.values())
    assert counts == sorted(counts, reverse=True)


def test_onnx_translation_exception_is_captured(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("translation exploded")

    monkeypatch.setattr(capture_mod.torch.onnx, "export", boom)

    result = capture(clean_mlp.make_model(), clean_mlp.make_inputs())

    assert result.success is False
    assert result.capture_strategy is not None  # torch.export itself succeeded
    assert isinstance(result.exception, RuntimeError)


def test_onnx_translation_returning_none_is_treated_as_failure(monkeypatch):
    monkeypatch.setattr(capture_mod.torch.onnx, "export", lambda *a, **kw: None)

    result = capture(clean_mlp.make_model(), clean_mlp.make_inputs())

    assert result.success is False
    assert result.capture_strategy is not None
    assert "returned None" in str(result.exception)
