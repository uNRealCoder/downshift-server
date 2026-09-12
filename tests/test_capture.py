"""capture(): the ONNX-translation failure paths, which never happen on the happy fixtures."""

from downshift.export import capture as capture_mod
from downshift.export.capture import capture
from tests.models import clean_mlp


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
