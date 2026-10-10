"""capture(): the failure paths of the ONNX translation. They never happen on the fixtures that work."""

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
    # Older exporters (torch 2.5) also emit a Transpose for each weight before each Gemm.
    assert result.op_types["Gemm"] == 2
    assert result.op_types["Relu"] == 1
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


def test_default_threshold_keeps_the_graph_in_memory():
    result = capture(clean_mlp.make_model(), clean_mlp.make_inputs())

    assert result.success is True
    assert result.onnx_bytes
    assert result.onnx_path is None
    assert result.tmpdir is None


def test_over_threshold_weights_go_to_an_external_data_file():
    import onnxruntime as ort

    result = capture(clean_mlp.make_model(), clean_mlp.make_inputs(), external_data_threshold=0)

    assert result.success is True
    assert result.onnx_bytes == b""
    assert result.onnx_path is not None and result.onnx_path.name == "model.onnx"
    assert result.onnx_path.with_name("model.onnx.data").is_file()
    assert result.op_types["Gemm"] == 2
    assert result.opset is not None
    session = ort.InferenceSession(str(result.onnx_path), providers=["CPUExecutionProvider"])
    assert session.get_inputs()[0].name == "x"


def test_external_data_verdict_is_clean_and_keeps_its_directory_alive():
    from downshift.core.verdict import build_verdict, prepare_model

    prepared = prepare_model(clean_mlp.make_model(), clean_mlp.make_inputs())

    verdict = build_verdict(prepared, k=2, _external_data_threshold=0)

    assert verdict.status == "CLEAN", verdict.reason
    assert verdict.onnx_bytes == b""
    assert verdict.onnx_path is not None
    assert verdict.onnx_path.with_name("model.onnx.data").is_file()
    assert verdict._tmpdir is not None
    assert "_tmpdir" not in verdict.to_dict()
