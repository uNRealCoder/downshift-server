"""Direct unit coverage for verify.py's small helpers, exercised only indirectly elsewhere."""

import numpy as np
import onnxruntime as ort
import pytest
import torch

from downshift.core import verify as verify_mod
from downshift.core.verify import (
    OnnxRuntimeError,
    _as_tensor_list,
    _compare_sample,
    _resize_dim0,
    _to_session,
    verify,
)
from tests.models import bf16_weights, clean_mlp


def test_resize_dim0_is_a_noop_for_scalars_and_matching_sizes():
    scalar = torch.tensor(5.0)
    assert _resize_dim0(scalar, 3) is scalar

    tensor = torch.randn(4, 3)
    assert _resize_dim0(tensor, 4) is tensor


def test_to_session_accepts_raw_bytes(exported_mlp):
    onnx_path, _, _ = exported_mlp
    session = _to_session(onnx_path.read_bytes())
    assert isinstance(session, ort.InferenceSession)


def test_as_tensor_list_filters_a_tuple_and_rejects_other_types():
    t1, t2 = torch.zeros(1), torch.ones(1)
    assert _as_tensor_list((t1, "not a tensor", t2)) == [t1, t2]

    with pytest.raises(TypeError, match="can't compare"):
        _as_tensor_list({"a": 1})


def test_verify_requires_dynamic_shapes_or_vary_fn():
    with pytest.raises(ValueError, match="dynamic_shapes or an explicit vary_fn"):
        verify(clean_mlp.make_model(), "unused.onnx", clean_mlp.make_inputs())


class _FakeInput:
    name = "x"


class _EchoSession:
    """Mimics an ORT session whose graph is `x * 2`, so tests don't need a real export."""

    def get_inputs(self):
        return [_FakeInput()]

    def run(self, output_names, feeds):
        return [feeds["x"] * 2]


def test_compare_sample_uses_per_element_allclose_not_the_maxima():
    """A sample np.allclose would accept must PASS, even though the old max-abs/max-rel
    rule (each maximum can come from a different element) would have failed it."""
    atol, rtol = 1e-4, 1e-3
    expected = torch.tensor([1000.0, 1e-6])
    got = np.array([1000.5, 1e-6 + 5e-5])

    sample_abs, sample_rel, failed, note = _compare_sample([expected], [got], atol, rtol, 0)

    assert failed is False
    assert note is None
    # Sanity: this is exactly the case the old maxima-based rule got wrong.
    assert sample_abs > atol
    assert sample_rel > rtol


def test_compare_sample_fails_when_any_single_element_exceeds_allclose():
    atol, rtol = 1e-4, 1e-3
    expected = torch.tensor([1.0])
    got = np.array([2.0])  # abs_err 1.0 >> atol + rtol * |expected| == 1.1e-3

    _, _, failed, note = _compare_sample([expected], [got], atol, rtol, 0)

    assert failed is True
    assert note is None


def test_compare_sample_notes_a_shape_mismatch_instead_of_raising():
    expected = torch.zeros(2, 4)
    got = np.zeros((2, 3))

    _, _, failed, note = _compare_sample([expected], [got], 1e-4, 1e-3, 3)

    assert failed is True
    assert note == "sample 3: output_0 shape (2, 4) from torch vs (2, 3) from onnxruntime"


def test_compare_sample_notes_an_output_count_mismatch_instead_of_raising():
    expected = [torch.zeros(2), torch.zeros(2)]
    got = [np.zeros(2)]

    _, _, failed, note = _compare_sample(expected, got, 1e-4, 1e-3, 1)

    assert failed is True
    assert note == "sample 1: 2 torch outputs vs 1 onnxruntime outputs"


def test_verify_records_a_note_and_fails_when_output_shapes_differ(monkeypatch):
    model = clean_mlp.make_model()
    inputs = clean_mlp.make_inputs()  # forward output is shape (1, 4)

    class MismatchedSession:
        def get_inputs(self):
            return [_FakeInput()]

        def run(self, output_names, feeds):
            return [np.zeros((1, 3), dtype=np.float32)]

    monkeypatch.setattr(verify_mod, "_to_session", lambda onnx_model: MismatchedSession())

    report = verify(model, "unused.onnx", inputs, vary_fn=lambda i: inputs, k=1)

    assert report.failures == 1
    assert report.notes == ["sample 0: output_0 shape (1, 4) from torch vs (1, 3) from onnxruntime"]


def test_bf16_output_is_cast_to_float32_before_numpy_conversion():
    """torch.Tensor.numpy() raises on bfloat16; verify()'s comparison must not hit that."""
    model = bf16_weights.make_model()
    inputs = bf16_weights.make_inputs()
    with torch.inference_mode():
        expected = model(*inputs)
    assert expected.dtype is torch.bfloat16

    got = expected.float().numpy()
    sample_abs, sample_rel, failed, note = _compare_sample([expected], [got], 1.0, 1.0, 0)

    assert note is None
    assert failed is False
    assert sample_abs == pytest.approx(0.0)


def test_verify_wraps_a_session_creation_failure_as_onnx_runtime_error(monkeypatch):
    def boom(onnx_model):
        raise RuntimeError("bad graph\nsome onnxruntime detail")

    monkeypatch.setattr(verify_mod, "_to_session", boom)

    with pytest.raises(OnnxRuntimeError, match="bad graph"):
        verify(
            clean_mlp.make_model(),
            "unused.onnx",
            clean_mlp.make_inputs(),
            vary_fn=lambda i: clean_mlp.make_inputs(),
            k=1,
        )


def test_verify_wraps_a_session_run_failure_as_onnx_runtime_error(monkeypatch):
    class FailingSession:
        def get_inputs(self):
            return [_FakeInput()]

        def run(self, output_names, feeds):
            raise RuntimeError(
                "NOT_IMPLEMENTED : Could not find an implementation for Gemm(13)\nmore detail"
            )

    monkeypatch.setattr(verify_mod, "_to_session", lambda onnx_model: FailingSession())

    with pytest.raises(OnnxRuntimeError, match="Gemm"):
        verify(
            clean_mlp.make_model(),
            "unused.onnx",
            clean_mlp.make_inputs(),
            vary_fn=lambda i: clean_mlp.make_inputs(),
            k=1,
        )


def test_numerics_report_notes_default_to_empty_and_serialize():
    report = verify_mod.NumericsReport(
        samples_tested=8,
        max_abs_err=0.0,
        max_rel_err=0.0,
        failures=0,
        shape_generalization=True,
        tolerance_abs=1e-4,
        tolerance_rel=1e-3,
    )

    assert report.notes == []
    assert report.to_dict()["notes"] == []


def test_verify_wraps_a_torch_model_exception_as_value_error(monkeypatch):
    class BatchPicky(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if x.shape[0] != 1:
                raise RuntimeError("only batch 1 is supported")
            return x * 2

    monkeypatch.setattr(verify_mod, "_to_session", lambda onnx_model: _EchoSession())
    base_inputs = (torch.randn(1, 3),)

    def vary(i: int) -> tuple:
        return (torch.randn(2, 3),) if i == 1 else base_inputs

    with pytest.raises(ValueError, match=r"sample 1.*shape"):
        verify(BatchPicky(), "unused.onnx", base_inputs, vary_fn=vary, k=2)
