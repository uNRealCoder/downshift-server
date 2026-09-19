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
    default_tolerances,
    make_shared_axis0_vary_fn,
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


def test_verify_names_the_sampler_and_bounds_when_a_generated_sample_fails(monkeypatch):
    class BatchPicky(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if x.shape[0] != 1:
                raise RuntimeError("only batch 1 is supported")
            return x * 2

    monkeypatch.setattr(verify_mod, "_to_session", lambda onnx_model: _EchoSession())
    base_inputs = (torch.randn(1, 3),)
    dynamic_shapes = ({0: torch.export.Dim("n", min=1, max=8)},)

    def vary(i: int) -> tuple:
        return (torch.randn(2, 3),) if i == 1 else base_inputs

    with pytest.raises(ValueError, match="generated by downshift's sampler") as excinfo:
        verify(BatchPicky(), "unused.onnx", base_inputs, dynamic_shapes, vary_fn=vary, k=2)
    message = str(excinfo.value)
    assert "--vary" in message
    assert "axis 0 in (1, 8)" in message


def test_verify_keeps_the_original_wording_when_sample_0_fails(monkeypatch):
    class AlwaysBroken(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            raise RuntimeError("nope")

    monkeypatch.setattr(verify_mod, "_to_session", lambda onnx_model: _EchoSession())
    inputs = (torch.randn(1, 3),)

    with pytest.raises(ValueError, match="Try --dynamic or --inputs") as excinfo:
        verify(AlwaysBroken(), "unused.onnx", inputs, vary_fn=lambda i: inputs, k=1)
    assert "downshift's sampler" not in str(excinfo.value)


@pytest.mark.parametrize(
    ("cast", "expected_name", "expected"),
    [
        (lambda m: m, "float32", (1e-4, 1e-3)),
        (lambda m: m.double(), "float64", (1e-6, 1e-5)),
        (lambda m: m.half(), "float16", (1e-2, 1e-2)),
        (lambda m: m.bfloat16(), "bfloat16", (5e-2, 5e-2)),
    ],
    ids=["float32", "float64", "float16", "bfloat16"],
)
def test_default_tolerances_by_dtype(cast, expected_name, expected):
    model = cast(clean_mlp.make_model())
    name, atol, rtol = default_tolerances(model)
    assert (name, (atol, rtol)) == (expected_name, expected)


def test_default_tolerances_prefers_float32_over_float64_when_mixed():
    """float64 only wins outright when it's the sole floating dtype; mixed precision is
    only as good as its least precise dtype."""
    model = clean_mlp.make_model()
    model.net[0].weight.data = model.net[0].weight.data.double()
    name, atol, rtol = default_tolerances(model)
    assert (name, atol, rtol) == ("float32", 1e-4, 1e-3)


def test_verify_records_tolerance_dtype_and_seed_on_the_report(monkeypatch):
    monkeypatch.setattr(verify_mod, "_to_session", lambda onnx_model: _EchoSession())
    model = clean_mlp.make_model().double()
    inputs = tuple(
        t.double() if isinstance(t, torch.Tensor) else t for t in clean_mlp.make_inputs()
    )

    report = verify(model, "unused.onnx", inputs, vary_fn=lambda i: inputs, k=1, seed=3)

    assert report.tolerance_dtype == "float64"
    assert (report.tolerance_abs, report.tolerance_rel) == (1e-6, 1e-5)
    assert report.seed == 3


def test_atol_rtol_override_the_dtype_default(monkeypatch):
    monkeypatch.setattr(verify_mod, "_to_session", lambda onnx_model: _EchoSession())
    model = clean_mlp.make_model()
    inputs = clean_mlp.make_inputs()

    report = verify(model, "unused.onnx", inputs, vary_fn=lambda i: inputs, k=1, atol=1.0, rtol=2.0)

    assert (report.tolerance_abs, report.tolerance_rel) == (1.0, 2.0)
    assert report.tolerance_dtype == "float32"  # still records what the dtype would have picked


def test_seed_makes_samples_reproducible(monkeypatch):
    monkeypatch.setattr(verify_mod, "_to_session", lambda onnx_model: _EchoSession())
    model = clean_mlp.make_model()
    inputs = clean_mlp.make_inputs()
    dynamic_shapes = ({0: torch.export.Dim("n", min=1, max=64)},)

    r1 = verify(model, "unused.onnx", inputs, dynamic_shapes, k=4, seed=7)
    r2 = verify(model, "unused.onnx", inputs, dynamic_shapes, k=4, seed=7)

    assert r1.to_dict() == r2.to_dict()
    assert r1.seed == 7


def test_shared_axis0_vary_fn_never_exceeds_the_dims_max():
    base_inputs = (torch.randn(2, 4),)
    dynamic_shapes = ({0: torch.export.Dim("n", min=1, max=3)},)
    vary = make_shared_axis0_vary_fn(base_inputs, dynamic_shapes)

    for i in range(1, 20):
        (sample,) = vary(i)
        assert 1 <= sample.shape[0] <= 3


def test_resize_dim0_tiles_the_example_exactly_when_it_has_no_spread():
    base = torch.full((2, 4), 1000.0)
    resized = _resize_dim0(base, 5)
    assert resized.shape == (5, 4)
    assert torch.equal(resized, torch.full((5, 4), 1000.0))


def test_resize_dim0_stays_near_the_example_rather_than_pure_noise():
    torch.manual_seed(0)
    base = torch.tensor([[0.0], [10.0]])
    resized = _resize_dim0(base, 4)
    tiled = torch.tensor([[0.0], [10.0], [0.0], [10.0]])
    assert resized.shape == (4, 1)
    assert torch.all((resized - tiled).abs() < 5.0)
