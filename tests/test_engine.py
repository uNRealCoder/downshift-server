"""prepare_serving(): verdict -> backend selection -> warmed-up state, and the backends'
shared contract."""

import numpy as np
import pytest
import torch

import downshift
from downshift.export.verdict import ExportVerdict, prepare_model
from downshift.loading import LoadedModel, load_model
from downshift.serve import backends as backends_mod
from downshift.serve.backends import OnnxRuntimeBackend, TorchBackend
from downshift.serve.engine import ServeOptions, choose_backend, prepare_serving
from tests.models import clean_mlp


def test_clean_model_serves_via_onnxruntime(mlp_state):
    assert mlp_state.verdict.status == "CLEAN"
    assert mlp_state.backend.name == "onnxruntime"
    assert mlp_state.input_names == ("x",)
    assert mlp_state.ready is True
    assert mlp_state.notes == []


def test_degraded_model_falls_back_to_torch(serve_fixture):
    state = serve_fixture("scatter_include_self_false")
    assert state.verdict.status == "DEGRADED"
    assert state.backend.name == "torch"
    assert state.ready is True


def test_force_onnx_serves_degraded_graph_with_a_warning(serve_fixture):
    state = serve_fixture("scatter_include_self_false", force_onnx=True)
    assert state.verdict.status == "DEGRADED"
    assert state.backend.name == "onnxruntime"
    assert any("--force-onnx" in note for note in state.notes)


def test_failed_export_falls_back_to_torch(branch_state):
    assert branch_state.verdict.status == "FAILED"
    assert branch_state.backend.name == "torch"
    assert branch_state.ready is True


def test_torch_backend_option_skips_export(serve_fixture):
    state = serve_fixture("clean_mlp", backend="torch")
    assert state.verdict.status == "UNVERIFIED"
    assert "--backend torch" in state.verdict.reason
    assert state.verdict.onnx_program is None
    assert state.backend.name == "torch"
    assert state.input_names == ("x",)


def test_backends_agree_on_the_same_model():
    loaded = load_model("tests.models.clean_mlp:make_model")
    same = LoadedModel(
        source=loaded.source, model=loaded.model, example_inputs=loaded.example_inputs
    )
    ort_state = prepare_serving(loaded, ServeOptions(warmup=1, backend="onnxruntime"))
    torch_state = prepare_serving(same, ServeOptions(warmup=1, backend="torch"))
    assert ort_state.backend.name == "onnxruntime"
    assert torch_state.backend.name == "torch"

    feeds = {"x": np.random.randn(4, 16).astype(np.float32)}
    ort_out = ort_state.backend.infer(feeds)
    torch_out = torch_state.backend.infer(feeds)

    assert set(ort_out) == set(torch_out) == {"output_0"}
    np.testing.assert_allclose(ort_out["output_0"], torch_out["output_0"], atol=1e-5)


def test_onnx_file_without_reference_is_served_unverified(exported_mlp):
    path, _, _ = exported_mlp
    state = prepare_serving(load_model(str(path)), ServeOptions(warmup=1))

    assert state.verdict.status == "UNVERIFIED"
    assert state.backend.name == "onnxruntime"
    assert state.input_names == ("x",)
    assert state.ready is True
    out = state.backend.infer({"x": np.random.randn(2, 16).astype(np.float32)})
    assert out["output_0"].shape == (2, 4)


def test_onnx_file_with_reference_is_verified(exported_mlp):
    path, model, _ = exported_mlp
    reference = LoadedModel(source="ref", model=model, example_inputs=clean_mlp.make_inputs())

    state = prepare_serving(load_model(str(path)), ServeOptions(warmup=1), reference=reference)

    assert state.verdict.status == "CLEAN", state.verdict.reason
    assert state.backend.name == "onnxruntime"
    assert state.ready is True


def test_torch_backend_rejects_missing_input():
    backend = TorchBackend(clean_mlp.make_model(), ("x",), device="cpu")
    with pytest.raises(KeyError, match="x"):
        backend.infer({"y": np.zeros((1, 16), dtype=np.float32)})


class _TupleOutputModel(torch.nn.Module):
    def forward(self, x):
        return x, x * 2


def test_torch_backend_splits_tuple_outputs_positionally():
    backend = TorchBackend(_TupleOutputModel(), ("x",), device="cpu")
    out = backend.infer({"x": np.ones((2, 3), dtype=np.float32)})
    assert set(out) == {"output_0", "output_1"}
    np.testing.assert_allclose(out["output_1"], out["output_0"] * 2)


def test_onnxruntime_backend_defaults_to_max_graph_optimization(exported_mlp):
    path, _, _ = exported_mlp
    backend = OnnxRuntimeBackend(path, device="cpu")
    opts = backend.session.get_session_options()
    assert opts.graph_optimization_level == backends_mod.ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    assert opts.intra_op_num_threads == 0
    assert opts.inter_op_num_threads == 0


def test_onnxruntime_backend_applies_explicit_thread_counts(exported_mlp):
    path, _, _ = exported_mlp
    backend = OnnxRuntimeBackend(path, device="cpu", intra_op_threads=2, inter_op_threads=1)
    opts = backend.session.get_session_options()
    assert opts.intra_op_num_threads == 2
    assert opts.inter_op_num_threads == 1


def test_serve_options_thread_counts_reach_the_ort_session(exported_mlp):
    path, _, _ = exported_mlp
    state = prepare_serving(
        load_model(str(path)),
        ServeOptions(warmup=1, intra_op_threads=4, inter_op_threads=2),
    )
    opts = state.backend.session.get_session_options()
    assert opts.intra_op_num_threads == 4
    assert opts.inter_op_num_threads == 2


def test_ort_providers_prefers_cuda_when_available(monkeypatch):
    monkeypatch.setattr(
        backends_mod.ort,
        "get_available_providers",
        lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    assert backends_mod._ort_providers("cuda") == ["CUDAExecutionProvider", "CPUExecutionProvider"]


def _bare_verdict(**overrides) -> ExportVerdict:
    fields = dict(
        status="CLEAN",
        model_family="generic-torch",
        capture_strategy=None,
        opset=None,
        op_types=[],
        numerics=None,
        recommended_backend="onnxruntime",
        reason="",
    )
    fields.update(overrides)
    return ExportVerdict(**fields)


def test_choose_backend_raises_when_neither_backend_is_available():
    verdict = _bare_verdict(prepared=None)
    with pytest.raises(ValueError, match="no PyTorch model to run"):
        choose_backend(verdict, ServeOptions())


def test_choose_backend_falls_back_to_torch_when_no_onnx_is_available():
    prepared = prepare_model(clean_mlp.make_model(), clean_mlp.make_inputs())
    verdict = _bare_verdict(prepared=prepared)

    name, notes = choose_backend(verdict, ServeOptions())

    assert name == "torch"
    assert any("falling back to torch" in n for n in notes)


def test_onnxruntime_backend_metadata(exported_mlp):
    path, _, _ = exported_mlp
    meta = OnnxRuntimeBackend(path, device="cpu").metadata().to_dict()

    assert meta["name"] == "onnxruntime"
    assert meta["device"] == "CPUExecutionProvider"
    assert meta["inputs"][0]["name"] == "x"
    assert [o["name"] for o in meta["outputs"]] == ["output_0"]
    assert meta["outputs"][0]["shape"] is not None


def test_check_leaves_global_rng_alone():
    """verify() seeds its own sampler; it must not reseed the caller's RNG, or every
    model built after a check() would come out with the same weights."""
    torch.manual_seed(1234)
    before = torch.get_rng_state()

    downshift.check(clean_mlp.make_model(), clean_mlp.make_inputs(), k=2)
    first = clean_mlp.make_model()
    downshift.check(clean_mlp.make_model(), clean_mlp.make_inputs(), k=2)
    second = clean_mlp.make_model()

    assert not torch.equal(first.net[0].weight, second.net[0].weight)
    assert not torch.equal(before, torch.get_rng_state())  # the RNG advanced, wasn't reset
