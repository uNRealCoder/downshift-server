"""prepare_serving(): verdict -> backend selection -> warmed-up state, and the backends'
shared contract."""

import dataclasses
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
import torch

import downshift
from downshift.core.verdict import ExportVerdict, prepare_model
from downshift.loading import LoadedModel, LoadSpec, load_model
from downshift.serve import backends as backends_mod
from downshift.serve.backends import InferenceInputError, OnnxRuntimeBackend, TorchBackend
from downshift.serve.engine import (
    ServeOptions,
    WarmupStats,
    choose_backend,
    prepare_serving,
    synthesize_feeds,
)
from tests.models import clean_mlp


def test_clean_model_serves_via_onnxruntime(mlp_state):
    assert mlp_state.verdict.status == "CLEAN"
    assert mlp_state.backend.name == "onnxruntime"
    assert mlp_state.input_names == ("x",)
    assert mlp_state.ready is True
    assert mlp_state.notes == []


def test_serving_state_builds_an_executor_from_max_concurrency(mlp_state):
    assert isinstance(mlp_state.executor, ThreadPoolExecutor)
    assert mlp_state.executor._max_workers == 1
    assert mlp_state.in_flight == 0

    wider = dataclasses.replace(
        mlp_state, options=dataclasses.replace(mlp_state.options, max_concurrency=3)
    )
    assert wider.executor._max_workers == 3


def test_serving_state_builds_a_prep_pool_from_prep_threads(mlp_state):
    narrow = dataclasses.replace(
        mlp_state, options=dataclasses.replace(mlp_state.options, prep_threads=2)
    )
    assert isinstance(narrow.prep_executor, ThreadPoolExecutor)
    assert narrow.prep_executor._max_workers == 2
    assert narrow.prep_executor._thread_name_prefix == "downshift-prep"


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
    loaded = load_model(LoadSpec("tests.models.clean_mlp:make_model"))
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
    state = prepare_serving(load_model(LoadSpec(str(path))), ServeOptions(warmup=1))

    assert state.verdict.status == "UNVERIFIED"
    assert state.backend.name == "onnxruntime"
    assert state.input_names == ("x",)
    assert state.ready is True
    out = state.backend.infer({"x": np.random.randn(2, 16).astype(np.float32)})
    assert out["output_0"].shape == (2, 4)


def test_synthesize_feeds_fills_dynamic_axes_with_one(exported_mlp):
    path, _, _ = exported_mlp
    backend = OnnxRuntimeBackend(path, device="cpu")

    feeds = synthesize_feeds(backend)

    assert set(feeds) == {"x"}
    assert feeds["x"].shape == (1, 16)
    assert feeds["x"].dtype == np.float32


def test_bare_onnx_warmup_runs_on_synthesized_inputs(exported_mlp, monkeypatch):
    path, _, _ = exported_mlp
    calls: list[dict] = []
    real_infer = OnnxRuntimeBackend.infer

    def counting_infer(self, inputs):
        calls.append(inputs)
        return real_infer(self, inputs)

    monkeypatch.setattr(OnnxRuntimeBackend, "infer", counting_infer)

    state = prepare_serving(load_model(LoadSpec(str(path))), ServeOptions(warmup=3))

    assert len(calls) == 3
    assert calls[0]["x"].shape == (1, 16)
    assert state.warmup_stats == WarmupStats(
        count=3, mean_ms=state.warmup_stats.mean_ms, synthesized=True
    )


def test_warmup_stats_are_not_synthesized_when_example_inputs_exist(mlp_state):
    assert mlp_state.warmup_stats is not None
    assert mlp_state.warmup_stats.synthesized is False
    assert mlp_state.warmup_stats.count == 1


def test_prepare_serving_records_phase_timings(mlp_state):
    assert set(mlp_state.timings) >= {"export", "verify", "session", "warmup"}
    assert all(v >= 0 for v in mlp_state.timings.values())


def test_prepare_serving_records_only_verify_for_a_bare_onnx_with_reference(exported_mlp):
    path, model, _ = exported_mlp
    reference = LoadedModel(source="ref", model=model, example_inputs=clean_mlp.make_inputs())
    state = prepare_serving(
        load_model(LoadSpec(str(path))), ServeOptions(warmup=1), reference=reference
    )
    assert "export" not in state.timings
    assert set(state.timings) >= {"verify", "session", "warmup"}


def test_onnx_file_with_reference_is_verified(exported_mlp):
    path, model, _ = exported_mlp
    reference = LoadedModel(source="ref", model=model, example_inputs=clean_mlp.make_inputs())

    state = prepare_serving(
        load_model(LoadSpec(str(path))), ServeOptions(warmup=1), reference=reference
    )

    assert state.verdict.status == "CLEAN", state.verdict.reason
    assert state.backend.name == "onnxruntime"
    assert state.ready is True


def test_torch_backend_reports_only_the_adapters_dynamic_axes():
    """A graph model's edge_index is [2, E]: only E is dynamic. Reporting axis 0 as "batch"
    made /schema's example [1, E], which the model then rejected."""

    class TakesGraph(torch.nn.Module):
        def forward(self, x, edge_index):
            return x[edge_index[1]]

    x, edge_index = torch.zeros(4, 3), torch.zeros(2, 5, dtype=torch.long)
    nodes, edges = torch.export.Dim("num_nodes"), torch.export.Dim("num_edges")
    backend = TorchBackend(
        TakesGraph(),
        ("x", "edge_index"),
        device="cpu",
        example_inputs=(x, edge_index),
        dynamic_shapes=({0: nodes}, {1: edges}),
    )

    shapes = {spec.name: spec.shape for spec in backend.metadata().inputs}
    assert shapes == {"x": ["num_nodes", 3], "edge_index": [2, "num_edges"]}


def test_torch_backend_rejects_missing_input():
    backend = TorchBackend(clean_mlp.make_model(), ("x",), device="cpu")
    with pytest.raises(InferenceInputError, match="missing inputs"):
        backend.infer({"y": np.zeros((1, 16), dtype=np.float32)})


def test_torch_backend_wraps_shape_error_in_inference_input_error():
    backend = TorchBackend(clean_mlp.make_model(), ("x",), device="cpu")
    with pytest.raises(InferenceInputError, match="cannot be multiplied"):
        backend.infer({"x": np.zeros((1, 5), dtype=np.float32)})


def test_torch_backend_reraises_out_of_memory_untouched(monkeypatch):
    backend = TorchBackend(clean_mlp.make_model(), ("x",), device="cpu")

    def boom(*args):
        raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")

    monkeypatch.setattr(backend.module, "forward", boom)
    with pytest.raises(RuntimeError, match="out of memory"):
        backend.infer({"x": np.zeros((1, 16), dtype=np.float32)})


def test_onnxruntime_backend_wraps_shape_mismatch_in_inference_input_error(exported_mlp):
    path, _, _ = exported_mlp
    backend = OnnxRuntimeBackend(path, device="cpu")
    with pytest.raises(InferenceInputError) as excinfo:
        backend.infer({"x": np.zeros((1, 5), dtype=np.float32)})
    # First line of ORT's (often multi-line) message only.
    assert "\n" not in str(excinfo.value)
    assert "INVALID_ARGUMENT" in str(excinfo.value)


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
        load_model(LoadSpec(str(path))),
        ServeOptions(warmup=1, intra_op_threads=4, inter_op_threads=2),
    )
    opts = state.backend.session.get_session_options()
    assert opts.intra_op_num_threads == 4
    assert opts.inter_op_num_threads == 2


def test_torch_backend_sets_num_threads_when_given_a_budget(monkeypatch):
    calls: list[int] = []
    monkeypatch.setattr(torch, "set_num_threads", calls.append)
    TorchBackend(clean_mlp.make_model(), ("x",), device="cpu", intra_op_threads=4)
    assert calls == [4]


def test_torch_backend_leaves_thread_count_alone_by_default(monkeypatch):
    calls: list[int] = []
    monkeypatch.setattr(torch, "set_num_threads", calls.append)
    TorchBackend(clean_mlp.make_model(), ("x",), device="cpu")
    assert calls == []


def test_ort_providers_prefers_cuda_when_available(monkeypatch):
    monkeypatch.setattr(
        backends_mod.ort,
        "get_available_providers",
        lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    assert backends_mod._ort_providers("cuda") == ["CUDAExecutionProvider", "CPUExecutionProvider"]


class _FakeSession:
    """Stands in for an ort.InferenceSession that ended up on the given providers."""

    def __init__(self, providers: list[str]) -> None:
        self._providers = providers

    def get_providers(self) -> list[str]:
        return self._providers

    def get_inputs(self) -> list:
        return []

    def get_outputs(self) -> list:
        return []


def _fake_gpu_wheel(monkeypatch, session_providers: list[str]) -> None:
    """onnxruntime-gpu installed (CUDA listed as available), but the session it hands back runs
    on `session_providers`: the CPU alone when the CUDA/cuDNN libraries would not load."""
    monkeypatch.setattr(
        backends_mod.ort,
        "get_available_providers",
        lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    monkeypatch.setattr(
        backends_mod.ort, "InferenceSession", lambda *a, **k: _FakeSession(session_providers)
    )


def test_ort_backend_refuses_cuda_when_the_session_fell_back_to_the_cpu(monkeypatch):
    _fake_gpu_wheel(monkeypatch, ["CPUExecutionProvider"])

    with pytest.raises(ValueError, match=r"--device cuda: onnxruntime could not start CUDA.*CPU"):
        OnnxRuntimeBackend(b"model", device="cuda")


def test_ort_backend_accepts_cuda_when_the_session_runs_on_it(monkeypatch):
    _fake_gpu_wheel(monkeypatch, ["CUDAExecutionProvider", "CPUExecutionProvider"])

    backend = OnnxRuntimeBackend(b"model", device="cuda")

    assert backend.provider == "CUDAExecutionProvider"


def test_ort_backend_does_not_check_for_cuda_on_a_cpu_server(monkeypatch):
    _fake_gpu_wheel(monkeypatch, ["CPUExecutionProvider"])

    assert OnnxRuntimeBackend(b"model", device="cpu").provider == "CPUExecutionProvider"


def _bare_verdict(**overrides) -> ExportVerdict:
    fields = dict(
        status="CLEAN",
        model_family="generic-torch",
        capture_strategy=None,
        opset=None,
        op_types={},
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


def test_choose_backend_onnxruntime_on_degraded_is_an_error():
    verdict = _bare_verdict(status="DEGRADED")
    with pytest.raises(ValueError, match="--force-onnx"):
        choose_backend(verdict, ServeOptions(backend="onnxruntime"))


def test_choose_backend_onnxruntime_and_force_onnx_on_degraded_is_accepted():
    verdict = _bare_verdict(status="DEGRADED", onnx_path=Path("fake.onnx"))
    name, notes = choose_backend(verdict, ServeOptions(backend="onnxruntime", force_onnx=True))
    assert name == "onnxruntime"
    assert any("--force-onnx" in n for n in notes)


def test_onnxruntime_backend_metadata(exported_mlp):
    path, _, _ = exported_mlp
    meta = OnnxRuntimeBackend(path, device="cpu").metadata().to_dict()

    assert meta["name"] == "onnxruntime"
    assert meta["device"] == "CPUExecutionProvider"
    assert meta["inputs"][0]["name"] == "x"
    assert [o["name"] for o in meta["outputs"]] == ["output_0"]
    assert meta["outputs"][0]["shape"] is not None


def _count_sessions(monkeypatch) -> list[int]:
    """Subclass the real InferenceSession so every construction (not just the count of
    calls to the constructor function) is caught, then swap it in on the onnxruntime module
    itself: both verify.py and backends.py look up `ort.InferenceSession` by attribute at
    call time, so patching the module's attribute reaches both."""
    import onnxruntime

    calls: list[int] = []
    real_session = onnxruntime.InferenceSession

    class CountingSession(real_session):
        def __init__(self, *args, **kwargs):
            calls.append(1)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(onnxruntime, "InferenceSession", CountingSession)
    return calls


def test_clean_model_serves_from_exactly_one_inference_session(monkeypatch):
    calls = _count_sessions(monkeypatch)

    loaded = load_model(LoadSpec("tests.models.clean_mlp:make_model"))
    state = prepare_serving(loaded, ServeOptions(warmup=1))

    assert state.verdict.status == "CLEAN"
    assert len(calls) == 1


def test_explicit_intra_op_threads_builds_a_second_session(monkeypatch):
    calls = _count_sessions(monkeypatch)

    loaded = load_model(LoadSpec("tests.models.clean_mlp:make_model"))
    state = prepare_serving(loaded, ServeOptions(warmup=1, intra_op_threads=2))

    assert state.verdict.status == "CLEAN"
    assert len(calls) == 2


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


def test_verify_logs_one_progress_line_per_sample_with_every_input_shape(caplog):
    import logging

    from downshift.logs import REPORT_LOGGER

    with caplog.at_level(logging.INFO, logger=REPORT_LOGGER):
        verdict = downshift.check(clean_mlp.make_model(), clean_mlp.make_inputs(), k=3)

    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("verify:")]
    assert verdict.numerics is not None
    assert len(lines) == 3
    for i, line in enumerate(lines, start=1):
        assert line.startswith(f"verify: sample {i}/3, x [")
