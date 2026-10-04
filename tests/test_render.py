"""Smoke tests for CLI rendering: every table/banner branch gets executed at least once.

Text content is deliberately not asserted in detail (see tests/test_cli.py's docstring);
these tests just exercise the branches with constructed verdicts and serving states.
"""

import logging

import pytest

from downshift.adapters.embedding import EmbeddingRecipe
from downshift.adapters.text import TextIO
from downshift.cli import render
from downshift.core.verdict import ExportVerdict
from downshift.core.verify import NumericsReport, WorstMismatch
from downshift.serve.backends import BackendMeta
from downshift.serve.engine import ServeOptions, ServingState, WarmupStats
from downshift.serve.options import ExecutionChoice
from downshift.serve.schemas import OutputEncoding


@pytest.fixture(autouse=True)
def _capture_reports(caplog):
    """Reports go out as INFO on downshift.report and warnings/errors on downshift.cli, which
    only reach a handler once setup_logging has run; caplog stands in for that sink."""
    caplog.set_level(logging.INFO, logger="downshift.report")
    caplog.set_level(logging.DEBUG, logger="downshift.cli")


class _StubBackend:
    def __init__(
        self,
        name: str = "onnxruntime",
        device: str = "cpu",
        verified_provider: str | None = None,
    ) -> None:
        self.name = name
        self._device = device
        self.verified_provider = verified_provider

    def metadata(self) -> BackendMeta:
        return BackendMeta(self.name, self._device, [], [])

    def infer(self, inputs):
        raise NotImplementedError


def _numerics(
    passed: bool,
    shape_generalization: bool | None = True,
    baseline_failed: bool = False,
    **overrides,
) -> NumericsReport:
    fields = dict(
        samples_tested=8,
        max_abs_err=0.0 if passed else 1.0,
        max_rel_err=0.0 if passed else 1.0,
        failures=0 if passed else 3,
        shape_generalization=shape_generalization,
        baseline_failed=baseline_failed,
        tolerance_abs=1e-4,
        tolerance_rel=1e-3,
    )
    fields.update(overrides)
    return NumericsReport(**fields)


def _verdict(**overrides) -> ExportVerdict:
    fields = dict(
        status="CLEAN",
        model_family="generic",
        capture_strategy="strict=False",
        opset=18,
        op_types={"Gemm": 2, "Relu": 1},
        numerics=_numerics(True),
        recommended_backend="onnxruntime",
        reason="exported via strict=False; numerics ok",
    )
    fields.update(overrides)
    return ExportVerdict(**fields)


def test_print_verdict_clean_with_dynamic_dims():
    verdict = _verdict(dynamic_dims={"x": [0]})
    render.print_verdict(verdict, "model")


def test_print_verdict_shows_the_tolerance_row(caplog):
    render.print_verdict(_verdict(), "model")
    out = caplog.text
    assert "Tolerance" in out
    assert "float32" in out


def test_print_verdict_omits_the_tolerance_row_without_numerics():
    verdict = _verdict(status="UNVERIFIED", numerics=None, recommended_backend="onnxruntime")
    render.print_verdict(verdict, "model")


def test_print_verdict_unverified_has_no_numerics():
    verdict = _verdict(status="UNVERIFIED", numerics=None, recommended_backend="onnxruntime")
    render.print_verdict(verdict, "model")


def test_print_verdict_degraded_numerics_failed():
    verdict = _verdict(
        status="DEGRADED",
        numerics=_numerics(False, shape_generalization=False),
        recommended_backend="torch",
    )
    render.print_verdict(verdict, "model")


def test_print_verdict_failed_shows_unsupported_ops_and_warnings():
    verdict = _verdict(
        status="FAILED",
        numerics=None,
        recommended_backend="torch",
        unsupported_ops=["scatter_reduce"],
        warnings=["tied weights: b shares storage with a"],
    )
    render.print_verdict(verdict, "model")


def test_print_verdict_failed_reason_points_at_debug_logging(caplog):
    verdict = _verdict(status="FAILED", numerics=None, recommended_backend="torch", reason="nope")
    render.print_verdict(verdict, "model")
    out = caplog.text
    assert "--log-level debug" in out


def test_print_verdict_shows_overridden_tolerance(caplog):
    verdict = _verdict(numerics=_numerics(True, tolerance_overridden=True))
    render.print_verdict(verdict, "model")
    out = caplog.text
    assert "--atol/--rtol" in out
    assert "float32" not in out


def test_print_verdict_shows_shape_general_na_when_baseline_fails(caplog):
    verdict = _verdict(
        status="DEGRADED",
        numerics=_numerics(False, shape_generalization=None, baseline_failed=True),
        recommended_backend="torch",
    )
    render.print_verdict(verdict, "model")
    out = caplog.text
    assert "n/a (baseline fails)" in out


def test_print_verdict_shows_worst_and_samples_rows(caplog):
    worst = WorstMismatch(
        sample=3, output=0, index=(2, 5), expected=0.4113, got=-0.8327, input_shapes=[(12, 8)]
    )
    verdict = _verdict(
        status="DEGRADED",
        numerics=_numerics(
            False,
            shape_generalization=False,
            worst=worst,
            sample_shapes=[[(1, 8)], [(2, 8)], [(12, 8)]],
        ),
        recommended_backend="torch",
        input_names=("x",),
    )
    render.print_verdict(verdict, "model")
    out = caplog.text
    assert "Worst" in out
    assert "output_0[2, 5]" in out
    assert "(sample" in out
    assert "Samples" in out
    assert "(1,8)" in out and "(12,8)" in out


def test_print_verdict_omits_worst_row_when_clean():
    verdict = _verdict(numerics=_numerics(True, sample_shapes=[[(1, 16)], [(2, 16)]]))
    render.print_verdict(verdict, "model")


def test_print_artifacts_nothing_written(caplog):
    render.print_artifacts(None, None)
    assert "nothing written" in caplog.text


def test_print_artifacts_writes_onnx_and_manifest(tmp_path, caplog):
    onnx_path = tmp_path / "m.onnx"
    manifest_path = tmp_path / "m.manifest.json"
    render.print_artifacts(onnx_path, manifest_path)
    out = caplog.text
    assert "Wrote" in out
    assert "Manifest" in out


def test_print_artifacts_writes_onnx_without_manifest(tmp_path, caplog):
    render.print_artifacts(tmp_path / "m.onnx", None)
    out = caplog.text
    assert "Wrote" in out
    assert "Manifest" not in out


def _serving_state(verdict: ExportVerdict, **overrides) -> ServingState:
    fields = dict(
        source="tests.models.clean_mlp:make_model",
        verdict=verdict,
        backend=_StubBackend(),
        input_names=("x",),
        options=ServeOptions(),
        notes=[],
    )
    fields.update(overrides)
    return ServingState(**fields)


def test_print_banner_unverified_without_reference():
    verdict = _verdict(status="UNVERIFIED", numerics=None, recommended_backend="onnxruntime")
    state = _serving_state(verdict)
    render.print_banner(state, "127.0.0.1", 8000)


def test_print_banner_shows_output_encoding(caplog):
    state = _serving_state(_verdict(), options=ServeOptions(output_encoding=OutputEncoding.base64))
    render.print_banner(state, "127.0.0.1", 8000)
    out = caplog.text
    assert "Encoding" in out
    assert "base64" in out
    assert "output_encoding" in out
    assert "[fast]" not in out  # pybase64 is installed in the dev environment


def test_print_banner_shows_concurrency(caplog):
    state = _serving_state(_verdict(), options=ServeOptions(max_concurrency=4))
    render.print_banner(state, "127.0.0.1", 8000)
    out = caplog.text
    assert "Capacity" in out
    assert "4 inferences at a time" in out
    assert "--max-concurrency" in out


def test_print_banner_shows_threads_row_for_multiple_workers(caplog, monkeypatch):
    monkeypatch.setattr(render.settings, "usable_cpus", lambda: 16)
    state = _serving_state(_verdict(), options=ServeOptions(intra_op_threads=4))
    render.print_banner(state, "127.0.0.1", 8000, workers=4)
    out = caplog.text
    assert "Threads" in out
    assert "4 intra-op per worker" in out
    assert "16 logical / 4 workers" in out


def test_print_banner_omits_threads_row_for_a_single_worker(caplog):
    render.print_banner(_serving_state(_verdict()), "127.0.0.1", 8000)
    assert "Threads" not in caplog.text


def test_print_banner_tips_pybase64_when_missing(caplog, monkeypatch):
    monkeypatch.setattr(render, "BASE64_CODEC", "stdlib")
    render.print_banner(_serving_state(_verdict()), "127.0.0.1", 8000)
    assert "downshift-server[fast]" in caplog.text


def test_print_banner_shows_tolerance_row(caplog):
    render.print_banner(_serving_state(_verdict()), "127.0.0.1", 8000)
    out = caplog.text
    assert "Tolerance" in out
    assert "float32" in out


def test_print_banner_shows_worst_and_samples_rows_when_degraded(caplog):
    worst = WorstMismatch(
        sample=3, output=0, index=(2, 5), expected=0.4113, got=-0.8327, input_shapes=[(12, 8)]
    )
    verdict = _verdict(
        status="DEGRADED",
        numerics=_numerics(
            False,
            shape_generalization=False,
            worst=worst,
            sample_shapes=[[(1, 16)], [(2, 16)]],
        ),
        recommended_backend="torch",
        input_names=("x",),
    )
    render.print_banner(_serving_state(verdict, backend=_StubBackend("torch")), "127.0.0.1", 8000)
    out = caplog.text
    assert "Worst" in out
    assert "output_0[2, 5]" in out
    assert "Samples" in out
    assert "(1,16)" in out and "(2,16)" in out


def test_print_banner_omits_worst_row_when_clean(caplog):
    render.print_banner(_serving_state(_verdict()), "127.0.0.1", 8000)
    assert "Worst" not in caplog.text


def test_print_banner_shows_embedding_and_text_rows(caplog):
    state = _serving_state(_verdict())
    state.embedding = EmbeddingRecipe("mean", True, 256, "modules.json")
    state.text = TextIO(tokenizer=None, max_length=256)
    render.print_banner(state, "127.0.0.1", 8000)
    out = caplog.text
    assert "Embedding" in out
    assert "mean pooling, L2-normalised" in out
    assert "from modules.json" in out
    assert "up to 256 tokens a row" in out


def test_print_banner_omits_embedding_and_text_rows_for_other_models(caplog):
    render.print_banner(_serving_state(_verdict()), "127.0.0.1", 8000)
    out = caplog.text
    assert "Embedding" not in out
    assert "Text input" not in out


def test_print_banner_shows_warmup_row(caplog):
    state = _serving_state(
        _verdict(), warmup_stats=WarmupStats(count=3, mean_ms=0.02, synthesized=False)
    )
    render.print_banner(state, "127.0.0.1", 8000)
    out = caplog.text
    assert "Warmup" in out
    assert "3 inferences, 0.02 ms each" in out


def test_print_banner_shows_warmup_row_when_synthesized(caplog):
    state = _serving_state(
        _verdict(), warmup_stats=WarmupStats(count=3, mean_ms=0.0, synthesized=True)
    )
    render.print_banner(state, "127.0.0.1", 8000)
    out = caplog.text
    assert "3 inferences on synthesized inputs" in out


def test_print_banner_omits_warmup_row_without_warmup_stats(caplog):
    render.print_banner(_serving_state(_verdict()), "127.0.0.1", 8000)
    assert "Warmup" not in caplog.text


def test_print_banner_shows_queue_and_timeout_in_the_capacity_row(caplog):
    state = _serving_state(_verdict(), options=ServeOptions(max_queue=64, request_timeout=0))
    render.print_banner(state, "127.0.0.1", 8000)
    out = caplog.text
    assert "Capacity" in out
    assert "64 queued, no timeout" in out
    assert "--request-timeout" in out


def test_print_banner_shows_queue_and_timeout_in_the_capacity_row_with_a_timeout(caplog):
    state = _serving_state(_verdict(), options=ServeOptions(request_timeout=5))
    render.print_banner(state, "127.0.0.1", 8000)
    assert "5 s timeout" in caplog.text


def test_print_banner_shows_boot_row(caplog):
    state = _serving_state(_verdict())
    state.timings = {"load": 0.3, "export": 2.8, "verify": 0.2, "session": 0.1, "warmup": 0.1}
    render.print_banner(state, "127.0.0.1", 8000)
    out = caplog.text
    assert "Boot" in out
    assert "load 0.3" in out
    assert "export 2.8" in out
    assert "verify 0.2" in out
    assert "session 0.1" in out
    assert "warmup 0.1" in out


def test_print_banner_omits_boot_row_without_timings(caplog):
    render.print_banner(_serving_state(_verdict()), "127.0.0.1", 8000)
    assert "Boot" not in caplog.text


def test_print_banner_unverified_with_prepared_backend_torch_skip():
    verdict = _verdict(
        status="UNVERIFIED",
        numerics=None,
        recommended_backend="torch",
        reason="--backend torch: export skipped",
        prepared=object(),
    )
    state = _serving_state(
        verdict, backend=_StubBackend("torch"), options=ServeOptions(backend="torch")
    )
    render.print_banner(state, "127.0.0.1", 8000)


def test_print_banner_failed_shows_reason():
    verdict = _verdict(
        status="FAILED", numerics=None, recommended_backend="torch", prepared=object()
    )
    state = _serving_state(verdict, backend=_StubBackend("torch"))
    render.print_banner(state, "127.0.0.1", 8000)


def test_print_banner_degraded_without_numerics_uses_reason():
    verdict = _verdict(
        status="DEGRADED",
        numerics=None,
        recommended_backend="torch",
        reason="exported but numerics diverge",
    )
    state = _serving_state(verdict, backend=_StubBackend("torch"))
    render.print_banner(state, "127.0.0.1", 8000)


def test_print_banner_degraded_forced_onnx_notes_and_warnings():
    verdict = _verdict(
        status="DEGRADED",
        numerics=_numerics(False, shape_generalization=False),
        recommended_backend="torch",
        dynamic_dims={"x": [0]},
        warnings=["model was in training mode; switched to eval() for export"],
    )
    options = ServeOptions(force_onnx=True)
    state = _serving_state(
        verdict,
        backend=_StubBackend("onnxruntime"),
        options=options,
        notes=[
            "--force-onnx: serving a DEGRADED graph; outputs may be wrong",
            "auto-selected backend note",
        ],
    )
    render.print_banner(state, "127.0.0.1", 8000)


def test_warn_and_error(caplog):
    render.warn("careful")
    render.error("bad thing")
    err = caplog.text
    assert "careful" in err
    assert "bad thing" in err


def test_print_traceback(caplog):
    try:
        raise ValueError("boom")
    except ValueError:
        render.print_traceback()
    assert "boom" in caplog.text


def test_print_banner_shows_the_provider_the_numerics_ran_on(caplog):
    state = _serving_state(
        _verdict(), backend=_StubBackend(verified_provider="CPUExecutionProvider")
    )
    render.print_banner(state, "127.0.0.1", 8000)
    assert "Verified on" in caplog.text
    assert "CPUExecutionProvider" in caplog.text
    assert "only verified on the CPU" not in caplog.text


def test_print_banner_says_numerics_did_not_cover_a_cuda_server(caplog):
    backend = _StubBackend(device="cuda:0", verified_provider="CPUExecutionProvider")
    render.print_banner(_serving_state(_verdict(), backend=backend), "127.0.0.1", 8000)
    assert "serving on cuda:0; numerics were only verified on the CPU" in caplog.text


def test_print_banner_omits_verified_on_when_numerics_never_ran(caplog):
    render.print_banner(_serving_state(_verdict()), "127.0.0.1", 8000)
    assert "Verified on" not in caplog.text


def test_print_banner_endpoint_names_the_client_url_for_a_wildcard_host(caplog):
    render.print_banner(_serving_state(_verdict()), "0.0.0.0", 8000)
    assert "http://localhost:8000  (bound to 0.0.0.0;" in caplog.text


def test_print_banner_endpoint_is_plain_for_a_specific_host(caplog):
    render.print_banner(_serving_state(_verdict()), "127.0.0.1", 8000)
    assert "http://127.0.0.1:8000  (GET /schema" in caplog.text
    assert "bound to" not in caplog.text


def test_print_booting_says_where_it_will_listen(caplog):
    render.print_booting("some.model:make", "127.0.0.1", 9000)
    assert "loading some.model:make" in caplog.text
    assert "will listen on http://127.0.0.1:9000 (not ready yet)" in caplog.text


def test_print_ready_reports_the_total_boot_time(caplog):
    state = _serving_state(_verdict())
    state.timings = {"load": 0.5, "export": 1.0}
    render.print_ready(state)
    assert "ready in 1.5 s" in caplog.text


def test_print_ready_without_timings(caplog):
    render.print_ready(_serving_state(_verdict()))
    assert "ready" in caplog.text
    assert " in " not in caplog.text.split("ready", 1)[1]


def _fact(name: str, served_max: int, sampled_max: int | None, input: str = "x"):
    from downshift.core.axes import AxisFact

    return AxisFact(input, 0, name, 1, served_max, None if sampled_max is None else 2, sampled_max)


def test_axes_text_warns_on_a_wide_unverified_axis(caplog):
    render.print_verdict(_verdict(axes=[_fact("num_nodes", 65536, 23)]), "model")

    assert "`num_nodes` (x[0])  sampled 2-23, serves 1-65536" in caplog.text
    assert "unverified above 23" in caplog.text


def test_axes_text_never_warns_on_the_batch_axis(caplog):
    render.print_verdict(_verdict(axes=[_fact("batch", 4096, 8, "input_ids")]), "model")

    assert "`batch` (input_ids[0])  sampled 2-8, serves 1-4096" in caplog.text
    assert "unverified" not in caplog.text


def test_axes_text_says_so_when_nothing_was_sampled(caplog):
    render.print_verdict(_verdict(axes=[_fact("seq", 512, None)]), "model")

    assert "not verified, serves 1-512" in caplog.text
    assert "unverified above" not in caplog.text


def test_axes_text_falls_back_to_the_dynamic_dims(caplog):
    render.print_verdict(_verdict(dynamic_dims={"x": [0]}), "model")

    assert "x[0]" in caplog.text


def test_print_banner_shows_prep_threads_in_the_capacity_row(caplog):
    state = _serving_state(_verdict(), options=ServeOptions(prep_threads=3))
    render.print_banner(state, "127.0.0.1", 8000)
    assert "3 prep threads" in caplog.text


def test_print_banner_execution_row_threadpool(caplog):
    render.print_banner(_serving_state(_verdict()), "127.0.0.1", 8000)
    assert "Execution" in caplog.text
    assert "threadpool" in caplog.text
    assert "event loop" not in caplog.text


def test_print_banner_inline_execution_warns(caplog):
    state = _serving_state(_verdict(), options=ServeOptions(execution=ExecutionChoice.inline))
    render.print_banner(state, "127.0.0.1", 8000)
    out = caplog.text
    assert "inline for small JSON bodies" in out
    assert "under ~1 ms" in out
    assert "/health and /ready" in out
