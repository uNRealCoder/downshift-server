"""CLI contract: exit codes and JSON. Banner and table text are deliberately not asserted."""

import json
import os
import sys
from pathlib import Path

import pytest
import torch
import uvicorn
from fastapi.testclient import TestClient
from typer.testing import CliRunner

import downshift
from downshift.cli import main, runtime
from downshift.cli.main import app
from downshift.loading import LoadSpec
from downshift.serve.options import ServeOptions
from downshift.serve.schemas import OutputEncoding
from tests.models import clean_mlp

CLEAN = "tests.models.clean_mlp:make_model"
DEGRADED = "tests.models.scatter_include_self_false:make_model"
FAILED = "tests.models.data_dependent_branch:make_model"
BROKEN = "tests.models.broken_factory:make_model"

runner = CliRunner()


def run(*args: str):
    return runner.invoke(app, list(args))


def parse(result) -> dict:
    assert result.exit_code in (0, 1, 2, 3), result.output
    return json.loads(result.stdout)


@pytest.fixture(scope="module")
def exported(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("export")
    result = run("export", CLEAN, "-o", str(out), "--json")
    assert result.exit_code == 0, result.output
    assert parse(result)["manifest_path"] == str(out / "clean_mlp.manifest.json")
    return out


def test_check_clean():
    result = run("check", CLEAN, "--json")
    assert result.exit_code == 0, result.output
    assert parse(result)["status"] == "CLEAN"


def test_check_degraded():
    result = run("check", DEGRADED, "--json")
    assert result.exit_code == 2, result.output
    assert parse(result)["status"] == "DEGRADED"


def test_check_failed():
    result = run("check", FAILED, "--json")
    assert result.exit_code == 1, result.output
    assert parse(result)["status"] == "FAILED"


def test_check_bad_spec_is_usage_error():
    result = run("check", "tests.models.does_not_exist:make_model", "--json")
    assert result.exit_code == 4, result.output
    assert result.stdout == ""


def test_check_imports_a_model_module_from_the_current_directory(tmp_path, monkeypatch):
    # The `downshift` console script, unlike `python -m downshift`, does not put the current
    # directory on sys.path by itself; the CLI does.
    (tmp_path / "cwd_model.py").write_text(
        "from tests.models.clean_mlp import make_inputs, make_model\n"
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "path", [p for p in sys.path if p not in ("", str(tmp_path))])

    result = run("check", "cwd_model:make_model", "--json")

    assert result.exit_code == 0, result.output
    assert parse(result)["status"] == "CLEAN"


def test_check_file_name_spec_says_to_use_the_module_name():
    result = run("check", "my_model.py:model", "--json")
    assert result.exit_code == main.EXIT_USAGE, result.output
    assert "my_model:model" in result.output


def test_check_table_output():
    result = run("check", CLEAN)
    assert result.exit_code == 0, result.output


def test_check_atol_rtol_override():
    result = run("check", CLEAN, "--atol", "1", "--rtol", "1", "--json")
    assert result.exit_code == 0, result.output
    numerics = parse(result)["numerics"]
    assert numerics["tolerance_abs"] == 1.0
    assert numerics["tolerance_rel"] == 1.0


def test_check_records_seed_and_tolerance_dtype_in_json():
    result = run("check", CLEAN, "--seed", "3", "--json")
    assert result.exit_code == 0, result.output
    numerics = parse(result)["numerics"]
    assert numerics["seed"] == 3
    assert numerics["tolerance_dtype"] == "float32"


def test_check_vary_accepts_an_import_spec():
    result = run("check", CLEAN, "--vary", "tests.test_cli:_custom_vary", "--json")
    assert result.exit_code == 0, result.output
    assert parse(result)["status"] == "CLEAN"


def _custom_vary(i: int) -> tuple:
    from tests.models import clean_mlp

    return clean_mlp.make_inputs()


def test_check_unknown_adapter_is_a_usage_error():
    result = run("check", CLEAN, "--adapter", "doesnotexist", "--json")
    assert result.exit_code == main.EXIT_USAGE, result.output
    assert "LoadError" not in result.output
    assert "doesnotexist" in result.output


def test_check_crash_with_debug_prints_traceback():
    result = run("check", BROKEN, "--log-level", "debug", "--json")
    assert result.exit_code == main.EXIT_CRASH, result.output


def test_check_failed_table_points_at_debug_logging():
    result = run("check", FAILED)
    assert result.exit_code == 1, result.output
    assert "--log-level debug" in result.output


def test_check_failed_debug_logs_every_strategys_traceback():
    result = run("check", FAILED, "--log-level", "debug")
    assert result.exit_code == 1, result.output
    assert "strict=False failed" in result.output
    assert "strict=True failed" in result.output


def test_check_unsafe_load_prints_a_warning(tmp_path: Path):
    path = tmp_path / "full.pt"
    torch.save(clean_mlp.make_model(), path)

    result = run("check", str(path), "--unsafe-load", "--json")

    assert result.exit_code in (0, 1, 2, 3), result.output
    assert "arbitrary code" in result.output


def test_export_rejects_an_onnx_model_as_input(exported: Path, tmp_path: Path):
    result = run("export", str(exported / "clean_mlp.onnx"), "-o", str(tmp_path), "--json")
    assert result.exit_code == main.EXIT_USAGE, result.output


def test_export_table_output_prints_artifacts(tmp_path: Path):
    result = run("export", CLEAN, "-o", str(tmp_path))
    assert result.exit_code == 0, result.output
    assert "Wrote" in result.output


def test_export_writes_artifact_and_manifest(exported: Path):
    assert (exported / "clean_mlp.onnx").exists()
    manifest = json.loads((exported / "clean_mlp.manifest.json").read_text())
    assert manifest["verdict"]["status"] == "CLEAN"
    assert manifest["onnx_file"] == "clean_mlp.onnx"


def test_check_onnx_without_reference_is_unverified(exported: Path):
    result = run("check", str(exported / "clean_mlp.onnx"), "--json")
    assert result.exit_code == 3, result.output
    assert parse(result)["status"] == "UNVERIFIED"


def test_check_onnx_against_different_init_is_degraded(exported: Path):
    # make_model() builds a fresh random init, so the graph on disk can't match it.
    result = run("check", str(exported / "clean_mlp.onnx"), "--reference", CLEAN, "--json")
    assert result.exit_code == 2, result.output
    data = parse(result)
    assert data["status"] == "DEGRADED"
    assert data["numerics"]["failures"] > 0


def test_export_no_verify(tmp_path: Path):
    result = run("export", CLEAN, "--no-verify", "-o", str(tmp_path), "--json")
    assert result.exit_code == 3, result.output
    data = parse(result)
    assert data["status"] == "UNVERIFIED"
    assert data["numerics"] is None
    assert (tmp_path / "clean_mlp.onnx").exists()
    manifest = json.loads((tmp_path / "clean_mlp.manifest.json").read_text())
    assert manifest["verdict"]["status"] == "UNVERIFIED"


def test_export_failed_writes_nothing(tmp_path: Path):
    result = run("export", FAILED, "-o", str(tmp_path), "--json")
    assert result.exit_code == 1, result.output
    assert parse(result)["manifest_path"] is None
    assert not list(tmp_path.iterdir())


def test_export_custom_name(tmp_path: Path):
    result = run("export", CLEAN, "-o", str(tmp_path), "--name", "mlp", "--json")
    assert result.exit_code == 0, result.output
    assert (tmp_path / "mlp.onnx").exists()
    assert (tmp_path / "mlp.manifest.json").exists()


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        (CLEAN, "clean_mlp"),
        ("./gat_fraud_v3.pt", "gat_fraud_v3"),
        ("models/bert-base/", "bert-base"),
        ("model.onnx", "model"),
    ],
)
def test_slug(spec: str, expected: str):
    assert main.slug(spec) == expected


def _fake_uvicorn_server(monkeypatch, captured: dict) -> None:
    """Single-worker `serve` now binds via uvicorn.Server directly (not uvicorn.run), so
    should_exit is reachable from the loader thread. Stand in for it: capture the config,
    and drive the app's lifespan the way a real server would (so the loader thread the
    plan describes actually runs and lands app.state.serving), without opening a socket.
    """

    def fake_init(self, config) -> None:
        self.config = config
        self.should_exit = False
        captured.update(
            app=config.app,
            host=config.host,
            port=config.port,
            log_level=config.log_level,
            access_log=config.access_log,
        )

    def fake_run(self) -> None:
        with TestClient(self.config.app):
            thread = getattr(self.config.app.state, "loader_thread", None)
            if thread is not None:
                thread.join(timeout=30)

    monkeypatch.setattr(uvicorn.Server, "__init__", fake_init)
    monkeypatch.setattr(uvicorn.Server, "run", fake_run)


def _serve_captured(monkeypatch, *extra_args: str) -> tuple:
    """Run `serve CLEAN --warmup 1 <extra_args>` with uvicorn stubbed out: uvicorn.run for
    `--workers` > 1 (unchanged), uvicorn.Server for the single-worker bind-first path.

    Returns (CliRunner result, the captured config values plus the app).
    """
    pytest.importorskip("downshift.serve.app")
    captured: dict = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: captured.update(app=app, **kw))
    _fake_uvicorn_server(monkeypatch, captured)
    result = run("serve", CLEAN, "--warmup", "1", *extra_args)
    assert result.exit_code == 0, result.output
    return result, captured


def test_serve_builds_app(monkeypatch):
    _, captured = _serve_captured(monkeypatch, "--port", "9999")
    assert captured["port"] == 9999
    serving = captured["app"].state.serving
    assert serving.verdict.status == "CLEAN"
    # Tensor-IO options not given on the command line come from settings.
    assert serving.options.output_encoding == main.settings.OUTPUT_ENCODING == "json"
    assert serving.options.max_input_bytes == main.settings.MAX_INPUT_BYTES


def test_serve_passes_thread_options_to_the_ort_session(monkeypatch):
    _, captured = _serve_captured(monkeypatch, "--intra-op-threads", "3", "--inter-op-threads", "2")
    session_opts = captured["app"].state.serving.backend.session.get_session_options()
    assert session_opts.intra_op_num_threads == 3
    assert session_opts.inter_op_num_threads == 2


def test_serve_passes_tensor_io_options_to_serve_options(monkeypatch):
    result, captured = _serve_captured(
        monkeypatch, "--output-encoding", "base64", "--max-input-bytes", "4096"
    )
    options = captured["app"].state.serving.options
    assert options.output_encoding == "base64"
    assert options.max_input_bytes == 4096
    assert "Encoding" in result.output


def test_serve_passes_max_body_bytes_and_max_concurrency(monkeypatch):
    result, captured = _serve_captured(
        monkeypatch, "--max-body-bytes", "8192", "--max-concurrency", "4"
    )
    options = captured["app"].state.serving.options
    assert options.max_body_bytes == 8192
    assert options.max_concurrency == 4
    assert "Capacity" in result.output


def test_serve_passes_max_queue_and_request_timeout(monkeypatch):
    result, captured = _serve_captured(monkeypatch, "--max-queue", "8", "--request-timeout", "2.5")
    options = captured["app"].state.serving.options
    assert options.max_queue == 8
    assert options.request_timeout == 2.5
    assert "Capacity" in result.output


def _app_access_log(api) -> bool:
    """uvicorn's own access log is always off; the app's RequestIdMiddleware writes the line."""
    middleware = next(m for m in api.user_middleware if m.cls.__name__ == "RequestIdMiddleware")
    return middleware.kwargs["access_log"]


def test_serve_access_log_defaults_to_enabled(monkeypatch):
    _, captured = _serve_captured(monkeypatch)
    assert captured["access_log"] is False
    assert _app_access_log(captured["app"]) is True


def test_serve_no_access_log_disables_it(monkeypatch):
    _, captured = _serve_captured(monkeypatch, "--no-access-log")
    assert _app_access_log(captured["app"]) is False


@pytest.mark.parametrize("value", ["hex", "safetensors"])
def test_serve_rejects_unknown_output_encoding(value):
    result = run("serve", CLEAN, "--output-encoding", value)
    assert result.exit_code == 2, result.output  # typer usage error: not a choice


def test_serve_load_failure_on_the_loader_thread_exits_with_the_usual_code(monkeypatch):
    """A model that fails to load in the background thread must still end the process with
    the exit code _exit_on_error would give it synchronously, not serve 503 forever."""
    captured: dict = {}
    _fake_uvicorn_server(monkeypatch, captured)

    result = run("serve", CLEAN, "--adapter", "doesnotexist")

    assert result.exit_code == main.EXIT_USAGE, result.output
    assert "LoadError" not in result.output
    assert "doesnotexist" in result.output


def test_serve_backend_onnxruntime_on_degraded_is_a_usage_error(monkeypatch):
    captured: dict = {}
    _fake_uvicorn_server(monkeypatch, captured)

    result = run("serve", DEGRADED, "--backend", "onnxruntime")

    assert result.exit_code == main.EXIT_USAGE, result.output
    assert "--force-onnx" in result.output


def test_serve_backend_onnxruntime_and_force_onnx_on_degraded_is_accepted(monkeypatch):
    captured: dict = {}
    _fake_uvicorn_server(monkeypatch, captured)

    result = run("serve", DEGRADED, "--backend", "onnxruntime", "--force-onnx", "--warmup", "0")

    assert result.exit_code == 0, result.output
    assert captured["app"].state.serving.backend.name == "onnxruntime"


def test_serve_workers_uses_an_import_string_factory(monkeypatch):
    _, captured = _serve_captured(monkeypatch, "--workers", "2")
    assert captured["app"] == "downshift.cli.runtime:_serve_app_factory"
    assert captured["workers"] == 2
    assert captured["factory"] is True

    args = main.ServeArgs.from_json(os.environ[main._SERVE_ARGS_ENV])
    assert args.load.model == CLEAN


def test_serve_workers_splits_threads_across_the_cpu_count(monkeypatch):
    monkeypatch.setattr(main.settings, "usable_cpus", lambda: 16)
    _, captured = _serve_captured(monkeypatch, "--workers", "4")
    assert captured["workers"] == 4

    args = main.ServeArgs.from_json(os.environ[main._SERVE_ARGS_ENV])
    assert args.options.intra_op_threads == 4


def test_serve_workers_explicit_intra_op_threads_wins(monkeypatch):
    monkeypatch.setattr(main.settings, "usable_cpus", lambda: 16)
    _serve_captured(monkeypatch, "--workers", "4", "--intra-op-threads", "7")

    args = main.ServeArgs.from_json(os.environ[main._SERVE_ARGS_ENV])
    assert args.options.intra_op_threads == 7


def _rebuilt_app():
    """What a --workers worker does: rebuild the app from the env, then let the loader run."""
    api = runtime._serve_app_factory()
    with TestClient(api):
        api.state.loader_thread.join(timeout=60)
    return api


def test_serve_app_factory_rebuilds_the_app_from_env(monkeypatch):
    pytest.importorskip("downshift.serve.app")
    args = main.ServeArgs(
        load=LoadSpec(CLEAN),
        options=ServeOptions(
            k=1,
            device="cpu",
            warmup=1,
            output_encoding=OutputEncoding.base64,
            max_input_bytes=1024,
            max_body_bytes=2048,
            max_concurrency=2,
        ),
        reference=None,
        middleware=None,
        log_level="warning",
    )
    monkeypatch.setenv(main._SERVE_ARGS_ENV, args.to_json())

    api = _rebuilt_app()
    assert api.state.serving.verdict.status == "CLEAN"
    assert api.state.serving.options.output_encoding == "base64"
    assert api.state.serving.options.max_input_bytes == 1024
    assert api.state.serving.options.max_body_bytes == 2048
    assert api.state.serving.options.max_concurrency == 2


def test_a_worker_that_cannot_load_exits_with_uvicorns_startup_failure_code(monkeypatch):
    """uvicorn's worker supervisor respawns a worker that dies with any other code, which
    would reload the same failing model forever; STARTUP_FAILURE makes it stop the server."""
    pytest.importorskip("downshift.serve.app")
    from uvicorn.config import STARTUP_FAILURE

    from downshift.cli import runtime

    args = main.ServeArgs(
        load=LoadSpec(CLEAN),
        options=ServeOptions(k=1, device="cpu", warmup=1),
        reference=None,
        middleware=None,
        log_level="warning",
    )
    monkeypatch.setenv(main._SERVE_ARGS_ENV, args.to_json())

    def fail(_args):
        raise MemoryError("not enough RAM for another copy")

    exits: list[int] = []
    monkeypatch.setattr(runtime, "_build_serving_state", fail)
    monkeypatch.setattr(os, "_exit", exits.append)

    _rebuilt_app()

    assert exits == [STARTUP_FAILURE]


def test_serve_workers_writes_and_ships_the_onnx_artifact(monkeypatch):
    """Parent-side: the .onnx the parent exported exists on disk, and its bytes plus the
    verdict are what the workers get, before the (stubbed) uvicorn.run's `finally` cleans
    the temp file up."""
    seen: dict = {}

    def fake_run(app, **kw):
        seen["app"] = app
        seen.update(kw)
        args = main.ServeArgs.from_json(os.environ[main._SERVE_ARGS_ENV])
        seen["args"] = args
        assert args.artifact.onnx_path is not None
        seen["onnx_bytes"] = Path(args.artifact.onnx_path).read_bytes()

    monkeypatch.setattr(uvicorn, "run", fake_run)
    result = run("serve", CLEAN, "--warmup", "1", "--workers", "2", "--no-access-log")
    assert result.exit_code == 0, result.output

    assert seen["workers"] == 2
    assert seen["access_log"] is False
    args = seen["args"]
    assert args.artifact.onnx_path is not None  # the parent serves the graph
    assert args.artifact.verdict is not None
    assert args.artifact.verdict["status"] == "CLEAN"
    assert args.artifact.input_names == ["x"]
    assert len(seen["onnx_bytes"]) > 0


def test_export_with_external_data_writes_both_files_and_lists_the_data_file(
    tmp_path: Path, monkeypatch
):
    from downshift.core import capture as capture_mod

    monkeypatch.setattr(capture_mod, "EXTERNAL_DATA_THRESHOLD", 0)
    result = run("export", CLEAN, "-o", str(tmp_path), "--json")
    assert result.exit_code == 0, result.output

    assert (tmp_path / "clean_mlp.onnx").is_file()
    assert (tmp_path / "clean_mlp.onnx.data").is_file()
    manifest = json.loads((tmp_path / "clean_mlp.manifest.json").read_text())
    assert [entry["file"] for entry in manifest["external_data"]] == ["clean_mlp.onnx.data"]


def test_serve_workers_hands_an_external_data_export_to_the_workers(monkeypatch):
    """The parent drops its state before the workers load; the .onnx and the data file beside
    it must still be there, and a worker rebuilt from the handoff must serve."""
    from downshift.core import capture as capture_mod

    monkeypatch.setattr(capture_mod, "EXTERNAL_DATA_THRESHOLD", 0)
    seen: dict = {}

    def fake_run(app, **kw):
        args = main.ServeArgs.from_json(os.environ[main._SERVE_ARGS_ENV])
        onnx_path = Path(args.artifact.onnx_path)
        seen["files"] = sorted(p.name for p in onnx_path.parent.iterdir() if p.suffix != ".npz")
        seen["api"] = _rebuilt_app()

    monkeypatch.setattr(uvicorn, "run", fake_run)
    result = run("serve", CLEAN, "--warmup", "1", "--workers", "2", "--no-access-log")
    assert result.exit_code == 0, result.output

    assert seen["files"] == ["model.onnx", "model.onnx.data"]
    assert seen["api"].state.serving.backend.name == "onnxruntime"


def test_serve_workers_degraded_ships_a_torch_artifact_and_warns(monkeypatch):
    seen: dict = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: seen.update(app=app, **kw))

    result = run("serve", DEGRADED, "--warmup", "0", "--workers", "2")

    assert result.exit_code == 0, result.output
    assert "each worker independently reloads and re-warms" in result.output
    args = main.ServeArgs.from_json(os.environ[main._SERVE_ARGS_ENV])
    assert args.artifact.verdict is not None
    assert args.artifact.verdict["status"] == "DEGRADED"
    assert args.artifact.onnx_path is None


def test_serve_app_factory_from_onnx_artifact_never_calls_capture(monkeypatch, tmp_path):
    """Worker rebuild from an onnxruntime artifact must not re-run capture()/verify(): the
    parent already did that once."""
    pytest.importorskip("downshift.serve.app")
    onnx_path = tmp_path / "clean_mlp.onnx"
    verdict = downshift.export(clean_mlp.make_model(), onnx_path, clean_mlp.make_inputs())
    assert verdict.status == "CLEAN", verdict.reason

    from downshift.core import verdict as verdict_mod

    def boom(*args, **kwargs):
        raise AssertionError("capture() must not run when serving a parent's artifact")

    monkeypatch.setattr(verdict_mod, "capture", boom)

    args = main.ServeArgs(
        load=LoadSpec(CLEAN),
        options=ServeOptions(
            k=1, device="cpu", warmup=1, max_input_bytes=1024, max_body_bytes=2048
        ),
        reference=None,
        middleware=None,
        log_level="warning",
        artifact=main.ArtifactHandoff(
            verdict=verdict.to_dict(),
            input_names=list(verdict.input_names),
            onnx_path=str(onnx_path),
            kind="import-spec",
        ),
    )
    monkeypatch.setenv(main._SERVE_ARGS_ENV, args.to_json())

    api = _rebuilt_app()

    assert api.state.serving.verdict.status == "CLEAN"
    assert api.state.serving.backend.name == "onnxruntime"


def test_version_eager_flag():
    result = run("--version")
    assert result.exit_code == 0
    assert result.stdout.strip() == main.__version__


def test_help_still_works_with_no_args_is_help():
    result = run("--help")
    assert result.exit_code == 0
    assert "check" in result.output
    assert "serve" in result.output


def test_check_axis_max_lowers_the_served_bound():
    result = run("check", CLEAN, "--axis-max", "dim0=20", "--json")
    assert result.exit_code == 0, result.output
    assert parse(result)["axes"][0]["served_max"] == 20


@pytest.mark.parametrize(
    "value",
    ["seq=4", "dim0=999999", "dim0", "dim0=abc", "dim0=0"],
    ids=["unknown-name", "above-ceiling", "no-value", "non-int", "zero"],
)
def test_check_bad_axis_max_is_usage_error(value):
    result = run("check", CLEAN, "--axis-max", value, "--json")
    assert result.exit_code == 4, result.output
    assert result.stdout == ""


def test_serve_axis_max_round_trips_through_serve_args():
    from downshift.cli.runtime import ServeArgs

    args = ServeArgs(
        load=LoadSpec(CLEAN),
        options=ServeOptions(axis_max={"dim0": 20}),
        reference=None,
        middleware=None,
        log_level="warning",
    )
    assert ServeArgs.from_json(args.to_json()).options.axis_max == {"dim0": 20}


def test_serve_passes_prep_threads(monkeypatch):
    result, captured = _serve_captured(monkeypatch, "--prep-threads", "3")
    assert captured["app"].state.serving.options.prep_threads == 3
    assert "3 prep threads" in result.output


def test_serve_execution_defaults_to_threadpool(monkeypatch):
    _, captured = _serve_captured(monkeypatch)
    assert captured["app"].state.serving.options.execution == "threadpool"


def test_serve_passes_execution_inline(monkeypatch):
    result, captured = _serve_captured(monkeypatch, "--execution", "inline")
    assert captured["app"].state.serving.options.execution == "inline"
    assert "Execution" in result.output
    assert "event loop" in result.output


def test_serve_rejects_an_unknown_execution_mode(monkeypatch):
    result = run("serve", CLEAN, "--execution", "auto")
    assert result.exit_code == 2, result.output


def test_serve_execution_round_trips_through_serve_args():
    from downshift.cli.runtime import ServeArgs
    from downshift.serve.options import ExecutionChoice

    args = ServeArgs(
        load=LoadSpec(CLEAN),
        options=ServeOptions(execution=ExecutionChoice.inline),
        reference=None,
        middleware=None,
        log_level="warning",
    )
    restored = ServeArgs.from_json(args.to_json()).options.execution
    assert restored is ExecutionChoice.inline
