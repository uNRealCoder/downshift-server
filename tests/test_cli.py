"""CLI contract: exit codes and JSON. Banner and table text are deliberately not asserted."""

import json
import logging
import os
import sys
from pathlib import Path

import pytest
import torch
import uvicorn
from typer.testing import CliRunner

from downshift.cli import main
from downshift.cli.main import app
from tests.models import clean_mlp

CLEAN = "tests.models.clean_mlp:make_model"
DEGRADED = "tests.models.scatter_include_self_false:make_model"
FAILED = "tests.models.data_dependent_branch:make_model"

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


def test_check_table_output():
    result = run("check", CLEAN)
    assert result.exit_code == 0, result.output


def test_check_log_format_json_is_accepted():
    result = run("check", CLEAN, "--log-format", "json", "--json")
    assert result.exit_code == 0, result.output


def test_json_formatter_serialises_exc_info():
    try:
        raise ValueError("boom")
    except ValueError:
        exc_info = sys.exc_info()
    record = logging.LogRecord("test", logging.ERROR, __file__, 1, "failed", (), exc_info)

    payload = json.loads(main._JsonFormatter().format(record))

    assert payload["level"] == "ERROR"
    assert "ValueError" in payload["exc_info"]


def test_check_unknown_adapter_is_an_unexpected_crash():
    result = run("check", CLEAN, "--adapter", "doesnotexist", "--json")
    assert result.exit_code == main.EXIT_CRASH, result.output
    assert "KeyError" in result.output


def test_check_crash_with_debug_prints_traceback():
    result = run("check", CLEAN, "--adapter", "doesnotexist", "--log-level", "debug", "--json")
    assert result.exit_code == main.EXIT_CRASH, result.output


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
        ("org/repo", "repo"),
        ("model.onnx", "model"),
    ],
)
def test_slug(spec: str, expected: str):
    assert main.slug(spec) == expected


def _serve_captured(monkeypatch, *extra_args: str) -> tuple:
    """Run `serve CLEAN --warmup 1 <extra_args>` with uvicorn.run stubbed out.

    Returns (CliRunner result, the kwargs uvicorn.run received plus its `app`).
    """
    pytest.importorskip("downshift.serve.app")
    captured: dict = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: captured.update(app=app, **kw))
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
    assert "Concurrency" in result.output


def test_serve_rejects_unknown_output_encoding():
    result = run("serve", CLEAN, "--output-encoding", "hex")
    assert result.exit_code == 2, result.output  # typer usage error: not a choice


def test_serve_workers_uses_an_import_string_factory(monkeypatch):
    _, captured = _serve_captured(monkeypatch, "--workers", "2")
    assert captured["app"] == "downshift.cli.main:_serve_app_factory"
    assert captured["workers"] == 2
    assert captured["factory"] is True

    args = main.ServeArgs(**json.loads(os.environ[main._SERVE_ARGS_ENV]))
    assert args.model == CLEAN


def test_serve_app_factory_rebuilds_the_app_from_env(monkeypatch):
    pytest.importorskip("downshift.serve.app")
    args = main.ServeArgs(
        model=CLEAN,
        inputs=None,
        model_class=None,
        unsafe_load=False,
        adapter=None,
        k=1,
        dynamic=None,
        reference=None,
        middleware=None,
        backend="auto",
        force_onnx=False,
        device="cpu",
        warmup=1,
        intra_op_threads=0,
        inter_op_threads=0,
        output_encoding="base64",
        max_input_bytes=1024,
        max_body_bytes=2048,
        max_concurrency=2,
        log_level="warning",
        log_format="text",
    )
    monkeypatch.setenv(main._SERVE_ARGS_ENV, json.dumps(main.asdict(args)))

    api = main._serve_app_factory()
    assert api.state.serving.verdict.status == "CLEAN"
    assert api.state.serving.options.output_encoding == "base64"
    assert api.state.serving.options.max_input_bytes == 1024
    assert api.state.serving.options.max_body_bytes == 2048
    assert api.state.serving.options.max_concurrency == 2


def test_version():
    result = run("version")
    assert result.exit_code == 0
    assert main.__version__ in result.stdout


def test_version_eager_flag():
    result = run("--version")
    assert result.exit_code == 0
    assert main.__version__ in result.stdout


def test_help_still_works_with_no_args_is_help():
    result = run("--help")
    assert result.exit_code == 0
    assert "check" in result.output
    assert "serve" in result.output
