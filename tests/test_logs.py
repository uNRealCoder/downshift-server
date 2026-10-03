"""downshift.logs (the one logging sink) and the per-request line the ASGI middleware writes on
the `downshift.access` logger."""

import io
import logging
import re
import subprocess
import sys
import warnings

import pytest
from fastapi.testclient import TestClient

from downshift.loading import LoadSpec, load_model
from downshift.logs import request_id_var, setup_logging
from downshift.serve.app import build_app
from downshift.serve.engine import ServeOptions, prepare_serving
from tests.conftest import subprocess_env

MLP_INPUT = {"inputs": {"x": [[0.0] * 16]}}
_UVICORN = ("uvicorn", "uvicorn.error", "uvicorn.access")
_TOUCHED = ("", *_UVICORN, "downshift.report", "downshift.x", "onnxscript", "onnx_ir")
_LINE = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} (\w+) ([\w.]+): (.*)$")


@pytest.fixture
def restore_logging():
    """setup_logging is process-global (force=True on the root logger, captured warnings);
    put every logger it touches back so it cannot leak into other tests."""
    saved = {}
    for name in _TOUCHED:
        target = logging.getLogger(name)
        saved[name] = (list(target.handlers), target.level, target.propagate)
    showwarning = warnings.showwarning
    yield
    logging.captureWarnings(False)
    warnings.showwarning = showwarning
    for name, (handlers, level, propagate) in saved.items():
        target = logging.getLogger(name)
        target.handlers[:] = handlers
        target.setLevel(level)
        target.propagate = propagate


def _lines(stream: io.StringIO) -> list[str]:
    return [line for line in stream.getvalue().splitlines() if line]


def test_setup_logging_writes_text_lines_to_the_stream(restore_logging):
    stream = io.StringIO()
    setup_logging("info", stream=stream)

    logging.getLogger("downshift.x").info("hello %s", "world")

    (line,) = _lines(stream)
    match = _LINE.match(line)
    assert match, line
    assert match.groups() == ("INFO", "downshift.x", "hello world")


def test_setup_logging_appends_the_request_id_when_one_is_bound(restore_logging):
    stream = io.StringIO()
    setup_logging("info", stream=stream)

    token = request_id_var.set("abc123")
    try:
        logging.getLogger("downshift.x").info("inside")
    finally:
        request_id_var.reset(token)
    logging.getLogger("downshift.x").info("outside")

    inside, outside = _lines(stream)
    assert inside.endswith("downshift.x: inside request_id=abc123")
    assert "request_id" not in outside


def test_setup_logging_twice_does_not_duplicate_lines(restore_logging):
    stream = io.StringIO()
    setup_logging("info", stream=stream)
    setup_logging("info", stream=stream)

    logging.getLogger("downshift.x").info("once")

    assert len(_lines(stream)) == 1
    assert len(logging.getLogger().handlers) == 1


def test_setup_logging_captures_warnings(restore_logging):
    stream = io.StringIO()
    setup_logging("info", stream=stream)

    warnings.warn("careful-with-this-one", UserWarning, stacklevel=1)

    text = stream.getvalue()
    assert " WARNING py.warnings: " in text
    assert "UserWarning: careful-with-this-one" in text


def test_uvicorn_loggers_share_the_handler(restore_logging):
    for name in _UVICORN:  # what uvicorn's own dictConfig leaves behind
        own = logging.getLogger(name)
        own.handlers[:] = [logging.NullHandler()]
        own.propagate = False
    stream = io.StringIO()
    setup_logging("info", stream=stream)

    for name in _UVICORN:
        assert logging.getLogger(name).propagate is True
        logging.getLogger(name).info("from %s", name)

    lines = _lines(stream)
    assert len(lines) == 3
    for name, line in zip(_UVICORN, lines, strict=True):
        assert f" INFO {name}: from {name}" in line


def test_report_logger_prints_at_warning_level_but_ordinary_info_does_not(restore_logging):
    stream = io.StringIO()
    setup_logging("warning", stream=stream)

    logging.getLogger("downshift.report").info("the banner")
    logging.getLogger("downshift.x").info("chatter")
    logging.getLogger("downshift.x").warning("a warning")

    messages = [_LINE.match(line).group(3) for line in _lines(stream)]
    assert messages == ["the banner", "a warning"]


def test_importing_the_cli_module_pulls_in_neither_torch_nor_rich():
    code = (
        "import downshift.cli.main, sys; "
        "assert 'torch' not in sys.modules, 'torch'; "
        "assert 'rich' not in sys.modules, 'rich'"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=subprocess_env()
    )
    assert result.returncode == 0, result.stderr


def _access_records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.name == "downshift.access"]


@pytest.fixture(scope="module")
def mlp_state():
    return prepare_serving(
        load_model(LoadSpec("tests.models.clean_mlp:make_model")), ServeOptions(warmup=1)
    )


@pytest.fixture(scope="module")
def client(mlp_state) -> TestClient:
    return TestClient(build_app(mlp_state, api_key=None))


def test_predict_writes_one_request_line_with_timings(client, caplog):
    with caplog.at_level(logging.INFO, logger="downshift.access"):
        resp = client.post("/predict", json=MLP_INPUT)

    assert resp.status_code == 200, resp.text
    (record,) = _access_records(caplog)
    assert record.levelno == logging.INFO
    assert re.fullmatch(r"POST /predict 200 \d+\.\d ms", record.getMessage())
    assert (record.method, record.path, record.status) == ("POST", "/predict", 200)
    assert record.duration_ms > 0
    assert set(record.timings_ms) == {
        "parse",
        "prep_wait",
        "prep",
        "infer_wait",
        "infer",
        "encode",
    }


def test_health_is_debug_only(client, caplog):
    with caplog.at_level(logging.INFO, logger="downshift.access"):
        client.get("/health")
        client.get("/ready")
    assert _access_records(caplog) == []

    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger="downshift.access"):
        client.get("/health")
    (record,) = _access_records(caplog)
    assert record.levelno == logging.DEBUG
    assert record.getMessage().startswith("GET /health 200 ")
    assert not hasattr(record, "timings_ms")


def test_a_client_error_is_logged_at_warning(client, caplog):
    with caplog.at_level(logging.INFO, logger="downshift.access"):
        resp = client.post("/predict", json={"inputs": {}})

    assert resp.status_code == 400
    (record,) = _access_records(caplog)
    assert record.levelno == logging.WARNING
    assert record.status == 400
    assert not hasattr(record, "timings_ms")


def test_access_log_off_logs_nothing_but_still_echoes_the_request_id(mlp_state, caplog):
    quiet = TestClient(build_app(mlp_state, api_key=None, access_log=False))

    with caplog.at_level(logging.DEBUG, logger="downshift.access"):
        generated = quiet.post("/predict", json=MLP_INPUT)
        supplied = quiet.get("/health", headers={"X-Request-Id": "trace-me"})

    assert _access_records(caplog) == []
    assert generated.headers["x-request-id"]
    assert supplied.headers["x-request-id"] == "trace-me"


def test_a_client_supplied_request_id_is_echoed_on_a_logged_request(client, caplog):
    with caplog.at_level(logging.INFO, logger="downshift.access"):
        resp = client.post("/predict", json=MLP_INPUT, headers={"X-Request-Id": "caller-7"})

    assert resp.headers["x-request-id"] == "caller-7"
    (record,) = _access_records(caplog)
    assert record.status == 200
