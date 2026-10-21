"""settings.py reads the DOWNSHIFT_* environment variables at import. These tests therefore
reload the module around each case. They do not import the constants one time at collection.
"""

import importlib
import json
import subprocess
import sys

import pytest

from downshift import settings
from tests.conftest import subprocess_env


@pytest.fixture(autouse=True)
def _reload_after() -> None:
    """Undo the reload this test triggers, so later tests see the un-overridden defaults."""
    yield
    importlib.reload(settings)


def test_defaults_with_no_env_set(monkeypatch: pytest.MonkeyPatch) -> None:
    names = (
        "HOST",
        "PORT",
        "DEVICE",
        "BACKEND",
        "WARMUP",
        "SAMPLES",
        "OUTPUT_ENCODING",
        "MAX_INPUT_BYTES",
        "MAX_BODY_BYTES",
        "MAX_CONCURRENCY",
        "MAX_QUEUE",
        "REQUEST_TIMEOUT",
    )
    for name in names:
        monkeypatch.delenv(f"DOWNSHIFT_{name}", raising=False)
    mod = importlib.reload(settings)
    assert tuple(getattr(mod, name) for name in names) == (
        "127.0.0.1",
        8000,
        "auto",
        "auto",
        3,
        8,
        "json",
        256 * 1024 * 1024,
        32 * 1024 * 1024,
        4,
        64,
        30.0,
    )


def test_env_vars_override_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DOWNSHIFT_HOST", "127.0.0.1")
    monkeypatch.setenv("DOWNSHIFT_PORT", "9000")
    monkeypatch.setenv("DOWNSHIFT_SAMPLES", "16")
    monkeypatch.setenv("DOWNSHIFT_OUTPUT_ENCODING", "base64")
    monkeypatch.setenv("DOWNSHIFT_MAX_INPUT_BYTES", "4096")
    monkeypatch.setenv("DOWNSHIFT_MAX_BODY_BYTES", "8192")
    monkeypatch.setenv("DOWNSHIFT_MAX_CONCURRENCY", "2")
    mod = importlib.reload(settings)
    assert mod.HOST == "127.0.0.1"
    assert mod.PORT == 9000
    assert mod.SAMPLES == 16
    assert mod.OUTPUT_ENCODING == "base64"
    assert mod.MAX_INPUT_BYTES == 4096
    assert mod.MAX_BODY_BYTES == 8192
    assert mod.MAX_CONCURRENCY == 2


@pytest.mark.parametrize(
    ("name", "value", "kind"),
    [
        ("DOWNSHIFT_PORT", "not-a-number", "integer"),
        ("DOWNSHIFT_TOL_FLOAT32_ATOL", "not-a-float", "float"),
        ("DOWNSHIFT_EXECUTION", "auto", "execution mode"),
    ],
)
def test_bad_env_var_raises_clear_error(monkeypatch: pytest.MonkeyPatch, name, value, kind) -> None:
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=f"{name}.*{value}.*{kind}"):
        importlib.reload(settings)


def test_per_dtype_tolerance_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DOWNSHIFT_TOL_FLOAT16_ATOL", "0.05")
    mod = importlib.reload(settings)
    assert mod.TOLERANCES["float16"] == (0.05, 1e-2)
    # Dtypes that nobody changed keep their defaults.
    assert mod.TOLERANCES["float32"] == (1e-4, 1e-3)


def test_axis_max_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DOWNSHIFT_AXIS_MAX", "seq=4096, num_nodes=500")
    assert importlib.reload(settings).AXIS_MAX == {"seq": 4096, "num_nodes": 500}


def test_axis_max_defaults_to_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DOWNSHIFT_AXIS_MAX", raising=False)
    assert importlib.reload(settings).AXIS_MAX == {}


@pytest.mark.parametrize("raw", ["seq", "seq=", "=4", "seq=x", "seq=0", "seq=-1"])
def test_bad_axis_max_env_var_raises_clear_error(monkeypatch: pytest.MonkeyPatch, raw: str) -> None:
    monkeypatch.setenv("DOWNSHIFT_AXIS_MAX", raw)
    with pytest.raises(ValueError, match="DOWNSHIFT_AXIS_MAX"):
        importlib.reload(settings)


def test_usable_cpus_reads_a_cgroup_v2_quota(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    quota = tmp_path / "cpu.max"
    quota.write_text("200000 100000\n")
    monkeypatch.setattr(settings, "_CGROUP_CPU_MAX", str(quota))
    assert settings.usable_cpus() == 2

    quota.write_text("50000 100000\n")
    assert settings.usable_cpus() == 1


def test_usable_cpus_falls_back_without_a_quota(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    quota = tmp_path / "cpu.max"
    quota.write_text("max 100000\n")
    monkeypatch.setattr(settings, "_CGROUP_CPU_MAX", str(quota))
    monkeypatch.setattr(settings.os, "process_cpu_count", lambda: 6, raising=False)
    assert settings.usable_cpus() == 6

    monkeypatch.setattr(settings, "_CGROUP_CPU_MAX", str(tmp_path / "missing"))
    assert settings.usable_cpus() == 6


def test_prep_threads_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DOWNSHIFT_PREP_THREADS", "7")
    importlib.reload(settings)
    assert settings.PREP_THREADS == 7


def test_execution_defaults_to_threadpool(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DOWNSHIFT_EXECUTION", raising=False)
    assert importlib.reload(settings).EXECUTION == "threadpool"


def test_execution_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DOWNSHIFT_EXECUTION", "inline")
    assert importlib.reload(settings).EXECUTION == "inline"


def test_result_cache_rows_is_unset_by_default_and_read_from_the_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DOWNSHIFT_RESULT_CACHE_ROWS", raising=False)
    assert importlib.reload(settings).RESULT_CACHE_ROWS is None

    monkeypatch.setenv("DOWNSHIFT_RESULT_CACHE_ROWS", "250")
    assert importlib.reload(settings).RESULT_CACHE_ROWS == 250


def test_the_body_and_timeout_defaults():
    assert settings.DEFAULT_MAX_BODY_BYTES == 32 * 1024 * 1024
    assert settings.DEFAULT_REQUEST_TIMEOUT == 30.0


def _serve_options_in_a_child(**env: str) -> dict:
    """ServeOptions() as a new interpreter sees it. Downshift reads its defaults from settings
    when downshift.serve.options is first imported. A reload in the same process cannot show
    them."""
    code = (
        "import json; from downshift.serve.options import ServeOptions; o = ServeOptions(); "
        "print(json.dumps({'max_body_bytes': o.max_body_bytes, 'request_timeout': o.request_timeout}))"
    )
    child_env = {k: v for k, v in subprocess_env().items() if not k.startswith("DOWNSHIFT_")}
    child_env.update(env)
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=child_env
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_serve_options_pick_up_the_new_defaults():
    assert _serve_options_in_a_child() == {
        "max_body_bytes": 32 * 1024 * 1024,
        "request_timeout": 30.0,
    }


def test_the_environment_overrides_reach_serve_options():
    options = _serve_options_in_a_child(
        DOWNSHIFT_MAX_BODY_BYTES="4096", DOWNSHIFT_REQUEST_TIMEOUT="2.5"
    )
    assert options == {"max_body_bytes": 4096, "request_timeout": 2.5}


def test_request_id_header_defaults_to_x_request_id(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DOWNSHIFT_REQUEST_ID_HEADER", raising=False)
    assert importlib.reload(settings).REQUEST_ID_HEADER == "X-Request-Id"


@pytest.mark.parametrize(
    ("raw", "expected"), [("X-Correlation-Id", "X-Correlation-Id"), ("", "X-Request-Id")]
)
def test_request_id_header_is_read_from_the_env(
    monkeypatch: pytest.MonkeyPatch, raw, expected
) -> None:
    monkeypatch.setenv("DOWNSHIFT_REQUEST_ID_HEADER", raw)
    assert importlib.reload(settings).REQUEST_ID_HEADER == expected


def test_a_bad_request_id_header_name_says_what_to_use(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DOWNSHIFT_REQUEST_ID_HEADER", "X Request Id")
    with pytest.raises(ValueError, match="not a valid HTTP header name.*X-Correlation-Id"):
        importlib.reload(settings)


def test_server_timing_is_off_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DOWNSHIFT_SERVER_TIMING", raising=False)
    assert importlib.reload(settings).SERVER_TIMING is False


@pytest.mark.parametrize(
    ("raw", "expected"), [("1", True), ("On", True), ("0", False), ("no", False)]
)
def test_server_timing_is_read_from_the_env(monkeypatch: pytest.MonkeyPatch, raw, expected) -> None:
    monkeypatch.setenv("DOWNSHIFT_SERVER_TIMING", raw)
    assert importlib.reload(settings).SERVER_TIMING is expected


def test_a_bad_boolean_env_var_raises_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DOWNSHIFT_SERVER_TIMING", "maybe")
    with pytest.raises(ValueError, match="DOWNSHIFT_SERVER_TIMING.*maybe.*boolean"):
        importlib.reload(settings)


def test_cache_header_defaults_and_reads_the_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DOWNSHIFT_CACHE_HEADER", raising=False)
    assert importlib.reload(settings).CACHE_HEADER == "X-Downshift-Cache"
    monkeypatch.setenv("DOWNSHIFT_CACHE_HEADER", "X-Result-Cache")
    assert importlib.reload(settings).CACHE_HEADER == "X-Result-Cache"
