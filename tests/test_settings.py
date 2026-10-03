"""settings.py reads DOWNSHIFT_* env vars at import time, so these tests reload the module
around each case rather than importing the constants once at collection time.
"""

import importlib

import pytest

from downshift import settings


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
        1,
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
    monkeypatch.setenv("DOWNSHIFT_MAX_CONCURRENCY", "4")
    mod = importlib.reload(settings)
    assert mod.HOST == "127.0.0.1"
    assert mod.PORT == 9000
    assert mod.SAMPLES == 16
    assert mod.OUTPUT_ENCODING == "base64"
    assert mod.MAX_INPUT_BYTES == 4096
    assert mod.MAX_BODY_BYTES == 8192
    assert mod.MAX_CONCURRENCY == 4


def test_bad_int_env_var_raises_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DOWNSHIFT_PORT", "not-a-number")
    with pytest.raises(ValueError, match="DOWNSHIFT_PORT.*not-a-number.*integer"):
        importlib.reload(settings)


def test_bad_float_env_var_raises_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DOWNSHIFT_TOL_FLOAT32_ATOL", "not-a-float")
    with pytest.raises(ValueError, match="DOWNSHIFT_TOL_FLOAT32_ATOL.*not-a-float.*float"):
        importlib.reload(settings)


def test_per_dtype_tolerance_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DOWNSHIFT_TOL_FLOAT16_ATOL", "0.05")
    mod = importlib.reload(settings)
    assert mod.TOLERANCES["float16"] == (0.05, 1e-2)
    # Untouched dtypes keep their defaults.
    assert mod.TOLERANCES["float32"] == (1e-4, 1e-3)
