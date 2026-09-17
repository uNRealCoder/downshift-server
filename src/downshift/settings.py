"""Defaults for the CLI and library, overridable via DOWNSHIFT_* environment variables.

Everything here is just a fallback: a CLI flag or a function kwarg always wins over the
environment, and the environment always wins over the default below. Read once at import
time, so setting an env var means restarting the process (or CLI invocation) that reads it.
"""

import os
from collections.abc import Callable
from typing import TypeVar

import torch

from downshift.serve.schemas import DEFAULT_MAX_INPUT_BYTES

T = TypeVar("T")


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_cast(name: str, default: T, cast: Callable[[str], T], type_name: str) -> T:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return cast(raw)
    except ValueError:
        raise ValueError(f"{name}={raw!r} is not a valid {type_name}") from None


def _env_int(name: str, default: int) -> int:
    return _env_cast(name, default, int, "integer")


def _env_float(name: str, default: float) -> float:
    return _env_cast(name, default, float, "float")


HOST = _env_str("DOWNSHIFT_HOST", "127.0.0.1")
PORT = _env_int("DOWNSHIFT_PORT", 8000)
DEVICE = _env_str("DOWNSHIFT_DEVICE", "auto")
BACKEND = _env_str("DOWNSHIFT_BACKEND", "auto")
WARMUP = _env_int("DOWNSHIFT_WARMUP", 3)
SAMPLES = _env_int("DOWNSHIFT_SAMPLES", 8)

# 0 means "let ONNX Runtime pick" (its own default: physical cores for intra-op, 1 for inter-op).
INTRA_OP_THREADS = _env_int("DOWNSHIFT_INTRA_OP_THREADS", 0)
INTER_OP_THREADS = _env_int("DOWNSHIFT_INTER_OP_THREADS", 0)

# Default encoding of response tensors ("json" lists or "base64" buffers); a request's
# output_encoding field overrides it.
OUTPUT_ENCODING = _env_str("DOWNSHIFT_OUTPUT_ENCODING", "json")

# Largest decoded size accepted for one base64 tensor input; bigger ones get a 400.
MAX_INPUT_BYTES = _env_int("DOWNSHIFT_MAX_INPUT_BYTES", DEFAULT_MAX_INPUT_BYTES)

# >1 spawns that many uvicorn worker processes, each independently loading/exporting/warming
# the model, so memory and startup time scale with this number.
WORKERS = _env_int("DOWNSHIFT_WORKERS", 1)

# (atol, rtol) by the widest float dtype involved in numerics verification. Loosen one
# without touching the rest, e.g. DOWNSHIFT_TOL_FLOAT16_ATOL=0.05 for a noisier fp16 model.
_TOLERANCE_DEFAULTS = {
    "float32": (1e-4, 1e-3),
    "float64": (1e-6, 1e-5),
    "float16": (1e-2, 1e-2),
    "bfloat16": (5e-2, 5e-2),
}

TOLERANCES: dict[torch.dtype, tuple[float, float]] = {
    getattr(torch, name): (
        _env_float(f"DOWNSHIFT_TOL_{name.upper()}_ATOL", atol),
        _env_float(f"DOWNSHIFT_TOL_{name.upper()}_RTOL", rtol),
    )
    for name, (atol, rtol) in _TOLERANCE_DEFAULTS.items()
}
