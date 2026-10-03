"""Defaults for the CLI and for library serving (ServeOptions, app_for()), overridable via
DOWNSHIFT_* environment variables.

Everything here is just a fallback: a CLI flag or a function kwarg always wins over the
environment, and the environment always wins over the default below. Read once at import
time, so setting an env var means restarting the process (or CLI invocation) that reads it.
"""

import math
import os
from collections.abc import Callable, Iterable

# Decoded size a single base64 tensor input may reach; the server passes its own limit.
DEFAULT_MAX_INPUT_BYTES = 256 * 1024 * 1024

# Total request body size the server will read before parsing it as JSON.
DEFAULT_MAX_BODY_BYTES = 32 * 1024 * 1024

# Verification samples for check()/export()/intake()/build_verdict() and ServeOptions.k.
DEFAULT_SAMPLES = 8


_CGROUP_CPU_MAX = "/sys/fs/cgroup/cpu.max"


def usable_cpus() -> int:
    """CPUs this process may actually use: a cgroup v2 quota (a container's --cpus) first,
    then the scheduler affinity mask, then the machine's logical count."""
    try:
        with open(_CGROUP_CPU_MAX) as f:
            quota, period = f.read().split()[:2]
        if quota != "max":
            return max(1, math.ceil(int(quota) / int(period)))
    except (OSError, ValueError, ZeroDivisionError):
        pass
    process_cpu_count = getattr(os, "process_cpu_count", None)
    if process_cpu_count is not None:
        return process_cpu_count() or 1
    return os.cpu_count() or 1


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_cast[T](name: str, default: T, cast: Callable[[str], T], type_name: str) -> T:
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


def _execution(raw: str) -> str:
    if raw not in ("threadpool", "inline"):
        raise ValueError(raw)
    return raw


def parse_axis_max(items: Iterable[str]) -> dict[str, int]:
    """ "seq=4096,num_nodes=500" (or one NAME=N per item, as --axis-max repeats) into
    {name: N}. ValueError on anything else, so the CLI reports it as a usage error."""
    result: dict[str, int] = {}
    for item in items:
        for entry in filter(None, (e.strip() for e in item.split(","))):
            name, _, value = entry.partition("=")
            if not name.strip() or not value.strip().isdecimal() or int(value) < 1:
                raise ValueError(
                    f"bad axis max {entry!r}; expected NAME=N with N a positive integer"
                )
            result[name.strip()] = int(value)
    return result


def _env_axis_max(name: str, default: dict[str, int]) -> dict[str, int]:
    return _env_cast(name, default, lambda raw: parse_axis_max([raw]), "list of NAME=N")


HOST = _env_str("DOWNSHIFT_HOST", "127.0.0.1")
PORT = _env_int("DOWNSHIFT_PORT", 8000)
DEVICE = _env_str("DOWNSHIFT_DEVICE", "auto")
BACKEND = _env_str("DOWNSHIFT_BACKEND", "auto")
WARMUP = _env_int("DOWNSHIFT_WARMUP", 3)
SAMPLES = _env_int("DOWNSHIFT_SAMPLES", DEFAULT_SAMPLES)

# 0 means "let ONNX Runtime pick" (its own default: physical cores for intra-op, 1 for inter-op).
INTRA_OP_THREADS = _env_int("DOWNSHIFT_INTRA_OP_THREADS", 0)
INTER_OP_THREADS = _env_int("DOWNSHIFT_INTER_OP_THREADS", 0)

# Default encoding of response tensors ("json" lists or "base64" buffers); a request's
# output_encoding field overrides it.
OUTPUT_ENCODING = _env_str("DOWNSHIFT_OUTPUT_ENCODING", "json")

# Largest decoded size accepted for one base64 tensor input; bigger ones get a 400.
MAX_INPUT_BYTES = _env_int("DOWNSHIFT_MAX_INPUT_BYTES", DEFAULT_MAX_INPUT_BYTES)

# Largest request body the server will read before parsing it as JSON; bigger ones get a 413.
MAX_BODY_BYTES = _env_int("DOWNSHIFT_MAX_BODY_BYTES", DEFAULT_MAX_BODY_BYTES)

# Inferences allowed to run at once per worker process. Measured on all-MiniLM-L6-v2, batch 8,
# 8 concurrent clients: 52 req/s at 1, 134 req/s at 4. Small encoders usually gain from 2-4,
# because one small inference does not fill every core; a large model that already does gains
# nothing. Each extra concurrent inference holds its own activation memory.
MAX_CONCURRENCY = _env_int("DOWNSHIFT_MAX_CONCURRENCY", 1)

# "threadpool" runs every request's parse, inference and encode on worker threads; "inline"
# runs small JSON bodies on the event loop itself, which only pays off for very fast models.
EXECUTION = _env_cast("DOWNSHIFT_EXECUTION", "threadpool", _execution, "execution mode")

# Largest size to serve per named axis ("seq=4096,num_nodes=500"); the names are the ones the
# boot banner and /schema list. Empty means each axis keeps the bound the adapter exported.
AXIS_MAX = _env_axis_max("DOWNSHIFT_AXIS_MAX", {})

# A directory verified exports are saved in and reused from on the next boot (`serve` and
# `export`). Unset means nothing is ever written to disk: the in-process memo is the whole cache.
EXPORT_CACHE_DIR = os.environ.get("DOWNSHIFT_EXPORT_CACHE_DIR") or None

# Threads converting request bodies to arrays and responses to bytes, apart from the
# inference threads, so a slow encode never holds an inference slot.
PREP_THREADS = _env_int("DOWNSHIFT_PREP_THREADS", min(4, usable_cpus()))

# Predicts allowed to wait past max_concurrency before a new one gets a fast 503 instead of
# joining the queue.
DEFAULT_MAX_QUEUE = 64
MAX_QUEUE = _env_int("DOWNSHIFT_MAX_QUEUE", DEFAULT_MAX_QUEUE)

# Seconds a predict may wait admitted-but-not-running before it gets a 503 instead of an
# inference; 0 means no limit.
DEFAULT_REQUEST_TIMEOUT = 30.0
REQUEST_TIMEOUT = _env_float("DOWNSHIFT_REQUEST_TIMEOUT", DEFAULT_REQUEST_TIMEOUT)

# >1 spawns that many uvicorn worker processes, each independently loading/exporting/warming
# the model, so memory and startup time scale with this number.
WORKERS = _env_int("DOWNSHIFT_WORKERS", 1)

# When set, every route but /health and /ready requires "Authorization: Bearer <this value>"
# (U7). None (unset) means unauthenticated, with a startup warning saying so.
API_KEY = os.environ.get("DOWNSHIFT_SERVER_API_KEY")

# (atol, rtol) by the widest float dtype involved in numerics verification. Loosen one
# without touching the rest, e.g. DOWNSHIFT_TOL_FLOAT16_ATOL=0.05 for a noisier fp16 model.
_TOLERANCE_DEFAULTS = {
    "float32": (1e-4, 1e-3),
    "float64": (1e-6, 1e-5),
    "float16": (1e-2, 1e-2),
    "bfloat16": (5e-2, 5e-2),
}

TOLERANCES: dict[str, tuple[float, float]] = {
    name: (
        _env_float(f"DOWNSHIFT_TOL_{name.upper()}_ATOL", atol),
        _env_float(f"DOWNSHIFT_TOL_{name.upper()}_RTOL", rtol),
    )
    for name, (atol, rtol) in _TOLERANCE_DEFAULTS.items()
}
