"""Defaults for the CLI and for library serving (ServeOptions and app_for()). The DOWNSHIFT_*
environment variables override them.

Each value here is only a fallback. A CLI flag or a function keyword argument has priority over
the environment. The environment has priority over the default below. Downshift reads the
variables one time, at import. To change a variable, restart the process (or the CLI call)
that reads it.
"""

import math
import os
from collections.abc import Callable, Iterable

# The decoded size that one base64 tensor input can reach. The server passes its own limit.
DEFAULT_MAX_INPUT_BYTES = 256 * 1024 * 1024

# The total size of a request body that the server reads before it parses the JSON.
DEFAULT_MAX_BODY_BYTES = 32 * 1024 * 1024

# The verification samples for check(), export(), intake(), build_verdict() and ServeOptions.k.
DEFAULT_SAMPLES = 8


_CGROUP_CPU_MAX = "/sys/fs/cgroup/cpu.max"


def usable_cpus() -> int:
    """The CPUs that this process can use. The order is: a cgroup v2 quota (the --cpus of a
    container), then the scheduler affinity mask, then the logical count of the machine."""
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


def _env_choice(name: str, default: str, choices: tuple[str, ...], type_name: str) -> str:
    """One of `choices`, so a typo in the environment fails at startup with its own name."""

    def cast(raw: str) -> str:
        if raw not in choices:
            raise ValueError(raw)
        return raw

    return _env_cast(name, default, cast, f"{type_name} (one of {', '.join(choices)})")


def parse_axis_max(items: Iterable[str]) -> dict[str, int]:
    """Turn "seq=4096,num_nodes=500" (or one NAME=N for each item, as --axis-max repeats) into
    {name: N}. Any other input raises ValueError, so the CLI reports a usage error."""
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
BACKEND = _env_choice("DOWNSHIFT_BACKEND", "auto", ("auto", "onnxruntime", "torch"), "backend")
WARMUP = _env_int("DOWNSHIFT_WARMUP", 3)
SAMPLES = _env_int("DOWNSHIFT_SAMPLES", DEFAULT_SAMPLES)

# 0 means "the default of the backend". For ONNX Runtime, this is the physical cores for
# intra-op and 1 for inter-op. For torch, it is the own thread count of torch. Intra-op applies
# to both backends. Inter-op applies to ONNX Runtime only.
INTRA_OP_THREADS = _env_int("DOWNSHIFT_INTRA_OP_THREADS", 0)
INTER_OP_THREADS = _env_int("DOWNSHIFT_INTER_OP_THREADS", 0)

# The default encoding of response tensors ("json" lists or "base64" buffers). The
# output_encoding field of a request overrides it.
OUTPUT_ENCODING = _env_choice(
    "DOWNSHIFT_OUTPUT_ENCODING", "json", ("json", "base64"), "output encoding"
)

# The largest decoded size that downshift accepts for one base64 tensor input. A larger one gets a 400.
MAX_INPUT_BYTES = _env_int("DOWNSHIFT_MAX_INPUT_BYTES", DEFAULT_MAX_INPUT_BYTES)

# The largest request body that the server reads before it parses the JSON. A larger one gets a 413.
MAX_BODY_BYTES = _env_int("DOWNSHIFT_MAX_BODY_BYTES", DEFAULT_MAX_BODY_BYTES)

# The number of inferences that can run at the same time in each worker process. Measured with
# 4 to 8 concurrent clients: all-MiniLM-L6-v2 (batch 8) 52 req/s at 1 and 134 at 4.
# Qwen3-Embedding-0.6B (one query) 7.2 at 1 and 14.6 at 4. A single request is not slower.
# One inference does not usually fill all cores. Each extra concurrent inference holds its own
# activation memory. Set 1 to limit it.
MAX_CONCURRENCY = _env_int("DOWNSHIFT_MAX_CONCURRENCY", 4)

# "threadpool" runs the parse, the inference and the encoding of each request on worker
# threads. "inline" runs small JSON bodies on the event loop. This helps only very fast models.
EXECUTION = _env_choice(
    "DOWNSHIFT_EXECUTION", "threadpool", ("threadpool", "inline"), "execution mode"
)

# The largest size to serve for each named axis ("seq=4096,num_nodes=500"). The names are the
# names that the boot banner and /schema list. If it is empty, each axis keeps the bound that
# the adapter exported.
AXIS_MAX = _env_axis_max("DOWNSHIFT_AXIS_MAX", {})

# A directory where downshift saves verified exports and reuses them at the next boot (`serve`
# and `export`). If it is unset, downshift never writes to disk. The in-process memo is then the
# whole cache.
EXPORT_CACHE_DIR = os.environ.get("DOWNSHIFT_EXPORT_CACHE_DIR") or None

# The threads that convert request bodies to arrays. They are separate from the inference
# threads, so a slow conversion never holds an inference slot.
PREP_THREADS = _env_int("DOWNSHIFT_PREP_THREADS", min(4, usable_cpus()))

# The number of predicts that can wait beyond max_concurrency. After that, a new predict gets a
# fast 503 and does not join the queue.
DEFAULT_MAX_QUEUE = 64
MAX_QUEUE = _env_int("DOWNSHIFT_MAX_QUEUE", DEFAULT_MAX_QUEUE)

# The number of seconds that an admitted predict can wait without a start. After this time, it
# gets a 503 and no inference. 0 means no limit.
DEFAULT_REQUEST_TIMEOUT = 30.0
REQUEST_TIMEOUT = _env_float("DOWNSHIFT_REQUEST_TIMEOUT", DEFAULT_REQUEST_TIMEOUT)

# A value of more than 1 starts that number of uvicorn worker processes. Each one loads,
# exports and warms up the model on its own. Memory and startup time increase with this number.
WORKERS = _env_int("DOWNSHIFT_WORKERS", 1)

# If it is set, every route except /health and /ready needs "Authorization: Bearer <this value>".
# None (unset) means no authentication, with a startup warning that says so.
API_KEY = os.environ.get("DOWNSHIFT_SERVER_API_KEY")

# (atol, rtol) for the narrowest float dtype in the numerics verification. You can loosen one
# value and not change the others. Example: DOWNSHIFT_TOL_FLOAT16_ATOL=0.05 for an fp16 model
# with more noise.
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
