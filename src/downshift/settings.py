"""The defaults for the CLI and library serving, and the DOWNSHIFT_* variables that override
them. A flag or keyword argument wins over the environment, which wins over the default. The
variables are read once, at import.
"""

import math
import os
import re
from collections.abc import Callable, Iterable

# The decoded size that one base64 tensor input can reach. The server passes its own limit.
DEFAULT_MAX_INPUT_BYTES = 256 * 1024 * 1024

# The total size of a request body that the server reads before it parses the JSON.
DEFAULT_MAX_BODY_BYTES = 32 * 1024 * 1024

# The verification samples for check(), export(), intake(), build_verdict() and ServeOptions.k.
DEFAULT_SAMPLES = 8


_CGROUP_CPU_MAX = "/sys/fs/cgroup/cpu.max"


def usable_cpus() -> int:
    """The CPUs this process can use: the cgroup v2 quota, else the affinity mask, else the
    machine's count."""
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


def _env_bool(name: str, default: bool) -> bool:
    """1, true, yes or on; 0, false, no or off (any case)."""

    def cast(raw: str) -> bool:
        value = raw.strip().lower()
        if value in ("1", "true", "yes", "on"):
            return True
        if value in ("0", "false", "no", "off"):
            return False
        raise ValueError(raw)

    return _env_cast(name, default, cast, "boolean (1 or 0)")


def _env_choice(name: str, default: str, choices: tuple[str, ...], type_name: str) -> str:
    """One of `choices`, so a typo in the environment fails at startup with its own name."""

    def cast(raw: str) -> str:
        if raw not in choices:
            raise ValueError(raw)
        return raw

    return _env_cast(name, default, cast, f"{type_name} (one of {', '.join(choices)})")


# The characters of an HTTP header name (the RFC 9110 token).
_HEADER_NAME = re.compile(r"[!#$%&'*+.^_`|~0-9A-Za-z-]+")


def _env_header(name: str, default: str) -> str:
    """A header name from the environment. Unset or empty means the default."""
    raw = os.environ.get(name)
    if not raw:
        return default
    if not _HEADER_NAME.fullmatch(raw):
        raise ValueError(
            f"{name}={raw!r} is not a valid HTTP header name. Use letters, digits and '-' "
            "only, for example X-Correlation-Id"
        )
    return raw


def parse_axis_max(items: Iterable[str]) -> dict[str, int]:
    """ "seq=4096,num_nodes=500" (or one NAME=N per item) -> {name: N}, else ValueError."""
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


HOST = os.environ.get("DOWNSHIFT_HOST", "127.0.0.1")
PORT = _env_int("DOWNSHIFT_PORT", 8000)
DEVICE = os.environ.get("DOWNSHIFT_DEVICE", "auto")
BACKEND = _env_choice("DOWNSHIFT_BACKEND", "auto", ("auto", "onnxruntime", "torch"), "backend")
WARMUP = _env_int("DOWNSHIFT_WARMUP", 3)
SAMPLES = _env_int("DOWNSHIFT_SAMPLES", DEFAULT_SAMPLES)

# 0 means "the default of the backend". For ONNX Runtime, this is the physical cores for
# intra-op and 1 for inter-op. For torch, it is the thread count of torch. Intra-op applies
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
AXIS_MAX: dict[str, int] = _env_cast(
    "DOWNSHIFT_AXIS_MAX", {}, lambda raw: parse_axis_max([raw]), "list of NAME=N"
)

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

# The number of rows that the result cache keeps in each worker process. A row that arrives a
# second time then skips the model. The cache counts rows and not bytes. Set a number that fits
# the size of one output row. 0 turns the cache off. If the variable is unset, downshift uses
# DEFAULT_RESULT_CACHE_ROWS for a model with independent rows and no cache for other models.
DEFAULT_RESULT_CACHE_ROWS = 100
RESULT_CACHE_ROWS: int | None = (
    _env_int("DOWNSHIFT_RESULT_CACHE_ROWS", DEFAULT_RESULT_CACHE_ROWS)
    if "DOWNSHIFT_RESULT_CACHE_ROWS" in os.environ
    else None
)

# The number of seconds that an admitted predict can wait without a start. After this time, it
# gets a 503 and no inference. 0 means no limit.
DEFAULT_REQUEST_TIMEOUT = 30.0
REQUEST_TIMEOUT = _env_float("DOWNSHIFT_REQUEST_TIMEOUT", DEFAULT_REQUEST_TIMEOUT)

# A value of more than 1 starts that number of uvicorn worker processes. The parent exports and
# verifies one time. Each worker loads that result and warms up. Memory and startup time
# increase with this number.
WORKERS = _env_int("DOWNSHIFT_WORKERS", 1)

# If it is set, every route except /health and /ready needs "Authorization: Bearer <this value>".
# None (unset) means no authentication, with a startup warning that says so.
API_KEY = os.environ.get("DOWNSHIFT_SERVER_API_KEY")

# The header that carries the request ID: the server reads it from a request and sets it on
# each response.
REQUEST_ID_HEADER = _env_header("DOWNSHIFT_REQUEST_ID_HEADER", "X-Request-Id")

# The header of a /predict response that says hit, partial or miss for the result cache.
CACHE_HEADER = _env_header("DOWNSHIFT_CACHE_HEADER", "X-Downshift-Cache")

# Add the Server-Timing header (the milliseconds of each stage) to each predict response. It is
# off by default: the timings tell a client how the server works, and a fast cache hit tells it
# that another client sent the same row. The access log has the timings in both cases.
SERVER_TIMING = _env_bool("DOWNSHIFT_SERVER_TIMING", False)

# HTTPS. If SSL_CERTFILE is set, `serve` uses TLS. The key can be in the cert file, or in its
# own file. The password of an encrypted key is an environment variable only, so that it does
# not show in the process list.
SSL_CERTFILE = os.environ.get("DOWNSHIFT_SSL_CERTFILE") or None
SSL_KEYFILE = os.environ.get("DOWNSHIFT_SSL_KEYFILE") or None
SSL_KEYFILE_PASSWORD = os.environ.get("DOWNSHIFT_SSL_KEYFILE_PASSWORD") or None

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
