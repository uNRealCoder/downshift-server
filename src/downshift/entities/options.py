"""The serve options as data: the choices (backend, execution, output encoding) and ServeOptions.

ServeOptions is the one record of all `serve` settings. The CLI fills it from its flags, `app_for()`
fills it from keyword arguments, and both start from the defaults in settings.py.
Used by cli/main.py, cli/options.py, serve/engine.py, serve/reuse.py, serve/app.py and core/memo.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from downshift import settings


# What the caller asked for with --backend. `auto` is never a running backend:
# serve/engine.py `choose_backend` turns it into a BackendName.
class BackendChoice(StrEnum):
    """What the caller asked for; "auto" defers to the verdict's recommendation."""

    auto = "auto"
    onnxruntime = "onnxruntime"
    torch = "torch"


# --execution: run the parse, the inference and the encoding on pool threads (threadpool),
# or on the event loop for a small JSON body (inline). serve/app.py reads it.
class ExecutionChoice(StrEnum):
    """Where a predict's parse, inference and encode run."""

    threadpool = "threadpool"
    inline = "inline"


# The default encoding of response tensors (--output-encoding). A request can override it.
class OutputEncoding(StrEnum):
    """How response tensors are written: nested lists, or {data, dtype, shape} base64 dicts."""

    json = "json"
    base64 = "base64"


# What the `output_encoding` field of a request can name: OutputEncoding, and safetensors.
class RequestOutputEncoding(StrEnum):
    """What the `output_encoding` of a request can name: OutputEncoding plus safetensors.
    Safetensors is a format for the whole response (an `Accept` header can also ask for it).
    It is therefore not a default of the server."""

    json = "json"
    base64 = "base64"
    safetensors = "safetensors"


# All `serve` settings in one record. Add a field here for each new serve option. cli/main.py
# copies each field from the `serve_cmd` parameter with the same name.
@dataclass
class ServeOptions:
    """The defaults come from downshift.settings. A DOWNSHIFT_* environment variable therefore
    reaches app_for() and all other library callers, and not only the CLI. (The CLI passes its
    own flag values here instead, when they are set.)"""

    backend: BackendChoice = BackendChoice(settings.BACKEND)
    force_onnx: bool = False  # serve a DEGRADED graph through ORT
    device: str = settings.DEVICE
    warmup: int = settings.WARMUP
    samples: int = settings.SAMPLES
    adapter: str | None = None
    dynamic: dict[str, list[int]] | None = None
    # --axis-max: {axis name: largest size to serve}. DOWNSHIFT_AXIS_MAX, if you do not give it
    axis_max: dict[str, int] | None = field(default_factory=lambda: dict(settings.AXIS_MAX) or None)
    # --export-cache-dir: where downshift saves verified exports for the next boot. None = never write
    export_cache_dir: str | None = settings.EXPORT_CACHE_DIR
    intra_op_threads: int = (
        settings.INTRA_OP_THREADS
    )  # ORT SessionOptions or torch.set_num_threads. 0 = the default of the backend
    inter_op_threads: int = settings.INTER_OP_THREADS
    output_encoding: OutputEncoding = OutputEncoding(settings.OUTPUT_ENCODING)
    max_input_bytes: int = settings.MAX_INPUT_BYTES  # limit on one decoded base64 tensor input
    max_body_bytes: int = settings.MAX_BODY_BYTES  # limit on the whole request body
    max_concurrency: int = settings.MAX_CONCURRENCY  # inferences at the same time, per worker
    execution: ExecutionChoice = ExecutionChoice(settings.EXECUTION)
    prep_threads: int = settings.PREP_THREADS  # threads that parse and convert requests
    max_queue: int = settings.MAX_QUEUE  # admitted predicts that can wait beyond max_concurrency
    request_timeout: float = settings.REQUEST_TIMEOUT  # seconds for the whole request. 0 = none
    # Rows in the result cache. None = the default, but only for a model with independent rows. 0 = off
    result_cache_rows: int | None = settings.RESULT_CACHE_ROWS
    server_timing: bool = settings.SERVER_TIMING  # add the Server-Timing header to predicts
    metrics: bool = settings.METRICS  # count requests and serve GET /metrics
    served_model_name: str | None = settings.SERVED_MODEL_NAME  # None = from the model source
    matryoshka: bool = settings.MATRYOSHKA  # /v1/embeddings takes `dimensions`
    tf32: bool = settings.TF32  # allow TF32 math on the GPU
    atol: float | None = None  # None means "by the floating dtype of the model"
    rtol: float | None = None
    seed: int = 0  # makes the verification samples reproducible
    vary: str | None = None  # pkg.module:fn that replaces the vary_fn of the adapter
    pooling: str | None = None  # a PoolingChoice value. It overrides the repo recipe
    normalize: bool | None = None  # L2-normalise the embedding. None = the setting of the recipe
