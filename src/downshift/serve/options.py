"""Serving options and the backend choice: no torch import, so the CLI can parse `--backend`
and `--output-encoding` without paying torch's import cost just to print `--help`.
"""

from dataclasses import dataclass, field
from enum import StrEnum

from downshift import settings
from downshift.serve.schemas import OutputEncoding

__all__ = ["BackendChoice", "ExecutionChoice", "ServeOptions"]


class BackendChoice(StrEnum):
    """What the caller asked for; "auto" defers to the verdict's recommendation."""

    auto = "auto"
    onnxruntime = "onnxruntime"
    torch = "torch"


class ExecutionChoice(StrEnum):
    """Where a predict's parse, inference and encode run."""

    threadpool = "threadpool"
    inline = "inline"


@dataclass
class ServeOptions:
    """Defaults come from downshift.settings, so a DOWNSHIFT_* environment variable reaches
    app_for() and any other library caller, not just the CLI (which passes its own flag
    values here instead when they were set)."""

    backend: BackendChoice = BackendChoice(settings.BACKEND)
    force_onnx: bool = False  # serve a DEGRADED graph via ORT anyway
    device: str = settings.DEVICE
    warmup: int = settings.WARMUP
    k: int = settings.SAMPLES
    adapter: str | None = None
    dynamic: dict[str, list[int]] | None = None
    # --axis-max: {axis name: largest size to serve}; DOWNSHIFT_AXIS_MAX when not given
    axis_max: dict[str, int] | None = field(default_factory=lambda: dict(settings.AXIS_MAX) or None)
    # --export-cache-dir: where verified exports are saved for the next boot; None = never write
    export_cache_dir: str | None = settings.EXPORT_CACHE_DIR
    intra_op_threads: int = settings.INTRA_OP_THREADS  # ORT SessionOptions; 0 = let ORT choose
    inter_op_threads: int = settings.INTER_OP_THREADS
    output_encoding: OutputEncoding = OutputEncoding(settings.OUTPUT_ENCODING)  # per-call override
    max_input_bytes: int = settings.MAX_INPUT_BYTES  # cap on one decoded base64 tensor input
    max_body_bytes: int = settings.MAX_BODY_BYTES  # cap on the whole request body
    max_concurrency: int = settings.MAX_CONCURRENCY  # inferences allowed to run at once per worker
    execution: ExecutionChoice = ExecutionChoice(settings.EXECUTION)
    prep_threads: int = settings.PREP_THREADS  # request decode/encode threads, not inference
    max_queue: int = settings.MAX_QUEUE  # admitted predicts allowed to wait past max_concurrency
    request_timeout: float = settings.REQUEST_TIMEOUT  # seconds queued before a 503; 0 = no limit
    atol: float | None = None  # None means "by the model's floating dtype"
    rtol: float | None = None
    seed: int = 0  # makes verification samples reproducible
    vary: str | None = None  # pkg.module:fn overriding the adapter's own vary_fn
    pooling: str | None = None  # a PoolingChoice value overriding the repo's embedding recipe
    normalize: bool | None = None  # L2-normalise the embedding; None = whatever the recipe says
