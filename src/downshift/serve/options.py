"""The serving options and the backend choice. This module has no torch import. The CLI can
therefore parse `--backend` and `--output-encoding` without the import cost of torch, only to
print `--help`.
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
    """The defaults come from downshift.settings. A DOWNSHIFT_* environment variable therefore
    reaches app_for() and all other library callers, and not only the CLI. (The CLI passes its
    own flag values here instead, when they are set.)"""

    backend: BackendChoice = BackendChoice(settings.BACKEND)
    force_onnx: bool = False  # serve a DEGRADED graph through ORT
    device: str = settings.DEVICE
    warmup: int = settings.WARMUP
    k: int = settings.SAMPLES
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
    request_timeout: float = settings.REQUEST_TIMEOUT  # queue seconds. 0 = none
    atol: float | None = None  # None means "by the floating dtype of the model"
    rtol: float | None = None
    seed: int = 0  # makes the verification samples reproducible
    vary: str | None = None  # pkg.module:fn that replaces the vary_fn of the adapter
    pooling: str | None = None  # a PoolingChoice value. It overrides the repo recipe
    normalize: bool | None = None  # L2-normalise the embedding. None = the setting of the recipe
