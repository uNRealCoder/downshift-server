"""Serving options and the backend choice: no torch import, so the CLI can parse `--backend`
and `--output-encoding` without paying torch's import cost just to print `--help`.
"""

from dataclasses import dataclass
from enum import StrEnum

from downshift.serve.schemas import OutputEncoding
from downshift.settings import (
    DEFAULT_MAX_BODY_BYTES,
    DEFAULT_MAX_INPUT_BYTES,
    DEFAULT_MAX_QUEUE,
    DEFAULT_REQUEST_TIMEOUT,
)


class BackendChoice(StrEnum):
    """What the caller asked for; "auto" defers to the verdict's recommendation."""

    auto = "auto"
    onnxruntime = "onnxruntime"
    torch = "torch"


@dataclass
class ServeOptions:
    backend: BackendChoice = BackendChoice.auto
    force_onnx: bool = False  # serve a DEGRADED graph via ORT anyway
    device: str = "auto"
    warmup: int = 3
    k: int = 8
    adapter: str | None = None
    dynamic: dict[str, list[int]] | None = None
    intra_op_threads: int = 0  # ORT SessionOptions; 0 = let ONNX Runtime choose
    inter_op_threads: int = 0
    output_encoding: OutputEncoding = OutputEncoding.json  # requests may override per call
    max_input_bytes: int = DEFAULT_MAX_INPUT_BYTES  # cap on one decoded base64 tensor input
    max_body_bytes: int = DEFAULT_MAX_BODY_BYTES  # cap on the whole request body
    max_concurrency: int = 1  # inferences allowed to run at once per worker process
    max_queue: int = DEFAULT_MAX_QUEUE  # admitted predicts allowed to wait past max_concurrency
    request_timeout: float = DEFAULT_REQUEST_TIMEOUT  # seconds queued before a 503; 0 = no limit
    atol: float | None = None  # None means "by the model's floating dtype"
    rtol: float | None = None
    seed: int = 0  # makes verification samples reproducible
    vary: str | None = None  # pkg.module:fn overriding the adapter's own vary_fn
