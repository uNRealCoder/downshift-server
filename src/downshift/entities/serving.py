"""ServingState: everything that one server process holds, and the two small records next to it.

This module is the one that is not light: ServingState starts the thread pools and the result
cache, so it imports the backends and the cache. Used by serve/engine.py (it builds the state)
and by every route and renderer that reads it.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from functools import cached_property

from downshift.entities.adapters import EmbeddingRecipe, TextIO
from downshift.entities.axes import AxisBounds
from downshift.entities.backend import IOSpec
from downshift.entities.options import BackendChoice, ServeOptions
from downshift.entities.verdict import ExportVerdict, Status
from downshift.serve.backends import Backend
from downshift.serve.result_cache import ResultCache
from downshift.settings import DEFAULT_RESULT_CACHE_ROWS
from downshift.sources import UNKNOWN_SOURCE, display_source


# The result of the serving check on a GPU: the serving backend against the CPU result.
# serve/numerics.py makes it. /metadata shows it.
@dataclass
class ServingNumerics:
    provider: str  # the execution provider, or "torch"
    device: str
    tf32: bool
    samples: int  # the samples that the check replayed
    samples_total: int  # the samples of the verification. Fewer replayed: a size limit
    max_abs_err: float
    min_cosine: float | None  # for an embedding model only
    ok: bool
    atol: float
    rtol: float

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        cosine = "" if self.min_cosine is None else f", min cosine {self.min_cosine:.6f}"
        of = "" if self.samples_total == self.samples else f" of {self.samples_total}"
        return f"max abs err {self.max_abs_err:.1e} over {self.samples}{of} samples{cosine}"


# The count and the mean time of the warm-up inferences. serve/engine.py `warmup` makes it.
@dataclass
class WarmupStats:
    count: int
    mean_ms: float
    synthesized: bool  # True if there was no example input, so warmup() made one


# Everything that one server process holds: the verdict, the backend, the options, the
# thread pools, the result cache and the boot timings. serve/engine.py builds it at boot.
# Every route, the banner and /metadata read it.
@dataclass
class ServingState:
    source: str
    verdict: ExportVerdict
    backend: Backend
    input_names: tuple[str, ...]
    options: ServeOptions
    example_inputs: tuple | None = None
    ready: bool = False
    notes: list[str] = field(default_factory=list)  # for the banner
    # The accepted form that `source` named (downshift.sources). /schema and the banner show it.
    source_kind: str = UNKNOWN_SOURCE
    # "memory" or "disk" if this boot reused an earlier export.
    reused: str | None = None
    # The HF repo directory for the tokenizer, pooling and labels: `source` itself, or
    # --tokenizer-from. None skips them.
    hf_source: str | None = None
    text: TextIO | None = None  # lets /predict take `text`
    embedding: EmbeddingRecipe | None = None  # the pooling in the graph, for /schema
    # Lets /predict range-check input_ids: ORT wraps negative indices.
    vocab_size: int | None = None
    warmup_stats: WarmupStats | None = None
    # The serving check on a GPU. None if the backend serves on the CPU.
    serving_numerics: ServingNumerics | None = None
    # Seconds for each Phase that ran, for the Boot row and /metadata.
    timings: dict[str, float] = field(default_factory=dict)
    # For each input and dynamic axis: the (name, min, max) that the export traced. Empty for
    # a bare .onnx file.
    axis_bounds: AxisBounds = field(default_factory=dict, repr=False)
    # Inferences run on `executor` (max_concurrency threads). Parsing, conversion and encoding
    # run on `prep_executor`, so they never hold an inference slot. The event loop does
    # neither, so /health and /ready never wait.
    executor: ThreadPoolExecutor = field(init=False, repr=False, compare=False)
    prep_executor: ThreadPoolExecutor = field(init=False, repr=False, compare=False)
    # Admitted predicts that have not finished. _admission_lock guards it.
    in_flight: int = field(init=False, repr=False, compare=False, default=0)
    # None if the cache is off. Then `cache_off_reason` says why, unless the user turned it off.
    result_cache: ResultCache | None = field(init=False, repr=False, compare=False, default=None)
    cache_off_reason: str | None = field(init=False, repr=False, compare=False, default=None)
    _admission_lock: threading.Lock = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self._set_up_result_cache()
        self.executor = ThreadPoolExecutor(max_workers=self.options.max_concurrency)
        self.prep_executor = ThreadPoolExecutor(
            max_workers=self.options.prep_threads, thread_name_prefix="downshift-prep"
        )
        self._admission_lock = threading.Lock()

    def _set_up_result_cache(self) -> None:
        """The cache needs independent rows. Asking for it on another model is a usage error;
        the default turns it off and the banner says why."""
        wanted = self.options.result_cache_rows
        rows = DEFAULT_RESULT_CACHE_ROWS if wanted is None else wanted
        if rows <= 0:
            return
        if self.verdict.row_independent:
            self.result_cache = ResultCache(rows)
        elif wanted is not None:
            raise ValueError(
                f"--result-cache-rows needs a model with independent rows. This model: "
                f"{self.verdict.rows_note}. Remove --result-cache-rows, or set it to 0."
            )
        else:
            self.cache_off_reason = self.verdict.rows_note

    def try_admit(self) -> bool:
        """Claim one of max_concurrency + max_queue slots, or refuse."""
        with self._admission_lock:
            if self.in_flight >= self.options.max_concurrency + self.options.max_queue:
                return False
            self.in_flight += 1
            return True

    def release(self) -> None:
        with self._admission_lock:
            self.in_flight -= 1

    @property
    def backend_auto_selected(self) -> bool:
        return self.options.backend == BackendChoice.auto and not self.forced_onnx

    @property
    def forced_onnx(self) -> bool:
        return self.options.force_onnx and self.verdict.status == Status.DEGRADED

    @cached_property
    def input_specs(self) -> dict[str, IOSpec]:
        return {spec.name: spec for spec in self.backend.metadata().inputs}

    @cached_property
    def declared_dtypes(self) -> dict[str, str | None]:
        return {name: spec.dtype for name, spec in self.input_specs.items()}

    @cached_property
    def served_name(self) -> str:
        """--served-model-name, or the name of the model source (a directory gives its last part)."""
        return self.options.served_model_name or display_source(self.source, self.source_kind)

    @cached_property
    def embedding_width(self) -> int | None:
        """The size of the last axis of the first output, or None if it is not a fixed number."""
        shape = self.backend.metadata().outputs[0].shape
        return shape[-1] if shape and isinstance(shape[-1], int) else None
