"""From a loaded model (or a reused export) to a warmed-up backend in a ServingState."""

import gc
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np
import torch

from downshift.adapters.base import Prepared
from downshift.adapters.embedding import EmbeddingRecipe
from downshift.adapters.text import TextIO
from downshift.core.axes import AxisBounds, axis_bounds_from_json
from downshift.core.feeds import example_feeds
from downshift.core.memo import ExportEntry
from downshift.core.phase import Phase, report
from downshift.core.shapes import dynamic_bounds
from downshift.core.verdict import (
    BackendName,
    ExportVerdict,
    Status,
    build_verdict,
    intake,
    prepare_model,
    unverified_verdict,
)
from downshift.loading import LoadedModel, LoadError
from downshift.serve.backends import (
    Backend,
    IOSpec,
    OnnxRuntimeBackend,
    TorchBackend,
    concrete_dim,
    resolve_device,
    verified_provider_for,
)
from downshift.serve.graphs import eager_output_axes
from downshift.serve.options import BackendChoice, ExecutionChoice, ServeOptions
from downshift.serve.result_cache import ResultCache
from downshift.serve.schemas import normalize_dtype
from downshift.settings import DEFAULT_RESULT_CACHE_ROWS
from downshift.sources import HF_REPO_DIR, UNKNOWN_SOURCE

# Returns the loaded MODEL and its --reference model. Called only when a model is needed.
Loader = Callable[[], tuple[LoadedModel, LoadedModel | None]]


@dataclass
class WarmupStats:
    count: int
    mean_ms: float
    synthesized: bool  # True if there was no example input, so warmup() made one


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
    def execution(self) -> ExecutionChoice:
        return self.options.execution

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


def _prepare(loaded: LoadedModel, opts: ServeOptions) -> Prepared:
    assert loaded.model is not None
    return prepare_model(
        loaded.model,
        loaded.example_inputs,
        opts.adapter or loaded.adapter_hint,
        opts.dynamic,
        vary=opts.vary,
        axis_max=opts.axis_max,
    )


def _verdict_for(
    loaded: LoadedModel,
    reference: LoadedModel | None,
    opts: ServeOptions,
    timings: dict[str, float] | None = None,
) -> ExportVerdict:
    if loaded.onnx_path is not None:
        return intake(
            loaded.onnx_path,
            reference.model if reference else None,
            reference.example_inputs if reference else None,
            opts.adapter or loaded.adapter_hint,
            k=opts.k,
            dynamic=opts.dynamic,
            atol=opts.atol,
            rtol=opts.rtol,
            seed=opts.seed,
            vary=opts.vary,
            axis_max=opts.axis_max,
            timings=timings,
        )
    prepared = _prepare(loaded, opts)
    if opts.backend == BackendChoice.torch:
        return unverified_verdict(
            prepared, "--backend torch: export skipped", eager_output_axes(prepared)
        )
    floats = {p.dtype for p in prepared.model.parameters() if p.is_floating_point()}
    if opts.backend == BackendChoice.auto and floats == {torch.bfloat16}:
        # ORT has no bf16 CPU kernels: the export would take minutes and then fail to load.
        return unverified_verdict(
            prepared,
            "bfloat16 weights: ONNX Runtime has no bf16 kernels, so the export was skipped; "
            "serving torch (pass --backend torch to silence this)",
            eager_output_axes(prepared),
        )
    verdict = build_verdict(
        prepared, k=opts.k, atol=opts.atol, rtol=opts.rtol, seed=opts.seed, timings=timings
    )
    if verdict.status == Status.FAILED and not verdict.output_axes:
        # The export never reached verification: classify on the eager model instead.
        verdict.output_axes = eager_output_axes(prepared)
    return verdict


def choose_backend(
    verdict: ExportVerdict, opts: ServeOptions, *, has_torch: bool | None = None
) -> tuple[BackendName, list[str]]:
    """(backend, notes for the banner). `has_torch` defaults to whether the verdict has a
    Prepared. A reused export passes True: its caller loads the model if it must."""
    notes: list[str] = []
    wanted = (
        verdict.recommended_backend
        if opts.backend == BackendChoice.auto
        else BackendName(opts.backend)
    )
    degraded = verdict.status == Status.DEGRADED
    if opts.backend == BackendChoice.onnxruntime and degraded and not opts.force_onnx:
        raise ValueError(
            "--backend onnxruntime: the verdict is DEGRADED. Pass --force-onnx to serve the "
            "ONNX graph"
        )
    if opts.force_onnx and degraded:
        wanted = BackendName.onnxruntime
        notes.append("--force-onnx: serving a DEGRADED graph; outputs may be wrong")

    has_onnx = any((verdict.onnx_program, verdict.onnx_path, verdict.onnx_bytes))
    if has_torch is None:
        has_torch = verdict.prepared is not None
    if wanted == BackendName.onnxruntime and not has_onnx:
        notes.append("no ONNX graph available; falling back to torch")
        wanted = BackendName.torch
    if wanted == BackendName.torch and not has_torch:
        raise ValueError(
            "torch backend requested but there is no PyTorch model to run. Give "
            "--reference with the PyTorch model, or use --backend onnxruntime"
        )
    return wanted, notes


_TEXT_UNAVAILABLE_PREFIX = "text input unavailable:"
_NO_TOKENIZER_NOTE = "no tokenizer files in the repo directory; /predict takes tensors only"
_POOLING_IGNORED_NOTE = (
    "--pooling/--normalize ignored: they only apply to a Hugging Face repo served directly; "
    "a --tokenizer-from companion supplies the tokenizer, not the graph"
)


def attach_hf_metadata(state: ServingState) -> None:
    """Fill state.text, state.embedding and state.vocab_size from state.hf_source.

    Text input is optional: a missing tokenizer or [hf] extra only adds a banner note. A bad
    --pooling or a config that does not parse fails the boot, as for the repo served directly.
    """
    if state.hf_source is None:
        return
    opts = state.options
    try:
        from downshift import hf_repo
    except ImportError as exc:
        state.notes.append(f"{_TEXT_UNAVAILABLE_PREFIX} {type(exc).__name__}: {exc}")
        return

    try:
        config = hf_repo.load_config(state.hf_source)
        vocab = getattr(config, "vocab_size", None)
        state.vocab_size = int(vocab) if vocab is not None else None
        # --pooling changes the export of a repo served directly. A --tokenizer-from graph is
        # already built, so for it the flags would only mislabel /schema.
        direct = state.source_kind == HF_REPO_DIR
        pooling, normalize = (opts.pooling, opts.normalize) if direct else (None, None)
        recipe = hf_repo.embedding_recipe(state.hf_source, config, pooling, normalize)
    except (OSError, ValueError) as exc:  # ValueError includes RecipeError
        raise LoadError(f"{state.hf_source}: {exc}") from exc

    try:
        text = hf_repo.load_text_io(state.hf_source, config, recipe)
    except Exception as exc:  # noqa: BLE001 - tokenizer files are optional
        state.notes.append(f"{_TEXT_UNAVAILABLE_PREFIX} {type(exc).__name__}: {exc}")
        return
    if not direct and (opts.pooling is not None or opts.normalize is not None):
        state.notes.append(_POOLING_IGNORED_NOTE)
    # A --tokenizer-from graph can be the bare encoder: claim the recipe only if output_0 is
    # one vector per row.
    outputs = state.backend.metadata().outputs
    pooled = bool(outputs) and outputs[0].shape is not None and len(outputs[0].shape) == 2
    if direct or pooled:
        state.embedding = recipe
    state.text = text
    if text is None:
        state.notes.append(_NO_TOKENIZER_NOTE)


def _reusable_verify_session(verdict: ExportVerdict, opts: ServeOptions) -> Any:
    """The session of verify, if the serving options need the same one (CPU, default
    threads). The verdict lets go of it in both cases."""
    session = verdict.take_session()
    if resolve_device(opts.device) != "cpu":
        return None
    if opts.intra_op_threads != 0 or opts.inter_op_threads != 0:
        return None
    return session


def _build_backend(
    name: BackendName, verdict: ExportVerdict, opts: ServeOptions, timings: dict[str, float]
) -> Backend:
    report(Phase.session)
    start = time.perf_counter()
    backend: Backend
    if name == BackendName.onnxruntime:
        session = _reusable_verify_session(verdict, opts)
        if session is not None:
            backend = OnnxRuntimeBackend(session=session)
        else:
            source: bytes | Path = (
                verdict.onnx_path if verdict.onnx_path is not None else verdict.onnx_bytes
            )
            backend = OnnxRuntimeBackend(
                source, opts.device, opts.intra_op_threads, opts.inter_op_threads
            )
    else:
        prepared = verdict.prepared
        assert prepared is not None
        backend = TorchBackend(
            prepared.model,
            prepared.input_names,
            opts.device,
            prepared.inputs,
            opts.intra_op_threads,
            prepared.dynamic_shapes,
        )
    backend.verified_provider = verified_provider_for(verdict)
    timings[Phase.session] = time.perf_counter() - start
    return backend


def _finish(state: ServingState, timings: dict[str, float]) -> ServingState:
    """The last steps of each builder: the HF metadata, then a timed warmup."""
    attach_hf_metadata(state)
    state.timings = timings
    report(Phase.warmup)
    start = time.perf_counter()
    warmup(state, state.options.warmup)
    timings[Phase.warmup] = time.perf_counter() - start
    return state


def _drop_models(*loaded: LoadedModel | None) -> None:
    for item in loaded:
        if item is not None:
            item.model = None


def prepare_serving(
    loaded: LoadedModel,
    opts: ServeOptions | None = None,
    reference: LoadedModel | None = None,
    tokenizer_from: str | None = None,
) -> ServingState:
    """Export and verify `loaded` (or take its .onnx file), choose a backend, warm it up."""
    opts = opts or ServeOptions()
    timings: dict[str, float] = {}
    report(Phase.load)
    verdict = _verdict_for(loaded, reference, opts, timings)
    name, notes = choose_backend(verdict, opts)
    backend = _build_backend(name, verdict, opts, timings)
    verdict.take_session()  # a torch backend did not take it

    prepared = verdict.prepared
    if prepared is not None:
        input_names, example_inputs = prepared.input_names, prepared.inputs
        axis_bounds = dynamic_bounds(prepared.input_names, prepared.dynamic_shapes)
    else:
        input_names, example_inputs, axis_bounds = tuple(backend.input_names), None, {}

    if name == BackendName.onnxruntime and verdict.status == Status.CLEAN:
        # The session serves alone: drop every reference to the torch weights before warmup.
        _drop_models(loaded, reference)
        verdict.prepared = None
        verdict.onnx_program = None
        gc.collect()

    state = ServingState(
        loaded.source,
        verdict,
        backend,
        input_names,
        opts,
        example_inputs,
        notes=notes,
        source_kind=loaded.kind,
        axis_bounds=axis_bounds,
        hf_source=loaded.source if loaded.kind == HF_REPO_DIR else tokenizer_from,
    )
    return _finish(state, timings)


def state_from_entry(
    entry: ExportEntry,
    load: Loader,
    opts: ServeOptions,
    tokenizer_from: str | None,
    *,
    source: str,
    kind: str,
    reused: str | None = None,
    loaded: LoadedModel | None = None,
    reference: LoadedModel | None = None,
) -> ServingState:
    """A ServingState from an earlier export: a cache hit (`reused` names the tier), or a
    `--workers N` worker with its parent's export. The backend is chosen again from the
    verdict. `load()` runs only if torch serves and `loaded` has no model."""
    verdict = ExportVerdict.from_dict(entry.verdict)
    verdict.onnx_bytes = entry.onnx_bytes
    verdict.onnx_path = entry.onnx_path
    name, notes = choose_backend(verdict, opts, has_torch=True)
    hf_source = source if kind == HF_REPO_DIR else tokenizer_from
    timings: dict[str, float] = {}
    example_inputs: tuple | None

    if name == BackendName.torch:
        # Eager serving needs the model: the entry has only the export and its verdict.
        if loaded is None:
            loaded, reference = load()
        report(Phase.load)
        prepared = _prepare(loaded, opts)
        verdict.prepared = prepared
        source, kind = loaded.source, loaded.kind
        input_names, example_inputs = prepared.input_names, prepared.inputs
        axis_bounds = dynamic_bounds(prepared.input_names, prepared.dynamic_shapes)
    else:
        _drop_models(loaded, reference)
        input_names, example_inputs = tuple(entry.input_names), entry.example_inputs()
        axis_bounds = axis_bounds_from_json(entry.axis_bounds)

    backend = _build_backend(name, verdict, opts, timings)
    state = ServingState(
        source,
        verdict,
        backend,
        input_names,
        opts,
        example_inputs,
        notes=notes,
        source_kind=kind,
        axis_bounds=axis_bounds,
        hf_source=hf_source,
        reused=reused,
    )
    return _finish(state, timings)


def synthesize_feeds(backend: Backend) -> dict[str, np.ndarray]:
    """One dummy array for each declared input, to warm up a backend with no example inputs.
    Dynamic axes become 1. Floats are randn, other dtypes zero."""
    feeds: dict[str, np.ndarray] = {}
    for spec in backend.metadata().inputs:
        shape = tuple(concrete_dim(dim) for dim in (spec.shape or [1]))
        dtype = np.dtype(normalize_dtype(spec.dtype) or "float32")
        if dtype.kind == "f":
            feeds[spec.name] = np.random.randn(*shape).astype(dtype)
        else:
            feeds[spec.name] = np.zeros(shape, dtype=dtype)
    return feeds


def warmup(state: ServingState, n: int) -> WarmupStats:
    """Run a few inferences before /ready turns 200, so users do not pay first-call costs."""
    example_inputs = state.example_inputs
    mean_ms = 0.0
    if n > 0:
        feeds = (
            synthesize_feeds(state.backend)
            if example_inputs is None
            else example_feeds(state.input_names, example_inputs)
        )
        start = time.perf_counter()
        for _ in range(n):
            state.backend.infer(feeds)
        mean_ms = (time.perf_counter() - start) / n * 1000
    state.warmup_stats = WarmupStats(n, mean_ms, synthesized=example_inputs is None)
    state.ready = True
    return state.warmup_stats
