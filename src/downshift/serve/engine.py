"""From a loaded model to a warmed-up backend. The CLI's `serve` is render(prepare_serving())."""

import gc
import threading
import time
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
from downshift.core.axes import AxisBounds
from downshift.core.feeds import example_feeds
from downshift.core.phase import Phase, report
from downshift.core.prevalidated import intake
from downshift.core.shapes import dynamic_bounds
from downshift.core.verdict import (
    BackendName,
    ExportVerdict,
    Status,
    build_verdict,
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
from downshift.serve.schemas import normalize_dtype
from downshift.sources import HF_REPO_DIR, UNKNOWN_SOURCE


@dataclass
class WarmupStats:
    """What warmup() did, for the banner (Phase 4) and /metadata to report."""

    count: int
    mean_ms: float
    synthesized: bool  # True when there was no example input, so warmup() made one up


@dataclass
class ServingState:
    source: str
    verdict: ExportVerdict
    backend: Backend
    input_names: tuple[str, ...]
    options: ServeOptions
    example_inputs: tuple | None = None
    ready: bool = False
    notes: list[str] = field(default_factory=list)  # things the banner should say
    # Which accepted form `source` named (downshift.loading's ONNX_FILE, TORCH_CHECKPOINT,
    # HF_REPO_DIR, IMPORT_SPEC, IN_PROCESS_MODULE). Reported under /schema's `source`, and
    # labelled on the boot banner, so it is visible that nothing was fetched to serve this.
    source_kind: str = UNKNOWN_SOURCE
    # "memory" or "disk" when this boot reused an earlier export instead of running export and
    # verify (the in-process memo, or --export-cache-dir); None for a fresh export.
    reused: str | None = None
    # The Hugging Face repo directory attach_hf_metadata should read tokenizer/pooling/label
    # metadata from, or None to skip that. Usually `source` itself (kind HF_REPO_DIR), but a
    # bare .onnx MODEL takes it from --tokenizer-from instead, kept separate from --reference
    # (numerics only) so each flag does exactly one job.
    hf_source: str | None = None
    # Tokenizer and label metadata when the source is a Hugging Face repo directory that has
    # them; what lets /predict take `text` and answer with class probabilities.
    text: TextIO | None = None
    # How token vectors become one embedding per text, when the repo declares (or --pooling
    # gives) a recipe; already part of the graph, kept here so /schema can say what it is.
    embedding: EmbeddingRecipe | None = None
    # The model's vocabulary size, when the source is a Hugging Face repo directory (B3): lets
    # the predict path range-check input_ids before infer instead of letting ORT wrap negatives.
    vocab_size: int | None = None
    warmup_stats: WarmupStats | None = None
    # Phase wall-clock seconds: "load" (CLI-only, added after prepare_serving returns),
    # "export", "verify", "session", "warmup" - whichever phases actually ran. Read by the
    # CLI's Boot banner row and /metadata's `boot` field.
    timings: dict[str, float] = field(default_factory=dict)
    # Per input, per dynamic axis: the (name, min, max) downshift's own export traced (U2).
    # Empty for a bare .onnx with no reference model - there is no Prepared to read it from.
    axis_bounds: AxisBounds = field(default_factory=dict, repr=False)
    # Not JSON-able and not part of a serving state's identity: rebuilt from options.max_concurrency.
    # `infer` runs on these threads, never on the event loop, so /health and /ready are never
    # stuck behind a queue of slow predicts.
    executor: ThreadPoolExecutor = field(init=False, repr=False, compare=False)
    # Request conversion and response encoding (and the body's JSON parse): the CPU work around
    # an inference, kept off `executor` so it never holds an inference slot.
    prep_executor: ThreadPoolExecutor = field(init=False, repr=False, compare=False)
    # Admitted-but-not-yet-finished predicts (running in the executor or still queued there),
    # guarded by _admission_lock since it's read-then-written from the event loop and written
    # again from whichever executor thread finishes a request.
    in_flight: int = field(init=False, repr=False, compare=False, default=0)
    _admission_lock: threading.Lock = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.executor = ThreadPoolExecutor(max_workers=self.options.max_concurrency)
        self.prep_executor = ThreadPoolExecutor(
            max_workers=self.options.prep_threads, thread_name_prefix="downshift-prep"
        )
        self._admission_lock = threading.Lock()

    def try_admit(self) -> bool:
        """Atomically claim one of max_concurrency + max_queue slots, or refuse."""
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
        """The inputs the backend advertises, read once rather than per request."""
        return {spec.name: spec for spec in self.backend.metadata().inputs}

    @cached_property
    def declared_dtypes(self) -> dict[str, str | None]:
        return {name: spec.dtype for name, spec in self.input_specs.items()}


def _prepare(loaded: LoadedModel, opts: ServeOptions) -> Prepared:
    """The one place a loaded torch model meets its adapter, --dynamic and --vary, so the
    --workers torch path prepares it exactly as the single-process path does."""
    assert loaded.model is not None
    return prepare_model(
        loaded.model,
        loaded.example_inputs,
        opts.adapter or loaded.adapter_hint,
        opts.dynamic,
        vary=opts.vary,
        axis_max=opts.axis_max,
    )


def _all_bfloat16(model: torch.nn.Module) -> bool:
    dtypes = {p.dtype for p in model.parameters() if p.is_floating_point()}
    return dtypes == {torch.bfloat16}


def _verdict_for(
    loaded: LoadedModel,
    reference: LoadedModel | None,
    opts: ServeOptions,
    timings: dict[str, float] | None = None,
) -> ExportVerdict:
    if loaded.onnx_path is not None:
        ref_model = reference.model if reference else None
        ref_inputs = reference.example_inputs if reference else None
        return intake(
            loaded.onnx_path,
            ref_model,
            ref_inputs,
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
        # Skip the export entirely; the user asked for eager.
        return unverified_verdict(
            prepared, "--backend torch: export skipped", eager_output_axes(prepared)
        )
    if opts.backend == BackendChoice.auto and _all_bfloat16(prepared.model):
        # ORT's CPU kernels have no bf16 Gemm, so the export would run for a minute or more
        # and then fail to load. Say so now and serve eagerly.
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
        # A failed export never reached verify, so classify on the eager model it falls back to.
        verdict.output_axes = eager_output_axes(prepared)
    return verdict


def choose_backend(
    verdict: ExportVerdict, opts: ServeOptions, *, has_torch: bool | None = None
) -> tuple[BackendName, list[str]]:
    """Return (backend name, notes for the banner). `has_torch` says whether a PyTorch model
    can be had when the verdict carries none (a reused export, whose caller loads it on
    demand); by default that is whether the verdict has a Prepared."""
    notes: list[str] = []
    wanted: BackendName = (
        verdict.recommended_backend
        if opts.backend == BackendChoice.auto
        else BackendName(opts.backend)
    )
    onnx_requested = opts.backend == BackendChoice.onnxruntime
    if onnx_requested and verdict.status == Status.DEGRADED and not opts.force_onnx:
        raise ValueError(
            "--backend onnxruntime: the verdict is DEGRADED; pass --force-onnx to serve the "
            "ONNX graph anyway"
        )
    if opts.force_onnx and verdict.status == Status.DEGRADED:
        wanted = BackendName.onnxruntime
        notes.append("--force-onnx: serving a DEGRADED graph; outputs may be wrong")

    has_onnx = (
        verdict.onnx_program is not None
        or verdict.onnx_path is not None
        or bool(verdict.onnx_bytes)
    )
    if has_torch is None:
        has_torch = verdict.prepared is not None
    if wanted == BackendName.onnxruntime and not has_onnx:
        notes.append("no ONNX graph available; falling back to torch")
        wanted = BackendName.torch
    if wanted == BackendName.torch and not has_torch:
        raise ValueError("torch backend requested but there is no PyTorch model to run")
    return wanted, notes


_TEXT_UNAVAILABLE_PREFIX = "text input unavailable:"
_NO_TOKENIZER_NOTE = "no tokenizer files in the repo directory; /predict takes tensors only"
_POOLING_IGNORED_NOTE = (
    "--pooling/--normalize ignored: they only apply to a Hugging Face repo served directly; "
    "a --tokenizer-from companion supplies the tokenizer, not the graph"
)


def attach_hf_metadata(state: ServingState) -> None:
    """Fill in state.text and state.embedding from state.hf_source, when there is one: the
    served model itself for a Hugging Face repo directory, or a companion HF repo directory
    named by --tokenizer-from when the served model is a bare .onnx.

    Every place that builds a ServingState calls this (through finish_state), a reused
    export's too. The embedding recipe is read again from the repo with the same overrides the
    load used, so it needs no hand-off from the loader. Text input is optional by nature: a
    repo with no tokenizer, or a missing [hf] extra, only costs the `text` input, so the
    reason goes on the banner rather than failing the boot.

    A bad --pooling/--normalize, or a repo AutoConfig can't parse, hard-fails the boot here
    exactly as it would for the same repo served directly (loading._load_hf): the recipe is
    not optional the way text input is, so --tokenizer-from must not turn that misconfiguration
    into a soft banner note just because it was reached through a different flag.
    """
    if state.hf_source is None:
        return
    opts = state.options
    try:
        from downshift import hf_repo
    except ImportError as exc:
        state.notes.append(f"{_TEXT_UNAVAILABLE_PREFIX} {type(exc).__name__}: {exc}")
        return

    # One config read for everything below. vocab_size is set before anything that needs a
    # tokenizer or a recipe, so the token-id range check (B3) still works when those fail.
    try:
        config = hf_repo.load_config(state.hf_source)
        vocab = getattr(config, "vocab_size", None)
        state.vocab_size = int(vocab) if vocab is not None else None
        # --pooling/--normalize reshape the hf adapter's own export; a --tokenizer-from
        # companion's graph is already built, so for it they would only mislabel /schema.
        direct = state.source_kind == HF_REPO_DIR
        pooling, normalize = (opts.pooling, opts.normalize) if direct else (None, None)
        recipe = hf_repo.embedding_recipe(state.hf_source, config, pooling, normalize)
    except (OSError, ValueError) as exc:  # ValueError also covers RecipeError
        raise LoadError(f"{state.hf_source}: {exc}") from exc

    try:
        text = hf_repo.load_text_io(state.hf_source, config, recipe)
    except Exception as exc:  # noqa: BLE001 - no tokenizer files is optional, unlike the recipe
        state.notes.append(f"{_TEXT_UNAVAILABLE_PREFIX} {type(exc).__name__}: {exc}")
        return
    if not direct and (opts.pooling is not None or opts.normalize is not None):
        state.notes.append(_POOLING_IGNORED_NOTE)
    # The hf adapter's own export applies the recipe (a PoolingHead in the graph). A
    # --tokenizer-from companion's recipe still sets the text path's max_seq_length, but the
    # served graph may be the bare encoder: only claim the recipe when its output is pooled.
    if direct or _pooled_output(state.backend):
        state.embedding = recipe
    state.text = text
    if text is None:
        state.notes.append(_NO_TOKENIZER_NOTE)


def _pooled_output(backend: Backend) -> bool:
    """Whether output_0 is one vector per row ([batch, hidden]) rather than token-level."""
    outputs = backend.metadata().outputs
    return bool(outputs) and outputs[0].shape is not None and len(outputs[0].shape) == 2


def _reusable_verify_session(verdict: ExportVerdict, opts: ServeOptions) -> Any:
    """The session verify() already built, if the serving options mean the same thing:
    resolved device cpu and both thread counts 0, which is exactly what verify's own
    session (ORT's defaults, CPU-only) already is. Anything else needs its own session.
    The verdict lets go of it either way: an unused one is a second copy of the weights."""
    session = verdict.take_session()
    if resolve_device(opts.device) != "cpu":
        return None
    if opts.intra_op_threads != 0 or opts.inter_op_threads != 0:
        return None
    return session


def _axis_bounds(verdict: ExportVerdict) -> AxisBounds:
    """Per input, per dynamic axis: the (name, min, max) an adapter's export traced it for
    (U2). Empty when there is no Prepared to read it from (a bare .onnx with no --reference)."""
    prepared = verdict.prepared
    if prepared is None:
        return {}
    return dynamic_bounds(prepared.input_names, prepared.dynamic_shapes)


def _build_backend(
    name: BackendName, verdict: ExportVerdict, opts: ServeOptions, timings: dict[str, float]
) -> Backend:
    """The one place a Backend is constructed, for every serving-state builder below: sets
    verified_provider here too, so no caller can build a backend and forget it. The build
    time goes into `timings` as Phase.session."""
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


def _new_state(
    source: str,
    verdict: ExportVerdict,
    backend: Backend,
    input_names: tuple[str, ...],
    opts: ServeOptions,
    example_inputs: tuple | None,
    timings: dict[str, float],
    *,
    notes: list[str] | None = None,
    kind: str = UNKNOWN_SOURCE,
    axis_bounds: AxisBounds | None = None,
    hf_source: str | None = None,
    reused: str | None = None,
) -> ServingState:
    """The single ServingState constructor every builder below funnels through, so
    finish_state (and so attach_hf_metadata) never gets skipped by a builder that forgot.
    `axis_bounds` overrides what the verdict's Prepared would give (a worker has none)."""
    state = ServingState(
        source,
        verdict,
        backend,
        input_names,
        opts,
        example_inputs,
        notes=list(notes or []),
        source_kind=kind,
        axis_bounds=_axis_bounds(verdict) if axis_bounds is None else axis_bounds,
        hf_source=hf_source,
        reused=reused,
    )
    return finish_state(state, timings)


def prepare_serving(
    loaded: LoadedModel,
    opts: ServeOptions | None = None,
    reference: LoadedModel | None = None,
    tokenizer_from: str | None = None,
) -> ServingState:
    opts = opts or ServeOptions()
    timings: dict[str, float] = {}
    # core/verdict.py and core/prevalidated.py report export and verify from inside this call.
    report(Phase.load)
    verdict = _verdict_for(loaded, reference, opts, timings)
    name, notes = choose_backend(verdict, opts)
    backend = _build_backend(name, verdict, opts, timings)

    if verdict.prepared is not None:
        input_names = verdict.prepared.input_names
        example_inputs = verdict.prepared.inputs
    else:
        input_names = tuple(backend.input_names)
        example_inputs = None

    verdict.take_session()  # a torch backend never took it; drop it all the same

    axis_bounds = _axis_bounds(verdict)
    if name == BackendName.onnxruntime and verdict.status == Status.CLEAN:
        _release_torch_model(loaded, reference, verdict)

    return _new_state(
        loaded.source,
        verdict,
        backend,
        input_names,
        opts,
        example_inputs,
        timings,
        notes=notes,
        kind=loaded.kind,
        axis_bounds=axis_bounds,
        hf_source=loaded.source if loaded.kind == HF_REPO_DIR else tokenizer_from,
    )


def _release_torch_model(
    loaded: LoadedModel, reference: LoadedModel | None, verdict: ExportVerdict
) -> None:
    """A verified ONNX graph is served by its session alone: drop every reference to the torch
    model (the loaded one, the --reference one, the exporter's program and the adapter's shim)
    before warmup, so peak memory is not the weights held twice or three times. The axis
    bounds and example inputs are already read off the Prepared by then."""
    loaded.model = None
    if reference is not None:
        reference.model = None
    verdict.prepared = None
    verdict.onnx_program = None
    gc.collect()


def serving_state_from_artifact(
    source: str,
    verdict: ExportVerdict,
    opts: ServeOptions,
    input_names: tuple[str, ...],
    notes: list[str] | None = None,
    example_inputs: tuple | None = None,
    axis_bounds: AxisBounds | None = None,
    kind: str = UNKNOWN_SOURCE,
    hf_source: str | None = None,
    reused: str | None = None,
) -> ServingState:
    """A ServingState over an ONNX graph an earlier export already verified (a cache hit, or
    a `serve --workers N` parent's): no capture, no verify, just a session over the graph at
    verdict.onnx_path, or the verdict's own onnx_bytes when that is None. The verdict carries
    no `prepared`, so `input_names`, `kind` and `hf_source` come from the caller;
    `example_inputs` are saved feeds when there were any, else warmup() synthesizes them.
    `reused` says which cache the verdict came from, for the banner."""
    timings: dict[str, float] = {}
    backend = _build_backend(BackendName.onnxruntime, verdict, opts, timings)
    return _new_state(
        source,
        verdict,
        backend,
        input_names,
        opts,
        example_inputs,
        timings,
        notes=notes,
        kind=kind,
        axis_bounds=axis_bounds,
        hf_source=hf_source,
        reused=reused,
    )


def serving_state_from_torch_artifact(
    loaded: LoadedModel,
    verdict: ExportVerdict,
    opts: ServeOptions,
    notes: list[str] | None = None,
    hf_source: str | None = None,
    reused: str | None = None,
) -> ServingState:
    """Rebuild a ServingState in a `serve --workers N` worker when the parent's export chose
    torch: the worker still loads and prepares the model itself (torch weights aren't shipped
    between processes), but takes the parent's already-verified verdict as given rather than
    re-exporting. `loaded` is the caller's own reload of the same model spec the parent used;
    `hf_source` is the parent's own resolved value, carried the same way as in
    serving_state_from_artifact.
    """
    report(Phase.load)
    prepared = _prepare(loaded, opts)
    verdict.prepared = prepared
    timings: dict[str, float] = {}
    backend = _build_backend(BackendName.torch, verdict, opts, timings)
    return _new_state(
        loaded.source,
        verdict,
        backend,
        prepared.input_names,
        opts,
        prepared.inputs,
        timings,
        notes=notes,
        kind=loaded.kind,
        hf_source=hf_source,
        reused=reused,
    )


def finish_state(state: ServingState, timings: dict[str, float]) -> ServingState:
    """The tail every ServingState builder shares: attach the Hugging Face metadata, record
    the phase timings so far, then warm up (timed) and flip ready. A builder that skipped
    attach_hf_metadata would silently lose text input and the embedding metadata."""
    attach_hf_metadata(state)
    state.timings = timings
    report(Phase.warmup)
    warmup_start = time.perf_counter()
    warmup(state, state.options.warmup)
    timings[Phase.warmup] = time.perf_counter() - warmup_start
    return state


def synthesize_feeds(backend: Backend) -> dict[str, np.ndarray]:
    """One dummy array per input the backend declares, for warming up a backend with no
    example inputs (a bare .onnx served without --reference). Dynamic or unknown axes
    become 1; floats are randn, integers and bools are zero/false."""
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
    """Run a few inferences before /ready flips; first-call costs shouldn't hit users."""
    example_inputs = state.example_inputs
    synthesized = example_inputs is None
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
    stats = WarmupStats(count=n, mean_ms=mean_ms, synthesized=synthesized)
    state.warmup_stats = stats
    state.ready = True
    return stats
