"""From a loaded model to a warmed-up backend. The `serve` command of the CLI is render(prepare_serving())."""

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
    notes: list[str] = field(default_factory=list)  # what the banner must say
    # The accepted form that `source` named (ONNX_FILE, TORCH_CHECKPOINT, HF_REPO_DIR,
    # IMPORT_SPEC and IN_PROCESS_MODULE of downshift.loading). /schema reports it under `source`.
    # The boot banner also labels it. It is then visible that downshift fetched nothing to serve
    # this.
    source_kind: str = UNKNOWN_SOURCE
    # "memory" or "disk" if this boot reused an earlier export and did not run the export and
    # the verification (the in-process memo, or --export-cache-dir). None for a new export.
    reused: str | None = None
    # The Hugging Face repo directory from which attach_hf_metadata reads the tokenizer, the
    # pooling and the label metadata. None skips this. It is usually `source` itself (kind
    # HF_REPO_DIR). A bare .onnx MODEL takes it from --tokenizer-from instead. Downshift keeps
    # it separate from --reference (numerics only), so each flag does exactly one job.
    hf_source: str | None = None
    # The tokenizer and the label metadata, if the source is a Hugging Face repo directory that
    # has them. They let /predict take `text` and answer with class probabilities.
    text: TextIO | None = None
    # How token vectors become one embedding for each text, if the repo declares a recipe (or
    # --pooling gives one). It is already part of the graph. It is kept here so that /schema can
    # say what it is.
    embedding: EmbeddingRecipe | None = None
    # The vocabulary size of the model, if the source is a Hugging Face repo directory (B3). It
    # lets the predict path check the range of input_ids before the inference. ORT would wrap
    # negative values.
    vocab_size: int | None = None
    warmup_stats: WarmupStats | None = None
    # The wall-clock seconds of each phase: "load" (CLI only, added after prepare_serving
    # returns), "export", "verify", "session" and "warmup", for the phases that ran. The Boot row
    # of the CLI banner and the `boot` field of /metadata read it.
    timings: dict[str, float] = field(default_factory=dict)
    # For each input and each dynamic axis: the (name, min, max) that the own export of downshift
    # traced (U2). Empty for a bare .onnx file with no reference model. There is no Prepared from
    # which to read it.
    axis_bounds: AxisBounds = field(default_factory=dict, repr=False)
    # Not JSON-able, and not part of the identity of a serving state. Downshift builds it again
    # from options.max_concurrency. `infer` runs on these threads and never on the event loop.
    # /health and /ready therefore never wait behind a queue of slow predicts.
    executor: ThreadPoolExecutor = field(init=False, repr=False, compare=False)
    # The conversion of requests and the encoding of responses (and the JSON parse of the body).
    # This is the CPU work around an inference. It stays off `executor`, so it never holds an
    # inference slot.
    prep_executor: ThreadPoolExecutor = field(init=False, repr=False, compare=False)
    # Predicts that are admitted and not finished (they run in the executor or wait in its
    # queue). _admission_lock guards it. The event loop reads and then writes it, and the
    # executor thread that finishes a request writes it again.
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
    """The one place where a loaded torch model meets its adapter, --dynamic and --vary. The
    --workers torch path therefore prepares it in the same way as the single-process path."""
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
        # Skip the export. The user asked for eager mode.
        return unverified_verdict(
            prepared, "--backend torch: export skipped", eager_output_axes(prepared)
        )
    if opts.backend == BackendChoice.auto and _all_bfloat16(prepared.model):
        # The CPU kernels of ORT have no bf16 Gemm. The export would run for a minute or more
        # and then fail to load. Say this now and serve eagerly.
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
        # A failed export never reached the verification. Classify on the eager model that it uses instead.
        verdict.output_axes = eager_output_axes(prepared)
    return verdict


def choose_backend(
    verdict: ExportVerdict, opts: ServeOptions, *, has_torch: bool | None = None
) -> tuple[BackendName, list[str]]:
    """Return (backend name, notes for the banner). `has_torch` says whether a PyTorch model is
    available if the verdict has none (a reused export, whose caller loads it when it needs it).
    By default, this is whether the verdict has a Prepared."""
    notes: list[str] = []
    wanted: BackendName = (
        verdict.recommended_backend
        if opts.backend == BackendChoice.auto
        else BackendName(opts.backend)
    )
    onnx_requested = opts.backend == BackendChoice.onnxruntime
    if onnx_requested and verdict.status == Status.DEGRADED and not opts.force_onnx:
        raise ValueError(
            "--backend onnxruntime: the verdict is DEGRADED. Pass --force-onnx to serve the "
            "ONNX graph"
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
    """Fill in state.text and state.embedding from state.hf_source, if there is one. It is the
    served model itself for a Hugging Face repo directory. Or it is a companion HF repo
    directory that --tokenizer-from names, if the served model is a bare .onnx file.

    Each place that builds a ServingState calls this (through finish_state), also for a reused
    export. Downshift reads the embedding recipe again from the repo with the same overrides
    that the load used. It therefore needs no handoff from the loader. Text input is optional
    by nature. If a repo has no tokenizer, or the [hf] extra is missing, only the `text` input
    is lost. The reason then goes on the banner, and the boot does not fail.

    A bad --pooling or --normalize, or a repo that AutoConfig cannot parse, makes the boot fail
    here. This is the same as for the same repo served directly (loading._load_hf). The recipe
    is not optional in the way that text input is. --tokenizer-from must not turn that
    misconfiguration into a soft note on the banner only because it came through a different
    flag.
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
    # tokenizer or a recipe. The range check of the token ID (B3) then still works if those fail.
    try:
        config = hf_repo.load_config(state.hf_source)
        vocab = getattr(config, "vocab_size", None)
        state.vocab_size = int(vocab) if vocab is not None else None
        # --pooling and --normalize change the own export of the hf adapter. The graph of a
        # --tokenizer-from companion is already built. For it, they would only give a wrong
        # label in /schema.
        direct = state.source_kind == HF_REPO_DIR
        pooling, normalize = (opts.pooling, opts.normalize) if direct else (None, None)
        recipe = hf_repo.embedding_recipe(state.hf_source, config, pooling, normalize)
    except (OSError, ValueError) as exc:  # ValueError also includes RecipeError
        raise LoadError(f"{state.hf_source}: {exc}") from exc

    try:
        text = hf_repo.load_text_io(state.hf_source, config, recipe)
    except Exception as exc:  # noqa: BLE001 - tokenizer files are optional, but the recipe is not
        state.notes.append(f"{_TEXT_UNAVAILABLE_PREFIX} {type(exc).__name__}: {exc}")
        return
    if not direct and (opts.pooling is not None or opts.normalize is not None):
        state.notes.append(_POOLING_IGNORED_NOTE)
    # The own export of the hf adapter applies the recipe (a PoolingHead in the graph). The
    # recipe of a --tokenizer-from companion still sets max_seq_length for the text path. The
    # served graph can be the bare encoder. Claim the recipe only if its output is pooled.
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
    """The session that verify() already built, if the serving options mean the same: the device
    resolves to cpu and both thread counts are 0. This is exactly the own session of verify (the
    defaults of ORT, CPU only). All other options need their own session. The verdict lets go of
    it in both cases. A session that nobody uses is a second copy of the weights."""
    session = verdict.take_session()
    if resolve_device(opts.device) != "cpu":
        return None
    if opts.intra_op_threads != 0 or opts.inter_op_threads != 0:
        return None
    return session


def _axis_bounds(verdict: ExportVerdict) -> AxisBounds:
    """For each input and each dynamic axis: the (name, min, max) that the export of an adapter
    traced it for (U2). Empty if there is no Prepared from which to read it (a bare .onnx file
    with no --reference)."""
    prepared = verdict.prepared
    if prepared is None:
        return {}
    return dynamic_bounds(prepared.input_names, prepared.dynamic_shapes)


def _build_backend(
    name: BackendName, verdict: ExportVerdict, opts: ServeOptions, timings: dict[str, float]
) -> Backend:
    """The one place that constructs a Backend, for each serving-state builder below. It also
    sets verified_provider here, so no caller can build a backend and forget it. The build time
    goes into `timings` as Phase.session."""
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
    """The one ServingState constructor that each builder below uses. A builder that forgot
    finish_state (and so attach_hf_metadata) cannot skip it. `axis_bounds` overrides what the
    Prepared of the verdict would give (a worker has none)."""
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
    # core/verdict.py and core/prevalidated.py report the export and the verification from inside this call.
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

    verdict.take_session()  # a torch backend never took it. Drop it in all cases

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
    """A verified ONNX graph is served by its session alone. Drop each reference to the torch
    model (the loaded one, the --reference one, the program of the exporter and the shim of the
    adapter) before the warmup. The peak memory is then not the weights held two or three
    times. At this time, downshift has already read the axis bounds and the example inputs from
    the Prepared."""
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
    """A ServingState over an ONNX graph that an earlier export already verified (a cache hit,
    or the export of a `serve --workers N` parent). There is no capture and no verification. It
    only makes a session over the graph at verdict.onnx_path, or over the own onnx_bytes of the
    verdict if that is None. The verdict has no `prepared`, so `input_names`, `kind` and
    `hf_source` come from the caller. `example_inputs` are the saved feeds, if there were any.
    Otherwise, warmup() synthesizes them. `reused` says which cache the verdict came from, for
    the banner."""
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
    """Build a ServingState again in a `serve --workers N` worker, if the export of the parent
    chose torch. The worker still loads and prepares the model itself, because downshift does
    not send torch weights between processes. It uses the verdict that the parent already
    verified, and it does not export again. `loaded` is the own reload by the caller of the same
    model spec that the parent used. `hf_source` is the own resolved value of the parent. It is
    passed in the same way as in serving_state_from_artifact.
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
    """The last part that each ServingState builder shares. It attaches the Hugging Face
    metadata, records the phase timings until now, then warms up (timed) and sets ready. A
    builder that skipped attach_hf_metadata would lose the text input and the embedding metadata
    without a message."""
    attach_hf_metadata(state)
    state.timings = timings
    report(Phase.warmup)
    warmup_start = time.perf_counter()
    warmup(state, state.options.warmup)
    timings[Phase.warmup] = time.perf_counter() - warmup_start
    return state


def synthesize_feeds(backend: Backend) -> dict[str, np.ndarray]:
    """One dummy array for each input that the backend declares. It warms up a backend that has
    no example inputs (a bare .onnx file served without --reference). Dynamic or unknown axes
    become 1. Floats are randn. Integers and bools are zero or false."""
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
