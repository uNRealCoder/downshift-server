"""From a loaded model (or a reused export) to a warmed-up backend in a ServingState.

There are two checks, and they are not the same:

- The verdict check (core/verdict.py) runs on the CPU. It compares ONNX Runtime with torch and
  makes the ExportVerdict.
- The serving check (`_build_checked`) runs at boot, only if the backend serves off the CPU
  (a GPU). It replays the verification samples through the serving backend and compares the
  result with the CPU result. If it fails, torch serves (`_falls_back`).

There are two boot builders, and both end in `_finish`:

- `prepare_serving`: runs the whole gate (export, verify), then builds the backend.
- `state_from_entry`: boots from an earlier export. It skips the gate.

`choose_backend` turns the verdict and the options into a backend name.
"""

import gc
import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import torch

from downshift.core.axes import axis_bounds_from_json
from downshift.core.capture import has_half_weights
from downshift.core.feeds import example_feeds
from downshift.core.phase import report
from downshift.core.shapes import dynamic_bounds
from downshift.core.verdict import build_verdict, intake, prepare_model, unverified_verdict
from downshift.core.verify import resolve_tolerances
from downshift.entities.adapters import Prepared
from downshift.entities.cache import ExportEntry
from downshift.entities.loading import LoadedModel
from downshift.entities.options import BackendChoice, ServeOptions
from downshift.entities.phase import Phase
from downshift.entities.serving import ServingNumerics, ServingState, WarmupStats
from downshift.entities.verdict import BackendName, ExportVerdict, Status
from downshift.loading import LoadError
from downshift.serve.backends import (
    Backend,
    OnnxRuntimeBackend,
    TorchBackend,
    concrete_dim,
    resolve_device,
)
from downshift.serve.graphs import eager_output_axes
from downshift.serve.numerics import check_serving, serves_off_cpu
from downshift.serve.schemas import normalize_dtype
from downshift.sources import HF_REPO_DIR

logger = logging.getLogger("downshift.serve")

# Returns the loaded MODEL and its --reference model. Called only when a model is needed.
Loader = Callable[[], tuple[LoadedModel, LoadedModel | None]]


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
            samples=opts.samples,
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
        prepared,
        samples=opts.samples,
        atol=opts.atol,
        rtol=opts.rtol,
        seed=opts.seed,
        timings=timings,
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
    if resolve_device(opts.device, ort_backend=True) != "cpu":
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
                source, opts.device, opts.intra_op_threads, opts.inter_op_threads, tf32=opts.tf32
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
            opts.tf32,
        )
    # The provider that verify() compared against (always the CPU), or None without numerics.
    backend.verified_provider = "CPUExecutionProvider" if verdict.numerics is not None else None
    timings[Phase.session] = time.perf_counter() - start
    return backend


def _serving_tolerances(verdict: ExportVerdict, opts: ServeOptions) -> tuple[float, float]:
    """The tolerance of the verdict. Without a verdict check: float32 (float16 for a graph with
    fp16 weights), or --atol and --rtol."""
    if verdict.numerics is not None:
        return verdict.numerics.tolerance_abs, verdict.numerics.tolerance_rel
    half = verdict.onnx_path is not None and has_half_weights(
        onnx.load(str(verdict.onnx_path), load_external_data=False)
    )
    return resolve_tolerances("float16" if half else "float32", opts.atol, opts.rtol)


def _cpu_references(
    verdict: ExportVerdict, feeds: list[dict[str, np.ndarray]], *, torch_model: bool
) -> list[dict[str, np.ndarray]]:
    """The CPU result of each feed: torch for a torch backend, ONNX Runtime for an ONNX one
    (the verdict compared that one with torch)."""
    reference: Backend
    if torch_model:
        prepared = verdict.prepared
        assert prepared is not None
        reference = TorchBackend(prepared.model, prepared.input_names, "cpu")
    else:
        source = verdict.onnx_path if verdict.onnx_path is not None else verdict.onnx_bytes
        reference = OnnxRuntimeBackend(source, "cpu")
    return [reference.infer(sample) for sample in feeds]


def _build_checked(
    name: BackendName,
    verdict: ExportVerdict,
    opts: ServeOptions,
    timings: dict[str, float],
    example: dict[str, np.ndarray] | None,
) -> tuple[Backend, ServingNumerics | None]:
    """The backend, and its serving check if it serves off the CPU and there are feeds. The
    check replays the verification samples (the example input, if the verdict kept none).
    The check runs at boot only."""
    feeds = verdict.sample_feeds or ([] if example is None else [example])
    device = resolve_device(opts.device, ort_backend=name == BackendName.onnxruntime)
    start = time.perf_counter()
    # A torch model moves to the device in place: take its CPU result first.
    references = None
    if feeds and name == BackendName.torch and device.startswith("cuda"):
        references = _cpu_references(verdict, feeds, torch_model=True)
    backend = _build_backend(name, verdict, opts, timings)
    if not feeds or not serves_off_cpu(backend):
        return backend, None
    if references is None:
        references = _cpu_references(verdict, feeds, torch_model=False)
    atol, rtol = _serving_tolerances(verdict, opts)
    embedding = verdict.numerics is not None and verdict.numerics.tolerance_cosine is not None
    total = verdict.numerics.samples_tested if verdict.numerics is not None else len(feeds)
    numerics = check_serving(
        backend,
        references,
        feeds,
        atol,
        rtol,
        device=device,
        embedding=embedding,
        tf32=opts.tf32,
        total=total,
    )
    timings[Phase.session] = time.perf_counter() - start
    return backend, numerics


def _falls_back(
    numerics: ServingNumerics | None, name: BackendName, opts: ServeOptions, has_torch: bool
) -> bool:
    """Log a failed serving check. True if torch must serve instead. A DEGRADED verdict goes to
    torch in the same way, unless --backend or --force-onnx names ONNX Runtime."""
    if numerics is None or numerics.ok:
        return False
    fall_back = (
        name == BackendName.onnxruntime
        and opts.backend == BackendChoice.auto
        and not opts.force_onnx
        and has_torch
    )
    steps = ["pass --device cpu to serve on the CPU"]
    if numerics.tf32:
        steps.insert(0, "remove --tf32")
    if fall_back:
        outcome = "Serving torch instead."
        steps.append("pass --force-onnx to serve ONNX Runtime anyway")
    elif name == BackendName.torch:
        outcome = "No other backend can serve, so this one serves."
    else:
        outcome = "Serving it as asked."
    logger.warning(
        "serving check failed: %s on %s differs from the CPU result (%s; atol %g, rtol %g). "
        "%s To change this, %s.",
        numerics.provider,
        numerics.device,
        numerics.summary(),
        numerics.atol,
        numerics.rtol,
        outcome,
        " or ".join(steps),
    )
    return fall_back


# Makes a torch model available (sets verdict.prepared) and returns its example feeds.
ToTorch = Callable[[], dict[str, np.ndarray] | None]


def _serve_checked(
    name: BackendName,
    verdict: ExportVerdict,
    opts: ServeOptions,
    timings: dict[str, float],
    notes: list[str],
    example: dict[str, np.ndarray] | None,
    to_torch: ToTorch | None,
) -> tuple[BackendName, Backend, ServingNumerics | None]:
    """Build the backend and run its serving check. If the check fails and torch can serve
    instead, add a note and build the torch backend. `to_torch` is None if there is no torch
    model. The result is the backend that serves, its name and its serving check."""
    if name == BackendName.torch:
        assert to_torch is not None
        example = to_torch()
    backend, numerics = _build_checked(name, verdict, opts, timings, example)
    if _falls_back(numerics, name, opts, to_torch is not None):
        assert to_torch is not None
        notes.append("serving check failed: serving torch")
        name = BackendName.torch
        backend, numerics = _build_checked(name, verdict, opts, timings, to_torch())
        _falls_back(numerics, name, opts, True)
    return name, backend, numerics


def _attach_torch(
    verdict: ExportVerdict, opts: ServeOptions, load: Loader, loaded: LoadedModel | None
) -> dict[str, np.ndarray]:
    """Eager serving of a reused export needs the model: the entry has only the export and
    its verdict."""
    if loaded is None or loaded.model is None:
        loaded, _ = load()
    report(Phase.load)
    prepared = _prepare(loaded, opts)
    verdict.prepared = prepared
    return example_feeds(prepared.input_names, prepared.inputs)


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
    prepared = verdict.prepared
    feeds = example_feeds(prepared.input_names, prepared.inputs) if prepared is not None else None
    # Build the backend. If the serving check fails on a GPU and torch can serve, this returns
    # the torch backend and its name in place of the first one.
    name, backend, numerics = _serve_checked(
        name,
        verdict,
        opts,
        timings,
        notes,
        feeds,
        None if prepared is None else lambda: feeds,
    )
    verdict.take_session()  # a torch backend did not take it

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
        serving_numerics=numerics,
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
    verdict.sample_feeds = entry.samples
    name, notes = choose_backend(verdict, opts, has_torch=True)
    hf_source = source if kind == HF_REPO_DIR else tokenizer_from
    timings: dict[str, float] = {}
    if name != BackendName.torch:
        _drop_models(loaded, reference)
    example_inputs = None if name == BackendName.torch else entry.example_inputs()
    feeds = (
        None if example_inputs is None else example_feeds(tuple(entry.input_names), example_inputs)
    )
    name, backend, numerics = _serve_checked(
        name,
        verdict,
        opts,
        timings,
        notes,
        feeds,
        lambda: _attach_torch(verdict, opts, load, loaded),
    )
    prepared = verdict.prepared
    if prepared is not None:
        input_names, example_inputs = prepared.input_names, prepared.inputs
        axis_bounds = dynamic_bounds(prepared.input_names, prepared.dynamic_shapes)
    else:
        input_names = tuple(entry.input_names)
        axis_bounds = axis_bounds_from_json(entry.axis_bounds)
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
        serving_numerics=numerics,
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
