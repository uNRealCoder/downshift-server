"""From a loaded model to a warmed-up backend. The CLI's `serve` is render(prepare_serving())."""

import threading
import time
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np
import torch

from downshift.core.prevalidated import intake
from downshift.core.verdict import BackendName, ExportVerdict, build_verdict, prepare_model
from downshift.loading import LoadedModel
from downshift.serve.backends import Backend, OnnxRuntimeBackend, TorchBackend, resolve_device
from downshift.serve.options import BackendChoice, ServeOptions
from downshift.serve.schemas import normalize_dtype


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
    warmup_stats: WarmupStats | None = None
    # Not JSON-able and not part of a serving state's identity: rebuilt from options.max_concurrency.
    inference_semaphore: threading.Semaphore = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.inference_semaphore = threading.Semaphore(self.options.max_concurrency)

    @property
    def backend_auto_selected(self) -> bool:
        return self.options.backend == BackendChoice.auto and not self.forced_onnx

    @property
    def forced_onnx(self) -> bool:
        return self.options.force_onnx and self.verdict.status == "DEGRADED"

    @cached_property
    def declared_dtypes(self) -> dict[str, str | None]:
        """Input dtypes the backend advertises, computed once rather than per request."""
        return {spec.name: spec.dtype for spec in self.backend.metadata().inputs}


def _verdict_for(
    loaded: LoadedModel, reference: LoadedModel | None, opts: ServeOptions
) -> ExportVerdict:
    adapter = opts.adapter or loaded.adapter_hint
    if loaded.onnx_path is not None:
        ref_model = reference.model if reference else None
        ref_inputs = reference.example_inputs if reference else None
        return intake(
            loaded.onnx_path,
            ref_model,
            ref_inputs,
            adapter,
            k=opts.k,
            dynamic=opts.dynamic,
            atol=opts.atol,
            rtol=opts.rtol,
            seed=opts.seed,
            vary=opts.vary,
        )
    assert loaded.model is not None
    prepared = prepare_model(
        loaded.model, loaded.example_inputs, adapter, opts.dynamic, vary=opts.vary
    )
    if opts.backend == BackendChoice.torch:
        # Skip the export entirely; the user asked for eager.
        return ExportVerdict(
            status="UNVERIFIED",
            model_family=prepared.family,
            capture_strategy=None,
            opset=None,
            op_types={},
            numerics=None,
            recommended_backend=BackendName.torch,
            reason="--backend torch: export skipped",
            input_names=prepared.input_names,
            dynamic_dims=prepared.dynamic_dims,
            prepared=prepared,
        )
    return build_verdict(prepared, k=opts.k, atol=opts.atol, rtol=opts.rtol, seed=opts.seed)


def choose_backend(verdict: ExportVerdict, opts: ServeOptions) -> tuple[BackendName, list[str]]:
    """Return (backend name, notes for the banner)."""
    notes: list[str] = []
    wanted: BackendName = (
        verdict.recommended_backend
        if opts.backend == BackendChoice.auto
        else BackendName(opts.backend)
    )
    if opts.force_onnx and verdict.status == "DEGRADED":
        wanted = BackendName.onnxruntime
        notes.append("--force-onnx: serving a DEGRADED graph; outputs may be wrong")

    has_onnx = verdict.onnx_program is not None or verdict.onnx_path is not None
    has_torch = verdict.prepared is not None
    if wanted == BackendName.onnxruntime and not has_onnx:
        notes.append("no ONNX graph available; falling back to torch")
        wanted = BackendName.torch
    if wanted == BackendName.torch and not has_torch:
        raise ValueError("torch backend requested but there is no PyTorch model to run")
    return wanted, notes


def _reusable_verify_session(verdict: ExportVerdict, opts: ServeOptions) -> Any:
    """The session verify() already built, if the serving options mean the same thing:
    resolved device cpu and both thread counts 0, which is exactly what verify's own
    session (ORT's defaults, CPU-only) already is. Anything else needs its own session."""
    numerics = verdict.numerics
    if numerics is None or numerics.session is None:
        return None
    if resolve_device(opts.device) != "cpu":
        return None
    if opts.intra_op_threads != 0 or opts.inter_op_threads != 0:
        return None
    return numerics.session


def _build_backend(name: BackendName, verdict: ExportVerdict, opts: ServeOptions) -> Backend:
    if name == BackendName.onnxruntime:
        session = _reusable_verify_session(verdict, opts)
        if session is not None:
            return OnnxRuntimeBackend(session=session)
        source: bytes | Path = (
            verdict.onnx_path if verdict.onnx_path is not None else verdict.onnx_bytes
        )
        return OnnxRuntimeBackend(source, opts.device, opts.intra_op_threads, opts.inter_op_threads)
    prepared = verdict.prepared
    assert prepared is not None
    return TorchBackend(prepared.model, prepared.input_names, opts.device, prepared.inputs)


def prepare_serving(
    loaded: LoadedModel, opts: ServeOptions | None = None, reference: LoadedModel | None = None
) -> ServingState:
    opts = opts or ServeOptions()
    verdict = _verdict_for(loaded, reference, opts)
    name, notes = choose_backend(verdict, opts)
    backend = _build_backend(name, verdict, opts)

    if verdict.prepared is not None:
        input_names = verdict.prepared.input_names
        example_inputs = verdict.prepared.inputs
    else:
        input_names = tuple(backend.input_names)
        example_inputs = None

    state = ServingState(
        loaded.source, verdict, backend, input_names, opts, example_inputs, notes=notes
    )
    warmup(state, opts.warmup)
    return state


def serving_state_from_artifact(
    source: str,
    onnx_path: Path,
    verdict: ExportVerdict,
    opts: ServeOptions,
    input_names: tuple[str, ...],
    notes: list[str] | None = None,
    example_inputs: tuple | None = None,
) -> ServingState:
    """Rebuild a ServingState in a `serve --workers N` worker from the ONNX graph and
    verdict a parent process already exported and verified: no capture, no verify, just a
    session over the artifact. `input_names` and `notes` are the parent's own (the verdict
    it shipped has no `prepared` to derive them from); `example_inputs` are real feeds the
    parent saved alongside the graph when it had any, loaded by the caller from the .npz
    sidecar - otherwise warmup() synthesizes them (see synthesize_feeds).
    """
    verdict.onnx_path = onnx_path
    backend = _build_backend(BackendName.onnxruntime, verdict, opts)
    state = ServingState(
        source, verdict, backend, input_names, opts, example_inputs, notes=list(notes or [])
    )
    warmup(state, opts.warmup)
    return state


def synthesize_feeds(backend: Backend) -> dict[str, np.ndarray]:
    """One dummy array per input the backend declares, for warming up a backend with no
    example inputs (a bare .onnx served without --reference). Dynamic or unknown axes
    become 1; floats are randn, integers and bools are zero/false."""
    feeds: dict[str, np.ndarray] = {}
    for spec in backend.metadata().inputs:
        shape = tuple(dim if isinstance(dim, int) and dim > 0 else 1 for dim in (spec.shape or [1]))
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
            else {
                name: t.numpy() if isinstance(t, torch.Tensor) else np.asarray(t)
                for name, t in zip(state.input_names, example_inputs, strict=True)
            }
        )
        start = time.perf_counter()
        for _ in range(n):
            state.backend.infer(feeds)
        mean_ms = (time.perf_counter() - start) / n * 1000
    stats = WarmupStats(count=n, mean_ms=mean_ms, synthesized=synthesized)
    state.warmup_stats = stats
    state.ready = True
    return stats
