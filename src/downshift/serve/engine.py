"""From a loaded model to a warmed-up backend. The CLI's `serve` is render(prepare_serving())."""

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import torch

from downshift.export.prevalidated import intake
from downshift.export.verdict import ExportVerdict, build_verdict, prepare_model
from downshift.loading import LoadedModel
from downshift.serve.backends import Backend, OnnxRuntimeBackend, TorchBackend

BackendChoice = Literal["auto", "onnxruntime", "torch"]


@dataclass
class ServeOptions:
    backend: BackendChoice = "auto"
    force_onnx: bool = False  # serve a DEGRADED graph via ORT anyway
    device: str = "auto"
    warmup: int = 3
    k: int = 8
    adapter: str | None = None
    dynamic: dict[str, list[int]] | None = None


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

    @property
    def backend_auto_selected(self) -> bool:
        return self.options.backend == "auto" and not self.forced_onnx

    @property
    def forced_onnx(self) -> bool:
        return self.options.force_onnx and self.verdict.status == "DEGRADED"


def _verdict_for(
    loaded: LoadedModel, reference: LoadedModel | None, opts: ServeOptions
) -> ExportVerdict:
    adapter = opts.adapter or loaded.adapter_hint
    if loaded.onnx_path is not None:
        ref_model = reference.model if reference else None
        ref_inputs = reference.example_inputs if reference else None
        return intake(
            loaded.onnx_path, ref_model, ref_inputs, adapter, k=opts.k, dynamic=opts.dynamic
        )
    assert loaded.model is not None
    prepared = prepare_model(loaded.model, loaded.example_inputs, adapter, opts.dynamic)
    if opts.backend == "torch":
        # Skip the export entirely; the user asked for eager.
        return ExportVerdict(
            status="UNVERIFIED",
            model_family=prepared.family,
            capture_strategy=None,
            opset=None,
            op_types=[],
            numerics=None,
            recommended_backend="torch",
            reason="--backend torch: export skipped",
            input_names=prepared.input_names,
            dynamic_dims=prepared.dynamic_dims,
            prepared=prepared,
        )
    return build_verdict(prepared, k=opts.k)


def choose_backend(verdict: ExportVerdict, opts: ServeOptions) -> tuple[str, list[str]]:
    """Return (backend name, notes for the banner)."""
    notes: list[str] = []
    wanted = verdict.recommended_backend if opts.backend == "auto" else opts.backend
    if opts.force_onnx and verdict.status == "DEGRADED":
        wanted = "onnxruntime"
        notes.append("--force-onnx: serving a DEGRADED graph; outputs may be wrong")

    has_onnx = verdict.onnx_program is not None or verdict.onnx_path is not None
    has_torch = verdict.prepared is not None
    if wanted == "onnxruntime" and not has_onnx:
        notes.append("no ONNX graph available; falling back to torch")
        wanted = "torch"
    if wanted == "torch" and not has_torch:
        raise ValueError("torch backend requested but there is no PyTorch model to run")
    return wanted, notes


def _build_backend(name: str, verdict: ExportVerdict, opts: ServeOptions) -> Backend:
    if name == "onnxruntime":
        if verdict.onnx_path is not None:
            return OnnxRuntimeBackend(verdict.onnx_path, opts.device)
        program: Any = verdict.onnx_program
        return OnnxRuntimeBackend(program.model_proto.SerializeToString(), opts.device)
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


def warmup(state: ServingState, n: int) -> None:
    """Run a few inferences before /ready flips; first-call costs shouldn't hit users."""
    if state.example_inputs is not None:
        feeds = {
            name: t.numpy() if isinstance(t, torch.Tensor) else np.asarray(t)
            for name, t in zip(state.input_names, state.example_inputs, strict=True)
        }
        for _ in range(n):
            state.backend.infer(feeds)
    state.ready = True
