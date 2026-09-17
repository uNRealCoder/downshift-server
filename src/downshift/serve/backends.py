"""Inference backends. Both take and return dicts of numpy arrays keyed by input name,
so the HTTP layer doesn't care which one is behind it.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import onnxruntime as ort
import torch
from torch import nn

from downshift.export.verdict import BackendName

_CUDA_EP = "CUDAExecutionProvider"
_CPU_EP = "CPUExecutionProvider"

_ort_state = ort.capi.onnxruntime_pybind11_state
# ORT raises these for shape/dtype/graph mismatches the client caused; a Fail whose message
# mentions "shape" is the same story (e.g. a Reshape whose target size doesn't match the input).
_ORT_CLIENT_ERRORS = (_ort_state.InvalidArgument, _ort_state.InvalidGraph)


class InferenceInputError(ValueError):
    """The backend rejected the inputs themselves; safe to report to the client as a 400."""


@dataclass
class IOSpec:
    name: str
    dtype: str | None
    shape: list[int | str | None] | None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BackendMeta:
    name: BackendName
    device: str
    inputs: list[IOSpec]
    outputs: list[IOSpec]

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "device": self.device,
            "inputs": [i.to_dict() for i in self.inputs],
            "outputs": [o.to_dict() for o in self.outputs],
        }


class Backend(Protocol):
    name: BackendName
    input_names: list[str]

    def infer(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]: ...

    def metadata(self) -> BackendMeta: ...


def output_names(count: int) -> list[str]:
    return [f"output_{i}" for i in range(count)]


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _ort_providers(device: str) -> list[str]:
    available = ort.get_available_providers()
    if device == "cuda" and _CUDA_EP in available:
        return [_CUDA_EP, _CPU_EP]
    return [_CPU_EP]


def _session_options(intra_op_threads: int, inter_op_threads: int) -> ort.SessionOptions:
    """Max graph optimization always on. Thread counts of 0 mean "let ONNX Runtime choose",
    which is also its own default, so this is safe to set unconditionally."""
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.intra_op_num_threads = intra_op_threads
    options.inter_op_num_threads = inter_op_threads
    return options


class OnnxRuntimeBackend:
    name = BackendName.onnxruntime

    def __init__(
        self,
        model: bytes | str | Path,
        device: str = "auto",
        intra_op_threads: int = 0,
        inter_op_threads: int = 0,
    ) -> None:
        source = model if isinstance(model, bytes) else str(model)
        providers = _ort_providers(resolve_device(device))
        options = _session_options(intra_op_threads, inter_op_threads)
        self.session = ort.InferenceSession(source, sess_options=options, providers=providers)
        self.provider = self.session.get_providers()[0]
        self.input_names = [i.name for i in self.session.get_inputs()]
        self.onnx_output_names = [o.name for o in self.session.get_outputs()]
        # Outputs are keyed positionally so responses look the same from either backend.
        self.output_names = output_names(len(self.onnx_output_names))

    def infer(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        try:
            outputs = self.session.run(self.onnx_output_names, inputs)
        except _ORT_CLIENT_ERRORS as exc:
            raise InferenceInputError(str(exc).splitlines()[0]) from exc
        except _ort_state.Fail as exc:
            message = str(exc)
            if "shape" not in message.lower():
                raise
            raise InferenceInputError(message.splitlines()[0]) from exc
        return dict(zip(self.output_names, outputs, strict=True))

    def metadata(self) -> BackendMeta:
        def spec(name: str, node) -> IOSpec:
            return IOSpec(name, node.type, list(node.shape) if node.shape else None)

        return BackendMeta(
            name=self.name,
            device=self.provider,
            inputs=[spec(i.name, i) for i in self.session.get_inputs()],
            outputs=[
                spec(name, o)
                for name, o in zip(self.output_names, self.session.get_outputs(), strict=True)
            ],
        )


class TorchBackend:
    """Eager PyTorch. The fallback path, and a first-class one: same contract as ORT."""

    name = BackendName.torch

    def __init__(
        self,
        module: nn.Module,
        input_names: tuple[str, ...],
        device: str = "auto",
        example_inputs: tuple | None = None,
    ) -> None:
        self.device = resolve_device(device)
        self.module = module.eval().to(self.device)
        self.input_names = list(input_names)
        self._input_specs = [IOSpec(n, None, None) for n in self.input_names]
        self._output_specs: list[IOSpec] = []
        if example_inputs is not None:
            # One pass over the example fills in dtypes and shapes for /metadata.
            self._input_specs = [
                _spec_from_tensor(n, t)
                for n, t in zip(self.input_names, example_inputs, strict=True)
            ]
            feeds = {n: t.numpy() for n, t in zip(self.input_names, example_inputs, strict=True)}
            self._output_specs = [_spec_from_array(n, a) for n, a in self.infer(feeds).items()]

    def infer(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        missing = [n for n in self.input_names if n not in inputs]
        if missing:
            raise KeyError(f"missing inputs: {missing}")
        # torch.from_numpy needs a C-contiguous, writable array. np.require copies only when
        # one of those is missing (base64 inputs arrive as read-only frombuffer views).
        args = [
            torch.from_numpy(np.require(inputs[n], requirements=["C", "W"])).to(self.device)
            for n in self.input_names
        ]
        try:
            with torch.inference_mode():
                out = self.module(*args)
        except (RuntimeError, IndexError, ValueError) as exc:
            if "out of memory" in str(exc).lower():
                raise
            raise InferenceInputError(str(exc)) from exc
        if isinstance(out, torch.Tensor):
            tensors = [out]
        else:
            tensors = [t for t in out if isinstance(t, torch.Tensor)]
        return {
            name: t.detach().cpu().numpy()
            for name, t in zip(output_names(len(tensors)), tensors, strict=True)
        }

    def metadata(self) -> BackendMeta:
        return BackendMeta(self.name, self.device, self._input_specs, self._output_specs)


def _dynamic_shape(shape: tuple[int, ...]) -> list[int | str | None]:
    # Axis 0 is dynamic for anything we serve; report it the way ORT does.
    return ["batch", *shape[1:]] if shape else []


def _spec_from_tensor(name: str, t: torch.Tensor) -> IOSpec:
    dtype = str(t.dtype).removeprefix("torch.")
    return IOSpec(name, f"tensor({dtype})", _dynamic_shape(tuple(t.shape)))


def _spec_from_array(name: str, a: np.ndarray) -> IOSpec:
    return IOSpec(name, f"tensor({a.dtype.name})", _dynamic_shape(a.shape))
