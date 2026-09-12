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

_CUDA_EP = "CUDAExecutionProvider"
_CPU_EP = "CPUExecutionProvider"


@dataclass
class IOSpec:
    name: str
    dtype: str | None
    shape: list[int | str | None] | None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BackendMeta:
    name: str
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
    name: str
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


class OnnxRuntimeBackend:
    name = "onnxruntime"

    def __init__(self, model: bytes | str | Path, device: str = "auto") -> None:
        source = model if isinstance(model, bytes) else str(model)
        self.session = ort.InferenceSession(source, providers=_ort_providers(resolve_device(device)))
        self.provider = self.session.get_providers()[0]
        self.input_names = [i.name for i in self.session.get_inputs()]
        self.onnx_output_names = [o.name for o in self.session.get_outputs()]
        # Outputs are keyed positionally so responses look the same from either backend.
        self.output_names = output_names(len(self.onnx_output_names))

    def infer(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        outputs = self.session.run(self.onnx_output_names, inputs)
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

    name = "torch"

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
                _spec_from_tensor(n, t) for n, t in zip(self.input_names, example_inputs, strict=True)
            ]
            feeds = {n: t.numpy() for n, t in zip(self.input_names, example_inputs, strict=True)}
            self._output_specs = [_spec_from_array(n, a) for n, a in self.infer(feeds).items()]

    def infer(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        missing = [n for n in self.input_names if n not in inputs]
        if missing:
            raise KeyError(f"missing inputs: {missing}")
        args = [
            torch.from_numpy(np.ascontiguousarray(inputs[n])).to(self.device)
            for n in self.input_names
        ]
        with torch.inference_mode():
            out = self.module(*args)
        tensors = [out] if isinstance(out, torch.Tensor) else [t for t in out if isinstance(t, torch.Tensor)]
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
    return IOSpec(name, f"tensor({str(t.dtype).removeprefix('torch.')})", _dynamic_shape(tuple(t.shape)))


def _spec_from_array(name: str, a: np.ndarray) -> IOSpec:
    return IOSpec(name, f"tensor({a.dtype.name})", _dynamic_shape(a.shape))
