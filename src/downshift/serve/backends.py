"""Inference backends. Both take and return dicts of numpy arrays keyed by input name,
so the HTTP layer doesn't care which one is behind it.
"""

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import onnxruntime as ort
import torch
from torch import nn

from downshift.core.verdict import BackendName, ExportVerdict
from downshift.core.verify import quiet_ort_logging

_CUDA_EP = "CUDAExecutionProvider"
_CPU_EP = "CPUExecutionProvider"

# bf16/fp16 have no numpy dtype; the wire contract for both is float32 (B1).
_WIDEN_DTYPES = (torch.bfloat16, torch.float16)

_ort_state = ort.capi.onnxruntime_pybind11_state
# ORT raises these for shape/dtype/graph mismatches the client caused; a Fail whose message
# mentions "shape" is the same story (e.g. a Reshape whose target size doesn't match the input).
_ORT_CLIENT_ERRORS = (_ort_state.InvalidArgument, _ort_state.InvalidGraph)


# Torch has no exception class for "the client's tensor was the wrong shape or dtype": those
# are plain RuntimeError/ValueError with these words in the message. Anything without one
# (an unsupported op, a CUDA failure, a bug in the module) is the server's problem and stays a
# 500, as does an out-of-memory error, whose message also says "size".
_TORCH_CLIENT_MARKERS = (
    "shape",
    "size",
    "dimension",
    "dtype",
    "scalar type",
    "must match",
    "should be the same",
    "broadcast",
    "out of range",
)


def _is_client_input_message(message: str) -> bool:
    lowered = message.lower()
    return "out of memory" not in lowered and any(m in lowered for m in _TORCH_CLIENT_MARKERS)


class InferenceInputError(ValueError):
    """The backend rejected the inputs themselves; safe to report to the client as a 400."""


@dataclass
class IOSpec:
    name: str
    dtype: str | None
    shape: list[int | str | None] | None

    def to_dict(self) -> dict:
        return asdict(self)


def concrete_dim(dim: object) -> int:
    """One axis pinned to a size a request can actually have. A dynamic axis (ORT reports it
    as a name, torch as "batch", either may be None or a non-positive placeholder) becomes 1:
    the smallest thing that is still valid, so warmup feeds and /schema's example agree."""
    return dim if isinstance(dim, int) and dim > 0 else 1


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
    # The execution provider verify() compared the graph against (always CPU today, since
    # verify._to_session is CPU-only), or None when the verdict never ran numerics at all.
    # Set by engine.py's session-creation call sites, once the verdict is known; the banner
    # reads it to print a "Verified on" row (U6).
    verified_provider: str | None

    def infer(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]: ...

    def metadata(self) -> BackendMeta: ...


def verified_provider_for(verdict: ExportVerdict) -> str | None:
    """See `Backend.verified_provider`. Called once, from engine._build_backend, right after a
    backend is constructed, so every serving-state builder (including a `--workers` torch
    worker) gets it the same way."""
    return _CPU_EP if verdict.numerics is not None else None


# The positional name output_names() gives the first output; the one that TextIO.predictions
# reads (M5), so callers don't hand-carry the "output_0" literal.
FIRST_OUTPUT_NAME = "output_0"


def output_names(count: int) -> list[str]:
    return [f"output_{i}" for i in range(count)]


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _ort_providers(device: str) -> list[str]:
    available = ort.get_available_providers()
    if device == "cuda":
        if _CUDA_EP not in available:
            raise ValueError(
                "--device cuda: onnxruntime has no CUDA execution provider available "
                f"(providers found: {', '.join(available)}); install onnxruntime-gpu, or "
                "pass --device cpu"
            )
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
        model: bytes | str | Path | None = None,
        device: str = "auto",
        intra_op_threads: int = 0,
        inter_op_threads: int = 0,
        *,
        session: ort.InferenceSession | None = None,
    ) -> None:
        if session is not None:
            self.session = session
        else:
            assert model is not None
            quiet_ort_logging()  # must run before any session exists (P6)
            source = model if isinstance(model, bytes) else str(model)
            providers = _ort_providers(resolve_device(device))
            options = _session_options(intra_op_threads, inter_op_threads)
            self.session = ort.InferenceSession(source, sess_options=options, providers=providers)
        self.provider = self.session.get_providers()[0]
        self.input_names = [i.name for i in self.session.get_inputs()]
        self.onnx_output_names = [o.name for o in self.session.get_outputs()]
        # Outputs are keyed positionally so responses look the same from either backend.
        self.output_names = output_names(len(self.onnx_output_names))
        self.verified_provider: str | None = None

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
        intra_op_threads: int = 0,
    ) -> None:
        if intra_op_threads > 0:  # 0 means "leave torch's own default alone"
            torch.set_num_threads(intra_op_threads)
        self.device = resolve_device(device)
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise ValueError(
                f"--device {self.device}: CUDA is not available to torch "
                "(torch.cuda.is_available() is False); install a CUDA build of torch, or "
                "pass --device cpu"
            )
        self.module = module.eval().to(self.device)
        self.input_names = list(input_names)
        self.verified_provider: str | None = None
        # bf16/fp16 models take float32 on the wire; the module itself stays in its own
        # dtype, so floating inputs are cast to it here and outputs widened back on the way
        # out (B1). Mixed-dtype modules aren't a case downshift produces; the first parameter
        # found wins.
        self._param_dtype = next(
            (p.dtype for p in self.module.parameters() if p.is_floating_point()), torch.float32
        )
        self._input_specs = [IOSpec(n, None, None) for n in self.input_names]
        self._output_specs: list[IOSpec] = []
        if example_inputs is not None:
            # One pass over the example fills in dtypes and shapes for /metadata.
            self._input_specs = [
                _spec_from_tensor(n, t)
                for n, t in zip(self.input_names, example_inputs, strict=True)
            ]
            feeds = example_feeds(self.input_names, example_inputs)
            self._output_specs = [_spec_from_array(n, a) for n, a in self.infer(feeds).items()]

    def infer(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        missing = [n for n in self.input_names if n not in inputs]
        if missing:
            raise InferenceInputError(f"missing inputs: {missing}")
        # torch.from_numpy needs a C-contiguous, writable array. np.require copies only when
        # one of those is missing (base64 inputs arrive as read-only frombuffer views).
        args = []
        for n in self.input_names:
            t = torch.from_numpy(np.require(inputs[n], requirements=["C", "W"])).to(self.device)
            if t.is_floating_point() and t.dtype != self._param_dtype:
                t = t.to(self._param_dtype)
            args.append(t)
        try:
            with torch.inference_mode():
                out = self.module(*args)
        except IndexError as exc:
            raise InferenceInputError(str(exc)) from exc
        except (RuntimeError, ValueError) as exc:
            message = str(exc)
            if not _is_client_input_message(message):
                raise
            raise InferenceInputError(message) from exc
        if isinstance(out, torch.Tensor):
            tensors = [out]
        else:
            tensors = [t for t in out if isinstance(t, torch.Tensor)]
        return {
            name: widen_for_wire(t.detach().cpu()).numpy()
            for name, t in zip(output_names(len(tensors)), tensors, strict=True)
        }

    def metadata(self) -> BackendMeta:
        return BackendMeta(self.name, self.device, self._input_specs, self._output_specs)


def _dynamic_shape(shape: tuple[int, ...]) -> list[int | str | None]:
    # Axis 0 is dynamic for anything we serve; report it the way ORT does.
    return ["batch", *shape[1:]] if shape else []


def widen_for_wire(t: torch.Tensor) -> torch.Tensor:
    """bf16/fp16 -> float32, everything else unchanged (B1)."""
    return t.float() if t.dtype in _WIDEN_DTYPES else t


def example_feeds(input_names: Sequence[str], example_inputs: tuple) -> dict[str, np.ndarray]:
    """Example inputs as the numpy feeds a backend takes, keyed by forward-argument name.

    bf16/fp16 tensors have no numpy dtype, so they're widened to float32 first, same as the
    wire contract (B1).
    """
    return {
        name: widen_for_wire(t).numpy() if isinstance(t, torch.Tensor) else np.asarray(t)
        for name, t in zip(input_names, example_inputs, strict=True)
    }


def _wire_dtype_name(dtype: torch.dtype) -> str:
    wire = torch.float32 if dtype in _WIDEN_DTYPES else dtype
    return str(wire).removeprefix("torch.")


def _spec_from_tensor(name: str, t: torch.Tensor) -> IOSpec:
    return IOSpec(name, f"tensor({_wire_dtype_name(t.dtype)})", _dynamic_shape(tuple(t.shape)))


def _spec_from_array(name: str, a: np.ndarray) -> IOSpec:
    return IOSpec(name, f"tensor({a.dtype.name})", _dynamic_shape(a.shape))
