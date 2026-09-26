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


# Torch has no exception class for "the client's tensor was the wrong shape, dtype or index":
# those are plain RuntimeError/ValueError, in as many different phrasings as there are ops
# (a shape mismatch, a bad dtype, an out-of-bounds target, ...). Enumerating client-input
# phrasings is whack-a-mole and under-classifies; enumerating the server-side ones is a much
# shorter, more stable list - a hardware/driver fault or a torch-internal bug - so those are
# what stays a 500, and everything else is presumed to be the client's fault.
_TORCH_SERVER_MARKERS = (
    "out of memory",
    "cuda error",
    "cudnn error",
    "cublas",
    "illegal memory access",
    "internal assert",
    "not implemented for",  # a missing kernel/op, not a bad value
    "can't allocate memory",  # torch's CPU allocator ("DefaultCPUAllocator: can't allocate ...")
    "expected all tensors to be on the same device",  # a model bug, not the request's
)


def _is_client_input_message(message: str) -> bool:
    lowered = message.lower()
    return not any(m in lowered for m in _TORCH_SERVER_MARKERS)


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


def _require_cuda_provider(session: ort.InferenceSession, requested: list[str]) -> None:
    """ONNX Runtime lists CUDAExecutionProvider as available whenever the onnxruntime-gpu wheel
    is installed, and quietly drops it from a session when the CUDA or cuDNN libraries that wheel
    was built for cannot be loaded, so the session runs on the CPU. --device cuda is a promise
    about where the model runs; break it loudly, like torch's own missing-CUDA check."""
    if requested[0] != _CUDA_EP:
        return
    actual = session.get_providers()[0]
    if actual != _CUDA_EP:
        raise ValueError(
            f"--device cuda: onnxruntime could not start {_CUDA_EP} and would have served on "
            f"{actual}. onnxruntime-gpu needs the CUDA and cuDNN runtime libraries it was built "
            "for on this machine's library path (its release notes name the CUDA major version); "
            "downshift silences onnxruntime's own load errors, so run "
            "onnxruntime.preload_dlls() in Python to see them, or pass --device cpu"
        )


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
            _require_cuda_provider(self.session, providers)
        self.provider = self.session.get_providers()[0]
        self.input_names = [i.name for i in self.session.get_inputs()]
        self.onnx_output_names = [o.name for o in self.session.get_outputs()]
        # Outputs are keyed positionally so responses look the same from either backend.
        self.output_names = output_names(len(self.onnx_output_names))
        self.verified_provider: str | None = None

    def infer(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        # run_with_ort_values, not run(): a bf16 output has no numpy dtype of its own
        # (OrtValue.numpy() raises "No corresponding Numpy type"), so ordinary session.run()
        # can never return one. Only the OrtValue form exposes DLPack, which torch does
        # understand for bf16, letting _widen_ort_value reuse the same float32 wire
        # contract (B1) as TorchBackend.infer.
        ort_inputs = {name: ort.OrtValue.ortvalue_from_numpy(arr) for name, arr in inputs.items()}
        try:
            outputs = self.session.run_with_ort_values(self.onnx_output_names, ort_inputs)
        except _ORT_CLIENT_ERRORS as exc:
            raise InferenceInputError(str(exc).splitlines()[0]) from exc
        except _ort_state.Fail as exc:
            message = str(exc)
            if "shape" not in message.lower():
                raise
            raise InferenceInputError(message.splitlines()[0]) from exc
        return dict(zip(self.output_names, (_widen_ort_value(o) for o in outputs), strict=True))

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


def _widen_ort_value(value: ort.OrtValue) -> np.ndarray:
    """An ORT output as a wire-ready numpy array, bf16/fp16 widened to float32 (B1) like
    TorchBackend.infer. fp16 goes through numpy, which has the dtype. bf16 has none
    (OrtValue.numpy() raises "No corresponding Numpy type"), so it takes the DLPack bridge
    into torch, which only newer onnxruntime builds expose on OrtValue."""
    data_type = value.data_type()
    if data_type == "tensor(float16)":
        widened: np.ndarray = value.numpy().astype(np.float32)
        return widened
    if data_type == "tensor(bfloat16)":
        if not hasattr(value, "__dlpack__"):
            raise RuntimeError(
                f"this model returns bfloat16, and onnxruntime {ort.__version__} cannot hand a "
                "bfloat16 output back to Python; upgrade onnxruntime or serve with --backend torch"
            )
        return widen_for_wire(torch.from_dlpack(value)).numpy()
    array: np.ndarray = value.numpy()
    return array


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
