"""The inference backends. Both map {input name: numpy array} to {output name: numpy array},
so the HTTP layer does not know which one serves."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import onnxruntime as ort
import torch
from torch import nn

from downshift.core.feeds import WIDEN_DTYPES, example_feeds, widen_for_wire
from downshift.core.verdict import BackendName
from downshift.core.verify import first_line, new_session

_CUDA_EP = "CUDAExecutionProvider"
_CPU_EP = "CPUExecutionProvider"


_ort_state = ort.capi.onnxruntime_pybind11_state


class InferenceInputError(ValueError):
    """The backend rejected the inputs themselves (a 400). Only ORT's InvalidArgument and a
    missing input are this. Any other failure inside infer is a 500: predict.py checks names,
    dtypes, ranks and bounds before the inference."""


@dataclass
class IOSpec:
    name: str
    dtype: str | None
    shape: list[int | str | None] | None

    def to_dict(self) -> dict:
        return asdict(self)


def concrete_dim(dim: object) -> int:
    """A fixed axis keeps its size. A dynamic axis (a name, None or a placeholder) becomes 1, so
    the warmup feeds and the /schema example agree."""
    return dim if isinstance(dim, int) and dim > 0 else 1


@dataclass
class BackendMeta:
    name: BackendName
    device: str
    inputs: list[IOSpec]
    outputs: list[IOSpec]

    def to_dict(self) -> dict:
        return asdict(self)


class Backend(Protocol):
    name: BackendName
    input_names: list[str]
    # The execution provider that verify() compared against (CPU), or None without numerics.
    verified_provider: str | None

    def infer(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]: ...

    def metadata(self) -> BackendMeta: ...


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
                f"(providers found: {', '.join(available)}). Install onnxruntime-gpu, or "
                "pass --device cpu"
            )
        return [_CUDA_EP, _CPU_EP]
    return [_CPU_EP]


def _require_cuda_provider(session: ort.InferenceSession, requested: list[str]) -> None:
    """ORT drops the CUDA provider silently when the CUDA or cuDNN libraries do not load, and
    runs on the CPU. --device cuda promises the GPU, so fail instead."""
    if requested[0] != _CUDA_EP:
        return
    actual = session.get_providers()[0]
    if actual != _CUDA_EP:
        raise ValueError(
            f"--device cuda: onnxruntime could not start {_CUDA_EP} and would have served on "
            f"{actual}. onnxruntime-gpu needs the CUDA and cuDNN runtime libraries that it was "
            "built for on the library path of this machine. Its release notes name the CUDA "
            "major version. Downshift silences the load errors of onnxruntime. To see them, "
            "run onnxruntime.preload_dlls() in Python. Or pass --device cpu"
        )


def _session_options(intra_op_threads: int, inter_op_threads: int) -> ort.SessionOptions:
    """Full graph optimization. A thread count of 0 lets ORT choose."""
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
            providers = _ort_providers(resolve_device(device))
            options = _session_options(intra_op_threads, inter_op_threads)
            self.session = new_session(model, providers, options)
            _require_cuda_provider(self.session, providers)
        self.provider = self.session.get_providers()[0]
        self.input_names = [i.name for i in self.session.get_inputs()]
        self.onnx_output_names = [o.name for o in self.session.get_outputs()]
        # Positional output keys, the same for both backends.
        self.output_names = output_names(len(self.onnx_output_names))
        self.verified_provider: str | None = None

    def infer(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        # run() cannot return a bf16 output (no numpy dtype). OrtValues expose DLPack, which
        # _widen_ort_value takes through torch.
        ort_inputs = {name: ort.OrtValue.ortvalue_from_numpy(arr) for name, arr in inputs.items()}
        try:
            outputs = self.session.run_with_ort_values(self.onnx_output_names, ort_inputs)
        except _ort_state.InvalidArgument as exc:
            raise InferenceInputError(first_line(exc)) from exc
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
        dynamic_shapes: tuple | None = None,
    ) -> None:
        if intra_op_threads > 0:  # 0 means "keep the default of torch"
            torch.set_num_threads(intra_op_threads)
        self.device = resolve_device(device)
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise ValueError(
                f"--device {self.device}: CUDA is not available to torch "
                "(torch.cuda.is_available() is False). Install a CUDA build of torch, or "
                "pass --device cpu"
            )
        self.module = module.eval().to(self.device)
        self.input_names = list(input_names)
        self.verified_provider: str | None = None
        # bf16 and fp16 models take float32 on the wire: cast floating inputs to the dtype of
        # the first floating parameter, and widen the outputs again.
        self._param_dtype = next(
            (p.dtype for p in self.module.parameters() if p.is_floating_point()), torch.float32
        )
        self._input_specs = [IOSpec(n, None, None) for n in self.input_names]
        self._output_specs: list[IOSpec] = []
        if example_inputs is not None:
            # One pass over the example gives the dtypes and shapes for /metadata.
            specs = dynamic_shapes or (_BATCH_AXIS,) * len(self.input_names)
            self._input_specs = [
                _spec_from_tensor(n, t, spec)
                for n, t, spec in zip(self.input_names, example_inputs, specs, strict=True)
            ]
            feeds = example_feeds(self.input_names, example_inputs)
            self._output_specs = [_spec_from_array(n, a) for n, a in self.infer(feeds).items()]

    def infer(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        missing = [n for n in self.input_names if n not in inputs]
        if missing:
            raise InferenceInputError(f"missing inputs: {missing}")
        # torch.from_numpy needs a C-contiguous, writable array (base64 views are read-only).
        args = []
        for n in self.input_names:
            t = torch.from_numpy(np.require(inputs[n], requirements=["C", "W"])).to(self.device)
            if t.is_floating_point() and t.dtype != self._param_dtype:
                t = t.to(self._param_dtype)
            args.append(t)
        with torch.inference_mode():
            out = self.module(*args)
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


# Without an adapter's dynamic_shapes, axis 0 is the batch axis.
_BATCH_AXIS = {0: "batch"}


def _dynamic_shape(
    shape: tuple[int, ...], spec: dict | None = _BATCH_AXIS
) -> list[int | str | None]:
    """The shape as ORT reports it: a dynamic axis by its Dim name, others by size. `spec` is
    {axis: Dim} for this input (None: nothing is dynamic)."""
    dynamic = spec or {}
    return [
        getattr(dynamic[i], "__name__", str(dynamic[i])) if i in dynamic else size
        for i, size in enumerate(shape)
    ]


def _widen_ort_value(value: ort.OrtValue) -> np.ndarray:
    """An ORT output for the wire, with bf16 and fp16 widened to float32. bf16 has no numpy
    dtype, so it goes through DLPack into torch (newer onnxruntime builds only)."""
    data_type = value.data_type()
    if data_type == "tensor(float16)":
        widened: np.ndarray = value.numpy().astype(np.float32)
        return widened
    if data_type == "tensor(bfloat16)":
        if not hasattr(value, "__dlpack__"):
            raise RuntimeError(
                f"this model returns bfloat16, and onnxruntime {ort.__version__} cannot hand a "
                "bfloat16 output back to Python. Upgrade onnxruntime or serve with --backend torch"
            )
        return widen_for_wire(torch.from_dlpack(value)).numpy()
    array: np.ndarray = value.numpy()
    return array


def _wire_dtype_name(dtype: torch.dtype) -> str:
    wire = torch.float32 if dtype in WIDEN_DTYPES else dtype
    return str(wire).removeprefix("torch.")


def _spec_from_tensor(name: str, t: torch.Tensor, spec: dict | None = _BATCH_AXIS) -> IOSpec:
    dtype = f"tensor({_wire_dtype_name(t.dtype)})"
    return IOSpec(name, dtype, _dynamic_shape(tuple(t.shape), spec))


def _spec_from_array(name: str, a: np.ndarray) -> IOSpec:
    return IOSpec(name, f"tensor({a.dtype.name})", _dynamic_shape(a.shape))
