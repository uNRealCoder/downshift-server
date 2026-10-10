"""The inference backends. Both take and return dicts of numpy arrays. The key is the input
name. The HTTP layer therefore does not need to know which backend is behind it.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import onnxruntime as ort
import torch
from torch import nn

from downshift.core.feeds import WIDEN_DTYPES, example_feeds, widen_for_wire
from downshift.core.verdict import BackendName, ExportVerdict
from downshift.core.verify import first_line, new_session

_CUDA_EP = "CUDAExecutionProvider"
_CPU_EP = "CPUExecutionProvider"


_ort_state = ort.capi.onnxruntime_pybind11_state


class InferenceInputError(ValueError):
    """The backend rejected the inputs themselves. Downshift can report this to the client as a
    400.

    Only ORT says this by type (InvalidArgument). Each other failure inside infer is a fault of
    the server (a 500). predict.py checks the names, dtypes, ranks and bounds before infer. A
    request that gets this far is a request that the model declared it takes."""


@dataclass
class IOSpec:
    name: str
    dtype: str | None
    shape: list[int | str | None] | None

    def to_dict(self) -> dict:
        return asdict(self)


def concrete_dim(dim: object) -> int:
    """One axis pinned to a size that a request can really have. A dynamic axis (ORT reports it
    as a name, torch as "batch". Both can be None or a placeholder that is not positive) becomes
    1. This is the smallest value that is still valid. The warmup feeds and the example of
    /schema then agree."""
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
    # The execution provider that verify() compared the graph against (always CPU today,
    # because verify.load_session is CPU only). None if the verdict never ran numerics. The
    # places in engine.py that create sessions set it, when the verdict is known. The banner
    # reads it to print a "Verified on" row (U6).
    verified_provider: str | None

    def infer(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]: ...

    def metadata(self) -> BackendMeta: ...


def verified_provider_for(verdict: ExportVerdict) -> str | None:
    """See `Backend.verified_provider`. It is called one time, from engine._build_backend,
    directly after a backend is constructed. Each serving-state builder (including a `--workers`
    torch worker) therefore gets it in the same way."""
    return _CPU_EP if verdict.numerics is not None else None


# The positional name that output_names() gives to the first output. TextIO.predictions reads it
# (M5). Callers therefore do not need to carry the "output_0" literal by hand.
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
    """ONNX Runtime lists CUDAExecutionProvider as available when the onnxruntime-gpu wheel is
    installed. It removes it from a session without a message if the CUDA or cuDNN libraries
    that the wheel was built for cannot be loaded. The session then runs on the CPU. --device
    cuda is a promise about where the model runs. If the promise fails, fail with a clear error,
    like the check of torch for missing CUDA."""
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
    """Maximum graph optimization is always on. A thread count of 0 means "let ONNX Runtime
    choose". This is also its own default, so it is safe to always set it."""
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
        # The output keys are positional. The responses then look the same for both backends.
        self.output_names = output_names(len(self.onnx_output_names))
        self.verified_provider: str | None = None

    def infer(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        # Use run_with_ort_values and not run(). A bf16 output has no numpy dtype of its own
        # (OrtValue.numpy() raises "No corresponding Numpy type"). The normal session.run()
        # can therefore never return one. Only the OrtValue form exposes DLPack. Torch
        # understands DLPack for bf16. _widen_ort_value can then use the same float32 wire
        # contract (B1) as TorchBackend.infer.
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
        # bf16 and fp16 models take float32 on the wire. The module itself stays in its own
        # dtype. Floating inputs are cast to it here, and outputs are widened again on the way
        # out (B1). Downshift does not produce modules with mixed dtypes. The first parameter
        # that downshift finds wins.
        self._param_dtype = next(
            (p.dtype for p in self.module.parameters() if p.is_floating_point()), torch.float32
        )
        self._input_specs = [IOSpec(n, None, None) for n in self.input_names]
        self._output_specs: list[IOSpec] = []
        if example_inputs is not None:
            # One pass over the example fills in the dtypes and shapes for /metadata.
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
        # torch.from_numpy needs an array that is C-contiguous and writable. np.require copies
        # only if one of these is missing. (Base64 inputs arrive as frombuffer views that are
        # read-only.)
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


# Without the dynamic_shapes of an adapter, downshift treats axis 0 as the batch axis.
_BATCH_AXIS = {0: "batch"}


def _dynamic_shape(
    shape: tuple[int, ...], spec: dict | None = _BATCH_AXIS
) -> list[int | str | None]:
    """The shape as ORT reports it: a dynamic axis by its name, and each other axis by its size.
    `spec` is the {axis: torch.export.Dim} of an adapter for this input (None: nothing is
    dynamic). An input such as the edge_index [2, E] of a graph then keeps its fixed 2 at the
    start."""
    dynamic = spec or {}
    return [
        getattr(dynamic[i], "__name__", str(dynamic[i])) if i in dynamic else size
        for i, size in enumerate(shape)
    ]


def _widen_ort_value(value: ort.OrtValue) -> np.ndarray:
    """An ORT output as a numpy array that is ready for the wire. bf16 and fp16 are widened to
    float32 (B1), as in TorchBackend.infer. fp16 goes through numpy, which has the dtype. bf16
    has none (OrtValue.numpy() raises "No corresponding Numpy type"). It therefore takes the
    DLPack bridge into torch. Only newer onnxruntime builds expose this on OrtValue."""
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
