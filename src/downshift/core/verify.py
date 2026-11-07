"""Numerical verification. An export is not a success until this passes.

K samples run through torch and through the ONNX graph. The dynamic axes vary, so some samples
have shapes that the exporter did not see. This finds a graph that froze a shape or a branch.
"""

import logging
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from downshift.adapters.base import VaryFn
from downshift.core.shapes import alternative_sizes, dim_bounds, pick_size, resize_axis
from downshift.logs import REPORT_LOGGER
from downshift.settings import DEFAULT_SAMPLES, TOLERANCES

report_logger = logging.getLogger(REPORT_LOGGER)


class OnnxRuntimeError(RuntimeError):
    """ONNX Runtime could not load the graph or run a sample. A numeric mismatch is not this."""


@dataclass
class WorstMismatch:
    """The element with the largest absolute error, across all samples."""

    sample: int
    output: int
    index: tuple[int, ...]
    expected: float
    got: float
    input_shapes: list[tuple[int, ...]]


@dataclass
class NumericsReport:
    samples_tested: int
    max_abs_err: float
    max_rel_err: float
    failures: int
    # None if the baseline sample failed, False if only a varied shape failed.
    shape_generalization: bool | None
    tolerance_abs: float
    tolerance_rel: float
    tolerance_dtype: str = "float32"  # the dtype whose default tolerances applied
    tolerance_overridden: bool = False  # --atol or --rtol set the values
    baseline_failed: bool = False
    worst: WorstMismatch | None = None
    sample_shapes: list[list[tuple[int, ...]]] = field(default_factory=list)
    output_shapes: list[list[tuple[int, ...]]] = field(default_factory=list)  # [sample][output]
    seed: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.failures == 0

    def to_dict(self) -> dict:
        return asdict(self) | {"passed": self.passed}

    @classmethod
    def from_dict(cls, data: dict) -> "NumericsReport":
        """The inverse of to_dict()."""
        values = {f.name: data[f.name] for f in fields(cls) if f.name in data}
        if values.get("worst") is not None:
            worst = dict(values["worst"])
            worst["index"] = tuple(worst["index"])
            worst["input_shapes"] = [tuple(s) for s in worst["input_shapes"]]
            values["worst"] = WorstMismatch(**worst)
        for name in ("sample_shapes", "output_shapes"):
            values[name] = [[tuple(s) for s in sample] for sample in values.get(name, [])]
        return cls(**values)


def default_tolerances(model: torch.nn.Module) -> tuple[str, float, float]:
    """(dtype name, atol, rtol) for the least precise floating dtype of the parameters.
    float64 applies only if it is the only floating dtype."""
    dtypes = {p.dtype for p in model.parameters() if p.is_floating_point()}
    if torch.bfloat16 in dtypes:
        name = "bfloat16"
    elif torch.float16 in dtypes:
        name = "float16"
    elif dtypes == {torch.float64}:
        name = "float64"
    else:
        name = "float32"
    return (name, *TOLERANCES[name])


def make_shared_axis0_vary_fn(base_inputs: tuple, dynamic_shapes: tuple) -> VaryFn:
    """The default sampler: all dynamic tensors share one axis-0 size (the batch case)."""
    base_size = None
    axis0_spec = None
    for t, spec in zip(base_inputs, dynamic_shapes, strict=True):
        if spec:
            base_size, axis0_spec = t.shape[0], spec
            break
    lo, hi = dim_bounds(axis0_spec, 0)
    candidates = alternative_sizes(base_size, lo, hi) if base_size is not None else []

    def vary(i: int) -> tuple:
        if i == 0 or not candidates:
            return base_inputs
        size = pick_size(candidates)
        return tuple(
            resize_axis(t, 0, size) if isinstance(t, torch.Tensor) and spec else t
            for t, spec in zip(base_inputs, dynamic_shapes, strict=True)
        )

    return vary


def new_session(
    model: bytes | str | Path,
    providers: list[str] | None = None,
    options: ort.SessionOptions | None = None,
) -> ort.InferenceSession:
    """Build an InferenceSession (CPU by default). The C++ logger of ORT would print each
    client error to stderr. It can only be silenced before the first session."""
    ort.set_default_logger_severity(4)
    source = model if isinstance(model, bytes) else str(model)
    return ort.InferenceSession(
        source, sess_options=options, providers=providers or ["CPUExecutionProvider"]
    )


def load_session(onnx_model: bytes | str | Path | ort.InferenceSession) -> ort.InferenceSession:
    """The CPU session for verify(). Raises OnnxRuntimeError if ORT cannot load the graph."""
    if isinstance(onnx_model, ort.InferenceSession):
        return onnx_model
    try:
        return new_session(onnx_model)
    except Exception as exc:  # noqa: BLE001 - a verdict, not a crash
        raise OnnxRuntimeError(
            f"onnxruntime could not load the exported graph: {first_line(exc)}"
        ) from exc


def as_tensor_list(output) -> list[torch.Tensor]:
    if isinstance(output, torch.Tensor):
        return [output]
    if isinstance(output, (tuple, list)):
        return [t for t in output if isinstance(t, torch.Tensor)]
    raise TypeError(f"can't compare model output of type {type(output).__name__}")


def first_line(exc: Exception) -> str:
    text = str(exc)
    return text.splitlines()[0] if text else type(exc).__name__


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """bf16 and fp16 have no faithful numpy dtype: widen floats to float32 first."""
    tensor = tensor.detach()
    return (tensor.float() if tensor.is_floating_point() else tensor).numpy()


def _bounds_text(dynamic_shapes: tuple | None) -> str | None:
    parts = [
        f"axis {axis} in {dim_bounds(spec, axis)}"
        for spec in dynamic_shapes or ()
        if spec
        for axis in spec
    ]
    return ", ".join(parts) or None


# (output index, element index, expected, got, abs_err) of the worst element of one sample.
WorstCandidate = tuple[int, tuple[int, ...], float, float, float]


def _compare_sample(
    torch_outs: list[torch.Tensor],
    ort_outs: list,
    atol: float,
    rtol: float,
    sample_index: int,
) -> tuple[float, float, bool, str | None, WorstCandidate | None]:
    """(max_abs_err, max_rel_err, failed, note, worst) for one sample.

    An element fails by np.isclose(equal_nan=False) semantics: NaN never matches. A different
    output count or shape fails the sample with a note. `worst` is None if no output compared.
    """
    note: str | None = None
    if len(torch_outs) != len(ort_outs):
        note = (
            f"sample {sample_index}: {len(torch_outs)} torch outputs "
            f"vs {len(ort_outs)} onnxruntime outputs"
        )
        return 0.0, 0.0, True, note, None

    sample_abs = sample_rel = 0.0
    failed = False
    worst: WorstCandidate | None = None
    worst_abs = -1.0  # a NaN error never compares greater, so it is never the worst
    for idx, (expected, got) in enumerate(zip(torch_outs, ort_outs, strict=True)):
        expected_np = tensor_to_numpy(expected)
        got_np = np.asarray(got)
        if expected_np.shape != got_np.shape:
            failed = True
            note = (
                f"sample {sample_index}: output_{idx} shape {expected_np.shape} from torch "
                f"vs {got_np.shape} from onnxruntime"
            )
            continue

        expected64 = expected_np.astype(np.float64)
        got64 = got_np.astype(np.float64)
        abs_err = np.abs(expected64 - got64)
        output_abs = float(abs_err.max(initial=0.0))
        sample_abs = max(sample_abs, output_abs)
        rel_err = abs_err / (np.abs(expected64) + 1e-8)
        sample_rel = max(sample_rel, float(rel_err.max(initial=0.0)))
        if abs_err.size and output_abs > worst_abs:
            flat = int(np.argmax(abs_err))
            index = tuple(int(x) for x in np.unravel_index(flat, abs_err.shape))
            worst_abs = float(abs_err.flat[flat])
            worst = (idx, index, float(expected64.flat[flat]), float(got64.flat[flat]), worst_abs)
        if not np.isclose(got64, expected64, rtol=rtol, atol=atol, equal_nan=False).all():
            failed = True
    return sample_abs, sample_rel, failed, note, worst


def _shapes(sample: tuple) -> list[tuple[int, ...]]:
    return [tuple(t.shape) if isinstance(t, torch.Tensor) else () for t in sample]


def verify(
    model: torch.nn.Module,
    onnx_model: bytes | str | Path | ort.InferenceSession,
    base_inputs: tuple,
    dynamic_shapes: tuple | None = None,
    vary_fn: VaryFn | None = None,
    k: int = DEFAULT_SAMPLES,
    atol: float | None = None,
    rtol: float | None = None,
    seed: int = 0,
) -> NumericsReport:
    if vary_fn is None:
        if dynamic_shapes is None:
            raise ValueError("verify() needs either dynamic_shapes or an explicit vary_fn")
        vary_fn = make_shared_axis0_vary_fn(base_inputs, dynamic_shapes)

    tolerance_dtype, default_atol, default_rtol = default_tolerances(model)
    report = NumericsReport(
        samples_tested=k,
        max_abs_err=0.0,
        max_rel_err=0.0,
        failures=0,
        shape_generalization=None,
        tolerance_abs=default_atol if atol is None else atol,
        tolerance_rel=default_rtol if rtol is None else rtol,
        tolerance_dtype=tolerance_dtype,
        tolerance_overridden=atol is not None or rtol is not None,
        seed=seed,
    )
    session = load_session(onnx_model)
    input_names = [inp.name for inp in session.get_inputs()]
    model.eval()
    worst_abs = -1.0

    # A forked RNG leaves the random state of the caller unchanged.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        for i in range(k):
            sample = vary_fn(i)
            shapes = _shapes(sample)
            report.sample_shapes.append(shapes)
            report_logger.info(
                "verify: sample %d/%d, %s",
                i + 1,
                k,
                ", ".join(
                    f"{name} {list(shape)}"
                    for name, shape in zip(input_names, shapes, strict=False)
                ),
            )
            try:
                with torch.inference_mode():
                    torch_outs = as_tensor_list(model(*sample))
            except Exception as exc:  # noqa: BLE001 - a usage error, not a crash
                described = [
                    tuple(t.shape) if isinstance(t, torch.Tensor) else type(t).__name__
                    for t in sample
                ]
                if i == 0:
                    raise ValueError(
                        f"model raised on sample {i} (input shapes {described}): "
                        f"{first_line(exc)}. Try --dynamic or --inputs if the model does not "
                        "support this shape."
                    ) from exc
                bounds = _bounds_text(dynamic_shapes)
                drawn_from = f", drawn from bounds {bounds}" if bounds else ""
                raise ValueError(
                    f"model raised on sample {i} (input shapes {described}{drawn_from}), "
                    f"generated by the sampler of downshift: {first_line(exc)}. If the model "
                    "does not support this shape, pass --vary or a custom adapter to control "
                    "how downshift generates the verification samples."
                ) from exc
            report.output_shapes.append([tuple(t.shape) for t in torch_outs])

            try:
                # A bf16 input has no numpy dtype either. ORT would reject that graph anyway.
                feeds = {name: t.numpy() for name, t in zip(input_names, sample, strict=True)}
                ort_outs = session.run(None, feeds)
            except Exception as exc:  # noqa: BLE001 - a verdict, not a crash
                tensor_shapes = [tuple(t.shape) for t in sample if isinstance(t, torch.Tensor)]
                raise OnnxRuntimeError(
                    f"onnxruntime failed on sample {i} (input shapes {tensor_shapes}): "
                    f"{first_line(exc)}"
                ) from exc

            sample_abs, sample_rel, failed, note, candidate = _compare_sample(
                torch_outs, ort_outs, report.tolerance_abs, report.tolerance_rel, i
            )
            if note is not None:
                report.notes.append(note)
            if candidate is not None and candidate[4] > worst_abs:
                output_idx, index, expected, got, worst_abs = candidate
                report.worst = WorstMismatch(i, output_idx, index, expected, got, shapes)
            report.max_abs_err = max(report.max_abs_err, sample_abs)
            report.max_rel_err = max(report.max_rel_err, sample_rel)
            if failed:
                report.failures += 1
                report.baseline_failed = report.baseline_failed or i == 0

    report.shape_generalization = None if report.baseline_failed else report.failures == 0
    return report
