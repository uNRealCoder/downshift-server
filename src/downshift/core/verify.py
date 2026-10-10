"""Numerical verification. It is mandatory. An export is not a success until this passes.

It runs K samples through the torch model and through the ONNX graph. It varies the dynamic
dimensions, so at least some samples have shapes that the exporter did not see. This finds a
graph that traced without an error but froze a shape or specialized a data-dependent branch.
"""

import logging
from dataclasses import asdict, dataclass, field
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
    """ONNX Runtime itself failed. It could not load the graph, or it could not run a sample.

    This is different from a numeric mismatch (which is still a failure in a NumericsReport). It
    is also different from an error that the torch model raises (a ValueError, because it is a
    problem with a shape that the caller supplied and not a problem of ONNX Runtime).
    """


@dataclass
class WorstMismatch:
    """The single element with the largest absolute error, across every sample tried."""

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
    # None if the baseline sample itself failed (downshift did not evaluate the shape
    # generalization). False if the baseline passes but a varied shape fails. True otherwise.
    shape_generalization: bool | None
    tolerance_abs: float
    tolerance_rel: float
    tolerance_dtype: str = (
        "float32"  # the dtype whose default (tolerance_abs, tolerance_rel) applied
    )
    tolerance_overridden: bool = False  # --atol or --rtol set the values, not tolerance_dtype
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
        """Rebuild from to_dict()'s output; `passed` is derived, so it is ignored."""
        worst_data = data.get("worst")
        worst = (
            WorstMismatch(
                sample=worst_data["sample"],
                output=worst_data["output"],
                index=tuple(worst_data["index"]),
                expected=worst_data["expected"],
                got=worst_data["got"],
                input_shapes=[tuple(shape) for shape in worst_data["input_shapes"]],
            )
            if worst_data is not None
            else None
        )
        return cls(
            samples_tested=data["samples_tested"],
            max_abs_err=data["max_abs_err"],
            max_rel_err=data["max_rel_err"],
            failures=data["failures"],
            shape_generalization=data["shape_generalization"],
            tolerance_abs=data["tolerance_abs"],
            tolerance_rel=data["tolerance_rel"],
            tolerance_dtype=data.get("tolerance_dtype", "float32"),
            tolerance_overridden=data.get("tolerance_overridden", False),
            baseline_failed=data.get("baseline_failed", False),
            worst=worst,
            sample_shapes=[
                [tuple(shape) for shape in sample] for sample in data.get("sample_shapes", [])
            ],
            output_shapes=[
                [tuple(shape) for shape in sample] for sample in data.get("output_shapes", [])
            ],
            seed=data.get("seed", 0),
            notes=list(data.get("notes", [])),
        )


def default_tolerances(model: torch.nn.Module) -> tuple[str, float, float]:
    """(dtype name, atol, rtol), selected by the narrowest floating dtype that is present.

    If bfloat16 or float16 is in the parameters, it wins first. Their tolerances are the
    loosest. A model that mixes them with float32 has only the precision of its worst dtype.
    float64 wins only if it is the *only* floating dtype. A model that mixes float32 and float64
    still has the precision of float32. float32 is the fallback.
    """
    dtypes = {p.dtype for p in model.parameters() if p.is_floating_point()}
    if torch.bfloat16 in dtypes:
        name = "bfloat16"
    elif torch.float16 in dtypes:
        name = "float16"
    elif dtypes == {torch.float64}:
        name = "float64"
    else:
        name = "float32"
    atol, rtol = TOLERANCES[name]
    return name, atol, rtol


def make_shared_axis0_vary_fn(base_inputs: tuple, dynamic_shapes: tuple) -> VaryFn:
    """Default sampler: every dynamic tensor shares one axis-0 size (the batch case)."""
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


_ort_logging_quieted = False


def quiet_ort_logging() -> None:
    """Silence the C++ logger of ONNX Runtime. Otherwise, it prints each failure that a client
    caused to stderr, and it bypasses the logging of downshift. It has an effect only before
    the first session in the process is built (P6). Each place that creates a session
    therefore calls this first. You can call it more than one time.
    """
    global _ort_logging_quieted
    if _ort_logging_quieted:
        return
    ort.set_default_logger_severity(4)
    _ort_logging_quieted = True


def new_session(
    model: bytes | str | Path,
    providers: list[str] | None = None,
    options: ort.SessionOptions | None = None,
) -> ort.InferenceSession:
    """The one place that builds an InferenceSession (CPU, unless `providers` says otherwise).
    The logger of ORT is therefore always quiet first (P6)."""
    quiet_ort_logging()
    source = model if isinstance(model, bytes) else str(model)
    return ort.InferenceSession(
        source, sess_options=options, providers=providers or ["CPUExecutionProvider"]
    )


def _to_session(onnx_model: bytes | str | Path | ort.InferenceSession) -> ort.InferenceSession:
    if isinstance(onnx_model, ort.InferenceSession):
        return onnx_model
    return new_session(onnx_model)


def load_session(onnx_model: bytes | str | Path | ort.InferenceSession) -> ort.InferenceSession:
    """The CPU session that verify() runs the samples on. It raises OnnxRuntimeError if ORT
    cannot load the graph. The caller keeps it. The server can then reuse it and does not need
    to build a second one."""
    try:
        return _to_session(onnx_model)
    except Exception as exc:  # noqa: BLE001 - reported as a verdict and not as a crash
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


def _bounds_text(dynamic_shapes: tuple | None) -> str | None:
    if not dynamic_shapes:
        return None
    parts = [
        f"axis {axis} in {dim_bounds(spec, axis)}"
        for spec in dynamic_shapes
        if spec
        for axis in spec
    ]
    return ", ".join(parts) if parts else None


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    # bfloat16 tensors (and, for safety, float16 tensors) have no faithful numpy dtype. The own
    # .numpy() of torch raises an error on bfloat16. Widen floating outputs to float32 first.
    # Downshift does not change integer and bool outputs.
    if tensor.is_floating_point():
        return tensor.detach().float().numpy()
    return tensor.detach().numpy()


# (output index, unravelled element index, expected, got, abs_err) for the argmax element of one
# output. Downshift drops abs_err before it reaches WorstMismatch. It only ranks the candidates.
WorstCandidate = tuple[int, tuple[int, ...], float, float, float]


def _compare_sample(
    torch_outs: list[torch.Tensor],
    ort_outs: list,
    atol: float,
    rtol: float,
    sample_index: int,
) -> tuple[float, float, bool, str | None, WorstCandidate | None]:
    """The allclose rule for each element of one sample: (max_abs_err, max_rel_err, failed, note,
    worst).

    A sample fails if one element has abs_err > atol + rtol * |expected|. The rule uses the
    semantics of np.isclose(equal_nan=False). NaN against NaN is a mismatch. An inf against an
    inf of the same sign matches. A mismatch in shape or count between the torch outputs and
    the ORT outputs also fails the sample. Downshift gives a note that people can read, and it
    does not raise an error. `worst` is the element with the largest absolute error across all
    outputs that were comparable. It is None if none were comparable, for example a mismatch in
    shape or count on each output.
    """
    note: str | None
    if len(torch_outs) != len(ort_outs):
        note = (
            f"sample {sample_index}: {len(torch_outs)} torch outputs "
            f"vs {len(ort_outs)} onnxruntime outputs"
        )
        return 0.0, 0.0, True, note, None

    sample_abs = 0.0
    sample_rel = 0.0
    failed = False
    note = None
    worst: WorstCandidate | None = None
    worst_abs = -1.0
    for idx, (expected, got) in enumerate(zip(torch_outs, ort_outs, strict=True)):
        expected_np = _to_numpy(expected)
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
        rel_err = abs_err / (np.abs(expected64) + 1e-8)
        output_abs = float(abs_err.max(initial=0.0))
        sample_abs = max(sample_abs, output_abs)
        sample_rel = max(sample_rel, float(rel_err.max(initial=0.0)))

        if abs_err.size and output_abs > worst_abs:
            flat_index = int(np.argmax(abs_err))
            unravelled = tuple(int(x) for x in np.unravel_index(flat_index, abs_err.shape))
            worst_abs = float(abs_err.flat[flat_index])
            worst = (
                idx,
                unravelled,
                float(expected64.flat[flat_index]),
                float(got64.flat[flat_index]),
                worst_abs,
            )

        if np.any(~np.isclose(got64, expected64, rtol=rtol, atol=atol, equal_nan=False)):
            failed = True

    return sample_abs, sample_rel, failed, note, worst


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

    tolerance_overridden = atol is not None or rtol is not None
    tolerance_dtype, default_atol, default_rtol = default_tolerances(model)
    atol = default_atol if atol is None else atol
    rtol = default_rtol if rtol is None else rtol

    session = load_session(onnx_model)
    input_names = [inp.name for inp in session.get_inputs()]

    model.eval()
    max_abs_err = 0.0
    max_rel_err = 0.0
    failures = 0
    baseline_failed = False
    notes: list[str] = []
    sample_shapes: list[list[tuple[int, ...]]] = []
    output_shapes: list[list[tuple[int, ...]]] = []
    worst: WorstMismatch | None = None
    worst_abs = -1.0

    # Seed inside a forked RNG. The global random state of the caller then does not change.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        for i in range(k):
            sample = vary_fn(i)
            shapes_this_sample = [
                tuple(t.shape) if isinstance(t, torch.Tensor) else () for t in sample
            ]
            sample_shapes.append(shapes_this_sample)
            report_logger.info(
                "verify: sample %d/%d, %s",
                i + 1,
                k,
                ", ".join(
                    f"{name} {list(shape)}"
                    for name, shape in zip(input_names, shapes_this_sample, strict=False)
                ),
            )
            try:
                with torch.inference_mode():
                    raw_output = model(*sample)
            except Exception as exc:  # noqa: BLE001 - reported as a usage error and not as a crash
                shapes = [
                    tuple(t.shape) if isinstance(t, torch.Tensor) else type(t).__name__
                    for t in sample
                ]
                if i == 0:
                    raise ValueError(
                        f"model raised on sample {i} (input shapes {shapes}): "
                        f"{first_line(exc)}. Try --dynamic or --inputs if the model does not "
                        "support this shape."
                    ) from exc
                bounds = _bounds_text(dynamic_shapes)
                drawn_from = f", drawn from bounds {bounds}" if bounds else ""
                raise ValueError(
                    f"model raised on sample {i} (input shapes {shapes}{drawn_from}), "
                    f"generated by the sampler of downshift: {first_line(exc)}. If the model "
                    "does not support this shape, pass --vary or a custom adapter to control "
                    "how downshift generates the verification samples."
                ) from exc
            torch_outs = as_tensor_list(raw_output)
            output_shapes.append([tuple(t.shape) for t in torch_outs])

            try:
                # A bfloat16 input tensor also has no numpy equivalent. This TypeError can
                # arrive here (with the session.run failures). This is acceptable, because
                # onnxruntime would reject the graph for the same reason.
                feeds = {name: t.numpy() for name, t in zip(input_names, sample, strict=True)}
                ort_outs = session.run(None, feeds)
            except Exception as exc:  # noqa: BLE001 - reported as a verdict and not as a crash
                shapes = [tuple(t.shape) for t in sample if isinstance(t, torch.Tensor)]
                raise OnnxRuntimeError(
                    f"onnxruntime failed on sample {i} (input shapes {shapes}): {first_line(exc)}"
                ) from exc

            sample_abs, sample_rel, sample_failed, note, candidate = _compare_sample(
                torch_outs, ort_outs, atol, rtol, i
            )
            if note is not None:
                notes.append(note)
            if candidate is not None:
                output_idx, index, expected_val, got_val, candidate_abs = candidate
                if candidate_abs > worst_abs:
                    worst_abs = candidate_abs
                    worst = WorstMismatch(
                        sample=i,
                        output=output_idx,
                        index=index,
                        expected=expected_val,
                        got=got_val,
                        input_shapes=shapes_this_sample,
                    )

            max_abs_err = max(max_abs_err, sample_abs)
            max_rel_err = max(max_rel_err, sample_rel)
            if sample_failed:
                failures += 1
                if i == 0:
                    baseline_failed = True

    shape_generalization = None if baseline_failed else failures == 0
    return NumericsReport(
        samples_tested=k,
        max_abs_err=max_abs_err,
        max_rel_err=max_rel_err,
        failures=failures,
        shape_generalization=shape_generalization,
        tolerance_abs=atol,
        tolerance_rel=rtol,
        tolerance_dtype=tolerance_dtype,
        tolerance_overridden=tolerance_overridden,
        baseline_failed=baseline_failed,
        worst=worst,
        sample_shapes=sample_shapes,
        output_shapes=output_shapes,
        seed=seed,
        notes=notes,
    )
