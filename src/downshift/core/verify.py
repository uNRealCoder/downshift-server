"""Numerical verification. An export is not a success until this passes.

The samples run through torch and through the ONNX graph. The dynamic axes vary, so some samples
have shapes that the exporter did not see. This finds a graph that froze a shape or a branch.
"""

import logging
import math
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from downshift.core.shapes import alternative_sizes, dim_bounds, pick_size, resize_axis
from downshift.entities.adapters import VaryFn
from downshift.entities.verdict import NumericsReport, WorstMismatch
from downshift.logs import REPORT_LOGGER
from downshift.settings import DEFAULT_SAMPLES, TOLERANCE_COSINE, TOLERANCES

report_logger = logging.getLogger(REPORT_LOGGER)


class OnnxRuntimeError(RuntimeError):
    """ONNX Runtime could not load the graph or run a sample. A numeric mismatch is not this."""


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


def resolve_tolerances(
    dtype_name: str, atol: float | None, rtol: float | None
) -> tuple[float, float]:
    """(atol, rtol) of a dtype name. --atol and --rtol replace the default values."""
    default_atol, default_rtol = TOLERANCES[dtype_name]
    return (default_atol if atol is None else atol, default_rtol if rtol is None else rtol)


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


# A provider name, or (name, provider options) such as ("CUDAExecutionProvider", {"device_id": "1"}).
OrtProvider = str | tuple[str, dict[str, str]]


def new_session(
    model: bytes | str | Path,
    providers: list[OrtProvider] | None = None,
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
    tensor = tensor.detach().cpu()
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


def compare_sample(
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
    worst_abs = -1.0
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
        close = np.isclose(got64, expected64, rtol=rtol, atol=atol, equal_nan=False)
        # A NaN or inf error stays out of the error numbers. It has its own note and `worst`.
        abs_err = np.abs(expected64 - got64)
        rel_err = abs_err / (np.abs(expected64) + 1e-8)
        finite = np.isfinite(abs_err)
        abs_err = np.where(finite, abs_err, 0.0)
        output_abs = float(abs_err.max(initial=0.0))
        sample_abs = max(sample_abs, output_abs)
        sample_rel = max(sample_rel, float(np.where(finite, rel_err, 0.0).max(initial=0.0)))
        nonfinite = ~finite & ~close
        if nonfinite.any():
            flat = int(np.argmax(nonfinite))
            index = tuple(int(x) for x in np.unravel_index(flat, nonfinite.shape))
            got_nan, torch_nan = int(np.isnan(got64).sum()), int(np.isnan(expected64).sum())
            note = (
                f"sample {sample_index}: output_{idx} has {int(nonfinite.sum())} non-finite "
                f"mismatches ({got_nan} NaN from onnxruntime, {torch_nan} NaN from torch)"
            )
            if worst_abs < math.inf:
                worst_abs = math.inf  # the first non-finite mismatch is the worst element
                worst = (
                    idx,
                    index,
                    float(expected64.flat[flat]),
                    float(got64.flat[flat]),
                    worst_abs,
                )
        elif abs_err.size and output_abs > worst_abs:
            flat = int(np.argmax(abs_err))
            index = tuple(int(x) for x in np.unravel_index(flat, abs_err.shape))
            worst_abs = output_abs
            worst = (idx, index, float(expected64.flat[flat]), float(got64.flat[flat]), worst_abs)
        if not close.all():
            failed = True
    return sample_abs, sample_rel, failed, note, worst


def min_row_cosine(expected: np.ndarray, got: np.ndarray) -> float:
    """The smallest cosine similarity between a row of `expected` and the same row of `got`.
    A row is the last axis. A zero row on both sides counts as 1. A zero row on one side only
    counts as 0. A NaN counts as 0. Two arrays of different shapes are not compared."""
    if expected.ndim == 0:
        return 1.0
    a = expected.reshape(-1, expected.shape[-1]).astype(np.float64)
    b = got.reshape(-1, got.shape[-1]).astype(np.float64)
    norm_a, norm_b = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
    denominator = norm_a * norm_b
    with np.errstate(invalid="ignore", divide="ignore"):
        cosine = np.where(denominator > 0, (a * b).sum(axis=1) / denominator, 0.0)
    cosine = np.where((norm_a == 0) & (norm_b == 0), 1.0, cosine)
    return float(np.nan_to_num(cosine, nan=0.0).min(initial=1.0))


def fold_cosine(
    expected: np.ndarray, got: np.ndarray, lowest: float | None
) -> tuple[float | None, bool]:
    """(the lowest row cosine so far including this pair, True if this pair is under the
    cosine tolerance). Two arrays of different shapes are not compared."""
    if expected.shape != got.shape:
        return lowest, False
    cosine = min_row_cosine(expected, got)
    return (cosine if lowest is None else min(cosine, lowest)), cosine < TOLERANCE_COSINE


def _shapes(sample: tuple) -> list[tuple[int, ...]]:
    return [tuple(t.shape) if isinstance(t, torch.Tensor) else () for t in sample]


def verify(
    model: torch.nn.Module,
    onnx_model: bytes | str | Path | ort.InferenceSession,
    base_inputs: tuple,
    dynamic_shapes: tuple | None = None,
    vary_fn: VaryFn | None = None,
    samples: int = DEFAULT_SAMPLES,
    atol: float | None = None,
    rtol: float | None = None,
    seed: int = 0,
    embedding: bool = False,
    keep_feeds: list[dict[str, np.ndarray]] | None = None,
    fp16: bool = False,
) -> NumericsReport:
    """Compare torch and ONNX Runtime on `samples` inputs. If `embedding` is True, the first
    output is a pooled embedding. Then each of its rows must also reach the cosine tolerance.
    The input feeds of each sample are appended to `keep_feeds`, if you give it. With `fp16`,
    the graph is an fp16 copy of the fp32 `model`: the float16 tolerances apply."""
    if vary_fn is None:
        if dynamic_shapes is None:
            raise ValueError("verify() needs either dynamic_shapes or an explicit vary_fn")
        vary_fn = make_shared_axis0_vary_fn(base_inputs, dynamic_shapes)

    tolerance_dtype = default_tolerances(model)[0]
    if fp16 and tolerance_dtype in ("float32", "float64"):
        tolerance_dtype = "float16"
    tolerance_abs, tolerance_rel = resolve_tolerances(tolerance_dtype, atol, rtol)
    report = NumericsReport(
        samples_tested=samples,
        max_abs_err=0.0,
        max_rel_err=0.0,
        failures=0,
        shape_generalization=None,
        tolerance_abs=tolerance_abs,
        tolerance_rel=tolerance_rel,
        tolerance_dtype=tolerance_dtype,
        tolerance_overridden=atol is not None or rtol is not None,
        seed=seed,
        tolerance_cosine=TOLERANCE_COSINE if embedding else None,
    )
    session = load_session(onnx_model)
    input_names = [inp.name for inp in session.get_inputs()]
    model.eval()
    worst_abs = -1.0

    # A forked RNG leaves the random state of the caller unchanged.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        for i in range(samples):
            sample = vary_fn(i)
            shapes = _shapes(sample)
            report.sample_shapes.append(shapes)
            report_logger.info(
                "verify: sample %d/%d, %s",
                i + 1,
                samples,
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
                # Sample 0 is the example input. If torch fails on it, the model or the inputs
                # are wrong. A later sample comes from the sampler of downshift, so its message
                # names the sampler and the bounds.
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
                feeds = {
                    name: t.detach().cpu().numpy()
                    for name, t in zip(input_names, sample, strict=True)
                }
                if keep_feeds is not None:
                    keep_feeds.append(feeds)
                ort_outs = session.run(None, feeds)
            except Exception as exc:  # noqa: BLE001 - a verdict, not a crash
                tensor_shapes = [tuple(t.shape) for t in sample if isinstance(t, torch.Tensor)]
                raise OnnxRuntimeError(
                    f"onnxruntime failed on sample {i} (input shapes {tensor_shapes}): "
                    f"{first_line(exc)}"
                ) from exc

            sample_abs, sample_rel, failed, note, candidate = compare_sample(
                torch_outs, ort_outs, report.tolerance_abs, report.tolerance_rel, i
            )
            if note is not None:
                report.notes.append(note)
            if candidate is not None and candidate[4] > worst_abs:
                output_idx, index, expected, got, worst_abs = candidate
                report.worst = WorstMismatch(i, output_idx, index, expected, got, shapes)
            report.max_abs_err = max(report.max_abs_err, sample_abs)
            report.max_rel_err = max(report.max_rel_err, sample_rel)
            if embedding and ort_outs and torch_outs:
                report.min_cosine, low = fold_cosine(
                    tensor_to_numpy(torch_outs[0]), np.asarray(ort_outs[0]), report.min_cosine
                )
                failed = failed or low
            if failed:
                report.failures += 1
                report.baseline_failed = report.baseline_failed or i == 0

    if report.min_cosine is not None and report.min_cosine < TOLERANCE_COSINE:
        report.notes.append(
            f"embedding rows: minimum cosine {report.min_cosine:.5f} "
            f"(tolerance {TOLERANCE_COSINE:g})"
        )
    # If sample 0 failed, the fault is not about shapes, so the answer is "not known" (None).
    report.shape_generalization = None if report.baseline_failed else report.failures == 0
    return report
