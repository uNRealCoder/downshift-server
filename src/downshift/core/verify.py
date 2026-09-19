"""Numerical verification. Mandatory: an export isn't a success until this passes.

Runs K samples through the torch model and the ONNX graph, varying dynamic dims so at
least some samples have shapes the exporter never saw. That's what catches a graph that
traced fine but froze a shape or specialised a data-dependent branch.
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from downshift.adapters.base import VaryFn
from downshift.core.shapes import alternative_sizes, dim_bounds
from downshift.settings import TOLERANCES


class OnnxRuntimeError(RuntimeError):
    """ONNX Runtime itself failed: it couldn't load the graph or couldn't run a sample.

    Distinct from a numeric mismatch (still a NumericsReport failure) and from the torch
    model raising (a ValueError, since that's a caller-supplied-shape problem, not an
    ONNX Runtime one).
    """


@dataclass
class NumericsReport:
    samples_tested: int
    max_abs_err: float
    max_rel_err: float
    failures: int
    shape_generalization: bool  # did every non-baseline-shape sample also pass?
    tolerance_abs: float
    tolerance_rel: float
    tolerance_dtype: str = (
        "float32"  # the dtype whose default (tolerance_abs, tolerance_rel) applied
    )
    seed: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.failures == 0

    def to_dict(self) -> dict:
        return asdict(self) | {"passed": self.passed}


_TOLERANCES_BY_DTYPE: dict[torch.dtype, tuple[float, float]] = {
    getattr(torch, name): value for name, value in TOLERANCES.items()
}


def default_tolerances(model: torch.nn.Module) -> tuple[str, float, float]:
    """(dtype name, atol, rtol) picked by the narrowest floating dtype present.

    bfloat16 or float16 anywhere in the parameters wins first, since their tolerances are
    the loosest and a model that mixes them with float32 is only as precise as its worst
    dtype. float64 wins only when it's the *only* floating dtype - a model that mixes
    float32 and float64 is still bound by float32's precision. float32 is the fallback.
    """
    dtypes = {p.dtype for p in model.parameters() if p.is_floating_point()}
    for dtype in (torch.bfloat16, torch.float16):
        if dtype in dtypes:
            name = "bfloat16" if dtype is torch.bfloat16 else "float16"
            atol, rtol = _TOLERANCES_BY_DTYPE[dtype]
            return name, atol, rtol
    if dtypes == {torch.float64}:
        atol, rtol = _TOLERANCES_BY_DTYPE[torch.float64]
        return "float64", atol, rtol
    atol, rtol = _TOLERANCES_BY_DTYPE[torch.float32]
    return "float32", atol, rtol


def _resize_dim0(tensor: torch.Tensor, new_size: int) -> torch.Tensor:
    """Resize axis 0 to new_size, staying close to the example rather than pure noise.

    Floating tensors tile or slice the example's own rows and add small noise scaled to
    the example's own spread, so a varied sample looks like a plausible input rather than
    unrelated white noise. Integer tensors (usually indices) stay inside the observed range.
    """
    if tensor.ndim == 0 or tensor.shape[0] == new_size:
        return tensor
    if tensor.is_floating_point():
        base_size = tensor.shape[0]
        reps = -(-new_size // base_size)  # ceil division
        tiled = tensor.repeat(reps, *([1] * (tensor.ndim - 1)))[:new_size]
        noise_std = 0.1 * tensor.std(unbiased=False)  # unbiased is undefined for one row
        return tiled + noise_std * torch.randn(tiled.shape, dtype=tensor.dtype)
    shape = list(tensor.shape)
    shape[0] = new_size
    lo = int(tensor.min().item())
    hi = max(int(tensor.max().item()) + 1, lo + 1)
    return torch.randint(lo, hi, shape, dtype=tensor.dtype)


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
        size = candidates[torch.randint(len(candidates), ())]
        return tuple(
            _resize_dim0(t, size) if isinstance(t, torch.Tensor) and spec else t
            for t, spec in zip(base_inputs, dynamic_shapes, strict=True)
        )

    return vary


def _to_session(onnx_model) -> ort.InferenceSession:
    if isinstance(onnx_model, (str, Path)):
        source: str | bytes = str(onnx_model)
    elif isinstance(onnx_model, bytes):
        source = onnx_model
    else:  # torch.onnx.ONNXProgram
        source = onnx_model.model_proto.SerializeToString()
    return ort.InferenceSession(source, providers=["CPUExecutionProvider"])


def _as_tensor_list(output) -> list[torch.Tensor]:
    if isinstance(output, torch.Tensor):
        return [output]
    if isinstance(output, (tuple, list)):
        return [t for t in output if isinstance(t, torch.Tensor)]
    raise TypeError(f"can't compare model output of type {type(output).__name__}")


def _first_line(exc: Exception) -> str:
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
    # bfloat16 (and, for safety, float16) tensors have no faithful numpy dtype; torch's own
    # .numpy() raises on bfloat16, so widen floating outputs to float32 first. Integer/bool
    # outputs are left alone.
    if tensor.is_floating_point():
        return tensor.detach().float().numpy()
    return tensor.detach().numpy()


def _compare_sample(
    torch_outs: list[torch.Tensor],
    ort_outs: list,
    atol: float,
    rtol: float,
    sample_index: int,
) -> tuple[float, float, bool, str | None]:
    """Per-element allclose rule for one sample: (max_abs_err, max_rel_err, failed, note).

    A sample fails if any single element has abs_err > atol + rtol * |expected|, using
    np.isclose(equal_nan=False) semantics: NaN vs NaN is a mismatch, same-sign inf vs inf
    matches. Shape/count mismatches between the torch and ORT outputs also fail the sample,
    with a human-readable note, instead of raising.
    """
    note: str | None
    if len(torch_outs) != len(ort_outs):
        note = (
            f"sample {sample_index}: {len(torch_outs)} torch outputs "
            f"vs {len(ort_outs)} onnxruntime outputs"
        )
        return 0.0, 0.0, True, note

    sample_abs = 0.0
    sample_rel = 0.0
    failed = False
    note = None
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
        sample_abs = max(sample_abs, float(abs_err.max(initial=0.0)))
        sample_rel = max(sample_rel, float(rel_err.max(initial=0.0)))

        if np.any(~np.isclose(got64, expected64, rtol=rtol, atol=atol, equal_nan=False)):
            failed = True

    return sample_abs, sample_rel, failed, note


def verify(
    model: torch.nn.Module,
    onnx_model,
    base_inputs: tuple,
    dynamic_shapes: tuple | None = None,
    vary_fn: VaryFn | None = None,
    k: int = 8,
    atol: float | None = None,
    rtol: float | None = None,
    seed: int = 0,
) -> NumericsReport:
    if vary_fn is None:
        if dynamic_shapes is None:
            raise ValueError("verify() needs either dynamic_shapes or an explicit vary_fn")
        vary_fn = make_shared_axis0_vary_fn(base_inputs, dynamic_shapes)

    tolerance_dtype, default_atol, default_rtol = default_tolerances(model)
    atol = default_atol if atol is None else atol
    rtol = default_rtol if rtol is None else rtol

    try:
        session = _to_session(onnx_model)
        input_names = [inp.name for inp in session.get_inputs()]
    except Exception as exc:  # noqa: BLE001 - reported as a verdict, not a crash
        raise OnnxRuntimeError(
            f"onnxruntime could not load the exported graph: {_first_line(exc)}"
        ) from exc

    model.eval()
    max_abs_err = 0.0
    max_rel_err = 0.0
    failures = 0
    non_baseline_failures = 0
    notes: list[str] = []

    # Seed inside a forked RNG so callers' global random state is untouched afterwards.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        for i in range(k):
            sample = vary_fn(i)
            try:
                with torch.inference_mode():
                    raw_output = model(*sample)
            except Exception as exc:  # noqa: BLE001 - reported as a usage error, not a crash
                shapes = [
                    tuple(t.shape) if isinstance(t, torch.Tensor) else type(t).__name__
                    for t in sample
                ]
                if i == 0:
                    raise ValueError(
                        f"model raised on sample {i} (input shapes {shapes}): "
                        f"{_first_line(exc)}. Try --dynamic or --inputs if this shape isn't "
                        "one the model supports."
                    ) from exc
                bounds = _bounds_text(dynamic_shapes)
                drawn_from = f", drawn from bounds {bounds}" if bounds else ""
                raise ValueError(
                    f"model raised on sample {i} (input shapes {shapes}{drawn_from}), "
                    f"generated by downshift's sampler: {_first_line(exc)}. If the model "
                    "doesn't support this shape, pass --vary or a custom adapter to control "
                    "how verification samples are generated."
                ) from exc
            torch_outs = _as_tensor_list(raw_output)

            try:
                # A bfloat16 input tensor has no numpy equivalent either; letting that
                # TypeError land here (alongside session.run failures) is fine, since
                # onnxruntime would reject the graph for the same reason anyway.
                feeds = {name: t.numpy() for name, t in zip(input_names, sample, strict=True)}
                ort_outs = session.run(None, feeds)
            except Exception as exc:  # noqa: BLE001 - reported as a verdict, not a crash
                shapes = [tuple(t.shape) for t in sample if isinstance(t, torch.Tensor)]
                raise OnnxRuntimeError(
                    f"onnxruntime failed on sample {i} (input shapes {shapes}): {_first_line(exc)}"
                ) from exc

            sample_abs, sample_rel, sample_failed, note = _compare_sample(
                torch_outs, ort_outs, atol, rtol, i
            )
            if note is not None:
                notes.append(note)

            max_abs_err = max(max_abs_err, sample_abs)
            max_rel_err = max(max_rel_err, sample_rel)
            if sample_failed:
                failures += 1
                if i > 0:
                    non_baseline_failures += 1

    return NumericsReport(
        samples_tested=k,
        max_abs_err=max_abs_err,
        max_rel_err=max_rel_err,
        failures=failures,
        shape_generalization=non_baseline_failures == 0,
        tolerance_abs=atol,
        tolerance_rel=rtol,
        tolerance_dtype=tolerance_dtype,
        seed=seed,
        notes=notes,
    )
