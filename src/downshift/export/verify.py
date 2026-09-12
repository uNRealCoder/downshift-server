"""Numerical verification (IMPLEMENTATION_PLAN.md §5.5) — mandatory, not optional.

A successful export is not a success until numerics are verified. We run K samples
through both the original torch model and the exported ONNX graph, deliberately varying
dynamic dimensions (including sizes different from the export-time example) — this is
what catches a graph that "exported cleanly" but silently froze a shape or specialized a
data-dependent branch at trace time.

How a sample is regenerated for a given size is adapter-specific (a PyG edge_index must
stay within the *resampled* node count, which a naive shared-axis heuristic can't know),
so callers may supply their own `vary_fn`; the default here covers the plain
shared-axis-0 case (dataclass/generic adapters).
"""

import random
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import onnxruntime as ort
import torch

from downshift.export.shapes import alternative_sizes

DEFAULT_ATOL = 1e-4
DEFAULT_RTOL = 1e-3

VaryFn = Callable[[int], tuple]


@dataclass
class NumericsReport:
    samples_tested: int
    max_abs_err: float
    max_rel_err: float
    failures: int
    shape_generalization: bool  # did every non-original-size sample also pass?
    tolerance_abs: float = DEFAULT_ATOL
    tolerance_rel: float = DEFAULT_RTOL

    @property
    def passed(self) -> bool:
        return self.failures == 0


def _resize_dim0(tensor: torch.Tensor, new_size: int) -> torch.Tensor:
    if tensor.ndim == 0 or tensor.shape[0] == new_size:
        return tensor
    shape = list(tensor.shape)
    shape[0] = new_size
    if tensor.is_floating_point():
        return torch.randn(*shape, dtype=tensor.dtype)
    # Integer tensor (token ids, segment ids, ...): stay within the original sample's
    # observed value range so we don't manufacture an out-of-vocab / out-of-range index.
    lo = int(tensor.min().item())
    hi = max(int(tensor.max().item()) + 1, lo + 1)
    return torch.randint(lo, hi, shape, dtype=tensor.dtype)


def make_shared_axis0_vary_fn(base_inputs: tuple, dynamic_shapes: tuple, seed: int = 0) -> VaryFn:
    """Default vary_fn: every dynamic tensor shares one co-varying axis-0 size (the
    ordinary "batch dim" case). Sample 0 is always the untouched base_inputs.
    """
    base_size = next(
        (t.shape[0] for t, spec in zip(base_inputs, dynamic_shapes, strict=True) if spec is not None),
        None,
    )
    rng = random.Random(seed)
    candidate_sizes = alternative_sizes(base_size) if base_size is not None else []

    def vary(i: int) -> tuple:
        if i == 0 or base_size is None:
            return base_inputs
        size = rng.choice(candidate_sizes) if candidate_sizes else base_size
        return tuple(
            _resize_dim0(t, size) if isinstance(t, torch.Tensor) and spec is not None else t
            for t, spec in zip(base_inputs, dynamic_shapes, strict=True)
        )

    return vary


def verify(
    model: torch.nn.Module,
    onnx_program,
    base_inputs: tuple,
    dynamic_shapes: tuple | None = None,
    vary_fn: VaryFn | None = None,
    k: int = 8,
    atol: float = DEFAULT_ATOL,
    rtol: float = DEFAULT_RTOL,
    seed: int = 0,
) -> NumericsReport:
    if vary_fn is None:
        if dynamic_shapes is None:
            raise ValueError("verify() needs either dynamic_shapes or an explicit vary_fn")
        vary_fn = make_shared_axis0_vary_fn(base_inputs, dynamic_shapes, seed=seed)

    session = ort.InferenceSession(
        onnx_program.model_proto.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    input_names = [inp.name for inp in session.get_inputs()]

    model.eval()
    max_abs_err = 0.0
    max_rel_err = 0.0
    failures = 0
    non_baseline_failures = 0

    for i in range(k):
        sample = vary_fn(i)
        with torch.no_grad():
            torch_out = model(*sample)
        ort_inputs = {name: t.numpy() for name, t in zip(input_names, sample, strict=True)}
        (ort_out,) = session.run(None, ort_inputs)

        torch_out_np = torch_out.detach().numpy().astype(np.float64)
        abs_err = np.abs(torch_out_np - ort_out.astype(np.float64))
        rel_err = abs_err / (np.abs(torch_out_np) + 1e-8)

        sample_max_abs = float(abs_err.max())
        sample_max_rel = float(rel_err.max())
        max_abs_err = max(max_abs_err, sample_max_abs)
        max_rel_err = max(max_rel_err, sample_max_rel)

        if sample_max_abs > atol and sample_max_rel > rtol:
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
    )
