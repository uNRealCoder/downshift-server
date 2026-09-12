"""ExportVerdict (IMPLEMENTATION_PLAN.md §5.6) — orchestrates adapter prep, capture, and
mandatory numerical verification into a single CLEAN / DEGRADED / FAILED call.

UNVERIFIED (pre-optimized .onnx with no reference model, §5.7) isn't produced here — this
module only ever sees a real torch model, so there's always something to verify against.
"""

from dataclasses import dataclass, field
from typing import Any, Literal, NamedTuple

import torch

from downshift.adapters import generic
from downshift.export.capture import capture
from downshift.export.shapes import infer_dynamic_shapes, safe_capture_inputs
from downshift.export.verify import NumericsReport, VaryFn, verify

Status = Literal["CLEAN", "DEGRADED", "FAILED"]
Backend = Literal["onnxruntime", "torch"]


@dataclass
class ExportVerdict:
    status: Status
    model_family: str
    capture_strategy: str | None
    opset: int | None
    op_types: list[str]
    numerics: NumericsReport | None
    recommended_backend: Backend
    reason: str
    onnx_program: object | None = field(default=None, repr=False)  # torch.onnx.ONNXProgram


class Prepared(NamedTuple):
    model: torch.nn.Module
    inputs: tuple
    dynamic_shapes: tuple
    vary_fn: VaryFn | None
    model_family: str


def _prepare(model: torch.nn.Module, example_inputs: tuple) -> Prepared:
    """Dispatch to the PyG adapter when the input looks like one, else the generic
    dataclass-flattening adapter. torch_geometric is optional, so this import is lazy and
    only touched when a PyG Data instance actually shows up.
    """
    pyg: Any = None
    try:
        from downshift.adapters import pyg
    except ImportError:
        pass

    if pyg is not None and pyg.is_pyg_data(example_inputs):
        shim, flat_inputs, dynamic_shapes, field_names = pyg.prepare(model, example_inputs)
        vary_fn = pyg.make_vary_fn(flat_inputs, field_names)
        return Prepared(shim, flat_inputs, dynamic_shapes, vary_fn, "pyg")

    export_model, flat_inputs = generic.prepare(model, example_inputs)
    return Prepared(export_model, flat_inputs, infer_dynamic_shapes(flat_inputs), None, "generic")


def build_verdict(model: torch.nn.Module, example_inputs: tuple, k: int = 8) -> ExportVerdict:
    prepared = _prepare(model, example_inputs)
    capture_inputs = safe_capture_inputs(prepared.inputs, prepared.dynamic_shapes)

    result = capture(prepared.model, capture_inputs, prepared.dynamic_shapes)

    if not result.success:
        exc = result.exception
        return ExportVerdict(
            status="FAILED",
            model_family=prepared.model_family,
            capture_strategy=result.capture_strategy,
            opset=None,
            op_types=[],
            numerics=None,
            recommended_backend="torch",
            reason=f"{type(exc).__name__}: {exc}" if exc is not None else "export failed",
        )

    numerics = verify(
        prepared.model,
        result.onnx_program,
        prepared.inputs,
        prepared.dynamic_shapes,
        vary_fn=prepared.vary_fn,
        k=k,
    )

    if numerics.passed:
        status: Status = "CLEAN"
        backend: Backend = "onnxruntime"
        reason = (
            f"exported via {result.capture_strategy}; numerics ok across "
            f"{numerics.samples_tested} samples (max abs err {numerics.max_abs_err:.2e})"
        )
    else:
        status = "DEGRADED"
        backend = "torch"
        reason = (
            f"exported via {result.capture_strategy} but numerics diverge on "
            f"{numerics.failures}/{numerics.samples_tested} samples "
            f"(max abs err {numerics.max_abs_err:.2e})"
        )

    return ExportVerdict(
        status=status,
        model_family=prepared.model_family,
        capture_strategy=result.capture_strategy,
        opset=result.opset,
        op_types=result.op_types,
        numerics=numerics,
        recommended_backend=backend,
        reason=reason,
        onnx_program=result.onnx_program,
    )
