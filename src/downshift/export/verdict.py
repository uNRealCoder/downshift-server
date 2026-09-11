"""ExportVerdict (IMPLEMENTATION_PLAN.md §5.6) — orchestrates adapter prep, capture, and
mandatory numerical verification into a single CLEAN / DEGRADED / FAILED call.

UNVERIFIED (pre-optimized .onnx with no reference model, §5.7) isn't produced here — this
module only ever sees a real torch model, so there's always something to verify against.
"""

from dataclasses import dataclass, field
from typing import Literal

import torch

from downshift.adapters import generic
from downshift.export.capture import capture
from downshift.export.shapes import infer_dynamic_shapes, safe_capture_inputs
from downshift.export.verify import NumericsReport, verify

Status = Literal["CLEAN", "DEGRADED", "FAILED"]


@dataclass
class ExportVerdict:
    status: Status
    model_family: str
    capture_strategy: str | None
    opset: int | None
    op_types: list[str]
    numerics: NumericsReport | None
    recommended_backend: Literal["onnxruntime", "torch"]
    reason: str
    onnx_program: object | None = field(default=None, repr=False)  # torch.onnx.ONNXProgram


def _prepare(model: torch.nn.Module, example_inputs: tuple):
    """Dispatch to the PyG adapter when the input looks like one, else the generic
    dataclass-flattening adapter. torch_geometric is optional, so this import is lazy and
    only touched when a PyG Data instance actually shows up.
    """
    try:
        from downshift.adapters import pyg
    except ImportError:
        pyg = None

    if pyg is not None and pyg.is_pyg_data(example_inputs):
        export_model, flat_inputs, dynamic_shapes, field_names = pyg.prepare(model, example_inputs)
        vary_fn = pyg.make_vary_fn(flat_inputs, field_names)
        return export_model, flat_inputs, dynamic_shapes, vary_fn, "pyg"

    export_model, flat_inputs = generic.prepare(model, example_inputs)
    dynamic_shapes = infer_dynamic_shapes(flat_inputs)
    return export_model, flat_inputs, dynamic_shapes, None, "generic"


def build_verdict(model: torch.nn.Module, example_inputs: tuple, k: int = 8) -> ExportVerdict:
    export_model, flat_inputs, dynamic_shapes, vary_fn, model_family = _prepare(model, example_inputs)
    capture_inputs = safe_capture_inputs(flat_inputs, dynamic_shapes)

    result = capture(export_model, capture_inputs, dynamic_shapes)

    if not result.success:
        reason = (
            f"{type(result.exception).__name__}: {result.exception}"
            if result.exception is not None
            else "export failed"
        )
        return ExportVerdict(
            status="FAILED",
            model_family=model_family,
            capture_strategy=result.capture_strategy,
            opset=None,
            op_types=[],
            numerics=None,
            recommended_backend="torch",
            reason=reason,
        )

    numerics = verify(
        export_model, result.onnx_program, flat_inputs, dynamic_shapes, vary_fn=vary_fn, k=k
    )

    if numerics.passed:
        status: Status = "CLEAN"
        backend: Literal["onnxruntime", "torch"] = "onnxruntime"
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
        model_family=model_family,
        capture_strategy=result.capture_strategy,
        opset=result.opset,
        op_types=result.op_types,
        numerics=numerics,
        recommended_backend=backend,
        reason=reason,
        onnx_program=result.onnx_program,
    )
