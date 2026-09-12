"""Intake for a .onnx file someone else produced (Olive, a notebook, whatever).

No reference model -> UNVERIFIED. We serve it, we just say we never checked it.
With --reference -> the normal verify path, exactly as for a fresh export.
"""

from pathlib import Path

import onnx
import torch

from downshift.adapters.base import Adapter
from downshift.export.verdict import ExportVerdict, prepare_model
from downshift.export.verify import verify


def _graph_summary(onnx_path: Path) -> tuple[int | None, list[str], tuple[str, ...]]:
    proto = onnx.load(str(onnx_path), load_external_data=False)
    opset = next((imp.version for imp in proto.opset_import if imp.domain in ("", "ai.onnx")), None)
    initializers = {init.name for init in proto.graph.initializer}
    input_names = tuple(i.name for i in proto.graph.input if i.name not in initializers)
    return opset, [n.op_type for n in proto.graph.node], input_names


def intake(
    onnx_path: str | Path,
    reference: torch.nn.Module | None = None,
    example_inputs: tuple | None = None,
    adapter: Adapter | str | None = None,
    k: int = 8,
    dynamic: dict[str, list[int]] | None = None,
) -> ExportVerdict:
    onnx_path = Path(onnx_path)
    opset, op_types, input_names = _graph_summary(onnx_path)

    if reference is None:
        return ExportVerdict(
            status="UNVERIFIED",
            model_family="onnx",
            capture_strategy=None,
            opset=opset,
            op_types=op_types,
            numerics=None,
            recommended_backend="onnxruntime",
            reason="no reference model supplied; served as-is, numerics never checked",
            input_names=input_names,
            onnx_path=onnx_path,
        )

    prepared = prepare_model(reference, example_inputs, adapter, dynamic)
    numerics = verify(
        prepared.model, onnx_path, prepared.inputs, prepared.dynamic_shapes, prepared.vary_fn, k=k
    )
    if numerics.passed:
        status, backend = "CLEAN", "onnxruntime"
        reason = (
            f"pre-built ONNX matches reference across {numerics.samples_tested} samples "
            f"(max abs err {numerics.max_abs_err:.2e})"
        )
    else:
        status, backend = "DEGRADED", "torch"
        reason = (
            f"pre-built ONNX diverges from reference on {numerics.failures}/"
            f"{numerics.samples_tested} samples (max abs err {numerics.max_abs_err:.2e})"
        )
    return ExportVerdict(
        status=status,  # type: ignore[arg-type]
        model_family=prepared.family,
        capture_strategy=None,
        opset=opset,
        op_types=op_types,
        numerics=numerics,
        recommended_backend=backend,  # type: ignore[arg-type]
        reason=reason,
        input_names=prepared.input_names,
        dynamic_dims=prepared.dynamic_dims,
        onnx_path=onnx_path,
        prepared=prepared,
    )
