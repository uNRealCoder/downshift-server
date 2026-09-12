"""torch.onnx.export(dynamo=True) driver (IMPLEMENTATION_PLAN.md §5.1).

We call torch.export.export ourselves, in the same order torch.onnx.export's own
docstring documents for dynamo=True (try strict=False, then strict=True), so we get a
structured answer for *which* strategy succeeded instead of parsing verbose console
output — whose unicode symbols crash on non-UTF-8 Windows consoles regardless (see the
walking-skeleton fix; verbose=True corrupts even a redirected StringIO on this platform).
The actual ONNX translation is still fully delegated to torch.onnx.export, called with
the already-built ExportedProgram via its documented "already an ExportedProgram" path —
we do not reimplement any translation logic, only the strategy selection.

Note: as of torch 2.14, torch.onnx.export's own docstring documents only strict=False /
strict=True for dynamo=True — no draft_export step and no `fallback` TorchScript escape
hatch, both of which IMPLEMENTATION_PLAN.md §5.1 describes. That doc section appears to
describe an earlier torch release; verified empirically rather than assumed.
"""

from dataclasses import dataclass, field

import torch

_STRATEGIES: tuple[tuple[str, bool], ...] = (("strict=False", False), ("strict=True", True))


@dataclass
class CaptureResult:
    success: bool
    capture_strategy: str  # one of _STRATEGIES's names, or "failed"
    onnx_program: "torch.onnx.ONNXProgram | None" = None
    opset: int | None = None
    op_types: list[str] = field(default_factory=list)
    exception: Exception | None = None


def capture(
    model: torch.nn.Module,
    example_inputs: tuple,
    dynamic_shapes: tuple | None = None,
) -> CaptureResult:
    exported_program = None
    strategy_used = "failed"
    last_exception: Exception | None = None

    for name, strict in _STRATEGIES:
        try:
            exported_program = torch.export.export(
                model, example_inputs, dynamic_shapes=dynamic_shapes, strict=strict
            )
        except Exception as exc:  # noqa: BLE001 - a strategy failing just means try the next
            last_exception = exc
        else:
            strategy_used = name
            break

    if exported_program is None:
        return CaptureResult(success=False, capture_strategy="failed", exception=last_exception)

    try:
        onnx_program = torch.onnx.export(exported_program, verbose=False, report=False)
    except Exception as exc:  # noqa: BLE001 - report as a failed capture, not a crash
        return CaptureResult(success=False, capture_strategy=strategy_used, exception=exc)

    if onnx_program is None:
        # Only the legacy (non-ExportedProgram) call path can return None per the type
        # stub; passing an already-built ExportedProgram, as we do here, always returns
        # an ONNXProgram in practice. Guard anyway rather than assume.
        return CaptureResult(
            success=False,
            capture_strategy=strategy_used,
            exception=RuntimeError("torch.onnx.export returned None"),
        )

    model_proto = onnx_program.model_proto
    opset = model_proto.opset_import[0].version if model_proto.opset_import else None
    op_types = [node.op_type for node in model_proto.graph.node]
    return CaptureResult(
        success=True,
        capture_strategy=strategy_used,
        onnx_program=onnx_program,
        opset=opset,
        op_types=op_types,
    )
