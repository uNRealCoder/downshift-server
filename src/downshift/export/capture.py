"""Drive torch.export + torch.onnx.export and say which strategy worked.

We run torch.export ourselves (strict=False, then strict=True) so the winning strategy is
a value we return, not something scraped from console output. The ONNX translation is
still entirely torch.onnx.export's; we only hand it the ExportedProgram.

torch 2.14 note: the dynamo exporter documents only the two strict modes. There's no
draft_export step or TorchScript fallback any more.
"""

import contextlib
import io
import logging
from dataclasses import dataclass, field

import torch

_STRATEGIES: tuple[tuple[str, bool], ...] = (("strict=False", False), ("strict=True", True))

# torch.onnx logs a warning per missing torchvision op on every export. Not actionable.
logging.getLogger("torch.onnx._internal.exporter._registration").setLevel(logging.ERROR)


@dataclass
class CaptureResult:
    success: bool
    capture_strategy: str | None  # a _STRATEGIES name; None when nothing traced
    onnx_program: "torch.onnx.ONNXProgram | None" = None
    opset: int | None = None
    op_types: list[str] = field(default_factory=list)
    exception: Exception | None = None
    stderr: str = ""  # whatever torch printed while we tried; useful at debug level


def capture(
    model: torch.nn.Module,
    example_inputs: tuple,
    dynamic_shapes: tuple | None = None,
) -> CaptureResult:
    exported_program = None
    strategy_used: str | None = None
    last_exception: Exception | None = None
    # torch prints whole FX graphs straight to stderr when a data-dependent guard fails.
    # Keep that out of the user's terminal; the exception message is what matters.
    captured = io.StringIO()

    with contextlib.redirect_stderr(captured):
        for name, strict in _STRATEGIES:
            try:
                exported_program = torch.export.export(
                    model, example_inputs, dynamic_shapes=dynamic_shapes, strict=strict
                )
            except Exception as exc:  # noqa: BLE001 - a failed strategy means try the next
                last_exception = exc
            else:
                strategy_used = name
                break

        if exported_program is None:
            return CaptureResult(
                success=False,
                capture_strategy=None,
                exception=last_exception,
                stderr=captured.getvalue(),
            )

        try:
            onnx_program = torch.onnx.export(exported_program, verbose=False, report=False)
        except Exception as exc:  # noqa: BLE001 - a failed translation is a failed capture
            return CaptureResult(
                success=False,
                capture_strategy=strategy_used,
                exception=exc,
                stderr=captured.getvalue(),
            )

    if onnx_program is None:
        # The type stub allows None for the legacy path; with an ExportedProgram it never is.
        return CaptureResult(
            success=False,
            capture_strategy=strategy_used,
            exception=RuntimeError("torch.onnx.export returned None"),
        )

    proto = onnx_program.model_proto
    return CaptureResult(
        success=True,
        capture_strategy=strategy_used,
        onnx_program=onnx_program,
        opset=proto.opset_import[0].version if proto.opset_import else None,
        op_types=[node.op_type for node in proto.graph.node],
        stderr=captured.getvalue(),
    )
