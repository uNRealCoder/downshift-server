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
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field

import torch

_STRATEGIES: tuple[tuple[str, bool], ...] = (("strict=False", False), ("strict=True", True))

_REGISTRATION_LOGGER = "torch.onnx._internal.exporter._registration"


@contextlib.contextmanager
def _quiet_registration_warnings() -> Iterator[None]:
    """torch.onnx logs a warning per missing torchvision op on every export. Not actionable.

    Scoped to one capture() call rather than set at import time - a library shouldn't
    change another package's logging configuration just by being imported.
    """
    logger = logging.getLogger(_REGISTRATION_LOGGER)
    previous = logger.level
    logger.setLevel(logging.ERROR)
    try:
        yield
    finally:
        logger.setLevel(previous)


@dataclass
class CaptureResult:
    success: bool
    capture_strategy: str | None  # a _STRATEGIES name; None when nothing traced
    onnx_program: "torch.onnx.ONNXProgram | None" = None
    onnx_bytes: bytes = field(default=b"", repr=False)  # model_proto.SerializeToString(), once
    opset: int | None = None
    op_types: dict[str, int] = field(default_factory=dict)  # count-descending histogram
    exception: Exception | None = None  # the exception build_verdict quotes as the reason
    exceptions: list[tuple[str, Exception]] = field(default_factory=list)  # every strategy tried
    stderr: str = ""  # whatever torch printed while we tried; useful at debug level


def capture(
    model: torch.nn.Module,
    example_inputs: tuple,
    dynamic_shapes: tuple | None = None,
) -> CaptureResult:
    exported_program = None
    strategy_used: str | None = None
    exceptions: list[tuple[str, Exception]] = []
    # torch prints whole FX graphs straight to stderr when a data-dependent guard fails.
    # Keep that out of the user's terminal; the exception message is what matters.
    captured = io.StringIO()

    with contextlib.redirect_stderr(captured), _quiet_registration_warnings():
        for name, strict in _STRATEGIES:
            try:
                exported_program = torch.export.export(
                    model, example_inputs, dynamic_shapes=dynamic_shapes, strict=strict
                )
            except Exception as exc:  # noqa: BLE001 - a failed strategy means try the next
                exceptions.append((name, exc))
            else:
                strategy_used = name
                break

        if exported_program is None:
            return CaptureResult(
                success=False,
                capture_strategy=None,
                # The reason quotes the first failure (strict=False); unsupported_ops is
                # mined from all of them, since strict=True's message is often just "no
                # strategy worked" and loses whatever strict=False said about the real op.
                exception=exceptions[0][1] if exceptions else None,
                exceptions=exceptions,
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
    counts = Counter(node.op_type for node in proto.graph.node)
    return CaptureResult(
        success=True,
        capture_strategy=strategy_used,
        onnx_program=onnx_program,
        onnx_bytes=proto.SerializeToString(),
        opset=proto.opset_import[0].version if proto.opset_import else None,
        op_types=dict(sorted(counts.items(), key=lambda kv: kv[1], reverse=True)),
        stderr=captured.getvalue(),
    )
