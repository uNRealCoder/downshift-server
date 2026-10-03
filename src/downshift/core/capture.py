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
import tempfile
import threading
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import onnx
import torch

_STRATEGIES: tuple[tuple[str, bool], ...] = (("strict=False", False), ("strict=True", True))

# Weights above this go to an external data file: protobuf caps one message at 2 GiB.
EXTERNAL_DATA_THRESHOLD = 1_800_000_000

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


@contextlib.contextmanager
def _capture_torch_output(sink: io.StringIO) -> Iterator[None]:
    """Collect what torch prints while it exports into `sink`.

    torch prints whole FX graphs straight to stderr when a data-dependent guard fails; keep
    that out of the user's terminal, the exception message is what matters. Swapping
    sys.stderr is process-global, so it is only done on the main thread (the CLI's own
    check/export). The serve loader runs on another thread beside a live server, whose
    stderr writers must not be swallowed: there only torch's logging is collected.
    """
    if threading.current_thread() is threading.main_thread():
        with contextlib.redirect_stderr(sink):
            yield
        return
    handler = logging.StreamHandler(sink)
    torch_logger = logging.getLogger("torch")
    torch_logger.addHandler(handler)
    try:
        yield
    finally:
        torch_logger.removeHandler(handler)


@dataclass
class CaptureResult:
    success: bool
    capture_strategy: str | None  # a _STRATEGIES name; None when nothing traced
    onnx_program: "torch.onnx.ONNXProgram | None" = None
    onnx_bytes: bytes = field(default=b"", repr=False)  # model_proto.SerializeToString(), once
    # Set instead of onnx_bytes when the weights went to external data; `tmpdir` owns the files.
    onnx_path: Path | None = None
    tmpdir: tempfile.TemporaryDirectory | None = field(default=None, repr=False)
    opset: int | None = None
    op_types: dict[str, int] = field(default_factory=dict)  # count-descending histogram
    exception: Exception | None = None  # the exception build_verdict quotes as the reason
    exceptions: list[tuple[str, Exception]] = field(default_factory=list)  # every strategy tried
    stderr: str = ""  # whatever torch printed while we tried; useful at debug level


def op_type_histogram(nodes: Iterable[Any]) -> dict[str, int]:
    """Op types by count, descending. Both paths that produce a verdict - a fresh export here
    and intake()'s read of a pre-built .onnx - report this same shape on ExportVerdict.op_types,
    so they share one definition of it."""
    counts = Counter(node.op_type for node in nodes)
    return dict(sorted(counts.items(), key=lambda kv: kv[1], reverse=True))


def _weights_nbytes(exported_program: "torch.export.ExportedProgram") -> int:
    tensors = [*exported_program.state_dict.values(), *exported_program.constants.values()]
    return sum(t.numel() * t.element_size() for t in tensors if isinstance(t, torch.Tensor))


def capture(
    model: torch.nn.Module,
    example_inputs: tuple,
    dynamic_shapes: tuple | None = None,
    external_data_threshold: int | None = None,
) -> CaptureResult:
    exported_program = None
    strategy_used: str | None = None
    exceptions: list[tuple[str, Exception]] = []
    captured = io.StringIO()

    with _capture_torch_output(captured), _quiet_registration_warnings():
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

    limit = EXTERNAL_DATA_THRESHOLD if external_data_threshold is None else external_data_threshold
    onnx_bytes = b""
    onnx_path: Path | None = None
    tmpdir: tempfile.TemporaryDirectory | None = None
    if _weights_nbytes(exported_program) > limit:
        # A single protobuf is capped at 2 GiB: past the threshold the weights go to a data
        # file beside the .onnx, in a directory the result (then the verdict) keeps alive.
        tmpdir = tempfile.TemporaryDirectory(prefix="downshift-", ignore_cleanup_errors=True)
        onnx_path = Path(tmpdir.name) / "model.onnx"
        onnx_program.save(onnx_path, external_data=True)
        proto = onnx.load(str(onnx_path), load_external_data=False)
    else:
        proto = onnx_program.model_proto
        onnx_bytes = proto.SerializeToString()
    return CaptureResult(
        success=True,
        capture_strategy=strategy_used,
        onnx_program=onnx_program,
        onnx_bytes=onnx_bytes,
        onnx_path=onnx_path,
        tmpdir=tmpdir,
        opset=proto.opset_import[0].version if proto.opset_import else None,
        op_types=op_type_histogram(proto.graph.node),
        stderr=captured.getvalue(),
    )
