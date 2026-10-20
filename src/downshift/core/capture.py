"""Run torch.export (strict=False, then strict=True) and torch.onnx.export on the result, and
say which strategy worked.
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

# Weights above this size go to an external data file. Protobuf limits one message to 2 GiB.
EXTERNAL_DATA_THRESHOLD = 1_800_000_000

_REGISTRATION_LOGGER = "torch.onnx._internal.exporter._registration"


@contextlib.contextmanager
def _quiet_registration_warnings() -> Iterator[None]:
    """Silence the torch.onnx warning about missing torchvision ops, for one capture() only."""
    logger = logging.getLogger(_REGISTRATION_LOGGER)
    previous = logger.level
    logger.setLevel(logging.ERROR)
    try:
        yield
    finally:
        logger.setLevel(previous)


@contextlib.contextmanager
def _capture_torch_output(sink: io.StringIO) -> Iterator[None]:
    """Collect what torch prints during the export (whole FX graphs on a failed guard). The
    sys.stderr swap is process-wide, so it happens only on the main thread. On the serve loader
    thread, next to a live server, only the torch loggers are collected."""
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
    capture_strategy: str | None  # a _STRATEGIES name. None if nothing traced
    onnx_program: "torch.onnx.ONNXProgram | None" = None
    onnx_bytes: bytes = field(default=b"", repr=False)
    # In place of onnx_bytes if the weights went to external data. `tmpdir` owns the files.
    onnx_path: Path | None = None
    tmpdir: tempfile.TemporaryDirectory | None = field(default=None, repr=False)
    opset: int | None = None
    op_types: dict[str, int] = field(default_factory=dict)  # histogram, highest count first
    exception: Exception | None = None  # quoted as the reason
    exceptions: list[tuple[str, Exception]] = field(default_factory=list)  # each strategy tried
    stderr: str = ""  # what torch printed, for the debug log


def op_type_histogram(nodes: Iterable[Any]) -> dict[str, int]:
    """Operation types by count, highest first."""
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
            except Exception as exc:  # noqa: BLE001 - a failed strategy means: try the next one
                exceptions.append((name, exc))
            else:
                strategy_used = name
                break

        if exported_program is None:
            return CaptureResult(
                success=False,
                capture_strategy=None,
                # Quote strict=False: the strict=True message often hides the real op.
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
        # Only the legacy path returns None, but the type stub allows it.
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
