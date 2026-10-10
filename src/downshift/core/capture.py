"""Run torch.export and torch.onnx.export, and say which strategy worked.

Downshift runs torch.export itself (strict=False, then strict=True). The strategy that works
is then a value that downshift returns. It is not text that downshift takes from the console
output. torch.onnx.export does all of the ONNX translation. Downshift only gives it the
ExportedProgram.

Note for torch 2.14: the dynamo exporter documents only the two strict modes. There is no
draft_export step and no TorchScript fallback.
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
    """torch.onnx logs a warning for each missing torchvision operation on every export. The
    user cannot act on it.

    The scope is one capture() call. Downshift does not set it at import. A library must not
    change the logging configuration of another package only because someone imports it.
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
    """Collect what torch prints while it exports, into `sink`.

    When a data-dependent guard fails, torch prints complete FX graphs directly to stderr. Keep
    this out of the terminal of the user. The exception message is what matters. A swap of
    sys.stderr is global for the process, so downshift does it only on the main thread (the own
    check and export of the CLI). The serve loader runs on another thread next to a live server.
    The stderr writers of that server must not be swallowed. In that case, downshift collects
    only the logging of torch.
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
    capture_strategy: str | None  # a _STRATEGIES name. None if nothing traced
    onnx_program: "torch.onnx.ONNXProgram | None" = None
    onnx_bytes: bytes = field(default=b"", repr=False)  # model_proto.SerializeToString(), one time
    # Set in place of onnx_bytes if the weights went to external data. `tmpdir` owns the files.
    onnx_path: Path | None = None
    tmpdir: tempfile.TemporaryDirectory | None = field(default=None, repr=False)
    opset: int | None = None
    op_types: dict[str, int] = field(default_factory=dict)  # histogram, highest count first
    exception: Exception | None = None  # the exception that build_verdict quotes as the reason
    exceptions: list[tuple[str, Exception]] = field(default_factory=list)  # each strategy tried
    stderr: str = ""  # what torch printed during the attempt. Useful at debug level


def op_type_histogram(nodes: Iterable[Any]) -> dict[str, int]:
    """Operation types by count, in descending order. Two paths produce a verdict: a new export
    here, and the read of a pre-built .onnx file by intake(). Both report this same shape on
    ExportVerdict.op_types. They therefore share one definition of it."""
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
                # The reason quotes the first failure (strict=False). unsupported_ops comes
                # from all failures. The message of strict=True is often only "no strategy
                # worked". It loses what strict=False said about the real operation.
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
        # The type stub allows None for the legacy path. With an ExportedProgram, it is never None.
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
        # One protobuf has a limit of 2 GiB. Above the threshold, the weights go to a data file
        # next to the .onnx file. The directory stays alive through the result (then the
        # verdict).
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
