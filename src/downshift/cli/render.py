"""Everything the CLI reports, as plain text through `logging`. Commands hand over objects; this
module builds the text and logs it, so reports, warnings and errors share the one sink
`downshift.logs.setup_logging` installs (plain text on stdout, no colours).

A report (verdict, banner, artifacts) is one INFO record on `downshift.report`: a multi-line
`Label   value` block. One record, not one per row, so a log collector keeps a report
together. That logger is forced to INFO, so reports print at any `--log-level`.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

from downshift import __version__
from downshift.core.phase import Phase
from downshift.logs import REPORT_LOGGER
from downshift.serve.codec import BASE64_CODEC
from downshift.sources import HF_REPO_DIR, IN_PROCESS_MODULE, ONNX_FILE, TORCH_CHECKPOINT

if TYPE_CHECKING:
    from downshift.core.verdict import ExportVerdict
    from downshift.serve.engine import ServingState

report_logger = logging.getLogger(REPORT_LOGGER)
logger = logging.getLogger("downshift.cli")

_WARN = "!"
_SUB = "->"


class _Rows:
    """`Label   value` lines, values aligned in one column. A row with an empty label carries
    on from the row above it; a value with newlines keeps its continuation lines in the column."""

    def __init__(self) -> None:
        self._rows: list[tuple[str, str]] = []

    def add(self, label: str, value: str) -> None:
        self._rows.append((label, value))

    def note(self, text: str) -> None:
        self._rows.append(("", text))

    def lines(self) -> list[str]:
        width = max((len(label) for label, _ in self._rows), default=0) + 3
        lines = []
        for label, value in self._rows:
            first, *rest = value.split("\n")
            lines.append(f"{label:<{width}}{first}".rstrip())
            lines.extend(f"{'':<{width}}{line}".rstrip() for line in rest)
        return lines


def _log_report(rows: _Rows, title: str | None = None) -> None:
    lines = rows.lines()
    if title is not None:
        lines = [title] + [f"  {line}" for line in lines]
    report_logger.info("%s", "\n".join(lines))


def _status_text(verdict: ExportVerdict) -> str:
    details = [verdict.capture_strategy] if verdict.capture_strategy else []
    if verdict.opset is not None:
        details.append(f"opset {verdict.opset}")
    if details:
        return f"{verdict.status}  ({', '.join(details)})"
    return verdict.status


def _numerics_text(verdict: ExportVerdict) -> str:
    n = verdict.numerics
    if n is None:
        return "not checked"
    text = f"max abs err {n.max_abs_err:.2e} over {n.samples_tested} samples  "
    if n.passed:
        return text + "OK"
    return text + f"{n.failures}/{n.samples_tested} failed"


def _tolerance_text(verdict: ExportVerdict) -> str | None:
    n = verdict.numerics
    if n is None:
        return None
    picked_by = "--atol/--rtol" if n.tolerance_overridden else n.tolerance_dtype
    return f"atol {n.tolerance_abs:.0e}, rtol {n.tolerance_rel:.0e} ({picked_by})"


def _shape_tuple_text(shape: tuple[int, ...]) -> str:
    return "(" + ",".join(str(dim) for dim in shape) + ")"


def _input_names(verdict: ExportVerdict, count: int) -> tuple[str, ...]:
    """The model's input names, or input_0, input_1, ... when the verdict carries none."""
    return verdict.input_names or tuple(f"input_{i}" for i in range(count))


def _worst_text(verdict: ExportVerdict) -> str | None:
    n = verdict.numerics
    if n is None or n.worst is None:
        return None
    w = n.worst
    index = ", ".join(str(i) for i in w.index)
    names = _input_names(verdict, len(w.input_shapes))
    shapes = ", ".join(
        f"{name} {_shape_tuple_text(shape)}"
        for name, shape in zip(names, w.input_shapes, strict=True)
    )
    return (
        f"output_{w.output}[{index}]: torch {w.expected:.4f}, onnxruntime {w.got:.4f}"
        f"  (sample {w.sample}, {shapes})"
    )


def _samples_text(verdict: ExportVerdict) -> str | None:
    n = verdict.numerics
    if n is None or not n.sample_shapes:
        return None
    names = _input_names(verdict, len(n.sample_shapes[0]))
    lines = []
    for name, shapes in zip(names, zip(*n.sample_shapes, strict=True), strict=True):
        lines.append(f"{name}: " + " ".join(_shape_tuple_text(shape) for shape in shapes))
    return "\n".join(lines)


def _axes_text(verdict: ExportVerdict) -> str:
    """One line per dynamic axis: its Dim name (what --axis-max takes), the sizes verify ran,
    and the sizes the server accepts. The batch axis never warns: rows are independent."""
    if not verdict.axes:
        if not verdict.dynamic_dims:
            return "-"
        return ", ".join(
            f"{name}[{axis}]" for name, axes in verdict.dynamic_dims.items() for axis in axes
        )
    lines = []
    for fact in verdict.axes:
        if fact.sampled_min is None or fact.sampled_max is None:
            sampled = "not verified"
        else:
            sampled = f"sampled {fact.sampled_min}-{fact.sampled_max}"
        line = (
            f"`{fact.name}` ({fact.input}[{fact.axis}])  {sampled}, "
            f"serves {fact.served_min}-{fact.served_max}"
        )
        if (
            fact.name != "batch"
            and fact.sampled_max is not None
            and fact.served_max > 2 * fact.sampled_max
        ):
            line += f"  {_WARN} unverified above {fact.sampled_max}"
        lines.append(line)
    return "\n".join(lines)


def _warmup_text(state: ServingState) -> str | None:
    w = state.warmup_stats
    if w is None or w.count == 0:
        return None
    if w.synthesized:
        return f"{w.count} inferences on synthesized inputs"
    return f"{w.count} inferences, {w.mean_ms:.2f} ms each"


def _capacity_text(state: ServingState) -> str:
    opts = state.options
    inferences = "inference" if opts.max_concurrency == 1 else "inferences"
    timeout = "no timeout" if opts.request_timeout <= 0 else f"{opts.request_timeout:g} s timeout"
    return (
        f"{opts.max_concurrency} {inferences} at a time, {opts.max_queue} queued, {timeout}"
        "  (--max-concurrency, --max-queue, --request-timeout)"
    )


def _boot_text(state: ServingState) -> str | None:
    timings = state.timings
    if not timings:
        return None
    total = sum(timings.values())
    parts = ", ".join(f"{name} {timings[name]:.1f}" for name in Phase if name in timings)
    return f"{total:.1f} s: {parts}"


def _shape_text(verdict: ExportVerdict) -> str:
    n = verdict.numerics
    if n is None:
        return "-"
    if n.baseline_failed:
        return "n/a (baseline fails)"
    return "yes" if verdict.shape_generalization else "no"


def _add_numerics_rows(rows: _Rows, verdict: ExportVerdict) -> None:
    """Tolerance/Worst/Samples, the three rows that explain a verdict's numbers.

    `check`'s report and `serve`'s banner show them identically, so they are built once here
    rather than kept in step by hand in two places.
    """
    tolerance = _tolerance_text(verdict)
    if tolerance is not None:
        rows.add("Tolerance", tolerance)
    if verdict.status == "DEGRADED":
        worst = _worst_text(verdict)
        if worst is not None:
            rows.add("Worst", worst)
    samples = _samples_text(verdict)
    if samples is not None:
        rows.add("Samples", samples)


def print_verdict(verdict: ExportVerdict, model_name: str) -> None:
    rows = _Rows()
    rows.add("Model", model_name)
    rows.add("Family", verdict.model_family)
    rows.add("Export", _status_text(verdict))
    rows.add("Numerics", _numerics_text(verdict))
    _add_numerics_rows(rows, verdict)
    rows.add("Shape-general", _shape_text(verdict))
    rows.add("Dynamic dims", _axes_text(verdict))
    if verdict.unsupported_ops:
        rows.add("Unsupported ops", ", ".join(verdict.unsupported_ops))
    if verdict.warnings:
        rows.add("Warnings", "\n".join(verdict.warnings))
    rows.add("Backend", verdict.recommended_backend)
    reason = verdict.reason
    if verdict.status == "FAILED":
        reason += " (run with --log-level debug for the torch.export trace)"
    rows.add("Reason", reason)
    _log_report(rows, f"downshift v{__version__}")


def print_artifacts(onnx_path: Path | None, manifest_path: Path | None) -> None:
    if onnx_path is None:
        report_logger.info("nothing written: the export failed")
        return
    rows = _Rows()
    rows.add("Wrote", str(onnx_path))
    if manifest_path is not None:
        rows.add("Manifest", str(manifest_path))
    _log_report(rows)


# What the banner calls each artifact on disk downshift accepts, so it is obvious nothing
# was fetched to start this server. An import spec is left unlabelled: it says what it is,
# and the label would only push a long spec onto a second line.
_SOURCE_LABEL = {
    ONNX_FILE: "local .onnx file",
    TORCH_CHECKPOINT: "local PyTorch checkpoint",
    HF_REPO_DIR: "downloaded Hugging Face repo",
    IN_PROCESS_MODULE: "in-process nn.Module",
}


def _model_text(state: ServingState) -> str:
    label = _SOURCE_LABEL.get(state.source_kind)
    return state.source if label is None else f"{state.source}  ({label})"


def _backend_text(state: ServingState) -> str:
    meta = state.backend.metadata()
    label = "torch (eager)" if meta.name == "torch" else meta.name
    text = f"{label} | {meta.device}"
    if state.forced_onnx:
        return f"{text}  <- --force-onnx"
    if state.backend_auto_selected:
        return f"{text}  <- auto-selected"
    return text


def _verified_on_text(state: ServingState) -> str | None:
    """The execution provider the numerics check ran on, or None when it never ran. Numerics
    are CPU-only today, so a CUDA server says its verdict does not cover the device it serves on."""
    provider = state.backend.verified_provider
    if provider is None:
        return None
    device = state.backend.metadata().device
    if device.startswith("cpu"):
        return provider
    return f"{provider}  (serving on {device}; numerics were only verified on the CPU)"


_ANY_HOSTS = ("", "0.0.0.0", "::")


def _client_url(host: str, port: int) -> str:
    """The address a client would use: a wildcard bind is reached as localhost."""
    shown = "localhost" if host in _ANY_HOSTS else host
    if ":" in shown and not shown.startswith("["):
        shown = f"[{shown}]"
    return f"http://{shown}:{port}"


def _endpoint_text(host: str, port: int) -> str:
    url = _client_url(host, port)
    if host in _ANY_HOSTS:
        return f"{url}  (bound to {host or 'all interfaces'}; GET /schema for the input format)"
    return f"{url}  (GET /schema for the input format)"


def print_banner(state: ServingState, host: str, port: int, workers: int = 1) -> None:
    """Boot banner for `serve`. Says what is served, how it was judged, and where it listens."""
    verdict = state.verdict
    rows = _Rows()

    rows.add("Model", _model_text(state))
    rows.add("Family", verdict.model_family)

    unverified_onnx = verdict.status == "UNVERIFIED" and verdict.prepared is None
    if unverified_onnx:
        rows.add("Verdict", f"{_status_text(verdict)} - no reference model supplied")
        rows.note(f"{_SUB} served as-is; numerics were never checked")
        rows.add("Tip", "pass --reference <model> to verify")
    else:
        rows.add("Verdict", _status_text(verdict))
        if verdict.status in ("FAILED", "UNVERIFIED"):
            rows.note(f"{_SUB} {verdict.reason}")

    rows.add("Numerics", _numerics_text(verdict))

    if verdict.status == "DEGRADED":
        n = verdict.numerics
        if n is None:
            detail = verdict.reason
        else:
            detail = (
                f"numerics diverge on {n.failures}/{n.samples_tested} samples "
                f"(max abs err {n.max_abs_err:.2e})"
            )
        rows.note(f"{_WARN} {detail}")
        if state.backend.name != "onnxruntime":
            rows.add("Override", "--force-onnx to serve the ONNX graph anyway")

    _add_numerics_rows(rows, verdict)
    warmup_text = _warmup_text(state)
    if warmup_text is not None:
        rows.add("Warmup", warmup_text)
    if workers > 1:
        logical = os.cpu_count() or workers
        rows.add(
            "Threads",
            f"{state.options.intra_op_threads} intra-op per worker"
            f"  ({logical} logical / {workers} workers)",
        )
    boot = _boot_text(state)
    if boot is not None:
        rows.add("Boot", boot)

    rows.add("Backend", _backend_text(state))
    verified_on = _verified_on_text(state)
    if verified_on is not None:
        rows.add("Verified on", verified_on)
    for note in state.notes:  # backend-selection notes from the engine
        rows.note(f"{_WARN} {note}")
    if state.embedding is not None:
        rows.add("Embedding", f"{state.embedding.describe()}  (from {state.embedding.origin})")
    if state.text is not None:
        rows.add(
            "Text input",
            f'"text" accepted, up to {state.text.max_length} tokens a row  (longer rows are refused)',
        )
    rows.add("Dynamic dims", _axes_text(verdict))
    for warning in verdict.warnings:
        rows.note(f"{_WARN} {warning}")
    rows.add(
        "Encoding",
        f"{state.options.output_encoding.value}  (clients override with output_encoding)",
    )
    rows.add("Capacity", _capacity_text(state))
    if BASE64_CODEC == "stdlib":
        rows.add("Tip", "pip install 'downshift-server[fast]' for ~10x faster base64 tensor I/O")
    rows.add("Endpoint", _endpoint_text(host, port))

    _log_report(rows, f"downshift v{__version__}")


def print_booting(model: str, host: str, port: int) -> None:
    """Logged once, before uvicorn binds; the full banner (print_banner) follows once the
    loader thread lands a verdict."""
    report_logger.info("loading %s", model)
    report_logger.info("will listen on %s (not ready yet)", _client_url(host, port))


def print_ready(state: ServingState) -> None:
    """Logged by the loader as it hands the state over, which is when /ready flips."""
    if not state.timings:
        report_logger.info("ready")
        return
    report_logger.info("ready in %.1f s", sum(state.timings.values()))


def warn(msg: str) -> None:
    logger.warning("%s", msg)


def error(msg: str) -> None:
    logger.error("%s", msg)


def print_traceback() -> None:
    """The active exception's traceback, at debug: what `--log-level debug` adds to the one-line
    error `_exit_on_error` logs, whichever kind of error it was."""
    logger.debug("traceback", exc_info=True)
