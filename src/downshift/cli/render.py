"""Everything that the CLI reports, as plain text through `logging`. The commands give objects
to this module. It builds the text and logs it. Reports, warnings and errors therefore share the
one sink that `downshift.logs.setup_logging` installs (plain text on stdout, no colours).

A report (verdict, banner or artifacts) is one INFO record on `downshift.report`. It is a
block of lines in the form `Label   value`. It is one record and not one record for each row.
A log collector therefore keeps a report together. That logger is forced to INFO, so reports
print at all values of `--log-level`.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

from downshift import __version__, settings
from downshift.core.phase import Phase
from downshift.logs import REPORT_LOGGER
from downshift.serve.codec import BASE64_CODEC
from downshift.serve.options import ExecutionChoice
from downshift.sources import (
    HF_REPO_DIR,
    IN_PROCESS_MODULE,
    ONNX_FILE,
    TORCH_CHECKPOINT,
    display_source,
    path_basename,
)

if TYPE_CHECKING:
    from downshift.core.verdict import ExportVerdict
    from downshift.serve.engine import ServingState

report_logger = logging.getLogger(REPORT_LOGGER)
logger = logging.getLogger("downshift.cli")

_WARN = "!"
_SUB = "->"


class _Rows:
    """Lines in the form `Label   value`, with the values in one column. A row with an empty
    label continues the row above it. A value with newlines keeps its next lines in the column."""

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
    """The input names of the model, or input_0, input_1, ... if the verdict has none."""
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
    """One line for each dynamic axis: its Dim name (the name that --axis-max takes), the sizes
    that verification ran, and the sizes that the server accepts. The batch axis never gives a
    warning, because the rows are independent."""
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
        f"{opts.max_concurrency} {inferences} at a time, {opts.prep_threads} prep threads, "
        f"{opts.max_queue} queued, {timeout}"
        "  (--max-concurrency, --prep-threads, --max-queue, --request-timeout)"
    )


def _execution_text(state: ServingState) -> str:
    if state.execution == ExecutionChoice.inline:
        return "inline for small JSON bodies  (--execution inline)"
    return "threadpool  (--execution)"


_REUSED_TEXT = {"memory": "reused (in-process)", "disk": "reused from --export-cache-dir"}


def _boot_text(state: ServingState) -> str | None:
    timings = state.timings
    if not timings:
        return None
    total = sum(timings.values())
    parts = ", ".join(f"{name} {timings[name]:.1f}" for name in Phase if name in timings)
    if state.reused is not None:
        parts += ", " + _REUSED_TEXT[state.reused]
    return f"{total:.1f} s: {parts}"


def _shape_text(verdict: ExportVerdict) -> str:
    n = verdict.numerics
    if n is None:
        return "-"
    if n.baseline_failed:
        return "n/a (baseline fails)"
    return "yes" if verdict.shape_generalization else "no"


def _add_numerics_rows(rows: _Rows, verdict: ExportVerdict) -> None:
    """Tolerance, Worst and Samples: the three rows that explain the numbers of a verdict.

    The report of `check` and the banner of `serve` show them in the same way. Downshift
    therefore builds them one time here. Nobody must keep two places in step by hand.
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


# The banner uses these names for each artifact on disk that downshift accepts. They show that
# downshift fetched nothing to start this server. An import spec has no label. It already shows
# what it is, and a label would only push a long spec onto a second line.
_SOURCE_LABEL = {
    ONNX_FILE: "local .onnx file",
    TORCH_CHECKPOINT: "local PyTorch checkpoint",
    HF_REPO_DIR: "downloaded Hugging Face repo",
    IN_PROCESS_MODULE: "in-process nn.Module",
}


def _model_text(state: ServingState) -> str:
    label = _SOURCE_LABEL.get(state.source_kind)
    name = display_source(state.source, state.source_kind)
    return name if label is None else f"{name}  ({label})"


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
    """The execution provider that the numerics check ran on, or None if it did not run. The
    numerics run only on the CPU. A CUDA server therefore says that its verdict does not cover
    the device that it serves on."""
    provider = state.backend.verified_provider
    if provider is None:
        return None
    device = state.backend.metadata().device
    if device.startswith("cpu"):
        return provider
    return f"{provider}  (serving on {device}; numerics were only verified on the CPU)"


_ANY_HOSTS = ("", "0.0.0.0", "::")


def _client_url(host: str, port: int) -> str:
    """The address that a client uses. A client reaches a wildcard bind as localhost."""
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
    """The boot banner of `serve`. It says what is served, how downshift judged it, and where it listens."""
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
        logical = settings.usable_cpus()
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
    for note in state.notes:  # notes about the backend selection, from the engine
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
    rows.add("Execution", _execution_text(state))
    if state.execution == ExecutionChoice.inline:
        rows.note(
            f"{_WARN} --execution inline only helps very fast models (under ~1 ms per "
            "inference); a slow model stalls the event loop, including /health and /ready"
        )
    if BASE64_CODEC == "stdlib":
        rows.add("Tip", "pip install 'downshift-server[fast]' for ~10x faster base64 tensor I/O")
    rows.add("Endpoint", _endpoint_text(host, port))

    _log_report(rows, f"downshift v{__version__}")


def print_config(
    state: ServingState,
    host: str,
    port: int,
    *,
    workers: int = 1,
    log_level: str,
    access_log: bool,
    middleware: Sequence[str] = (),
    api_key_set: bool,
) -> None:
    """Every setting that the server runs with, as one `name = value` line for each setting. It
    logs at INFO (only `--log-level info` or lower shows it), directly before `ready`. The
    whole of ServeOptions goes out as it is. A new option therefore shows here without a
    change to this function. It shows names only. The model is its file name. The API key is
    `set` or `unset` and never its value."""
    values = {
        "model": display_source(state.source, state.source_kind),
        "host": host,
        "port": port,
        "workers": workers,
        "log_level": log_level,
        "access_log": access_log,
        "api_key": "set" if api_key_set else "unset",
        "middleware": ", ".join(middleware) or None,
        **asdict(state.options),
    }
    width = max(len(name) for name in values)
    lines = [
        f"  {name:<{width}} = {'-' if value is None else value}" for name, value in values.items()
    ]
    logger.info("serving with:\n%s", "\n".join(lines))


def print_booting(model: str, host: str, port: int) -> None:
    """Logged one time, before uvicorn binds. The full banner (print_banner) follows when the
    loader thread has a verdict."""
    report_logger.info("loading %s", path_basename(model))
    report_logger.info("will listen on %s (not ready yet)", _client_url(host, port))


def print_ready(state: ServingState) -> None:
    """Logged by the loader when it gives the state over. This is when /ready changes."""
    if not state.timings:
        report_logger.info("ready")
        return
    report_logger.info("ready in %.1f s", sum(state.timings.values()))


def warn(msg: str) -> None:
    logger.warning("%s", msg)


def error(msg: str) -> None:
    logger.error("%s", msg)


def print_traceback() -> None:
    """The traceback of the active exception, at debug. It is what `--log-level debug` adds to
    the one-line error that `_exit_on_error` logs, for all kinds of error."""
    logger.debug("traceback", exc_info=True)
