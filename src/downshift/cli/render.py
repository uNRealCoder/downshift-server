"""All rich output for the CLI lives here. Commands hand over objects; this module prints."""

from pathlib import Path

from rich import box
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from downshift import __version__
from downshift.export.verdict import ExportVerdict
from downshift.serve.codec import BASE64_CODEC
from downshift.serve.engine import ServingState

console = Console()
err_console = Console(stderr=True)

STATUS_STYLE = {
    "CLEAN": "bold green",
    "DEGRADED": "bold yellow",
    "FAILED": "bold red",
    "UNVERIFIED": "bold magenta",
}


def _sym(utf: str, ascii_: str) -> str:
    """Unicode glyph unless the console can't encode it (cp1252 Windows pipes)."""
    return ascii_ if console.options.ascii_only else utf


def _status_text(verdict: ExportVerdict) -> Text:
    text = Text(verdict.status, style=STATUS_STYLE[verdict.status])
    details = [verdict.capture_strategy] if verdict.capture_strategy else []
    if verdict.opset is not None:
        details.append(f"opset {verdict.opset}")
    if details:
        text.append(f"  ({', '.join(details)})", style="dim")
    return text


def _numerics_text(verdict: ExportVerdict) -> Text:
    n = verdict.numerics
    if n is None:
        return Text("not checked", style="dim")
    text = Text(f"max abs err {n.max_abs_err:.2e} over {n.samples_tested} samples  ")
    if n.passed:
        text.append(_sym("✓", "OK"), style="bold green")
    else:
        text.append(f"{_sym('✗', 'X')} {n.failures}/{n.samples_tested} failed", style="bold red")
    return text


def _dynamic_text(verdict: ExportVerdict) -> str:
    if not verdict.dynamic_dims:
        return _sym("—", "-")
    return ", ".join(
        f"{name}[{axis}]" for name, axes in verdict.dynamic_dims.items() for axis in axes
    )


def _shape_text(verdict: ExportVerdict) -> str:
    if verdict.shape_generalization is None:
        return _sym("—", "-")
    return "yes" if verdict.shape_generalization else "no"


def print_verdict(verdict: ExportVerdict, model_name: str) -> None:
    table = Table(show_header=False, box=box.ROUNDED, border_style=STATUS_STYLE[verdict.status])
    table.add_column(style="bold", no_wrap=True)
    table.add_column()
    table.add_row("Model", escape(model_name))
    table.add_row("Family", verdict.model_family)
    table.add_row("Export", _status_text(verdict))
    table.add_row("Numerics", _numerics_text(verdict))
    table.add_row("Shape-general", _shape_text(verdict))
    table.add_row("Dynamic dims", _dynamic_text(verdict))
    if verdict.unsupported_ops:
        table.add_row("Unsupported ops", Text(", ".join(verdict.unsupported_ops), style="red"))
    if verdict.warnings:
        table.add_row("Warnings", Text("\n".join(verdict.warnings), style="yellow"))
    table.add_row("Backend", verdict.recommended_backend)
    table.add_row("Reason", escape(verdict.reason))
    console.print(table)


def print_artifacts(onnx_path: Path | None, manifest_path: Path | None) -> None:
    if onnx_path is None:
        console.print("[red]nothing written[/]: the export failed")
        return
    console.print(f"[bold]Wrote[/]     {escape(str(onnx_path))}")
    if manifest_path is not None:
        console.print(f"[bold]Manifest[/]  {escape(str(manifest_path))}")


def _backend_text(state: ServingState) -> Text:
    meta = state.backend.metadata()
    label = "torch (eager)" if meta.name == "torch" else meta.name
    text = Text(f"{label} {_sym('·', '|')} {meta.device}")
    arrow = _sym("←", "<-")
    if state.forced_onnx:
        text.append(f"  {arrow} --force-onnx", style="yellow")
    elif state.backend_auto_selected:
        text.append(f"  {arrow} auto-selected", style="dim")
    return text


def print_banner(state: ServingState, host: str, port: int) -> None:
    """Boot banner for `serve`. Says what is served, how it was judged, and where it listens."""
    verdict = state.verdict
    sub = _sym("└", "\\")
    grid = Table.grid(padding=(0, 3))
    grid.add_column(style="bold cyan", no_wrap=True)
    grid.add_column()

    grid.add_row("Model", escape(state.source))
    grid.add_row("Family", verdict.model_family)

    unverified_onnx = verdict.status == "UNVERIFIED" and verdict.prepared is None
    if unverified_onnx:
        grid.add_row("Verdict", _status_text(verdict) + Text(" - no reference model supplied"))
        grid.add_row("", Text(f"{sub} served as-is; numerics were never checked", style="dim"))
        grid.add_row("Tip", "pass --reference <model> to verify")
    else:
        grid.add_row("Verdict", _status_text(verdict))
        if verdict.status in ("FAILED", "UNVERIFIED"):
            grid.add_row("", Text(f"{sub} {verdict.reason}", style="dim"))

    grid.add_row("Numerics", _numerics_text(verdict))

    if verdict.status == "DEGRADED":
        n = verdict.numerics
        if n is None:
            detail = verdict.reason
        else:
            detail = (
                f"numerics diverge on {n.failures}/{n.samples_tested} samples "
                f"(max abs err {n.max_abs_err:.2e})"
            )
        grid.add_row("", Text(f"{_sym('⚠', '!')} {detail}", style="yellow"))
        if state.backend.name != "onnxruntime":
            grid.add_row("Override", "--force-onnx to serve the ONNX graph anyway")

    grid.add_row("Backend", _backend_text(state))
    for note in state.notes:  # backend-selection notes from the engine
        style = "bold red" if "outputs may be wrong" in note else "yellow"
        grid.add_row("", Text(f"{_sym('⚠', '!')} {note}", style=style))
    grid.add_row("Dynamic dims", _dynamic_text(verdict))
    for warning in verdict.warnings:
        grid.add_row("", Text(f"{_sym('⚠', '!')} {warning}", style="yellow"))
    encoding = Text(state.options.output_encoding.value)
    encoding.append("  (clients override with output_encoding)", style="dim")
    grid.add_row("Encoding", encoding)
    if BASE64_CODEC == "stdlib":
        grid.add_row(
            "Tip", escape("pip install 'downshift-server[fast]' for ~10x faster base64 tensor I/O")
        )
    grid.add_row("Endpoint", f"http://{host}:{port}")

    console.print(
        Panel(
            grid,
            title=f"downshift v{__version__}",
            title_align="left",
            border_style=STATUS_STYLE[verdict.status],
            expand=False,
            padding=(1, 2),
        )
    )


def warn(msg: str) -> None:
    err_console.print(f"[bold yellow]warning:[/] {escape(msg)}")


def error(msg: str) -> None:
    err_console.print(f"[bold red]error:[/] {escape(msg)}")


def print_traceback() -> None:
    err_console.print_exception()
