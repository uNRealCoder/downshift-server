"""downshift CLI. Each command loads, calls the library, and hands the result to render."""

import json
import logging
import re
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from enum import Enum
from pathlib import Path
from typing import Annotated

import typer
import uvicorn

import downshift
from downshift import __version__
from downshift.cli import render
from downshift.export.manifest import manifest_path_for
from downshift.export.shapes import parse_dynamic_spec
from downshift.export.verdict import ExportVerdict
from downshift.loading import LoadedModel, LoadError, load_model
from downshift.serve.app import build_app
from downshift.serve.engine import ServeOptions, prepare_serving

EXIT_USAGE = 4  # bad model spec, bad option, unloadable file
EXIT_CRASH = 5

app = typer.Typer(
    help="Check whether a PyTorch model survives ONNX export, then serve it.",
    no_args_is_help=True,
    add_completion=False,
)


class LogLevel(str, Enum):
    debug = "debug"
    info = "info"
    warning = "warning"
    error = "error"


class LogFormat(str, Enum):
    text = "text"
    json = "json"


class BackendChoice(str, Enum):
    auto = "auto"
    onnxruntime = "onnxruntime"
    torch = "torch"


ModelArg = Annotated[
    str,
    typer.Argument(metavar="MODEL", help="model.onnx | pkg.module:attr | weights.pt | org/repo"),
]
InputsOpt = Annotated[
    str | None,
    typer.Option(
        "--inputs", metavar="pkg.module:fn", help="Example inputs: a tuple, or a factory for one"
    ),
]
ModelClassOpt = Annotated[
    str | None,
    typer.Option(
        "--model-class", metavar="pkg.module:Class", help="Class to load a state dict into"
    ),
]
UnsafeLoadOpt = Annotated[
    bool,
    typer.Option(
        "--unsafe-load", help="Allow torch.load(weights_only=False); runs code from the file"
    ),
]
AdapterOpt = Annotated[
    str | None,
    typer.Option("--adapter", help="Model-family adapter (generic, pyg, hf); default: detect"),
]
SamplesOpt = Annotated[
    int, typer.Option("-k", "--samples", min=1, help="Number of verification samples")
]
DynamicOpt = Annotated[
    str | None,
    typer.Option(
        "--dynamic",
        metavar="NAME:AXIS[,...]",
        help='Dynamic axes, e.g. "x:0,edge_index:1". Default: axis 0 of every input.',
    ),
]
ReferenceOpt = Annotated[
    str | None,
    typer.Option("--reference", metavar="MODEL", help="PyTorch model to verify a .onnx against"),
]
JsonOpt = Annotated[
    bool, typer.Option("--json", help="Print the verdict as JSON and nothing else")
]
LogLevelOpt = Annotated[LogLevel, typer.Option("--log-level")]
LogFormatOpt = Annotated[LogFormat, typer.Option("--log-format")]


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "time": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def _setup_logging(level: LogLevel, fmt: LogFormat) -> None:
    handler = logging.StreamHandler(sys.stderr)  # keep stdout clean for --json
    if fmt is LogFormat.json:
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    logging.basicConfig(level=level.value.upper(), handlers=[handler], force=True)
    # The ONNX optimizer passes log every rewrite at INFO. Only show them when debugging.
    if level is not LogLevel.debug:
        for name in ("onnxscript", "onnx_ir"):
            logging.getLogger(name).setLevel(logging.WARNING)


@contextmanager
def _exit_on_error(debug: bool) -> Iterator[None]:
    """User errors exit 4, anything else exits 5. typer.Exit passes through untouched."""
    try:
        yield
    except typer.Exit:
        raise
    except ValueError as exc:  # LoadError, bad --dynamic, bad backend combination
        render.error(str(exc))
        raise typer.Exit(EXIT_USAGE) from exc
    except Exception as exc:
        if debug:
            render.print_traceback()
        render.error(f"{type(exc).__name__}: {exc}")
        raise typer.Exit(EXIT_CRASH) from exc


def _load(spec: str, inputs: str | None, model_class: str | None, unsafe_load: bool) -> LoadedModel:
    if unsafe_load:
        render.warn(
            f"--unsafe-load: torch.load(weights_only=False) on {spec}; arbitrary code may run"
        )
    return load_model(spec, inputs=inputs, model_class=model_class, unsafe_load=unsafe_load)


_IMPORT_SPEC = re.compile(r"^[A-Za-z_][\w.]*:[A-Za-z_]\w*$")


def slug(spec: str) -> str:
    """tests.models.clean_mlp:make_model -> clean_mlp; ./gat_v3.pt -> gat_v3; org/repo -> repo."""
    if _IMPORT_SPEC.match(spec):
        return spec.partition(":")[0].rsplit(".", 1)[-1]
    return Path(spec).stem


def _emit(
    verdict: ExportVerdict, model_name: str, as_json: bool, extra: dict | None = None
) -> None:
    if as_json:
        typer.echo(json.dumps(verdict.to_dict() | (extra or {}), indent=2))
    else:
        render.print_verdict(verdict, model_name)


@app.command("check")
def check_cmd(
    model: ModelArg,
    json_out: JsonOpt = False,
    reference: ReferenceOpt = None,
    inputs: InputsOpt = None,
    model_class: ModelClassOpt = None,
    unsafe_load: UnsafeLoadOpt = False,
    adapter: AdapterOpt = None,
    k: SamplesOpt = 8,
    dynamic: DynamicOpt = None,
    log_level: LogLevelOpt = LogLevel.warning,
    log_format: LogFormatOpt = LogFormat.text,
) -> None:
    """Export in memory and verify numerics. Exit 0 CLEAN, 1 FAILED, 2 DEGRADED, 3 UNVERIFIED."""
    _setup_logging(log_level, log_format)
    with _exit_on_error(log_level is LogLevel.debug):
        loaded = _load(model, inputs, model_class, unsafe_load)
        dynamic_spec = parse_dynamic_spec(dynamic) if dynamic else None
        if loaded.onnx_path is not None:
            ref = _load(reference, inputs, model_class, unsafe_load) if reference else None
            verdict = downshift.intake(
                loaded.onnx_path,
                ref.model if ref else None,
                ref.example_inputs if ref else None,
                adapter or (ref.adapter_hint if ref else None),
                k=k,
                dynamic=dynamic_spec,
            )
        else:
            assert loaded.model is not None
            verdict = downshift.check(
                loaded.model,
                loaded.example_inputs,
                k=k,
                adapter=adapter or loaded.adapter_hint,
                dynamic=dynamic_spec,
            )
        _emit(verdict, model, json_out)
        raise typer.Exit(verdict.exit_code)


@app.command("export")
def export_cmd(
    model: ModelArg,
    output: Annotated[Path, typer.Option("-o", "--output", help="Output directory")],
    name: Annotated[
        str | None, typer.Option("--name", help="Artifact stem; default: model slug")
    ] = None,
    fp16: Annotated[
        bool, typer.Option("--fp16", help="Cast the model to fp16 before export")
    ] = False,
    no_verify: Annotated[
        bool, typer.Option("--no-verify", help="Skip numerics; the verdict is UNVERIFIED")
    ] = False,
    json_out: JsonOpt = False,
    inputs: InputsOpt = None,
    model_class: ModelClassOpt = None,
    unsafe_load: UnsafeLoadOpt = False,
    adapter: AdapterOpt = None,
    k: SamplesOpt = 8,
    dynamic: DynamicOpt = None,
    log_level: LogLevelOpt = LogLevel.warning,
    log_format: LogFormatOpt = LogFormat.text,
) -> None:
    """Export to DIR/NAME.onnx with a NAME.manifest.json sidecar. Nothing is written if FAILED."""
    _setup_logging(log_level, log_format)
    with _exit_on_error(log_level is LogLevel.debug):
        loaded = _load(model, inputs, model_class, unsafe_load)
        if loaded.model is None:
            raise LoadError(f"{model} is already ONNX; export needs a PyTorch model")
        onnx_path = output / f"{name or slug(model)}.onnx"
        if no_verify:
            render.warn("--no-verify: the graph is saved without checking its numerics")
        verdict = downshift.export(
            loaded.model,
            onnx_path,
            loaded.example_inputs,
            k=k,
            adapter=adapter or loaded.adapter_hint,
            dynamic=parse_dynamic_spec(dynamic) if dynamic else None,
            fp16=fp16,
            source_path=loaded.source_path,
            verify_numerics=not no_verify,
        )
        manifest = manifest_path_for(onnx_path) if verdict.onnx_path else None
        _emit(verdict, model, json_out, {"manifest_path": str(manifest) if manifest else None})
        if not json_out:
            render.print_artifacts(verdict.onnx_path, manifest)
        raise typer.Exit(verdict.exit_code)


@app.command("serve")
def serve_cmd(
    model: ModelArg,
    host: Annotated[str, typer.Option("--host")] = "0.0.0.0",
    port: Annotated[int, typer.Option("--port")] = 8000,
    backend: Annotated[BackendChoice, typer.Option("--backend")] = BackendChoice.auto,
    force_onnx: Annotated[
        bool, typer.Option("--force-onnx", help="Serve a DEGRADED graph via ONNX Runtime anyway")
    ] = False,
    device: Annotated[str, typer.Option("--device", help="auto | cpu | cuda")] = "auto",
    warmup: Annotated[
        int, typer.Option("--warmup", min=0, help="Warm-up inferences before /ready flips")
    ] = 3,
    reference: ReferenceOpt = None,
    middleware: Annotated[
        list[str] | None,
        typer.Option("--middleware", metavar="pkg.module:Attr", help="Middleware to attach; repeatable"),
    ] = None,
    inputs: InputsOpt = None,
    model_class: ModelClassOpt = None,
    unsafe_load: UnsafeLoadOpt = False,
    adapter: AdapterOpt = None,
    k: SamplesOpt = 8,
    dynamic: DynamicOpt = None,
    log_level: LogLevelOpt = LogLevel.info,
    log_format: LogFormatOpt = LogFormat.text,
) -> None:
    """Check the model, pick a backend from the verdict, and serve it over HTTP."""
    _setup_logging(log_level, log_format)
    with _exit_on_error(log_level is LogLevel.debug):
        loaded = _load(model, inputs, model_class, unsafe_load)
        ref = _load(reference, inputs, model_class, unsafe_load) if reference else None
        opts = ServeOptions(
            backend=backend.value,
            force_onnx=force_onnx,
            device=device,
            warmup=warmup,
            k=k,
            adapter=adapter,
            dynamic=parse_dynamic_spec(dynamic) if dynamic else None,
        )
        state = prepare_serving(loaded, opts, ref)
        render.print_banner(state, host, port)
        api = build_app(state, middleware or ())
        uvicorn.run(api, host=host, port=port, log_level=log_level.value)


@app.command()
def version() -> None:
    """Print the version."""
    typer.echo(f"downshift v{__version__}")


if __name__ == "__main__":
    app()
