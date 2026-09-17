"""downshift CLI. Each command loads, calls the library, and hands the result to render."""

import json
import logging
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer
import uvicorn
from fastapi import FastAPI

import downshift
from downshift import __version__, settings
from downshift.cli import render
from downshift.export.manifest import manifest_path_for
from downshift.export.shapes import parse_dynamic_spec
from downshift.export.verdict import ExportVerdict
from downshift.loading import LoadedModel, LoadError, is_import_spec, load_model
from downshift.serve.app import build_app
from downshift.serve.engine import BackendChoice, ServeOptions, ServingState, prepare_serving
from downshift.serve.schemas import OutputEncoding

EXIT_USAGE = 4  # bad model spec, bad option, unloadable file
EXIT_CRASH = 5

app = typer.Typer(
    help="Check whether a PyTorch model survives ONNX export, then serve it.",
    no_args_is_help=True,
    add_completion=False,
)


def _print_version(value: bool) -> None:
    if value:
        typer.echo(f"downshift v{__version__}")
        raise typer.Exit(0)


@app.callback()
def _main(
    version: Annotated[
        bool,
        typer.Option(
            "--version", is_eager=True, callback=_print_version, help="Print the version and exit"
        ),
    ] = False,
) -> None:
    pass


class LogLevel(StrEnum):
    debug = "debug"
    info = "info"
    warning = "warning"
    error = "error"


class LogFormat(StrEnum):
    text = "text"
    json = "json"


ModelArg = Annotated[
    str,
    typer.Argument(
        metavar="MODEL", help="model.onnx | pkg.module:attr | weights.pt | org/repo | hf-repo-dir/"
    ),
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
    typer.Option(
        "--adapter",
        metavar="NAME|path/to/adapter.py[:attr]",
        help="Model-family adapter: generic, pyg, hf, or your own adapter.py; default: detect",
    ),
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
IntraOpThreadsOpt = Annotated[
    int,
    typer.Option(
        "--intra-op-threads", min=0, help="ORT threads within one op; 0 = let ONNX Runtime choose"
    ),
]
InterOpThreadsOpt = Annotated[
    int,
    typer.Option(
        "--inter-op-threads", min=0, help="ORT threads across ops; 0 = let ONNX Runtime choose"
    ),
]
WorkersOpt = Annotated[
    int,
    typer.Option(
        "--workers",
        min=1,
        help="Uvicorn worker processes; each independently loads/exports/warms the model",
    ),
]
OutputEncodingOpt = Annotated[
    OutputEncoding,
    typer.Option(
        "--output-encoding",
        help="Default encoding of response tensors; clients override per request with "
        "output_encoding",
    ),
]
MaxInputBytesOpt = Annotated[
    int,
    typer.Option(
        "--max-input-bytes",
        min=1,
        help="Reject base64 tensor inputs larger than this once decoded",
    ),
]
MaxBodyBytesOpt = Annotated[
    int,
    typer.Option(
        "--max-body-bytes",
        min=1,
        help="Reject request bodies larger than this, before they are parsed as JSON",
    ),
]
MaxConcurrencyOpt = Annotated[
    int,
    typer.Option(
        "--max-concurrency",
        min=1,
        help="Inferences allowed to run at once per worker process; 1 means one at a time "
        "(ONNX Runtime's own intra-op threads still parallelise inside that one inference)",
    ),
]
JsonOpt = Annotated[bool, typer.Option("--json", help="Print the verdict as JSON and nothing else")]
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


def slug(spec: str) -> str:
    """tests.models.clean_mlp:make_model -> clean_mlp; ./gat_v3.pt -> gat_v3; org/repo -> repo."""
    if is_import_spec(spec):
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
    k: SamplesOpt = settings.SAMPLES,
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
    k: SamplesOpt = settings.SAMPLES,
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


@dataclass
class ServeArgs:
    """Everything needed to rebuild a ServingState + FastAPI app from scratch. Plain JSON-able
    types only: a multi-worker run ships this to each worker process via an env var."""

    model: str
    inputs: str | None
    model_class: str | None
    unsafe_load: bool
    adapter: str | None
    k: int
    dynamic: str | None
    reference: str | None
    middleware: list[str] | None
    backend: str
    force_onnx: bool
    device: str
    warmup: int
    intra_op_threads: int
    inter_op_threads: int
    output_encoding: str
    max_input_bytes: int
    max_body_bytes: int
    max_concurrency: int
    log_level: str
    log_format: str


_SERVE_ARGS_ENV = "_DOWNSHIFT_SERVE_ARGS"


def _build_serving_app(args: ServeArgs) -> tuple[ServingState, FastAPI]:
    loaded = _load(args.model, args.inputs, args.model_class, args.unsafe_load)
    ref = (
        _load(args.reference, args.inputs, args.model_class, args.unsafe_load)
        if args.reference
        else None
    )
    opts = ServeOptions(
        backend=BackendChoice(args.backend),
        force_onnx=args.force_onnx,
        device=args.device,
        warmup=args.warmup,
        k=args.k,
        adapter=args.adapter,
        dynamic=parse_dynamic_spec(args.dynamic) if args.dynamic else None,
        intra_op_threads=args.intra_op_threads,
        inter_op_threads=args.inter_op_threads,
        output_encoding=OutputEncoding(args.output_encoding),
        max_input_bytes=args.max_input_bytes,
        max_body_bytes=args.max_body_bytes,
        max_concurrency=args.max_concurrency,
    )
    state = prepare_serving(loaded, opts, ref)
    api = build_app(state, tuple(args.middleware or ()))
    return state, api


def _serve_app_factory() -> FastAPI:
    """Import-string target for uvicorn's multi-worker mode (`downshift.cli.main:_serve_app_factory`).
    Each worker process calls this on its own, independently reloading/re-exporting/re-warming
    the model from the args the parent process serialized into _SERVE_ARGS_ENV."""
    args = ServeArgs(**json.loads(os.environ[_SERVE_ARGS_ENV]))
    _setup_logging(LogLevel(args.log_level), LogFormat(args.log_format))
    _, api = _build_serving_app(args)
    return api


@app.command("serve")
def serve_cmd(
    model: ModelArg,
    host: Annotated[str, typer.Option("--host")] = settings.HOST,
    port: Annotated[int, typer.Option("--port")] = settings.PORT,
    backend: Annotated[BackendChoice, typer.Option("--backend")] = BackendChoice(settings.BACKEND),
    force_onnx: Annotated[
        bool, typer.Option("--force-onnx", help="Serve a DEGRADED graph via ONNX Runtime anyway")
    ] = False,
    device: Annotated[str, typer.Option("--device", help="auto | cpu | cuda")] = settings.DEVICE,
    warmup: Annotated[
        int, typer.Option("--warmup", min=0, help="Warm-up inferences before /ready flips")
    ] = settings.WARMUP,
    reference: ReferenceOpt = None,
    middleware: Annotated[
        list[str] | None,
        typer.Option(
            "--middleware", metavar="pkg.module:Attr", help="Middleware to attach; repeatable"
        ),
    ] = None,
    inputs: InputsOpt = None,
    model_class: ModelClassOpt = None,
    unsafe_load: UnsafeLoadOpt = False,
    adapter: AdapterOpt = None,
    k: SamplesOpt = settings.SAMPLES,
    dynamic: DynamicOpt = None,
    intra_op_threads: IntraOpThreadsOpt = settings.INTRA_OP_THREADS,
    inter_op_threads: InterOpThreadsOpt = settings.INTER_OP_THREADS,
    output_encoding: OutputEncodingOpt = OutputEncoding(settings.OUTPUT_ENCODING),
    max_input_bytes: MaxInputBytesOpt = settings.MAX_INPUT_BYTES,
    max_body_bytes: MaxBodyBytesOpt = settings.MAX_BODY_BYTES,
    max_concurrency: MaxConcurrencyOpt = settings.MAX_CONCURRENCY,
    workers: WorkersOpt = settings.WORKERS,
    log_level: LogLevelOpt = LogLevel.info,
    log_format: LogFormatOpt = LogFormat.text,
) -> None:
    """Check the model, pick a backend from the verdict, and serve it over HTTP."""
    _setup_logging(log_level, log_format)
    with _exit_on_error(log_level is LogLevel.debug):
        args = ServeArgs(
            model=model,
            inputs=inputs,
            model_class=model_class,
            unsafe_load=unsafe_load,
            adapter=adapter,
            k=k,
            dynamic=dynamic,
            reference=reference,
            middleware=list(middleware) if middleware else None,
            backend=backend.value,
            force_onnx=force_onnx,
            device=device,
            warmup=warmup,
            intra_op_threads=intra_op_threads,
            inter_op_threads=inter_op_threads,
            output_encoding=output_encoding.value,
            max_input_bytes=max_input_bytes,
            max_body_bytes=max_body_bytes,
            max_concurrency=max_concurrency,
            log_level=log_level.value,
            log_format=log_format.value,
        )
        if workers <= 1:
            state, api = _build_serving_app(args)
            render.print_banner(state, host, port)
            uvicorn.run(api, host=host, port=port, log_level=log_level.value)
        else:
            # Only for the banner/fail-fast check: each of the N workers rebuilds its own
            # backend anyway, so this throwaway copy skips warmup, it'll never serve traffic.
            state, _ = _build_serving_app(replace(args, warmup=0))
            render.print_banner(state, host, port)
            render.warn(
                f"--workers {workers}: each worker independently reloads, re-exports, and "
                "re-warms the model (memory and startup time scale with this number)"
            )
            os.environ[_SERVE_ARGS_ENV] = json.dumps(asdict(args))
            uvicorn.run(
                "downshift.cli.main:_serve_app_factory",
                host=host,
                port=port,
                workers=workers,
                log_level=log_level.value,
                factory=True,
            )


@app.command()
def version() -> None:
    """Print the version."""
    typer.echo(f"downshift v{__version__}")


if __name__ == "__main__":  # pragma: no cover
    app()
