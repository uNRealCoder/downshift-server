"""downshift CLI. Each command loads, calls the library, and hands the result to render.

Heavy imports (torch, onnxruntime, uvicorn, fastapi, downshift.core, downshift.serve,
downshift.loading) are deferred into the command bodies that need them, so `--help` and
`--version` stay fast and torch-free. `from __future__ import annotations` lets the type
hints below name those modules' types without importing them at module load time; the
`TYPE_CHECKING` block below is what makes mypy still see them.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import shutil
import sys
import tempfile
import traceback
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

import downshift
from downshift import __version__, settings
from downshift.cli import render
from downshift.serve.options import BackendChoice, ServeOptions
from downshift.serve.schemas import OutputEncoding

if TYPE_CHECKING:
    from fastapi import FastAPI

    from downshift.core.verdict import ExportVerdict
    from downshift.loading import LoadedModel
    from downshift.serve.engine import ServingState

EXIT_USAGE = 4  # bad model spec, bad option, unloadable file
EXIT_CRASH = 5

app = typer.Typer(
    help="Serve a PyTorch model over HTTP, with its ONNX export verified against PyTorch first.",
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
MaxQueueOpt = Annotated[
    int,
    typer.Option(
        "--max-queue",
        min=0,
        help="Predicts allowed to wait past --max-concurrency before a new one gets a fast 503",
    ),
]
RequestTimeoutOpt = Annotated[
    float,
    typer.Option(
        "--request-timeout",
        min=0,
        help="Seconds a predict may wait, unstarted, before a 503 instead of an inference; "
        "0 = no limit",
    ),
]
AtolOpt = Annotated[
    float | None,
    typer.Option("--atol", help="Absolute tolerance override; default: by output dtype"),
]
RtolOpt = Annotated[
    float | None,
    typer.Option("--rtol", help="Relative tolerance override; default: by output dtype"),
]
SeedOpt = Annotated[int, typer.Option("--seed", help="Seed for verification sample generation")]
VaryOpt = Annotated[
    str | None,
    typer.Option(
        "--vary",
        metavar="pkg.module:fn",
        help="fn(i) -> inputs for verification samples, overriding the adapter's own; "
        "fn(0) must return the example inputs",
    ),
]
JsonOpt = Annotated[bool, typer.Option("--json", help="Print the verdict as JSON and nothing else")]
LogLevelOpt = Annotated[LogLevel, typer.Option("--log-level")]
LogFormatOpt = Annotated[LogFormat, typer.Option("--log-format")]
AccessLogOpt = Annotated[
    bool, typer.Option("--access-log/--no-access-log", help="Uvicorn's per-request access log")
]


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
    from downshift.loading import load_model

    if unsafe_load:
        render.warn(
            f"--unsafe-load: torch.load(weights_only=False) on {spec}; arbitrary code may run"
        )
    return load_model(spec, inputs=inputs, model_class=model_class, unsafe_load=unsafe_load)


def slug(spec: str) -> str:
    """tests.models.clean_mlp:make_model -> clean_mlp; ./gat_v3.pt -> gat_v3; org/repo -> repo."""
    from downshift.loading import is_import_spec

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


def _log_capture_failure(verdict: ExportVerdict, log_level: LogLevel) -> None:
    """At --log-level debug, a FAILED verdict logs what the CLI table only summarises:
    torch's own stderr and every strategy's traceback, not just the first line."""
    if log_level is not LogLevel.debug or verdict.status != "FAILED":
        return
    logger = logging.getLogger("downshift.cli")
    if verdict.capture_stderr:
        logger.debug("torch.export/onnx stderr:\n%s", verdict.capture_stderr)
    for name, exc in verdict.capture_exceptions:
        trace = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        logger.debug("%s failed:\n%s", name, trace)


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
    atol: AtolOpt = None,
    rtol: RtolOpt = None,
    seed: SeedOpt = 0,
    vary: VaryOpt = None,
    log_level: LogLevelOpt = LogLevel.warning,
    log_format: LogFormatOpt = LogFormat.text,
) -> None:
    """Export in memory and verify numerics. Exit 0 CLEAN, 1 FAILED, 2 DEGRADED, 3 UNVERIFIED."""
    from downshift.core.shapes import parse_dynamic_spec

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
                atol=atol,
                rtol=rtol,
                seed=seed,
                vary=vary,
            )
        else:
            assert loaded.model is not None
            verdict = downshift.check(
                loaded.model,
                loaded.example_inputs,
                k=k,
                adapter=adapter or loaded.adapter_hint,
                dynamic=dynamic_spec,
                atol=atol,
                rtol=rtol,
                seed=seed,
                vary=vary,
            )
        _log_capture_failure(verdict, log_level)
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
    atol: AtolOpt = None,
    rtol: RtolOpt = None,
    seed: SeedOpt = 0,
    vary: VaryOpt = None,
    log_level: LogLevelOpt = LogLevel.warning,
    log_format: LogFormatOpt = LogFormat.text,
) -> None:
    """Export to DIR/NAME.onnx with a NAME.manifest.json sidecar. Nothing is written if FAILED."""
    from downshift.core.manifest import manifest_path_for
    from downshift.core.shapes import parse_dynamic_spec
    from downshift.loading import LoadError

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
            atol=atol,
            rtol=rtol,
            seed=seed,
            vary=vary,
        )
        _log_capture_failure(verdict, log_level)
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
    max_queue: int
    request_timeout: float
    atol: float | None
    rtol: float | None
    seed: int
    vary: str | None
    log_level: str
    log_format: str
    # Set only for `--workers N`: the parent already ran capture/verify once and ships the
    # result here so workers don't repeat it. artifact_backend is "onnxruntime" (workers
    # load the exported graph, no torch model needed) or "torch" (workers still load the
    # model and run prepare_model, but take the verdict as given rather than re-exporting).
    artifact_backend: str | None = None
    artifact_verdict: dict | None = None
    artifact_input_names: list[str] | None = None
    artifact_notes: list[str] | None = None
    artifact_onnx_path: str | None = None  # onnxruntime only: the temp .onnx to load
    artifact_feeds_path: str | None = None  # onnxruntime only: real example inputs, if any


_SERVE_ARGS_ENV = "_DOWNSHIFT_SERVE_ARGS"


def _serve_options(args: ServeArgs) -> ServeOptions:
    from downshift.core.shapes import parse_dynamic_spec

    return ServeOptions(
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
        max_queue=args.max_queue,
        request_timeout=args.request_timeout,
        atol=args.atol,
        rtol=args.rtol,
        seed=args.seed,
        vary=args.vary,
    )


def _build_from_onnx_artifact(args: ServeArgs, opts: ServeOptions) -> ServingState:
    import numpy as np

    from downshift.core.verdict import ExportVerdict
    from downshift.serve.engine import serving_state_from_artifact

    assert args.artifact_verdict is not None and args.artifact_onnx_path is not None
    verdict = ExportVerdict.from_dict(args.artifact_verdict)
    input_names = tuple(args.artifact_input_names or verdict.input_names)
    example_inputs = None
    if args.artifact_feeds_path:
        with np.load(args.artifact_feeds_path) as feeds:
            example_inputs = tuple(feeds[name] for name in input_names)
    return serving_state_from_artifact(
        args.model,
        Path(args.artifact_onnx_path),
        verdict,
        opts,
        input_names,
        args.artifact_notes,
        example_inputs,
    )


def _build_from_torch_artifact(args: ServeArgs, opts: ServeOptions) -> ServingState:
    from downshift.core.verdict import ExportVerdict, prepare_model
    from downshift.serve.backends import TorchBackend
    from downshift.serve.engine import ServingState, warmup

    assert args.artifact_verdict is not None
    verdict = ExportVerdict.from_dict(args.artifact_verdict)
    loaded = _load(args.model, args.inputs, args.model_class, args.unsafe_load)
    assert loaded.model is not None
    adapter = args.adapter or loaded.adapter_hint
    prepared = prepare_model(
        loaded.model, loaded.example_inputs, adapter, opts.dynamic, vary=args.vary
    )
    verdict.prepared = prepared
    backend = TorchBackend(
        prepared.model, prepared.input_names, opts.device, prepared.inputs, opts.intra_op_threads
    )
    state = ServingState(
        args.model,
        verdict,
        backend,
        prepared.input_names,
        opts,
        prepared.inputs,
        notes=list(args.artifact_notes or []),
    )
    warmup(state, opts.warmup)
    return state


def _build_serving_state(args: ServeArgs) -> ServingState:
    from downshift.serve.engine import prepare_serving

    opts = _serve_options(args)

    if args.artifact_backend == "onnxruntime":
        return _build_from_onnx_artifact(args, opts)
    if args.artifact_backend == "torch":
        return _build_from_torch_artifact(args, opts)
    loaded = _load(args.model, args.inputs, args.model_class, args.unsafe_load)
    ref = (
        _load(args.reference, args.inputs, args.model_class, args.unsafe_load)
        if args.reference
        else None
    )
    return prepare_serving(loaded, opts, ref)


def _build_serving_app(args: ServeArgs) -> tuple[ServingState, FastAPI]:
    from downshift.serve.app import build_app

    state = _build_serving_state(args)
    api = build_app(state, middleware=tuple(args.middleware or ()))
    return state, api


def _serve_app_factory() -> FastAPI:
    """Import-string target for uvicorn's multi-worker mode (`downshift.cli.main:_serve_app_factory`).
    Each worker process calls this on its own. With no artifact fields set, it independently
    reloads/re-exports/re-warms the model, exactly like the single-worker path; with them
    set, it loads the parent's already-verified export instead (see ServeArgs)."""
    args = ServeArgs(**json.loads(os.environ[_SERVE_ARGS_ENV]))
    _setup_logging(LogLevel(args.log_level), LogFormat(args.log_format))
    _, api = _build_serving_app(args)
    return api


def _write_onnx_artifact(state: ServingState) -> tuple[Path, Path | None, Path | None]:
    """(onnx_path, feeds_path, temp_dir) for a `--workers N` parent to hand its already-
    verified export to the workers. temp_dir is what to clean up afterwards, or None when
    nothing was written (an already-on-disk .onnx with no example inputs to save)."""
    import numpy as np

    verdict = state.verdict
    needs_copy = verdict.onnx_path is None
    needs_feeds = state.example_inputs is not None
    if not needs_copy and not needs_feeds:
        assert verdict.onnx_path is not None
        return verdict.onnx_path, None, None

    temp_dir = Path(tempfile.mkdtemp(prefix="downshift-"))
    onnx_path = verdict.onnx_path if verdict.onnx_path is not None else temp_dir / "model.onnx"
    if needs_copy:
        onnx_path.write_bytes(verdict.onnx_bytes)
    feeds_path = None
    if needs_feeds:
        assert state.example_inputs is not None
        feeds_path = temp_dir / "feeds.npz"
        feeds = {
            name: t.numpy() if hasattr(t, "numpy") else np.asarray(t)
            for name, t in zip(state.input_names, state.example_inputs, strict=True)
        }
        np.savez(feeds_path, **feeds)  # type: ignore[arg-type]  # numpy's stub misreads **kwds
    return onnx_path, feeds_path, temp_dir


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
    max_queue: MaxQueueOpt = settings.MAX_QUEUE,
    request_timeout: RequestTimeoutOpt = settings.REQUEST_TIMEOUT,
    workers: WorkersOpt = settings.WORKERS,
    atol: AtolOpt = None,
    rtol: RtolOpt = None,
    seed: SeedOpt = 0,
    vary: VaryOpt = None,
    log_level: LogLevelOpt = LogLevel.info,
    log_format: LogFormatOpt = LogFormat.text,
    access_log: AccessLogOpt = True,
) -> None:
    """Check the model, pick a backend from the verdict, and serve it over HTTP."""
    import uvicorn

    _setup_logging(log_level, log_format)
    with _exit_on_error(log_level is LogLevel.debug):
        if workers > 1 and intra_op_threads == 0:
            # Unset (0 means "let ONNX Runtime/torch choose") oversubscribes N-fold across
            # worker processes; split the logical cores instead. Explicit flags still win.
            intra_op_threads = max(1, (os.cpu_count() or 1) // workers)
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
            max_queue=max_queue,
            request_timeout=request_timeout,
            atol=atol,
            rtol=rtol,
            seed=seed,
            vary=vary,
            log_level=log_level.value,
            log_format=log_format.value,
        )
        if workers <= 1:
            from downshift.serve.app import build_app

            # Bind first, load in the background: a slow export/verify/warmup no longer
            # holds the port closed. failure[] and server_holder{} let the loader (which
            # runs on its own thread, started by build_app's lifespan once uvicorn actually
            # serves) reach back into the main thread on a crash: it stashes the exception
            # here and flips should_exit, so a failed load still exits the process instead
            # of serving 503 forever.
            failure: list[BaseException] = []
            server_holder: dict[str, uvicorn.Server] = {}

            def loader() -> ServingState:
                try:
                    state = _build_serving_state(args)
                except Exception as exc:
                    failure.append(exc)
                    server = server_holder.get("server")
                    if server is not None:
                        server.should_exit = True
                    raise
                render.print_banner(state, host, port)
                return state

            api = build_app(loader=loader, middleware=tuple(args.middleware or ()))
            render.print_booting(model, host, port)
            config = uvicorn.Config(
                api, host=host, port=port, log_level=log_level.value, access_log=access_log
            )
            server = uvicorn.Server(config)
            server_holder["server"] = server
            server.run()
            if failure:
                raise failure[0]
        else:
            # The one and only export: each worker loads this artifact instead of
            # capturing/verifying again. Skips warmup here; this throwaway copy never
            # serves traffic, only prints the banner and decides which backend to ship.
            state, _ = _build_serving_app(replace(args, warmup=0))
            render.print_banner(state, host, port, workers=workers)
            temp_dir: Path | None = None
            if state.backend.name == "onnxruntime":
                onnx_path, feeds_path, temp_dir = _write_onnx_artifact(state)
                if temp_dir is not None:
                    atexit.register(shutil.rmtree, temp_dir, ignore_errors=True)
                args = replace(
                    args,
                    artifact_backend="onnxruntime",
                    artifact_verdict=state.verdict.to_dict(),
                    artifact_input_names=list(state.input_names),
                    artifact_notes=state.notes,
                    artifact_onnx_path=str(onnx_path),
                    artifact_feeds_path=str(feeds_path) if feeds_path else None,
                )
            else:
                render.warn(
                    f"--workers {workers}: each worker independently reloads and re-warms "
                    "the model (memory and startup time scale with this number)"
                )
                args = replace(
                    args,
                    artifact_backend="torch",
                    artifact_verdict=state.verdict.to_dict(),
                    artifact_input_names=list(state.input_names),
                    artifact_notes=state.notes,
                )
            os.environ[_SERVE_ARGS_ENV] = json.dumps(asdict(args))
            try:
                uvicorn.run(
                    "downshift.cli.main:_serve_app_factory",
                    host=host,
                    port=port,
                    workers=workers,
                    log_level=log_level.value,
                    access_log=access_log,
                    factory=True,
                )
            finally:
                if temp_dir is not None:
                    shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":  # pragma: no cover
    app()
