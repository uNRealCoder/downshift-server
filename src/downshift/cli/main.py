"""The downshift CLI. Each command loads the model, calls the library, and gives the result to
render.

The command bodies do the heavy imports (torch, onnxruntime, uvicorn, fastapi, downshift.core,
downshift.serve and downshift.loading) when they need them. This keeps `--help` and `--version`
fast and free of torch. `from __future__ import annotations` lets the type hints below name the
types of those modules without an import at module load time. The `TYPE_CHECKING` block lets
mypy see them. `downshift.cli.options` (the typer option types) and `downshift.cli.runtime`
(ServeArgs and the `--workers` code) follow the same rule. An import of them here costs nothing.
"""

from __future__ import annotations

import atexit
import gc
import json
import os
import shutil
import sys
import traceback
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

import downshift
from downshift import __version__, settings
from downshift.cli import render
from downshift.cli.options import (
    AccessLogOpt,
    AdapterOpt,
    AtolOpt,
    AxisMaxOpt,
    DynamicOpt,
    ExecutionOpt,
    ExportCacheDirOpt,
    InputsOpt,
    InterOpThreadsOpt,
    IntraOpThreadsOpt,
    JsonOpt,
    LogLevel,
    LogLevelOpt,
    MaxBodyBytesOpt,
    MaxConcurrencyOpt,
    MaxInputBytesOpt,
    MaxQueueOpt,
    ModelArg,
    ModelClassOpt,
    NormalizeOpt,
    OutputEncodingOpt,
    PoolingOpt,
    PrepThreadsOpt,
    ReferenceOpt,
    RequestTimeoutOpt,
    RtolOpt,
    SamplesOpt,
    SeedOpt,
    TokenizerFromOpt,
    UnsafeLoadOpt,
    VaryOpt,
    WorkersOpt,
)
from downshift.cli.runtime import (
    _SERVE_ARGS_ENV,
    ArtifactHandoff,
    ServeArgs,
    _build_serving_state,
    _load,
    _setup_logging,
    _write_onnx_artifact,
)
from downshift.serve.options import BackendChoice, ExecutionChoice, ServeOptions
from downshift.serve.schemas import OutputEncoding

if TYPE_CHECKING:
    from downshift.core.verdict import ExportVerdict
    from downshift.serve.engine import ServingState

EXIT_USAGE = 4  # a bad model spec, a bad option, a file that cannot load
EXIT_CRASH = 5

app = typer.Typer(
    help=(
        "Serve a model that you already downloaded, over HTTP. Downshift verifies the ONNX "
        "export against PyTorch first. MODEL is always on this machine. It is a .onnx file, a "
        "PyTorch checkpoint, a Hugging Face repo directory (a directory with a config.json), "
        "or an importable package.module:attr. Downshift fetches nothing. It rejects a "
        "Hugging Face hub id and does not download it."
    ),
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_enable=False,  # errors go through logging and not through rich
    rich_markup_mode=None,
)


def _print_version(value: bool) -> None:
    if value:
        typer.echo(__version__)
        raise typer.Exit(0)


@app.callback()
def _main(
    version: Annotated[
        bool,
        typer.Option(
            "--version", is_eager=True, callback=_print_version, help="Print the version, then exit"
        ),
    ] = False,
) -> None:
    # As with `python -m downshift`, import specs (MODEL, --inputs, --model-class, --vary and
    # --middleware) also resolve against the current directory. Downshift appends the directory
    # and does not put it first. A local file therefore never hides the standard library or an
    # installed package. The --workers processes start with this sys.path.
    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.append(cwd)


@contextmanager
def _exit_on_error(debug: bool) -> Iterator[None]:
    """A user error exits with 4. Any other error exits with 5. typer.Exit passes through."""
    try:
        yield
    except typer.Exit:
        raise
    except ValueError as exc:  # LoadError, a bad --dynamic, a bad backend combination
        if debug:
            render.print_traceback()
        render.error(str(exc))
        raise typer.Exit(EXIT_USAGE) from exc
    except Exception as exc:
        if debug:
            render.print_traceback()
        render.error(f"{type(exc).__name__}: {exc}")
        raise typer.Exit(EXIT_CRASH) from exc


def _axis_max(values: list[str] | None) -> dict[str, int] | None:
    """--axis-max NAME=N (repeatable) as {name: N}. Without the flag, DOWNSHIFT_AXIS_MAX."""
    parsed = settings.parse_axis_max(values) if values else dict(settings.AXIS_MAX)
    return parsed or None


def slug(spec: str) -> str:
    """tests.models.clean_mlp:make_model -> clean_mlp, ./gat_v3.pt -> gat_v3, hf/bert/ -> bert."""
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
    """At --log-level debug, a FAILED verdict logs what the CLI table only summarizes: the own
    stderr of torch and the traceback of each strategy, not only the first line."""
    if log_level is not LogLevel.debug or verdict.status != "FAILED":
        return
    if verdict.capture_stderr:
        render.logger.debug("torch.export/onnx stderr:\n%s", verdict.capture_stderr)
    for name, exc in verdict.capture_exceptions:
        trace = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        render.logger.debug("%s failed:\n%s", name, trace)


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
    axis_max: AxisMaxOpt = None,
    atol: AtolOpt = None,
    rtol: RtolOpt = None,
    seed: SeedOpt = 0,
    vary: VaryOpt = None,
    pooling: PoolingOpt = None,
    normalize: NormalizeOpt = None,
    log_level: LogLevelOpt = LogLevel.warning,
) -> None:
    """Export in memory and verify the numbers. Exit 0 CLEAN, 1 FAILED, 2 DEGRADED, 3 UNVERIFIED."""
    from downshift.core.shapes import parse_dynamic_spec
    from downshift.loading import LoadSpec

    _setup_logging(log_level, stderr=json_out)
    with _exit_on_error(log_level is LogLevel.debug):
        spec = LoadSpec(model, inputs, model_class, unsafe_load, pooling, normalize)
        loaded = _load(spec)
        dynamic_spec = parse_dynamic_spec(dynamic) if dynamic else None
        axis_max_spec = _axis_max(axis_max)
        if loaded.onnx_path is not None:
            # --reference shares --pooling and --normalize with MODEL. The two agree on the
            # numbers only if the embedding recipe of the reference model matches the export.
            ref = _load(replace(spec, model=reference)) if reference else None
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
                axis_max=axis_max_spec,
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
                axis_max=axis_max_spec,
                cache=False,  # one export for each process: nothing could read the memo again
            )
        _log_capture_failure(verdict, log_level)
        _emit(verdict, model, json_out)
        raise typer.Exit(verdict.exit_code)


@app.command("export")
def export_cmd(
    model: ModelArg,
    output: Annotated[Path, typer.Option("-o", "--output", help="The output directory")],
    name: Annotated[
        str | None, typer.Option("--name", help="The artifact name. Default: the model slug")
    ] = None,
    fp16: Annotated[
        bool, typer.Option("--fp16", help="Cast the model to fp16 before the export")
    ] = False,
    no_verify: Annotated[
        bool, typer.Option("--no-verify", help="Skip the numerics check. The verdict is UNVERIFIED")
    ] = False,
    json_out: JsonOpt = False,
    inputs: InputsOpt = None,
    model_class: ModelClassOpt = None,
    unsafe_load: UnsafeLoadOpt = False,
    adapter: AdapterOpt = None,
    k: SamplesOpt = settings.SAMPLES,
    dynamic: DynamicOpt = None,
    axis_max: AxisMaxOpt = None,
    atol: AtolOpt = None,
    rtol: RtolOpt = None,
    seed: SeedOpt = 0,
    vary: VaryOpt = None,
    pooling: PoolingOpt = None,
    normalize: NormalizeOpt = None,
    export_cache_dir: ExportCacheDirOpt = settings.EXPORT_CACHE_DIR,
    log_level: LogLevelOpt = LogLevel.warning,
) -> None:
    """Export to DIR/NAME.onnx with a NAME.manifest.json file. If the verdict is FAILED, nothing is written."""
    from downshift.core.manifest import manifest_path_for
    from downshift.core.shapes import parse_dynamic_spec
    from downshift.loading import LoadError, LoadSpec

    _setup_logging(log_level, stderr=json_out)
    with _exit_on_error(log_level is LogLevel.debug):
        if export_cache_dir:
            from downshift.core.export_cache import check_dir

            check_dir(export_cache_dir)
        spec = LoadSpec(model, inputs, model_class, unsafe_load, pooling, normalize)
        loaded = _load(spec)
        if loaded.model is None:
            raise LoadError(f"{model} is already ONNX. The export needs a PyTorch model")
        onnx_path = output / f"{name or slug(model)}.onnx"
        if no_verify:
            render.warn("--no-verify: downshift saves the graph without a check of its numbers")
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
            axis_max=_axis_max(axis_max),
            # The key hashes every weight. In a process with one export, only the disk tier can use it.
            cache=bool(export_cache_dir),
            export_cache_dir=export_cache_dir,
        )
        _log_capture_failure(verdict, log_level)
        manifest = manifest_path_for(onnx_path) if verdict.onnx_path else None
        _emit(verdict, model, json_out, {"manifest_path": str(manifest) if manifest else None})
        if not json_out:
            render.print_artifacts(verdict.onnx_path, manifest)
        raise typer.Exit(verdict.exit_code)


@app.command("serve")
def serve_cmd(
    model: ModelArg,
    host: Annotated[str, typer.Option("--host")] = settings.HOST,
    port: Annotated[int, typer.Option("--port")] = settings.PORT,
    backend: Annotated[BackendChoice, typer.Option("--backend")] = BackendChoice(settings.BACKEND),
    force_onnx: Annotated[
        bool, typer.Option("--force-onnx", help="Serve a DEGRADED graph through ONNX Runtime")
    ] = False,
    device: Annotated[str, typer.Option("--device", help="auto | cpu | cuda")] = settings.DEVICE,
    warmup: Annotated[
        int, typer.Option("--warmup", min=0, help="Warm-up inferences before /ready changes")
    ] = settings.WARMUP,
    reference: ReferenceOpt = None,
    tokenizer_from: TokenizerFromOpt = None,
    middleware: Annotated[
        list[str] | None,
        typer.Option(
            "--middleware",
            metavar="pkg.module:Attr",
            help="A middleware to attach. Repeat to add more",
        ),
    ] = None,
    inputs: InputsOpt = None,
    model_class: ModelClassOpt = None,
    unsafe_load: UnsafeLoadOpt = False,
    adapter: AdapterOpt = None,
    k: SamplesOpt = settings.SAMPLES,
    dynamic: DynamicOpt = None,
    axis_max: AxisMaxOpt = None,
    intra_op_threads: IntraOpThreadsOpt = settings.INTRA_OP_THREADS,
    inter_op_threads: InterOpThreadsOpt = settings.INTER_OP_THREADS,
    output_encoding: OutputEncodingOpt = OutputEncoding(settings.OUTPUT_ENCODING),
    max_input_bytes: MaxInputBytesOpt = settings.MAX_INPUT_BYTES,
    max_body_bytes: MaxBodyBytesOpt = settings.MAX_BODY_BYTES,
    max_concurrency: MaxConcurrencyOpt = settings.MAX_CONCURRENCY,
    execution: ExecutionOpt = ExecutionChoice(settings.EXECUTION),
    prep_threads: PrepThreadsOpt = settings.PREP_THREADS,
    max_queue: MaxQueueOpt = settings.MAX_QUEUE,
    request_timeout: RequestTimeoutOpt = settings.REQUEST_TIMEOUT,
    workers: WorkersOpt = settings.WORKERS,
    export_cache_dir: ExportCacheDirOpt = settings.EXPORT_CACHE_DIR,
    atol: AtolOpt = None,
    rtol: RtolOpt = None,
    seed: SeedOpt = 0,
    vary: VaryOpt = None,
    pooling: PoolingOpt = None,
    normalize: NormalizeOpt = None,
    log_level: LogLevelOpt = LogLevel.warning,
    access_log: AccessLogOpt = True,
) -> None:
    """Check a downloaded model, select a backend from the verdict, and serve it over HTTP.

    GET /schema on the running server says what to POST: the name, dtype and shape of each input, plus an example body.
    """  # noqa: E501 - typer wraps the help text itself. A line break in the source would show in it
    import uvicorn

    _setup_logging(log_level)
    with _exit_on_error(log_level is LogLevel.debug):
        if export_cache_dir:
            from downshift.core.export_cache import check_dir

            check_dir(export_cache_dir)
        resolved_tokenizer_from = None
        if tokenizer_from is not None:
            from downshift.loading import resolve_tokenizer_source

            resolved_tokenizer_from = resolve_tokenizer_source(tokenizer_from)
        if workers > 1 and intra_op_threads == 0:
            # An unset value (0 means "let ONNX Runtime or torch choose") oversubscribes the
            # cores N times across the worker processes. Split the logical cores instead. A flag
            # that you set still has priority.
            intra_op_threads = max(1, settings.usable_cpus() // workers)
        from downshift.core.shapes import parse_dynamic_spec
        from downshift.loading import LoadSpec

        load = LoadSpec(
            model=model,
            inputs=inputs,
            model_class=model_class,
            unsafe_load=unsafe_load,
            pooling=pooling,
            normalize=normalize,
        )
        options = ServeOptions(
            backend=backend,
            force_onnx=force_onnx,
            device=device,
            warmup=warmup,
            k=k,
            adapter=adapter,
            dynamic=parse_dynamic_spec(dynamic) if dynamic else None,
            intra_op_threads=intra_op_threads,
            inter_op_threads=inter_op_threads,
            output_encoding=output_encoding,
            max_input_bytes=max_input_bytes,
            max_body_bytes=max_body_bytes,
            max_concurrency=max_concurrency,
            execution=execution,
            prep_threads=prep_threads,
            max_queue=max_queue,
            request_timeout=request_timeout,
            atol=atol,
            rtol=rtol,
            seed=seed,
            vary=vary,
            axis_max=_axis_max(axis_max),
            export_cache_dir=export_cache_dir,
            pooling=pooling,
            normalize=normalize,
        )
        args = ServeArgs(
            load=load,
            options=options,
            reference=reference,
            middleware=list(middleware) if middleware else None,
            log_level=log_level.value,
            access_log=access_log,
            tokenizer_from=resolved_tokenizer_from,
        )

        def print_config(state: ServingState, workers: int = 1) -> None:
            render.print_config(
                state,
                host,
                port,
                workers=workers,
                log_level=log_level.value,
                access_log=access_log,
                middleware=middleware or (),
                api_key_set=bool(settings.API_KEY),
            )

        if workers <= 1:
            from downshift.serve.app import build_app

            # Bind first, then load in the background. A slow export, verification or warmup
            # does not keep the port closed. The loader runs on its own thread. The lifespan
            # of build_app starts it when uvicorn begins to serve. If the loader crashes,
            # failure[] gives the main thread a way to see the error. The loader keeps the
            # exception here and sets should_exit. A failed load then still ends the process.
            # It does not serve 503 forever. `server` below is bound before the lifespan can
            # run the loader, and closures resolve names at call time, so the loader can read it.
            failure: list[BaseException] = []

            def loader() -> ServingState:
                try:
                    state = _build_serving_state(args)
                except Exception as exc:
                    failure.append(exc)
                    server.should_exit = True
                    raise
                render.print_banner(state, host, port)
                print_config(state)
                render.print_ready(state)
                return state

            api = build_app(
                loader=loader, middleware=tuple(args.middleware or ()), access_log=access_log
            )
            render.print_booting(model, host, port)
            # log_config=None: the loggers of uvicorn use the sink that _setup_logging
            # installed. The app logs its own record for each request (access_log), so the
            # access log of uvicorn stays off.
            config = uvicorn.Config(
                api,
                host=host,
                port=port,
                log_level=log_level.value,
                log_config=None,
                access_log=False,
            )
            server = uvicorn.Server(config)
            server.run()
            if failure:
                raise failure[0]
        else:
            # This is the only export. Each worker loads this artifact and does not capture
            # or verify again. This temporary copy never serves traffic. It only decides which
            # backend to send and prints the banner. It therefore skips the warmup and uses
            # the default threads of ONNX Runtime. This lets it reuse the session of the
            # verification and not build a second one. Downshift drops it before uvicorn.run
            # starts the workers.
            parent_options = replace(args.options, warmup=0, intra_op_threads=0, inter_op_threads=0)
            state = _build_serving_state(replace(args, options=parent_options))
            state.options = args.options  # the banner reports what the workers use
            render.print_banner(state, host, port, workers=workers)
            print_config(state, workers=workers)
            temp_dir: Path | None = None
            onnx_path_str: str | None = None
            feeds_path_str: str | None = None
            if state.backend.name == "onnxruntime":
                onnx_path, feeds_path, temp_dir = _write_onnx_artifact(state)
                if temp_dir is not None:
                    atexit.register(shutil.rmtree, temp_dir, ignore_errors=True)
                onnx_path_str = str(onnx_path)
                feeds_path_str = str(feeds_path) if feeds_path else None
            else:
                render.warn(
                    f"--workers {workers}: each worker independently reloads and re-warms "
                    "the model (memory and startup time increase with this number)"
                )
            from downshift.core.axes import axis_bounds_to_json

            artifact = ArtifactHandoff(
                verdict=state.verdict.to_dict(),
                input_names=list(state.input_names),
                kind=state.source_kind,
                axis_bounds=axis_bounds_to_json(state.axis_bounds),
                onnx_path=onnx_path_str,
                feeds_path=feeds_path_str,
            )
            del state
            gc.collect()
            # The verdict goes to the workers, so they never need --reference.
            args = replace(args, artifact=artifact, reference=None)
            os.environ[_SERVE_ARGS_ENV] = args.to_json()
            try:
                uvicorn.run(
                    "downshift.cli.runtime:_serve_app_factory",
                    host=host,
                    port=port,
                    workers=workers,
                    log_level=log_level.value,
                    log_config=None,
                    access_log=False,
                    factory=True,
                )
            finally:
                if temp_dir is not None:
                    shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":  # pragma: no cover
    app()
