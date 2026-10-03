"""Round-tripping a `serve` invocation through `--workers N`, and the logging setup every
command uses. ServeArgs/ArtifactHandoff are the JSON envelope a parent process ships to its
uvicorn worker processes; the builders below turn that envelope (or a fresh MODEL argument)
into a ServingState + FastAPI app.

Heavy imports (torch, onnxruntime, downshift.core, downshift.serve.engine/app,
downshift.loading) are deferred into the function bodies that need them, same discipline as
`downshift.cli.main`: this module is imported by main.py at load time to re-export these names,
so keeping it torch-free at module scope is what keeps `--help`/`--version` fast.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

from downshift.adapters.pooling import PoolingChoice
from downshift.cli import render
from downshift.cli.options import LogLevel
from downshift.core.phase import Phase
from downshift.logs import setup_logging
from downshift.serve.options import BackendChoice, ExecutionChoice, ServeOptions
from downshift.serve.schemas import OutputEncoding
from downshift.sources import UNKNOWN_SOURCE

if TYPE_CHECKING:
    from fastapi import FastAPI

    from downshift.loading import LoadedModel, LoadSpec
    from downshift.serve.engine import ServingState

_SERVE_ARGS_ENV = "_DOWNSHIFT_SERVE_ARGS"


def _setup_logging(level: LogLevel, *, stderr: bool = False) -> None:
    """The one logging sink, on stdout. `stderr=True` is for `--json`, which keeps stdout a
    single parseable JSON document."""
    setup_logging(level.value, stream=sys.stderr if stderr else None)


def _load(spec: LoadSpec) -> LoadedModel:
    from downshift.loading import load_model

    if spec.unsafe_load:
        render.warn(
            f"--unsafe-load: torch.load(weights_only=False) on {spec.model}; arbitrary code may run"
        )
    return load_model(spec)


@dataclass
class ArtifactHandoff:
    """Set only for `--workers N`: the parent already ran capture/verify once and ships the
    result here so workers don't repeat it. `backend` is "onnxruntime" (workers load the
    exported graph, no torch model needed) or "torch" (workers still load the model and run
    prepare_model, but take the verdict as given rather than re-exporting)."""

    backend: str | None = None
    verdict: dict | None = None
    input_names: list[str] | None = None
    notes: list[str] | None = None
    onnx_path: str | None = None  # onnxruntime only: the temp .onnx to load
    feeds_path: str | None = None  # onnxruntime only: real example inputs, if any
    # The parent's adapter axis names/bounds (engine.axis_bounds_to_json); the worker has no
    # Prepared to derive them from, and /schema and the 400 messages need them (U2).
    axis_bounds: dict[str, list[list]] | None = None
    # The parent's ServingState.source_kind: an ONNX worker has no LoadedModel to read it from.
    kind: str | None = None
    # The parent's ServingState.hf_source: the HF repo directory to
    # read tokenizer/pooling metadata from, if any. Carried rather than re-derived so a
    # worker never needs to re-validate --tokenizer-from itself.
    hf_source: str | None = None


@dataclass
class ServeArgs:
    """Everything needed to rebuild a ServingState + FastAPI app from scratch. Round-tripped
    through JSON (dataclasses.asdict + json.dumps/loads) for a `--workers N` run, which ships
    this to each worker process via an env var; from_json/to_json do the enum coercion that
    needs (json has no enum type, so BackendChoice/OutputEncoding come back as plain str)."""

    load: LoadSpec
    options: ServeOptions
    reference: str | None
    middleware: list[str] | None
    log_level: str
    artifact: ArtifactHandoff | None = None
    access_log: bool = True
    # A Hugging Face repo directory validated by loading.resolve_tokenizer_source, or None.
    # Independent of `reference`: this is the whole of --tokenizer-from's job.
    tokenizer_from: str | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, raw: str) -> ServeArgs:
        from downshift.loading import LoadSpec

        data = json.loads(raw)
        data["load"] = LoadSpec(**data["load"])
        options = data["options"]
        options["backend"] = BackendChoice(options["backend"])
        options["output_encoding"] = OutputEncoding(options["output_encoding"])
        options["execution"] = ExecutionChoice(options["execution"])
        data["options"] = ServeOptions(**options)
        if data.get("artifact") is not None:
            data["artifact"] = ArtifactHandoff(**data["artifact"])
        return cls(**data)


def _collect_serve_args(
    model: str,
    inputs: str | None,
    model_class: str | None,
    unsafe_load: bool,
    pooling: PoolingChoice | None,
    normalize: bool | None,
    adapter: str | None,
    k: int,
    dynamic: str | None,
    backend: BackendChoice,
    force_onnx: bool,
    device: str,
    warmup: int,
    intra_op_threads: int,
    inter_op_threads: int,
    output_encoding: OutputEncoding,
    max_input_bytes: int,
    max_body_bytes: int,
    max_concurrency: int,
    execution: ExecutionChoice,
    prep_threads: int,
    max_queue: int,
    request_timeout: float,
    atol: float | None,
    rtol: float | None,
    seed: int,
    vary: str | None,
    axis_max: dict[str, int] | None,
    export_cache_dir: str | None,
) -> tuple[LoadSpec, ServeOptions]:
    """serve_cmd's typer parameters, collected into the LoadSpec/ServeOptions pair ServeArgs
    carries. A new serve option is one field here, one on ServeOptions, and one typer
    parameter, instead of touching every layer in between."""
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
        axis_max=axis_max,
        export_cache_dir=export_cache_dir,
        pooling=pooling,
        normalize=normalize,
    )
    return load, options


def _build_from_onnx_artifact(args: ServeArgs, opts: ServeOptions) -> ServingState:
    import numpy as np

    from downshift.core.verdict import ExportVerdict
    from downshift.serve.engine import axis_bounds_from_json, serving_state_from_artifact

    assert args.artifact is not None
    assert args.artifact.verdict is not None and args.artifact.onnx_path is not None
    verdict = ExportVerdict.from_dict(args.artifact.verdict)
    input_names = tuple(args.artifact.input_names or verdict.input_names)
    example_inputs = None
    if args.artifact.feeds_path:
        with np.load(args.artifact.feeds_path) as feeds:
            example_inputs = tuple(feeds[name] for name in input_names)
    return serving_state_from_artifact(
        args.load.model,
        Path(args.artifact.onnx_path),
        verdict,
        opts,
        input_names,
        args.artifact.notes,
        example_inputs,
        axis_bounds_from_json(args.artifact.axis_bounds) if args.artifact.axis_bounds else None,
        kind=args.artifact.kind or UNKNOWN_SOURCE,
        hf_source=args.artifact.hf_source,
    )


def _build_from_torch_artifact(args: ServeArgs, opts: ServeOptions) -> ServingState:
    from downshift.core.verdict import ExportVerdict
    from downshift.serve.engine import serving_state_from_torch_artifact

    assert args.artifact is not None and args.artifact.verdict is not None
    verdict = ExportVerdict.from_dict(args.artifact.verdict)
    load_start = time.perf_counter()
    loaded = _load(args.load)
    load_s = time.perf_counter() - load_start
    state = serving_state_from_torch_artifact(
        loaded, verdict, opts, notes=args.artifact.notes, hf_source=args.artifact.hf_source
    )
    state.timings[Phase.load] = load_s
    return state


def _build_serving_state(args: ServeArgs) -> ServingState:
    from downshift.loading import hf_repo_dir
    from downshift.serve.reuse import prepare_serving_reusing

    opts = args.options

    if args.artifact is not None and args.artifact.backend == "onnxruntime":
        return _build_from_onnx_artifact(args, opts)
    if args.artifact is not None and args.artifact.backend == "torch":
        return _build_from_torch_artifact(args, opts)

    load_s = 0.0

    def load() -> tuple[LoadedModel, LoadedModel | None]:
        nonlocal load_s
        load_start = time.perf_counter()
        loaded = _load(args.load)
        # --reference shares --pooling/--normalize with MODEL: see check_cmd's comment.
        ref = _load(replace(args.load, model=args.reference)) if args.reference else None
        load_s += time.perf_counter() - load_start
        return loaded, ref

    state = prepare_serving_reusing(
        load,
        opts,
        args.tokenizer_from,
        repo=hf_repo_dir(args.load.model),
        inputs_spec=args.load.inputs,
    )
    state.timings[Phase.load] = load_s
    return state


def _serve_app_factory() -> FastAPI:
    """Import-string target for uvicorn's multi-worker mode
    (`downshift.cli.runtime:_serve_app_factory`). Each worker process calls this on its own,
    synchronously, *before* uvicorn's per-worker Server starts accepting on the socket the
    parent already bound (see uvicorn._subprocess.subprocess_started) - so, like the
    single-worker path, this builds the app with a loader rather than a ready-made state:
    without that, a worker would accept no connections at all, not even /health, for the
    whole load/export/verify/warmup instead of answering a fast /ready 503 in the meantime.
    With no artifact set, a worker independently reloads/re-exports/re-warms the model,
    exactly like the single-worker path; with one set, it loads the parent's already-verified
    export instead (see ServeArgs). Unlike the single-worker path, a failed load here has no
    uvicorn.Server to set should_exit on (that Server is built by uvicorn itself, after this
    function returns), so the loader below exits the process directly instead: a worker that
    can't load must not stay up quietly serving 503s forever with nothing to say why. It exits
    with uvicorn's STARTUP_FAILURE code, which tells uvicorn's worker supervisor to stop the
    whole server rather than respawn the worker into the same failure (and the same reload)
    forever.
    """
    from uvicorn.config import STARTUP_FAILURE

    from downshift.serve.app import build_app

    args = ServeArgs.from_json(os.environ[_SERVE_ARGS_ENV])
    _setup_logging(LogLevel(args.log_level))

    def loader() -> ServingState:
        try:
            state = _build_serving_state(args)
        except Exception as exc:
            if LogLevel(args.log_level) is LogLevel.debug:
                render.print_traceback()
            render.error(f"model failed to load in this worker: {type(exc).__name__}: {exc}")
            os._exit(STARTUP_FAILURE)
        render.print_ready(state)
        return state

    return build_app(
        loader=loader, middleware=tuple(args.middleware or ()), access_log=args.access_log
    )


_HANDOFF_KEEPALIVE: list[object] = []


def _write_onnx_artifact(state: ServingState) -> tuple[Path, Path | None, Path | None]:
    """(onnx_path, feeds_path, temp_dir) for a `--workers N` parent to hand its already-
    verified export to the workers. temp_dir is what to clean up afterwards, or None when
    nothing was written (an already-on-disk .onnx with no example inputs to save)."""
    import tempfile

    import numpy as np

    from downshift.serve.backends import example_feeds

    verdict = state.verdict
    if verdict._tmpdir is not None:
        # An external-data export lives in the verdict's own temp dir, data file beside the
        # .onnx. The parent drops its state before the workers load, which would delete that
        # dir, so it has to outlive the verdict (its finalizer then runs at exit).
        _HANDOFF_KEEPALIVE.append(verdict._tmpdir)
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
        feeds = example_feeds(state.input_names, state.example_inputs)
        np.savez(feeds_path, **feeds)  # type: ignore[arg-type]  # numpy's stub misreads **kwds
    return onnx_path, feeds_path, temp_dir
