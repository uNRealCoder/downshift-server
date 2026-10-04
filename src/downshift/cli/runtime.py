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
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING

from downshift.cli import render
from downshift.cli.options import LogLevel
from downshift.core.phase import Phase
from downshift.logs import setup_logging
from downshift.serve.options import BackendChoice, ExecutionChoice, ServeOptions
from downshift.serve.schemas import OutputEncoding

if TYPE_CHECKING:
    from fastapi import FastAPI

    from downshift.core.memo import ExportEntry
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
    """Set only for `--workers N`: the parent's export, so workers don't capture or verify
    again. A worker boots from it exactly as from an export-cache hit (serve.reuse): the
    backend is chosen from the verdict as the parent chose it, and a torch worker loads the
    model itself (torch weights aren't shipped between processes)."""

    verdict: dict
    input_names: list[str]
    kind: str  # the parent's ServingState.source_kind; an ONNX worker loads nothing to tell
    # [[axis, name, min, max], ...] per input (core.axes): an ONNX worker has no Prepared.
    axis_bounds: dict[str, list[list]] = field(default_factory=dict)
    onnx_path: str | None = None  # the graph, when the parent serves it on onnxruntime
    feeds_path: str | None = None  # real example inputs for warmup, if there were any

    def entry(self) -> ExportEntry:
        import numpy as np

        from downshift.core.memo import ExportEntry

        feeds = None
        if self.feeds_path:
            with np.load(self.feeds_path) as archive:
                feeds = {name: archive[name] for name in archive.files}
        return ExportEntry(
            verdict=self.verdict,
            input_names=self.input_names,
            axis_bounds=self.axis_bounds,
            feeds=feeds,
            onnx_path=Path(self.onnx_path) if self.onnx_path else None,
        )


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


def _build_serving_state(args: ServeArgs) -> ServingState:
    from downshift.loading import hf_repo_dir
    from downshift.serve.reuse import prepare_serving_reusing, state_from_entry

    opts = args.options
    load_s = 0.0

    def load() -> tuple[LoadedModel, LoadedModel | None]:
        nonlocal load_s
        load_start = time.perf_counter()
        loaded = _load(args.load)
        # --reference shares --pooling/--normalize with MODEL: see check_cmd's comment.
        ref = _load(replace(args.load, model=args.reference)) if args.reference else None
        load_s += time.perf_counter() - load_start
        return loaded, ref

    if args.artifact is not None:
        state = state_from_entry(
            args.artifact.entry(),
            load,
            opts,
            args.tokenizer_from,
            source=args.load.model,
            kind=args.artifact.kind,
        )
    else:
        state = prepare_serving_reusing(
            load,
            opts,
            args.tokenizer_from,
            repo=hf_repo_dir(args.load.model),
            inputs_spec=args.load.inputs,
            # One export per process: without the disk tier, a key (a hash of every weight)
            # could never be read back.
            cache=bool(opts.export_cache_dir),
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

    from downshift.core.feeds import example_feeds

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
