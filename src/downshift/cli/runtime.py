"""The code that sends a `serve` call through `--workers N`, and the logging setup that every
command uses. ServeArgs and ArtifactHandoff are the JSON envelope that a parent process sends
to its uvicorn worker processes. The builders below turn that envelope (or a new MODEL
argument) into a ServingState and a FastAPI app.

The function bodies do the heavy imports (torch, onnxruntime, downshift.core,
downshift.serve.engine and app, and downshift.loading) when they need them. This is the same
rule as in `downshift.cli.main`. main.py imports this module at load time to re-export these
names. This module must stay free of torch at module scope, so that `--help` and `--version`
stay fast.
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
    """The one logging sink, on stdout. `stderr=True` is for `--json`. It keeps stdout as one
    JSON document that a program can parse."""
    setup_logging(level.value, stream=sys.stderr if stderr else None)


def _load(spec: LoadSpec) -> LoadedModel:
    from downshift.loading import load_model

    if spec.unsafe_load:
        render.warn(
            f"--unsafe-load: torch.load(weights_only=False) on {spec.model}. Arbitrary code can run"
        )
    return load_model(spec)


@dataclass
class ArtifactHandoff:
    """Set only for `--workers N`: the export of the parent. Workers do not capture or verify
    again. A worker boots from it in the same way as from an export-cache hit (serve.reuse).
    The worker selects the backend from the verdict, as the parent did. A torch worker loads
    the model itself, because downshift does not send torch weights between processes."""

    verdict: dict
    input_names: list[str]
    kind: str  # ServingState.source_kind of the parent. An ONNX worker loads nothing to find it
    # [[axis, name, min, max], ...] for each input (core.axes). An ONNX worker has no Prepared.
    axis_bounds: dict[str, list[list]] = field(default_factory=dict)
    onnx_path: str | None = None  # the graph, if the parent serves it on onnxruntime
    feeds_path: str | None = None  # real example inputs for the warmup, if there are any

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
    """Everything that downshift needs to build a ServingState and a FastAPI app again. For a
    `--workers N` run, it goes through JSON (dataclasses.asdict, json.dumps and json.loads) and
    back. The parent sends it to each worker process in an environment variable. from_json and
    to_json do the enum conversion that this needs. JSON has no enum type, so BackendChoice
    and OutputEncoding come back as plain str."""

    load: LoadSpec
    options: ServeOptions
    reference: str | None
    middleware: list[str] | None
    log_level: str
    artifact: ArtifactHandoff | None = None
    access_log: bool = True
    # A Hugging Face repo directory that loading.resolve_tokenizer_source validated, or None.
    # It does not depend on `reference`. This is the whole job of --tokenizer-from.
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
        # --reference shares --pooling and --normalize with MODEL. See the comment in check_cmd.
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
            # One export for each process. Without the disk tier, nothing could read a key
            # (a hash of every weight) again.
            cache=bool(opts.export_cache_dir),
        )
    state.timings[Phase.load] = load_s
    return state


def _serve_app_factory() -> FastAPI:
    """The import-string target for the multi-worker mode of uvicorn
    (`downshift.cli.runtime:_serve_app_factory`). Each worker process calls it on its own,
    synchronously. The call happens before the Server of uvicorn for that worker starts to
    accept connections on the socket that the parent already bound (see
    uvicorn._subprocess.subprocess_started).

    As in the single-worker path, this function builds the app with a loader and not with a
    ready-made state. Otherwise, a worker would accept no connections, not even /health, for
    the whole load, export, verification and warmup. It would not answer a fast /ready 503.

    If no artifact is set, a worker reloads, exports and warms up the model on its own, as in
    the single-worker path. If an artifact is set, the worker loads the export that the parent
    already verified (see ServeArgs).

    A failed load here is different from the single-worker path. There is no uvicorn.Server on
    which to set should_exit. Uvicorn builds that Server itself, after this function returns.
    The loader below therefore exits the process directly. A worker that cannot load must not
    stay up and serve 503 forever without a reason. It exits with the STARTUP_FAILURE code of
    uvicorn. This code tells the worker supervisor of uvicorn to stop the whole server. The
    supervisor does not start the worker again into the same failure and the same reload.
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
    """Return (onnx_path, feeds_path, temp_dir). A `--workers N` parent uses them to give its
    verified export to the workers. temp_dir is the directory to clean up afterward. It is None
    if nothing was written (a .onnx file that is already on disk, with no example inputs to
    save)."""
    import tempfile

    import numpy as np

    from downshift.core.feeds import example_feeds

    verdict = state.verdict
    if verdict._tmpdir is not None:
        # An export with external data is in the own temporary directory of the verdict. The
        # data file is next to the .onnx file. The parent drops its state before the workers
        # load. That would delete the directory. It must therefore outlive the verdict (its
        # finalizer then runs at exit).
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
        np.savez(feeds_path, **feeds)  # type: ignore[arg-type]  # the numpy stub misreads **kwds
    return onnx_path, feeds_path, temp_dir
