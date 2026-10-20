"""The runtime of `serve`: the ServingState builder, TLS, logging, and `--workers N`.

For `--workers N`, the parent exports and verifies one time, then sends ServeArgs (with an
ArtifactHandoff of its export) as JSON to the uvicorn workers in an environment variable.

No torch at module scope: `--help` must stay fast.
"""

from __future__ import annotations

import atexit
import gc
import json
import os
import shutil
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

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
    """The one logging sink, on stdout. `stderr=True` keeps stdout clean for `--json`."""
    setup_logging(level.value, stream=sys.stderr if stderr else None)


def _tls_config(certfile: str | None, keyfile: str | None, password: str | None) -> dict[str, Any]:
    """The uvicorn keyword arguments for HTTPS, or {} for plain HTTP.

    The files load here first, so a bad file is a usage error before the model loads. The
    password callable stops OpenSSL from prompting on the terminal. ssl_ciphers=None keeps the
    stdlib cipher list (uvicorn's default also enables NULL and anonymous ciphers).
    """
    if certfile is None:
        if keyfile is not None:
            raise ValueError("--ssl-keyfile needs --ssl-certfile. Give the certificate too")
        return {}
    import ssl

    context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    try:
        context.load_cert_chain(certfile, keyfile, lambda: password or "")
    except ssl.SSLError as exc:
        raise ValueError(
            f"cannot load the TLS certificate {certfile} ({exc.reason or 'bad key or password'}). "
            "Use PEM files. Use the key of this certificate. For an encrypted key, set "
            "DOWNSHIFT_SSL_KEYFILE_PASSWORD"
        ) from None
    except OSError as exc:
        files = " and ".join(filter(None, (certfile, keyfile)))
        raise ValueError(f"cannot read {files}: {exc.strerror}") from None
    return {
        "ssl_certfile": certfile,
        "ssl_keyfile": keyfile,
        "ssl_keyfile_password": password,
        "ssl_ciphers": None,
    }


def _load(spec: LoadSpec) -> LoadedModel:
    from downshift.loading import load_model

    if spec.unsafe_load:
        render.warn(
            f"--unsafe-load: torch.load(weights_only=False) on {spec.model}. Arbitrary code can run"
        )
    return load_model(spec)


@dataclass
class ArtifactHandoff:
    """The parent's export for `--workers N`. A worker boots from it like from a cache hit. A
    torch worker loads the model itself: weights are not sent between processes."""

    verdict: dict
    input_names: list[str]
    kind: str  # the parent's source_kind: an ONNX worker loads nothing to find it
    axis_bounds: dict[str, list[list]] = field(default_factory=dict)
    onnx_path: str | None = None  # the graph, if the parent serves it on onnxruntime
    feeds_path: str | None = None  # example inputs for the warmup, if any

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
    """Everything needed to build the ServingState and app again, as JSON for the workers."""

    load: LoadSpec
    options: ServeOptions
    reference: str | None
    middleware: list[str] | None
    log_level: str
    artifact: ArtifactHandoff | None = None
    access_log: bool = True
    # The validated --tokenizer-from directory.
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
    from downshift.serve.engine import state_from_entry
    from downshift.serve.reuse import prepare_serving_reusing

    opts = args.options
    load_s = 0.0

    def load() -> tuple[LoadedModel, LoadedModel | None]:
        nonlocal load_s
        load_start = time.perf_counter()
        loaded = _load(args.load)
        # --reference shares --pooling and --normalize with MODEL.
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
            # The key hashes every weight: worth it only for the disk tier.
            cache=bool(opts.export_cache_dir),
        )
    state.timings[Phase.load] = load_s
    return state


def _serve_app_factory() -> FastAPI:
    """The app factory that each uvicorn worker process calls. Like the single-process path,
    the app loads in the background, so /health answers at once.

    A worker that cannot load exits with uvicorn's STARTUP_FAILURE code, which stops the whole
    server instead of a restart into the same failure.
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
    """(onnx_path, feeds_path, temp_dir) of the parent's export for the workers. temp_dir is
    None if nothing was written."""
    import tempfile

    import numpy as np

    from downshift.core.feeds import example_feeds

    verdict = state.verdict
    if verdict._tmpdir is not None:
        # The verdict's temporary directory holds external data. Keep it alive until exit.
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


def _run_workers(
    args: ServeArgs,
    workers: int,
    uvicorn_kwargs: dict[str, Any],
    announce: Callable[[ServingState, int], None],
) -> None:
    """Export and verify once in this process, then start `workers` uvicorn workers that boot
    from that export. This copy never serves: it skips the warmup and keeps the default
    threads, so it reuses the session of verify."""
    import uvicorn

    from downshift.core.axes import axis_bounds_to_json

    parent_options = replace(args.options, warmup=0, intra_op_threads=0, inter_op_threads=0)
    state = _build_serving_state(replace(args, options=parent_options))
    state.options = args.options  # the banner reports what the workers use
    announce(state, workers)
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
    # The workers take the verdict, so they never need --reference.
    os.environ[_SERVE_ARGS_ENV] = replace(args, artifact=artifact, reference=None).to_json()
    try:
        uvicorn.run(
            "downshift.cli.runtime:_serve_app_factory",
            workers=workers,
            factory=True,
            **uvicorn_kwargs,
        )
    finally:
        if temp_dir is not None:
            shutil.rmtree(temp_dir, ignore_errors=True)
