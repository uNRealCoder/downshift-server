"""The runtime of `serve`: the ServingState builder, TLS, logging, and `--workers N`.

For `--workers N`, the parent exports and verifies one time, then sends ServeArgs (with an
ArtifactHandoff of its export) as JSON to the uvicorn workers in an environment variable.

No torch at module scope: `--help` must stay fast.
"""

from __future__ import annotations

import atexit
import gc
import os
import shutil
import sys
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from downshift.cli import render
from downshift.cli.options import LogLevel
from downshift.entities.cli import ArtifactHandoff, ServeArgs
from downshift.entities.phase import Phase
from downshift.logs import setup_logging

if TYPE_CHECKING:
    from fastapi import FastAPI

    from downshift.entities.loading import LoadedModel, LoadSpec
    from downshift.entities.serving import ServingState

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


def release_graph(state: ServingState) -> None:
    """The backend has its own copy of the graph. Drop the bytes of the graph and the export
    memo. A process that serves calls this. The parent of the workers does not: it needs the
    bytes to write the file for the workers."""
    from downshift.core.memo import MEMO

    state.verdict.onnx_bytes = b""
    state.verdict.sample_feeds = []
    MEMO.clear()


def _build_serving_state(args: ServeArgs) -> ServingState:
    """Boot the state of `args`."""
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
    import uvicorn.config

    from downshift.serve.app import build_app

    # uvicorn's exit code for a failed start. Older uvicorn stubs do not declare the name, so read it
    # with getattr. 3 is its value in every release.
    startup_failure: int = getattr(uvicorn.config, "STARTUP_FAILURE", 3)
    args = ServeArgs.from_json(os.environ[_SERVE_ARGS_ENV])
    _setup_logging(LogLevel(args.log_level))

    def loader() -> ServingState:
        try:
            state = _build_serving_state(args)
        except Exception as exc:
            if LogLevel(args.log_level) is LogLevel.debug:
                render.print_traceback()
            render.error(f"model failed to load in this worker: {type(exc).__name__}: {exc}")
            os._exit(startup_failure)
        release_graph(state)
        render.print_ready(state)
        return state

    return build_app(
        loader=loader,
        middleware=tuple(args.middleware or ()),
        access_log=args.access_log,
        metrics=args.options.metrics,
    )


_HANDOFF_KEEPALIVE: list[object] = []


def _write_onnx_artifact(state: ServingState) -> tuple[Path, Path | None, Path | None]:
    """(onnx_path, feeds_path, temp_dir) of the parent's export for the workers. temp_dir is
    None if nothing was written."""
    import tempfile

    import numpy as np

    from downshift.core.export_cache import pack_feeds
    from downshift.core.feeds import example_feeds

    verdict = state.verdict
    if verdict._tmpdir is not None:
        # The verdict's temporary directory holds external data. Keep it alive until exit.
        _HANDOFF_KEEPALIVE.append(verdict._tmpdir)
    needs_copy = verdict.onnx_path is None
    needs_feeds = state.example_inputs is not None or bool(verdict.sample_feeds)
    if not needs_copy and not needs_feeds:
        assert verdict.onnx_path is not None
        return verdict.onnx_path, None, None

    temp_dir = Path(tempfile.mkdtemp(prefix="downshift-"))
    onnx_path = verdict.onnx_path if verdict.onnx_path is not None else temp_dir / "model.onnx"
    if needs_copy:
        onnx_path.write_bytes(verdict.onnx_bytes)
    feeds_path = None
    if needs_feeds:
        feeds_path = temp_dir / "feeds.npz"
        example = (
            example_feeds(state.input_names, state.example_inputs)
            if state.example_inputs is not None
            else None
        )
        np.savez(feeds_path, **pack_feeds(example, verdict.sample_feeds))  # type: ignore[arg-type]  # the numpy stub misreads **kwds
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
