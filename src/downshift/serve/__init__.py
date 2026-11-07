"""The serving package. `app_for` imports its dependencies lazily, so importing this package
is cheap."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from downshift import settings

if TYPE_CHECKING:
    import torch
    from fastapi import FastAPI

__all__ = ["app_for"]


def app_for(
    model: torch.nn.Module | str | Path,
    example_inputs: tuple | None = None,
    *,
    source: str | None = None,
    reference: torch.nn.Module | None = None,
    tokenizer_from: str | Path | None = None,
    middleware: Sequence[str] = (),
    api_key: str | None = settings.API_KEY,
    cache: bool = True,
    **options: Any,
) -> FastAPI:
    """A ready FastAPI app for `model`, to mount in your own service:
    `app.mount("/model", app_for(model))`.

    It runs the gate and the warmup before it returns, so there is no /ready 503 window.
    `model` is an `nn.Module`, a `.onnx` file on this machine (verified against `reference`,
    else served UNVERIFIED), or a downloaded HF repo directory. `tokenizer_from` is an HF repo
    directory for the tokenizer, pooling and labels (as --tokenizer-from). `source` is the
    label of /metadata and /schema. `options` are ServeOptions fields.

    `api_key` defaults to DOWNSHIFT_SERVER_API_KEY; None turns it off. An earlier export of
    the same model and options is reused from the in-process memo and from
    `export_cache_dir`; `cache=False` skips both.
    """
    from downshift.loading import (
        LoadedModel,
        LoadSpec,
        hf_repo_dir,
        load_model,
        resolve_tokenizer_source,
    )
    from downshift.serve.app import build_app
    from downshift.serve.engine import ServeOptions
    from downshift.serve.reuse import prepare_serving_reusing
    from downshift.sources import IN_PROCESS_MODULE, ONNX_FILE

    opts = ServeOptions(**options)
    repo = hf_repo_dir(str(model)) if isinstance(model, (str, Path)) else None
    loaded: LoadedModel | None = None
    if isinstance(model, (str, Path)):
        if repo is None:
            loaded = LoadedModel(
                source=source or str(model),
                onnx_path=Path(model),
                example_inputs=example_inputs,
                kind=ONNX_FILE,
            )
    else:
        loaded = LoadedModel(
            source=source or "model",
            model=model,
            example_inputs=example_inputs,
            kind=IN_PROCESS_MODULE,
        )
    ref = (
        LoadedModel(source="reference", model=reference, example_inputs=example_inputs)
        if reference is not None
        else None
    )
    resolved_tokenizer_from = (
        resolve_tokenizer_source(str(tokenizer_from)) if tokenizer_from is not None else None
    )

    def load() -> tuple[LoadedModel, LoadedModel | None]:
        if loaded is not None:
            return loaded, ref
        return load_model(
            LoadSpec(str(model), None, None, False, opts.pooling, opts.normalize)
        ), ref

    state = prepare_serving_reusing(load, opts, resolved_tokenizer_from, repo=repo, cache=cache)
    return build_app(state=state, middleware=tuple(middleware), api_key=api_key)
