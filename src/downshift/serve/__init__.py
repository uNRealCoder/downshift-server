"""The serving package. `app_for` is the one-line path from a model to a FastAPI app,
for a team that already has a service and wants downshift as a mounted component rather
than a whole process (`app.mount("/model", app_for(model))`) - see the README's Serve
section. Everything it calls is imported inside the function body, so importing this
package stays as cheap as importing `downshift.serve.options` already is.
"""

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
    **options: Any,
) -> FastAPI:
    """`LoadedModel` + `prepare_serving()` + `build_app(state=...)`, in one call.

    Runs synchronously: the export-and-verify gate and warmup happen before this
    returns, so the app is ready to serve immediately (no loader thread, no `/ready`
    503 window - see `build_app`'s `loader=` form for that instead). `options` are
    `ServeOptions` fields (`backend=`, `warmup=`, `max_concurrency=`, ...). `model` is
    usually a `torch.nn.Module` this process already built; a `str`/`Path` is an `.onnx`
    file already on this machine, which `reference` (a PyTorch model) verifies against,
    same as `--reference` on the CLI - without it a `.onnx` `model` is served UNVERIFIED.
    Nothing is downloaded here either: an `.onnx` path has to exist before this is called.

    `tokenizer_from` is a downloaded Hugging Face repo directory to load the tokenizer,
    pooling recipe and label metadata from, same as `--tokenizer-from` on the CLI - for a
    `.onnx`/checkpoint `model` that has none of its own. Independent of `reference`: it
    never affects verification, and the two may name the same directory or different ones.

    `source` is the label `/metadata` and `/schema` report; it defaults to the `.onnx`
    path (reported as its file name only), or to `"model"` for an `nn.Module`, which has no
    path to name.

    `api_key` defaults to DOWNSHIFT_SERVER_API_KEY (settings.API_KEY), same as the CLI (U7,
    ruling 5); pass a value, or None to force it off, to override that for this app.
    """
    from downshift.loading import (
        IN_PROCESS_MODULE,
        ONNX_FILE,
        LoadedModel,
        resolve_tokenizer_source,
    )
    from downshift.serve.app import build_app
    from downshift.serve.engine import ServeOptions, prepare_serving

    if isinstance(model, (str, Path)):
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
    state = prepare_serving(
        loaded, ServeOptions(**options), ref, tokenizer_from=resolved_tokenizer_from
    )
    return build_app(state=state, middleware=tuple(middleware), api_key=api_key)
