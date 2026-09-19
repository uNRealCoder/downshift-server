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

if TYPE_CHECKING:
    import torch
    from fastapi import FastAPI

__all__ = ["app_for"]


def app_for(
    model: torch.nn.Module | str | Path,
    example_inputs: tuple | None = None,
    *,
    source: str = "model",
    reference: torch.nn.Module | None = None,
    middleware: Sequence[str] = (),
    **options: Any,
) -> FastAPI:
    """`LoadedModel` + `prepare_serving()` + `build_app(state=...)`, in one call.

    Runs synchronously: the export-and-verify gate and warmup happen before this
    returns, so the app is ready to serve immediately (no loader thread, no `/ready`
    503 window - see `build_app`'s `loader=` form for that instead). `options` are
    `ServeOptions` fields (`backend=`, `warmup=`, `max_concurrency=`, ...). `model` is
    usually a `torch.nn.Module`; a `str`/`Path` is a pre-built `.onnx` file, which
    `reference` (a PyTorch model) verifies against, same as `--reference` on the CLI -
    without it a `.onnx` `model` is served UNVERIFIED.
    """
    from downshift.loading import LoadedModel
    from downshift.serve.app import build_app
    from downshift.serve.engine import ServeOptions, prepare_serving

    if isinstance(model, (str, Path)):
        loaded = LoadedModel(source=source, onnx_path=Path(model), example_inputs=example_inputs)
    else:
        loaded = LoadedModel(source=source, model=model, example_inputs=example_inputs)
    ref = (
        LoadedModel(source="reference", model=reference, example_inputs=example_inputs)
        if reference is not None
        else None
    )
    state = prepare_serving(loaded, ServeOptions(**options), ref)
    return build_app(state=state, middleware=tuple(middleware))
