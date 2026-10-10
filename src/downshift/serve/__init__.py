"""The serving package. `app_for` is the one-line path from a model to a FastAPI app. It is for
a team that already has a service and wants downshift as a mounted component and not as a whole
process (`app.mount("/model", app_for(model))`). Refer to the Serve section of the README.
`app_for` imports everything that it calls inside the function body. An import of this package
therefore costs only as much as an import of `downshift.serve.options`.
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
    cache: bool = True,
    **options: Any,
) -> FastAPI:
    """`LoadedModel`, `prepare_serving()` and `build_app(state=...)`, in one call.

    It runs synchronously. The export-and-verify gate and the warmup finish before it returns.
    The app is therefore ready to serve immediately. There is no loader thread and no `/ready`
    503 window. For that, see the `loader=` form of `build_app`. `options` are `ServeOptions`
    fields (`backend=`, `warmup=`, `max_concurrency=`, ...). `model` is usually a
    `torch.nn.Module` that this process already built. A `str` or `Path` is an `.onnx` file
    that is already on this machine. `reference` (a PyTorch model) verifies it. This is the
    same as `--reference` on the CLI. Without `reference`, downshift serves an `.onnx` `model`
    as UNVERIFIED. Downshift downloads nothing here. An `.onnx` path must exist before the call.

    `tokenizer_from` is a downloaded Hugging Face repo directory. Downshift loads the tokenizer,
    the pooling recipe and the label metadata from it. This is the same as `--tokenizer-from` on
    the CLI. Use it for an `.onnx` or checkpoint `model` that has none of its own. It does not
    depend on `reference`. It never changes the verification. Both can name the same directory
    or different directories.

    `source` is the label that `/metadata` and `/schema` report. The default is the `.onnx`
    path (reported as the file name only). For an `nn.Module`, the default is `"model"`,
    because it has no path to name.

    `api_key` defaults to DOWNSHIFT_SERVER_API_KEY (settings.API_KEY), the same as the CLI (U7,
    ruling 5). To override that for this app, pass a value, or pass None to turn it off.

    If this process already exported and verified a model (the same weights, code and options),
    downshift serves it from the in-process memo. There is no export phase and no verification
    phase. `model` can also be a downloaded Hugging Face repo directory. Downshift then looks it
    up before it loads anything. `export_cache_dir=` (a `ServeOptions` field. The default is
    DOWNSHIFT_EXPORT_CACHE_DIR) adds the disk tier that survives restarts. `cache=False` skips
    both, for reads and for writes.
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
