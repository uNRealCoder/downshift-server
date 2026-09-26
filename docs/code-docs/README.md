# downshift code docs

downshift serves a model you have already downloaded, over HTTP. Before it answers a single
request, it exports the model to ONNX, runs a numerics check against PyTorch on inputs the
exporter never saw, and serves eager PyTorch instead whenever the ONNX graph would give
wrong answers. The verdict from that check - CLEAN, DEGRADED, FAILED or UNVERIFIED - is what
picks the backend behind `/predict`, both from the CLI and from the library.

What it accepts is three kinds of artifact already on the machine it runs on - a `.onnx`
file, a PyTorch checkpoint, or a Hugging Face repo directory, recognised by the
`config.json` in it - plus an import spec naming a model already importable in the process.
Nothing is fetched; a hub id is rejected, not downloaded.

This directory is reference material: every public name, flag, route and field, read out of
the source and kept accurate to it. It is not a tutorial. For a walkthrough, start with the
repo [`README.md`](../../README.md). For running `downshift serve` in a real deployment
(probes, sizing, Kubernetes, thread budgeting), see [`docs/production.md`](../production.md).
For which model families and export hazards are known to work, see
[`docs/compatibility.md`](../compatibility.md).

## Pages

| Page | Covers |
|---|---|
| [`python-api.md`](python-api.md) | Every name in `downshift.__all__`: `check`, `export`, `intake`, `build_verdict`, `prepare_model`, `app_for`, the `ExportVerdict`/`NumericsReport`/`Adapter`/`Prepared`/`OnnxRuntimeError` types, `__version__`, and the lazy-import behaviour of `import downshift` itself. |
| [`cli.md`](cli.md) | `downshift check`, `downshift export`, `downshift serve`: every flag, its environment variable and default where it has one, logging, the full environment-variable table, and the process exit codes. |
| [`http-api.md`](http-api.md) | Every HTTP route, including `GET /schema` (what to POST, with an example body): request/response schemas, status codes, the nested-list and base64 wire formats, headers, and `curl` examples. |
| [`serving.md`](serving.md) | The programmatic serving layer beneath `serve`: `ServeOptions`, `ServingState`, `prepare_serving`, `build_app`, backend selection, and the `Backend` implementations. |
| [`adapters.md`](adapters.md) | The `Adapter` protocol, how adapters are registered and resolved, and how to write a custom one. |

Each page opens with one sentence saying what it covers, so you can jump straight to the
section you need rather than reading front to back.

## Module map

Everything is under `src/downshift/`. The stdlib-only leaf modules exist so that
`downshift --help` and `import downshift` never import torch, onnxruntime, fastapi or
uvicorn.

| Module | What lives there |
|---|---|
| `__init__.py` | The lazy (PEP 562) public API, `export()`, `__version__`. |
| `settings.py` | Defaults and the `DOWNSHIFT_*` environment variables (including `DOWNSHIFT_SERVER_API_KEY`), for the CLI and for library use. |
| `logs.py` | The one logging sink: `setup_logging`, the plain-text formatter, `request_id_var`. |
| `_imports.py` | `import_object`, `is_import_spec`, `LoadError` (stdlib only). `loading.py` re-exports them. |
| `sources.py` | The source-kind names (`onnx-file`, `hf-repo-dir`, ...) and their descriptions (stdlib only). |
| `loading.py` | Turns a `MODEL` argument into a `LoadedModel`: `LoadSpec`, `load_model`. |
| `hf_repo.py` | Reads a downloaded Hugging Face repo: `load_config`, `load_pretrained` (with its task head), the pooling recipe, and the tokenizer/labels for text input. Only imported when transformers is installed. |
| `cli/main.py` | The three commands (`check`, `export`, `serve`). |
| `cli/options.py` | The typer option types (`Annotated` aliases, `LogLevel`). |
| `cli/runtime.py` | `ServeArgs`, `_collect_serve_args`, and the `--workers` handoff. |
| `cli/render.py` | The banner, reports, warnings and errors, all logged as plain text. |
| `core/` | The gate: `verdict.py` (`check`, `prepare_model`, `build_verdict`), `capture.py`, `verify.py`, `prevalidated.py` (`intake`), `manifest.py`, `shapes.py`, `inputs.py`, and `phase.py` (the `Phase` enum: `load`, `export`, `verify`, `session`, `warmup`; drives the `Boot` timings, `/ready` and the banner). |
| `adapters/` | `base.py` (the `Adapter` protocol, `Prepared`, `Family`), `registry.py`, `generic.py`, `pyg.py`, `hf.py`, and the Hugging Face pieces: `text.py` (tokenizing and class probabilities), `pooling.py` (the `--pooling` choices), `embedding.py` (the pooling recipe and the graph that applies it). |
| `serve/options.py` | `ServeOptions`, `BackendChoice`. |
| `serve/engine.py` | `ServingState`, `prepare_serving`, warmup. |
| `serve/app.py` | `build_app`, routing, the request-id and API-key middleware. |
| `serve/predict.py` | The `/predict` and `/predict/graph` internals: conversion, bounds checks, the inference call. |
| `serve/describe.py` | What `GET /schema` answers. |
| `serve/backends.py`, `serve/schemas.py`, `serve/codec.py`, `serve/middleware.py` | The two backends, the request/response models, the base64 codec, and `--middleware` loading. |
