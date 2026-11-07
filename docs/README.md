# downshift docs

Downshift serves a model that you already downloaded, over HTTP. Before it answers a request, it exports the model to ONNX and compares the ONNX graph with PyTorch on inputs that the exporter did not see. The check gives a verdict: CLEAN, DEGRADED, FAILED or UNVERIFIED. The verdict selects the backend behind `/predict`.

For a first walkthrough, read the repo [`README.md`](../README.md). The pages below are reference material. Each one describes the code as it is.

| Page | Read it to |
|---|---|
| [`cli.md`](cli.md) | Use `downshift check`, `export` and `serve`: every flag, environment variable, log level and exit code. |
| [`http-api.md`](http-api.md) | Call a server: every route, body, status code, header and wire format (JSON, base64, safetensors, text, graphs). |
| [`production.md`](production.md) | Deploy a server: authentication, HTTPS, probes, workers, threads, the result cache, sizing, Kubernetes and GPUs. |
| [`python-api.md`](python-api.md) | Use downshift as a library: `check`, `export`, `intake`, `app_for` and the verdict types. |
| [`serving.md`](serving.md) | Change the serving layer: `ServeOptions`, `ServingState`, the boot builders, the request path and the backends. |
| [`adapters.md`](adapters.md) | Add a model family with your own adapter. |
| [`compatibility.md`](compatibility.md) | See which export hazards downshift catches. A script generates this page. |

## Module map

All modules are in `src/downshift/`. The modules that the CLI imports at start do not import torch, onnxruntime, fastapi or uvicorn. `downshift --help` and `import downshift` therefore stay fast.

| Module | Contents |
|---|---|
| `__init__.py` | The lazy public API, `export()` and `__version__`. |
| `settings.py` | The defaults and the `DOWNSHIFT_*` environment variables. |
| `logs.py` | The one logging sink, the plain-text formatter and `request_id_var`. |
| `_imports.py`, `sources.py` | Import specs, `LoadError`, and the names of the source kinds. Standard library only. |
| `loading.py` | Turns a `MODEL` argument into a `LoadedModel`. |
| `hf_repo.py` | Reads a downloaded Hugging Face repo: the model with its task head, the pooling recipe, the tokenizer and the labels. Needs `transformers`. |
| `cli/main.py` | The commands `check`, `export` and `serve`. |
| `cli/options.py` | The typer option types. |
| `cli/runtime.py` | `ServeArgs`, the `--workers` handoff (`ArtifactHandoff`, `_run_workers` and the worker app factory), the TLS settings and the logging setup. |
| `cli/render.py` | The banner and the reports, as plain text. |
| `core/verdict.py` | `ExportVerdict`, `prepare_model`, `build_verdict`, `check`, and `intake()` for a `.onnx` file that someone else made. |
| `core/capture.py`, `core/verify.py` | `torch.export` plus the ONNX translation, and the numerics check. |
| `core/rows.py` | The row check: do the rows of a batch change each other? |
| `core/shapes.py`, `core/axes.py` | Dynamic shapes, `--dynamic`, `--axis-max`, and the axis facts of the verdict. |
| `core/memo.py`, `core/export_cache.py` | The export memo of the process and the `--export-cache-dir` disk tier. |
| `core/manifest.py`, `core/feeds.py`, `core/phase.py` | The manifest of `export`, example feeds, and the boot phases. |
| `adapters/` | The `Adapter` protocol (`base.py`), the registry, and the `generic`, `pyg` and `hf` adapters. `embedding.py`, `pooling.py` and `text.py` are the Hugging Face parts. |
| `serve/__init__.py` | `app_for`. |
| `serve/options.py` | `ServeOptions`, `BackendChoice` and `ExecutionChoice`. |
| `serve/engine.py`, `serve/reuse.py` | `ServingState`, the two boot builders, the backend selection and the warmup. `reuse.py` looks up the export cache. |
| `serve/app.py` | `build_app`, the routes, the request-ID and API-key middleware, and the loading of `--middleware`. |
| `serve/predict.py`, `serve/result_cache.py` | The request path of `/predict` and `/predict/graph`, and the result cache. |
| `serve/describe.py`, `serve/schemas.py` | `GET /schema`, and the request and response models. |
| `serve/backends.py`, `serve/codec.py`, `serve/graphs.py` | The two backends, the base64 and safetensors codecs, and graph batching. |
