# downshift code docs

Downshift serves a model that you already downloaded, over HTTP. Before it answers a request, it exports the model to ONNX. It then compares the ONNX graph with PyTorch on inputs that the exporter did not see. If the ONNX graph gives wrong answers, downshift serves eager PyTorch instead. The check gives a verdict: CLEAN, DEGRADED, FAILED or UNVERIFIED. The verdict selects the backend behind `/predict`. This is true for the CLI and for the library.

Downshift accepts these forms. All of them must be on the machine that runs downshift:

- A `.onnx` file.
- A PyTorch checkpoint.
- A Hugging Face repo directory. Downshift identifies it by the `config.json` in it.
- An import spec that names a model that the process can already import.

Downshift fetches nothing. It rejects a hub id and does not download it.

This directory is reference material. It describes every public name, flag, route and field. Each description comes from the source and agrees with the source. It is not a tutorial.

- For a walkthrough, start with the repo [`README.md`](../../README.md).
- To run `downshift serve` in a real deployment (probes, sizing, Kubernetes, thread budgets), refer to [`docs/production.md`](../production.md).
- To see which model families and export hazards work, refer to [`docs/compatibility.md`](../compatibility.md).

## Pages

Each page starts with one sentence that says what it covers. Use it to go to the section that you need.

| Page | Covers |
|---|---|
| [`python-api.md`](python-api.md) | Every name in `downshift.__all__`: `check`, `export`, `intake`, `build_verdict`, `prepare_model`, `app_for`, the types `ExportVerdict`, `NumericsReport`, `Adapter`, `Prepared` and `OnnxRuntimeError`, and `__version__`. It also describes the lazy-import behaviour of `import downshift`. |
| [`cli.md`](cli.md) | `downshift check`, `downshift export` and `downshift serve`. For each flag: the environment variable and the default, if it has them. Also logging, the table of environment variables, and the process exit codes. |
| [`http-api.md`](http-api.md) | Every HTTP route, including `GET /schema` (what to POST, with an example body). The request and response schemas, the status codes, the nested-list and base64 wire formats, the headers, and `curl` examples. |
| [`serving.md`](serving.md) | The programmatic serving layer under `serve`: `ServeOptions`, `ServingState`, `prepare_serving`, `build_app`, the selection of the backend, and the `Backend` implementations. |
| [`adapters.md`](adapters.md) | The `Adapter` protocol, how downshift registers and resolves adapters, and how to write a custom adapter. |

## Module map

All modules are in `src/downshift/`. Some leaf modules use only the standard library. They make sure that `downshift --help` and `import downshift` never import torch, onnxruntime, fastapi or uvicorn.

| Module | What it contains |
|---|---|
| `__init__.py` | The lazy (PEP 562) public API, `export()` and `__version__`. |
| `settings.py` | The defaults and the `DOWNSHIFT_*` environment variables (including `DOWNSHIFT_SERVER_API_KEY`), for the CLI and for library use. |
| `logs.py` | The one logging sink: `setup_logging`, the plain-text formatter and `request_id_var`. |
| `_imports.py` | `import_object`, `is_import_spec` and `LoadError` (standard library only). `loading.py` re-exports them. |
| `sources.py` | The names of the source kinds (`onnx-file`, `hf-repo-dir`, and others) and their descriptions (standard library only). |
| `loading.py` | Turns a `MODEL` argument into a `LoadedModel`: `LoadSpec` and `load_model`. |
| `hf_repo.py` | Reads a downloaded Hugging Face repo: `load_config`, `load_pretrained` (with its task head), the pooling recipe, and the tokenizer and labels for text input. Downshift imports it only if transformers is installed. |
| `cli/main.py` | The three commands: `check`, `export` and `serve`. |
| `cli/options.py` | The typer option types (`Annotated` aliases and `LogLevel`). |
| `cli/runtime.py` | `ServeArgs`, `_collect_serve_args` and the `--workers` handoff. |
| `cli/render.py` | The banner, the reports, the warnings and the errors. All are logged as plain text. |
| `core/` | The gate. `verdict.py` has `check`, `prepare_model` and `build_verdict`. The other files are `capture.py`, `verify.py`, `prevalidated.py` (`intake`), `manifest.py`, `shapes.py`, `inputs.py` and `phase.py`. `phase.py` has the `Phase` enum (`load`, `export`, `verify`, `session`, `warmup`). The enum drives the `Boot` timings, `/ready` and the banner. |
| `adapters/` | `base.py` has the `Adapter` protocol, `Prepared` and `Family`. The other files are `registry.py`, `generic.py`, `pyg.py` and `hf.py`. The Hugging Face parts are `text.py` (tokenizing and class probabilities), `pooling.py` (the `--pooling` choices) and `embedding.py` (the pooling recipe and the graph that applies it). |
| `serve/options.py` | `ServeOptions` and `BackendChoice`. |
| `serve/engine.py` | `ServingState`, `prepare_serving` and the warmup. |
| `serve/app.py` | `build_app`, the routing, the request-id and API-key middleware, and `PredictRoute` (admission, body limits and content types for the predict routes). |
| `serve/predict.py` | The request path of `/predict` and `/predict/graph`. It has two thread hops. First, the prep pool parses and validates the body, and converts and checks it. Then the inference pool runs the inference and encodes the result. |
| `serve/describe.py` | The answer to `GET /schema`. |
| `serve/backends.py`, `serve/schemas.py`, `serve/codec.py`, `serve/middleware.py` | The two backends, the request and response models, the base64 and safetensors codecs, and the loading of `--middleware`. |
