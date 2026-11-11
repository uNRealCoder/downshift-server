# Architecture

This page shows how downshift works as a whole, and in which order to read the code. For the details of one part, follow the links to the reference pages. To run the code before you read it, start with [`walkthrough.md`](walkthrough.md).

## The one idea

A model that exports to ONNX can still give wrong numbers. Downshift does not trust the export. It compares the ONNX graph with PyTorch, on inputs that the exporter did not see. The result is one object, the **verdict**. All other code reads the verdict.

| Status | Meaning | Backend |
|---|---|---|
| `CLEAN` | The graph exports. The numbers match. The graph works on new shapes. | ONNX Runtime |
| `DEGRADED` | The graph exports. The numbers differ more than the tolerance. | torch (eager) |
| `FAILED` | The graph does not export, or ONNX Runtime cannot run it. | torch (eager) |
| `UNVERIFIED` | A `.onnx` file with no reference model, or `--no-verify`. | ONNX Runtime |

The status gives the exit code of `downshift check`: `0` CLEAN, `1` FAILED, `2` DEGRADED, `3` UNVERIFIED. A usage error is exit code `4`.

## The flow

```
MODEL argument
   |
   v
loading.py            Find the model. Local files only. Never the network.
   |
   v
adapters/             Pick the model family. Turn the model into a module that takes flat tensors.
   |                  Give example inputs, dynamic shapes and a sampler for verification.
   v
core/capture.py       torch.export (strict=False, then strict=True), then the ONNX translation.
   |
   v
core/verify.py        Run N samples through torch and through ONNX Runtime. Vary the dynamic axes.
core/rows.py          Check: does a row give the same output alone and in a batch?
   |
   v
core/verdict.py       ExportVerdict: status, reason, numerics, axes, recommended backend.
   |
   +--> downshift check    Print the verdict. Exit with its code.
   +--> downshift export   Write NAME.onnx and NAME.manifest.json (not for FAILED).
   +--> downshift serve    serve/engine.py picks a backend, warms up, then serves HTTP.
```

`check`, `export` and `serve` all use the same gate. `export()` is `check()` plus the write of the files. `serve` is `check()` plus a web server.

## Layers

| Layer | Path | Job | Imports torch? |
|---|---|---|---|
| CLI | `src/downshift/cli/` | Parse options. Call the library. Print the report. | No, until a command body runs |
| Settings | `settings.py` | Defaults and the `DOWNSHIFT_*` variables. | No |
| Entities | `entities/` | The data types: the dataclasses and the enums that they need. | No, except `serving.py` |
| Loading | `loading.py`, `hf_repo.py`, `sources.py` | Turn `MODEL` into a `LoadedModel`. | Inside functions |
| Adapters | `adapters/` | One class for each model family. | Yes |
| Core | `core/` | Export, verify, verdict, caches. | Yes |
| Serve | `serve/` | FastAPI app, request path, backends. | Inside functions |

The import rule has a reason. `downshift --help` and `import downshift` must stay fast. Modules that the CLI imports at start do not import torch, onnxruntime, fastapi or uvicorn. `downshift/__init__.py` loads its public names on first use (PEP 562). Refer to [`python-api.md`](python-api.md#lazy-import-pep-562).

## The parts, in the order that they run

### 1. Loading (`loading.py`)

`MODEL` has four forms: a `.onnx` file, a state dict (`.pt`, `.pth`, `.bin`, `.ckpt`) with `--model-class`, a downloaded Hugging Face directory (it has a `config.json`), or `package.module:attr`. A `make_inputs` function in the same module gives the example inputs. Downshift never downloads a model.

### 2. Adapters (`adapters/`)

An adapter knows one model family. The `Adapter` protocol (`adapters/base.py`) has three methods: `matches`, `example_inputs` and `prepare`. `prepare` returns a `Prepared` object. Its `model` takes flat tensors only. `registry.py` finds the adapter: `detect` tries them, `get` finds one by name.

| Adapter | File | Handles |
|---|---|---|
| `generic` | `generic.py`, `_flatten.py` | Any `nn.Module`. It flattens a dataclass argument. |
| `pyg` | `pyg.py` | PyTorch Geometric `Data`. Node count and edge count are separate dynamic axes. |
| `hf` | `hf.py`, `embedding.py`, `pooling.py`, `text.py` | Hugging Face models: `input_ids`, `attention_mask`, pooling, tokenizer. |

To add a family, read [`adapters.md`](adapters.md).

### 3. Capture (`core/capture.py`)

The function `capture` tries `torch.export` with `strict=False`. If that fails, it tries `strict=True`. Then it translates the result to ONNX. The verdict records which strategy worked (`capture_strategy`).

### 4. Verify (`core/verify.py`, `core/rows.py`)

`verify` runs the samples through torch and through ONNX Runtime. The dynamic axes change from sample to sample, so some shapes are new to the graph. This finds a graph that froze a shape or a branch. The tolerance comes from the dtype of the model (`settings.TOLERANCES`).

`check_rows` runs two rows alone and together. The result cache needs it, because the cache serves rows on their own. A model such as `x - x.mean(0)` fails this check.

`core/shapes.py` and `core/axes.py` hold the dynamic-shape rules and the axis facts of the verdict. By default, axis 0 of each input is dynamic and all inputs share one `Dim`.

### 5. Verdict (`core/verdict.py`)

`ExportVerdict` is a dataclass. `prepare_model` loads the model with its adapter. `build_verdict` runs capture and verify. `check` is the public function. `intake` is the gate for a `.onnx` file that someone else made. A FAILED or DEGRADED export is a verdict and not an exception.

### 6. Caches (`core/memo.py`, `core/export_cache.py`)

A cache key hashes the weights, the model code, the options and the toolchain. There are two tiers:

- **Memo.** In the process. It holds the last two results.
- **Disk.** `--export-cache-dir DIR`. Downshift never deletes an entry. To clear the cache, delete the directory.

The cache holds only CLEAN and DEGRADED results.

### 7. The serving check (`serve/engine.py`)

The verdict is made on the CPU. If the server serves on a GPU (ONNX Runtime CUDA or torch `cuda`), a second check runs at boot. It replays the verification samples through the serving backend and compares the result with the CPU result. If it fails and torch can serve, torch serves instead. On the CPU, this check does not run. Refer to "GPU" in [`production.md`](production.md#gpu).

### 8. Serving (`serve/`)

For a full description, read [`serving.md`](serving.md). In short:

- `prepare_serving` runs the whole gate. `state_from_entry` boots from an earlier export and skips the gate. `ServingState` holds everything that one server process needs: the verdict, the backend, the pools and the caches.
- With `loader=`, `build_app` binds the port first and loads the model on a background thread. `/health` is `200` at once. `/ready` is `503` until the model is ready.
- Per-request speed has priority over boot time, so all the export work happens at boot. The two backends (`OnnxRuntimeBackend` and `TorchBackend`) take and return numpy arrays.

## Design rules that explain the code

| Rule | What you see in the code |
|---|---|
| Speed of a request beats boot time. | Export, verify and warmup run at boot. The request path does only the work of the request. |
| Keep the code simple. | Plain functions and small dataclasses. There is no plugin framework. |
| The verdict is the one source of truth. | All code reads `ExportVerdict`. It does not repeat the checks. |
| A wrong export must not reach a client. | DEGRADED and FAILED models serve on torch. |
| Never fetch. | Loading takes local files only. |
| Each usage error has a next step. | A `ValueError` from a command is exit code `4`, and its message says what to change. |

## Glossary

| Term | Meaning |
|---|---|
| Verdict | The `ExportVerdict` object: status, reason, numbers and axis facts. |
| Gate | The export-and-verify step that makes a verdict. |
| Adapter | The code for one model family. |
| Prepared | The output of `adapter.prepare`: a flat-tensor module, inputs, names and dynamic shapes. |
| Capture | `torch.export` plus the ONNX translation. |
| Sample | One set of inputs for verification. Samples have different shapes. |
| Dynamic axis | An axis whose size can change. `sampled_*` is what verification ran. `served_*` is what the server accepts. |
| Strict | The `strict` argument of `torch.export`. |
| Row check | The check that rows of a batch do not change each other. |
| Memo | The in-process export cache. |
| Handoff | The way `serve --workers N` gives an export from the parent to each worker. |
| Intake | The gate for a `.onnx` file. It has no reference model, so the status is UNVERIFIED. |

## Where to read next

1. [`walkthrough.md`](walkthrough.md): run the three verdicts and call a server.
2. [`serving.md`](serving.md): the serving layer, in detail.
3. [`python-api.md`](python-api.md): use downshift as a library.
4. [`../AGENTS.md`](../AGENTS.md): how to check a change before you commit it.
