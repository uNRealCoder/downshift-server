# downshift

Check whether your PyTorch model survives ONNX export. Then serve it, falling back to eager PyTorch when ONNX would be lying to you.

```
$ downshift serve tests.models.scatter_include_self_false:make_model

┌─ downshift v0.1.0 ────────────────────────────────────────────────────────┐
│                                                                           │
│  Model          tests.models.scatter_include_self_false:make_model        │
│  Family         generic-torch                                             │
│  Verdict        DEGRADED  (strict=False, opset 20)                        │
│  Numerics       max abs err 1.16e+00 over 8 samples  ✗ 7/8 failed         │
│                 ⚠ numerics diverge on 7/8 samples (max abs err 1.16e+00)  │
│  Override       --force-onnx to serve the ONNX graph anyway               │
│  Backend        torch (eager) · cpu  ← auto-selected                      │
│  Dynamic dims   x[0], segment_ids[0]                                      │
│  Endpoint       http://0.0.0.0:8000                                       │
│                                                                           │
└───────────────────────────────────────────────────────────────────────────┘
```

This graph exported without a single error and produces wrong numbers on 7 of 8 inputs. The tool caught it and served PyTorch instead.

## Install

```bash
pip install downshift-server            # core: any nn.Module, any .onnx
pip install "downshift-server[gnn]"     # + PyTorch Geometric adapter
pip install "downshift-server[hf]"      # + Hugging Face encoder adapter
pip install "downshift-server[all]"
```

Until the PyPI release lands, install from a checkout with `pip install -e ".[all]"`.

Python 3.11 to 3.13. CPU-only is what this release was tested on. CUDA execution-provider selection exists (`--device cuda`) but is untested in this release.

## Quick start

### `check`: is the export trustworthy?

```
$ downshift check tests.models.scatter_include_self_false:make_model

┌───────────────┬─────────────────────────────────────────────────────────────┐
│ Model         │ tests.models.scatter_include_self_false:make_model          │
│ Family        │ generic-torch                                               │
│ Export        │ DEGRADED  (strict=False, opset 20)                          │
│ Numerics      │ max abs err 1.34e+00 over 8 samples  ✗ 7/8 failed           │
│ Shape-general │ no                                                          │
│ Dynamic dims  │ x[0], segment_ids[0]                                        │
│ Backend       │ torch                                                       │
│ Reason        │ exported via strict=False but numerics diverge on 7/8       │
│               │ samples (max abs err 1.34e+00)                              │
└───────────────┴─────────────────────────────────────────────────────────────┘
```

`check` exports in memory, runs `k` random samples (default 8) through both PyTorch and ONNX Runtime, and varies the dynamic axes so some samples have shapes the exporter never saw. Nothing is written to disk.

The exit code is the verdict, so it can gate CI: `0` CLEAN, `1` FAILED, `2` DEGRADED, `3` UNVERIFIED. (`4` is a usage error such as an unloadable model; `5` is a crash.) `--json` prints the full verdict as JSON and nothing else:

```bash
downshift check my_pkg.models:build --json -k 16 > verdict.json
```

Useful options: `-k/--samples`, `--dynamic "x:0,edge_index:1"` to override which axes are dynamic (default: axis 0 of every input), `--adapter generic|pyg|hf` to skip detection, `--inputs pkg.module:fn` to supply example inputs.

### `export`: write the artifact

```bash
downshift export my_pkg.models:build -o artifacts/ --name classifier
```

Writes `artifacts/classifier.onnx` and `artifacts/classifier.manifest.json`. The manifest records the SHA-256 of the artifact and of the source checkpoint (when the model came from a file), torch/onnx/onnxruntime versions, opset, the observed weight dtype, and the full verdict including the numerics report. A DEGRADED export is still written, because the manifest records exactly how far off it is; a FAILED export writes nothing.

`--fp16` casts the model to half before export (a plain `.half()`, not quantization). `--no-verify` skips the numerics check and marks the verdict UNVERIFIED, with a warning.

### `serve`: one endpoint, backend chosen by the verdict

```bash
downshift serve my_pkg.models:build --port 8000
```

Prints the banner above, then starts uvicorn. Routes:

| Route | What it does |
|---|---|
| `POST /predict` | Named tensor inputs, any model |
| `POST /predict/graph` | One graph: `x`, `edge_index`, optional `edge_attr` |
| `GET /health` | Liveness |
| `GET /ready` | `200` once warmup is done, `503` before |
| `GET /metadata` | Family, backend, full verdict, input names |

```bash
curl -s localhost:8000/predict -H 'content-type: application/json' \
  -d '{"inputs": {"x": [[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]]}}'
```

```json
{"outputs": {"output_0": [[0.199, -0.206, 0.561, 0.405]]},
 "shapes": {"output_0": [1, 4]},
 "dtypes": {"output_0": "float32"}}
```

Integer lists become `int64`, everything else `float32`. To be explicit, pass `{"data": [...], "dtype": "float16", "shape": [1, 16]}` instead of a bare list. For graph models:

```bash
curl -s localhost:8000/predict/graph -H 'content-type: application/json' \
  -d '{"x": [[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
             [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
             [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]],
       "edge_index": [[0, 1, 2], [1, 2, 0]]}'
```

The response has the same shape as `/predict`, with one output row per node.

Options that change what gets served:

- `--backend auto|onnxruntime|torch`. `auto` follows the verdict. `torch` skips the export entirely.
- `--force-onnx` serves a DEGRADED graph through ONNX Runtime anyway. The banner says so in red.
- `--reference model` verifies a pre-built `.onnx` against a PyTorch model; without it the verdict is UNVERIFIED.
- `--middleware pkg.module:Attr` (repeatable) attaches a `BaseHTTPMiddleware` subclass or an `async (request, call_next)` function. No middleware means no overhead.
- `--device auto|cpu|cuda`, `--warmup N` (inferences before `/ready` flips), `--host`, `--port`, `--log-level`, `--log-format json`.

## Accepted model forms

| Argument | Meaning |
|---|---|
| `model.onnx` | Pre-built ONNX, served as-is. UNVERIFIED unless `--reference` is given. |
| `pkg.module:attr` | Import spec. `attr` is an `nn.Module` instance or a zero-argument factory. A sibling `make_inputs` in the same module is picked up automatically; otherwise pass `--inputs pkg.module:fn`. |
| `weights.pt` | State dict. Needs `--model-class pkg.module:Class`. Also `.pth`, `.bin`, `.ckpt`. |
| `org/repo` | Hugging Face hub id. Needs the `[hf]` extra. |

Checkpoints are loaded with `torch.load(weights_only=True)`. A file that holds a pickled full module will not load that way; `--unsafe-load` switches to `weights_only=False`, which means running arbitrary code from the file. Only use it on files you would run as a script.

## The four verdicts

- **CLEAN**: exports, matches PyTorch on every sample, survives shapes it was not traced on. Served via ONNX Runtime.
- **DEGRADED**: exports without error, but numerics drift past tolerance on at least one sample. Served via eager PyTorch; `--force-onnx` overrides.
- **FAILED**: does not export. Served via eager PyTorch. Not an error, a supported path.
- **UNVERIFIED**: a `.onnx` with no reference model, or `--no-verify`. Served via ONNX Runtime and labelled as never checked.

## Compatibility matrix

Generated by `scripts/gen_matrix.py` from the fixture corpus in `tests/models/`, each fixture isolating one export hazard. CI regenerates it weekly against current torch and onnxruntime and opens a PR when it changes. Full file, with versions and legend: [docs/compatibility.md](docs/compatibility.md).

| Model | Hazard | Family | Export | Capture | Numerics | Shape-general | Backend |
|---|---|---|---|---|---|---|---|
| `clean_mlp` | Control fixture: no export hazards | generic-torch | CLEAN | strict=False | 1.2e-07 | ✓ | onnxruntime |
| `custom_autograd` | custom autograd.Function with no symbolic override | generic-torch | CLEAN | strict=False | 4.2e-07 | ✓ | onnxruntime |
| `data_dependent_branch` | data-dependent control flow | generic-torch | FAILED | — | — | — | torch |
| `dict_input` | dataclass container input | generic-torch | CLEAN | strict=False | 4.8e-07 | ✓ | onnxruntime |
| `dropout_model` | stochastic layer | generic-torch | CLEAN | strict=False | 3.6e-07 | ✓ | onnxruntime |
| `dynamic_batch_cnn` | batch-dim generalization | generic-torch | CLEAN | strict=False | 6.0e-08 | ✓ | onnxruntime |
| `gnn_gat` | GNN fixture: 3-layer GAT node classifier | pyg | CLEAN | strict=False | 1.2e-07 | ✓ | onnxruntime |
| `gnn_gcn` | GNN fixture: 2-layer GCN node classifier | pyg | CLEAN | strict=False | 2.4e-07 | ✓ | onnxruntime |
| `gnn_sage` | GNN fixture: 2-layer GraphSAGE node classifier | pyg | CLEAN | strict=False | 1.2e-07 | ✓ | onnxruntime |
| `scatter_include_self_false` | scatter_reduce(include_self=False) has no faithful ONNX translation | generic-torch | DEGRADED | strict=False | 1.3e+00 | ✗ | torch |
| `tied_weights` | tied embedding/output weight (GPT-2/OPT-style) | generic-torch | CLEAN | strict=False | 1.9e-06 | ✓ | onnxruntime |
| `tiny_bert` | HF fixture: a randomly initialised two-layer BERT encoder | hf-transformers | CLEAN | strict=False | 6.0e-07 | ✓ | onnxruntime |

Two rows worth reading twice. `custom_autograd` was expected to fail and is CLEAN, because `torch.export` traces straight through a `Function.forward` made of ordinary ops. `scatter_include_self_false` was expected to fail loudly and instead exports with zero errors and returns the wrong numbers; the only thing standing between that graph and production is the numerics check.

## What this is not

- **No quantization or graph optimization, ever.** Not deferred, cut. Run Olive, `onnxruntime.quantization`, or your own script, then hand the result to `downshift serve model.onnx --reference model.pt` and it gets verified against the original weights like any other export. `--fp16` is a cast before tracing, nothing lower exists here.
- **No LLM path.** No `onnxruntime-genai` backend, no OpenAI-compatible endpoints, no KV cache, no sampling loop. Encoder-only Hugging Face models work; causal LMs are not a target yet.
- **No continuous batching, no PagedAttention.** The boot banner is a visual homage to vLLM. That is the full extent of the resemblance.
- **No dynamic request batching yet.** One request, one inference.
- **No graph batching yet.** `/predict/graph` takes one graph. Concatenate graphs client-side with offset edge indices if you need more.
- **No Prometheus metrics, no Docker image.** `--middleware` is the hook for the former; pip plus version pins is the path for the latter.
- **No DGL adapter yet.** PyG only.

**vs. anydeploy.** `anydeploy` also does export, validate, and serve, with a pass/fail validation step and an edge/mobile focus. downshift differs in three places: the verdict is tiered, with DEGRADED as a real middle state between "works" and "crashes"; the eager PyTorch fallback sits behind the same endpoint so a FAILED or DEGRADED model still serves; and GNNs (PyTorch Geometric) are a supported family with independent node and edge dynamic dims.

## Writing your own adapter

An adapter knows one model family well enough to build example inputs when the user gave none, and to turn the model plus inputs into something `torch.export` can trace: a module with a flat tensor signature. Implement the `Adapter` protocol from `downshift.adapters.base`:

```python
from downshift.adapters.base import Prepared

class MyAdapter:
    name = "myfamily"
    family = "myfamily"

    def matches(self, model, example_inputs) -> bool: ...
    def example_inputs(self, model) -> tuple | None: ...   # None if you can't guess
    def prepare(self, model, example_inputs) -> Prepared: ...

ADAPTER = MyAdapter()
```

`Prepared` carries the export-ready module, the flat example inputs, their names, the per-input `dynamic_shapes` spec, an optional `vary_fn(i) -> inputs` that generates verification samples, and the family string. Register it under the `downshift.adapters` entry-point group in your own package:

```toml
[project.entry-points."downshift.adapters"]
myfamily = "my_pkg.adapter:ADAPTER"
```

Adapters are tried most-specific first; `generic` always goes last. An adapter whose optional dependency is missing is skipped silently.

## Development

```bash
pip install -e ".[dev,all]"
ruff check .
mypy src
pytest
python scripts/gen_matrix.py     # regenerates docs/compatibility.md
```

## License

MIT.
