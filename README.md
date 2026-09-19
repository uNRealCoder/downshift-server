# downshift

Serve a PyTorch model over HTTP with one command. Before the first request, downshift exports the model to ONNX, verifies the graph against PyTorch on inputs the exporter never saw, and serves eager PyTorch instead when the ONNX graph would be lying to you.

```
$ downshift serve downshift.demo.scatter_include_self_false:make_model

┌─ downshift v0.4.0 ────────────────────────────────────────────────────────────────────┐
│                                                                                       │
│  Model          downshift.demo.scatter_include_self_false:make_model                  │
│  Family         generic-torch                                                         │
│  Verdict        DEGRADED  (strict=False, opset 20)                                    │
│  Numerics       max abs err 1.39e+00 over 8 samples  ✗ 6/8 failed                     │
│                 ⚠ numerics diverge on 6/8 samples (max abs err 1.39e+00)              │
│  Override       --force-onnx to serve the ONNX graph anyway                           │
│  Backend        torch (eager) · cpu  ← auto-selected                                  │
│  Dynamic dims   x[0], segment_ids[0]                                                  │
│  Encoding       json  (clients override with output_encoding)                         │
│  Concurrency    1 inference at a time, 64 queued  (--max-concurrency, --max-queue)    │
│  Endpoint       http://127.0.0.1:8000                                                 │
│                                                                                       │
└───────────────────────────────────────────────────────────────────────────────────────┘
```

This graph exported without a single error and produces wrong numbers on 6 of 8 inputs. downshift caught it before the first request and is serving PyTorch instead.

## Install

```bash
pip install downshift-server            # core: any nn.Module, any .onnx
pip install "downshift-server[gnn]"     # + PyTorch Geometric adapter
pip install "downshift-server[hf]"      # + Hugging Face encoder adapter
pip install "downshift-server[fast]"    # + pybase64, ~12x faster binary tensor I/O
pip install "downshift-server[all]"
```

For development, install editable from a checkout instead: `pip install -e ".[dev,all]"`.

Python 3.11 to 3.14. CPU-only is what this release was tested on. CUDA execution-provider selection exists (`--device cuda`) but is untested in this release.

## Serve

```bash
downshift serve my_pkg.models:build --port 8000
```

`serve` binds the port first, then loads the model, runs the export-and-verify gate described in [The gate](#the-gate-export-and-verify-before-serving), picks a backend from the verdict, warms it up and prints the banner above, all on a background thread. `--workers N` is the exception: the parent still does that work before uvicorn binds, since every worker needs the export it produces. The routes are the same whichever backend is behind them.

| Route | What it does |
|---|---|
| `POST /predict` | Named tensor inputs, any model; `503` (with `Retry-After`) until the model is ready |
| `POST /predict/graph` | One graph: `x`, `edge_index`, optional `edge_attr`; same `503` until ready |
| `GET /health` | Liveness: `200` as soon as the process is up, even mid-load |
| `GET /ready` | `503` (`{"ready": false, "phase": "export"}`) until the model has loaded, exported, verified and warmed up; `200` after, and it never goes back to `503` without a restart |
| `GET /metadata` | Family, backend, full verdict, input names, limits; `503` until ready |

```bash
curl -s localhost:8000/predict -H 'content-type: application/json' \
  -d '{"inputs": {"x": [[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]]}}'
```

```json
{"outputs": {"output_0": [[0.199, -0.206, 0.561, 0.405]]},
 "shapes": {"output_0": [1, 4]},
 "dtypes": {"output_0": "float32"}}
```

Integer lists become `int64`, everything else `float32`. To be explicit, pass `{"data": [...], "dtype": "float16", "shape": [1, 16]}` instead of a bare list.

For graph models:

```bash
curl -s localhost:8000/predict/graph -H 'content-type: application/json' \
  -d '{"x": [[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
             [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
             [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]],
       "edge_index": [[0, 1, 2], [1, 2, 0]]}'
```

The response has the same shape as `/predict`, with one output row per node.

### Binary tensors

The same `{"data", "dtype", "shape"}` form also takes a base64 string in `data`: the raw little-endian, C-contiguous bytes of the array, standard alphabet, with padding. Detection is by type, a string is base64 and a list is the JSON path above, so existing clients keep working. Any input on `/predict` accepts it, as do `x`, `edge_index` and `edge_attr` on `/predict/graph`. Outputs stay JSON unless asked: `"output_encoding": "base64"` next to `inputs` (or next to `x` on `/predict/graph`) turns each `outputs[name]` into the same `{"data", "dtype", "shape"}` dict, and `shapes` and `dtypes` stay populated either way. A client needs only the standard library and numpy:

```python
import base64, json, urllib.request
import numpy as np


def encode(a):
    a = np.ascontiguousarray(a)
    return {
        "data": base64.b64encode(a.tobytes()).decode(),
        "dtype": str(a.dtype),
        "shape": list(a.shape),
    }


def decode(o):
    return np.frombuffer(base64.b64decode(o["data"]), dtype=o["dtype"]).reshape(o["shape"])


x = np.random.rand(32, 3, 64, 64).astype(np.float32)
body = json.dumps({"inputs": {"x": encode(x)}, "output_encoding": "base64"}).encode()
req = urllib.request.Request(
    "http://localhost:8000/predict", body, {"content-type": "application/json"}
)
y = decode(json.load(urllib.request.urlopen(req))["outputs"]["output_0"])
```

`downshift serve --output-encoding base64` (or `DOWNSHIFT_OUTPUT_ENCODING=base64`) makes base64 the default for every response without clients changing their requests; a per-request `output_encoding` still wins. The banner's `Encoding` row shows which default is in effect. Base64 input is validated, and each failure is a `400`:

- `dtype` and `shape` are mandatory. Shape is not inferable from bytes.
- The decoded length must equal `prod(shape) * itemsize`. The error says what was expected and what arrived.
- Little-endian only. Big-endian dtype strings such as `>f4` are rejected.
- The decoded size is capped by `--max-input-bytes` / `DOWNSHIFT_MAX_INPUT_BYTES` (default 256 MiB).

Install `downshift-server[fast]` to get `pybase64`, a SIMD base64 codec about 12x faster than the standard library's on both directions. Without it the server works the same; the banner prints a tip. Request bodies on every route are parsed with `orjson`, which is several times faster than the standard library on MiB-scale bodies, so clients that keep sending nested lists get a smaller win for free.

Do not expect this to help on small payloads: under roughly 100 KiB the JSON codec is not where the time goes, and base64 gains nothing. The win is on wide inputs and outputs. The in-process estimates from the benchmark corpus, not yet measured over HTTP, are around 5x on p50 for `cnn_large`-class inputs (a `32×3×64×64` float32 batch) and around 3.5x for `bert_small`-class outputs (hidden states). Treat those as expectations until the benchmark re-run replaces them.

See CHANGELOG.md for wire-format changes in 0.3.

### Options

Options that change what gets served:

- `--backend auto|onnxruntime|torch`. `auto` follows the verdict. `torch` skips the export entirely.
- `--force-onnx` serves a DEGRADED graph through ONNX Runtime anyway. The banner says so in red.
- `--reference model` verifies a pre-built `.onnx` against a PyTorch model; without it the verdict is UNVERIFIED.
- `--middleware pkg.module:Attr` (repeatable) attaches a `BaseHTTPMiddleware` subclass or an `async (request, call_next)` function. No middleware means no overhead.
- `--output-encoding json|base64` (env `DOWNSHIFT_OUTPUT_ENCODING`, default `json`) sets the response encoding for requests that do not send their own `output_encoding`.
- `--max-input-bytes N` (env `DOWNSHIFT_MAX_INPUT_BYTES`, default 256 MiB) caps the decoded size of one base64 input; larger is a `400`.
- `--max-body-bytes N` (env `DOWNSHIFT_MAX_BODY_BYTES`, default 256 MiB) caps every request body, checked before it is parsed as JSON; larger is a `413`.
- `--max-concurrency N` (env `DOWNSHIFT_MAX_CONCURRENCY`, default 1) caps inferences running at once per worker process, via a dedicated thread pool of that size; requests beyond it wait in a queue rather than run inline. One inference already uses every core through ONNX Runtime's intra-op threads, so on CPU raising this rarely adds throughput; it mostly adds contention. Use `--workers` for more processes instead.
- `--max-queue N` (env `DOWNSHIFT_MAX_QUEUE`, default 64) caps predicts waiting past `--max-concurrency`. Once `max-concurrency + max-queue` requests are admitted, a new one gets an immediate `503` with `Retry-After: 1` instead of joining the queue.
- `--request-timeout SECONDS` (env `DOWNSHIFT_REQUEST_TIMEOUT`, default 0, off) caps how long an admitted predict may wait for its turn before it gets a `503` instead of an inference. A request already running is never interrupted.
- `--workers N` (env `DOWNSHIFT_WORKERS`, default 1) starts that many uvicorn worker processes. Each one independently loads, exports, verifies and warms the model, so memory and startup time scale with `N`.
- `--device auto|cpu|cuda`, `--warmup N` (inferences before `/ready` flips), `--intra-op-threads N` and `--inter-op-threads N` (ONNX Runtime thread counts; 0 lets it choose), `--host`, `--port`, `--log-level`, `--log-format json`.
- `--access-log/--no-access-log` (default on) passes through to uvicorn's per-request access log.
- `--version` prints the installed version and exits.

Every response carries an `X-Request-Id` header (echoing the client's own if it sent one, otherwise a generated one) and every `/predict`/`/predict/graph` response carries `Server-Timing: codec;dur=<ms>, infer;dur=<ms>` splitting conversion/encoding time from the backend call. A `500` body includes the same `request_id`, and the server log line for it does too, so "see the server log" has a key to search for.

`serve` runs the same gate as `check`, so it also accepts `-k/--samples`, `--dynamic`, `--adapter`, `--inputs`, `--model-class`, `--unsafe-load`, `--atol`/`--rtol`, `--seed` and `--vary`, described below.

### Errors

| Status | Cause |
|---|---|
| `400` | Client-caused input problem: bad JSON shape/dtype, or an input the backend rejects (message like `input 'x': ...`). |
| `413` | Request body larger than `--max-body-bytes`. |
| `422` | Malformed JSON, or a required field is missing. |
| `500` | Server-side fault. The body is `{"detail": "inference failed on the server; see the server log", "request_id": "..."}`; the actual exception is logged (with the same `request_id`), not returned. |
| `503` | Server at capacity (`max-concurrency + max-queue` predicts already admitted; carries `Retry-After: 1`), or a queued predict waited past `--request-timeout`. |

## The gate: export and verify before serving

`torch.onnx.export` succeeding is not evidence that the graph computes the same function as the model. So before anything is served, downshift exports the model in memory, runs `k` random samples (default 8) through both PyTorch and ONNX Runtime, and varies the dynamic axes so some samples have shapes the exporter never saw. Nothing is written to disk. The result is one of four verdicts, and the verdict picks the backend:

- **CLEAN**: exports, matches PyTorch on every sample, survives shapes it was not traced on. Served via ONNX Runtime.
- **DEGRADED**: exports without error, but numerics drift past tolerance on at least one sample. Served via eager PyTorch; `--force-onnx` overrides.
- **FAILED**: does not export, or exports but ONNX Runtime cannot load or run the graph. Served via eager PyTorch. Not an error, a supported path.
- **UNVERIFIED**: a `.onnx` with no reference model, or `--no-verify`. Served via ONNX Runtime and labelled as never checked.

A sample passes when every output element satisfies `numpy.allclose`: `abs_err <= atol + rtol * |expected|`, the same rule numpy uses. Defaults are chosen by the narrowest floating dtype present in the model's parameters: bfloat16 or float16 win first if either appears (their tolerances are the loosest), float64 wins only when it's the *only* floating dtype present (a model mixing float32 and float64 is still bound by float32's precision), and float32 is the fallback. Overridable per dtype via `DOWNSHIFT_TOL_FLOAT32_ATOL=1e-3` / `DOWNSHIFT_TOL_FLOAT16_RTOL=0.05`, or outright with `--atol`/`--rtol` on `check`, `export` and `serve`. The `check` table's `Tolerance` row shows which dtype picked the default, or `(--atol/--rtol)` when either flag overrides it.

The gate also runs on its own, to gate CI and to write artifacts.

### `check`: is the export trustworthy?

```
$ downshift check downshift.demo.scatter_include_self_false:make_model

┌───────────────┬─────────────────────────────────────────────────────────────┐
│ Model         │ downshift.demo.scatter_include_self_false:make_model        │
│ Family        │ generic-torch                                               │
│ Export        │ DEGRADED  (strict=False, opset 20)                          │
│ Numerics      │ max abs err 1.35e+00 over 8 samples  ✗ 6/8 failed           │
│ Tolerance     │ atol 1e-04, rtol 1e-03 (float32)                            │
│ Worst         │ output_0[3, 0]: torch 0.0516, onnxruntime -1.2979  (sample  │
│               │ 3, x (7,8), segment_ids (7))                                │
│ Samples       │ x: (6,8) (12,8) (7,8) (7,8) (1,8) (12,8) (1,8) (7,8)        │
│               │ segment_ids: (6) (12) (7) (7) (1) (12) (1) (7)              │
│ Shape-general │ n/a (baseline fails)                                        │
│ Dynamic dims  │ x[0], segment_ids[0]                                        │
│ Backend       │ torch                                                       │
│ Reason        │ exported via strict=False but numerics diverge on 6/8       │
│               │ samples (max abs err 1.35e+00)                              │
└───────────────┴─────────────────────────────────────────────────────────────┘
```

The `Tolerance` row shows which dtype picked the default (or `--atol/--rtol` when either overrides it); `Worst` (DEGRADED only) is the single largest-error output element across every sample tried; `Samples` lists every sample's input shapes, so shape generalization has visible content; `Shape-general` is `yes`, `no`, or `n/a (baseline fails)` when the un-varied example itself didn't pass (shape generalization was never evaluated in that case).

The exit code is the verdict, so it can gate CI: `0` CLEAN, `1` FAILED, `2` DEGRADED, `3` UNVERIFIED. (`4` is a usage error such as an unloadable model; `5` is a crash.) `--json` prints the full verdict as JSON and nothing else:

```bash
downshift check my_pkg.models:build --json -k 16 > verdict.json
```

Useful options: `-k/--samples`, `--dynamic "x:0,edge_index:1"` to override which axes are dynamic (default: axis 0 of every input), `--adapter generic|pyg|hf` to skip detection, `--inputs pkg.module:fn` to supply example inputs, `--atol`/`--rtol` to override the tolerance, `--seed N` (default 0) to make the verification samples reproducible, `--vary pkg.module:fn` to supply your own `fn(i) -> inputs` instead of downshift's sampler (`fn(0)` must return the example inputs).

### `export`: write the artifact

```bash
downshift export my_pkg.models:build -o artifacts/ --name classifier
```

Writes `artifacts/classifier.onnx` and `artifacts/classifier.manifest.json`. The manifest records the SHA-256 of the artifact and of the source checkpoint (when the model came from a file), torch/onnx/onnxruntime versions, opset, the observed weight dtype, and the full verdict including the numerics report. A DEGRADED export is still written, because the manifest records exactly how far off it is; a FAILED export writes nothing.

`--fp16` casts the model to half before export (a plain `.half()`, not quantization). It works on a deep copy, so the model object you passed in is left untouched. `--no-verify` skips the numerics check and marks the verdict UNVERIFIED, with a warning.

## Accepted model forms

| Argument | Meaning |
|---|---|
| `model.onnx` | Pre-built ONNX, served as-is. UNVERIFIED unless `--reference` is given. |
| `pkg.module:attr` | Import spec. `attr` is an `nn.Module` instance or a zero-argument factory. A sibling `make_inputs` in the same module is picked up automatically; otherwise pass `--inputs pkg.module:fn`. |
| `weights.pt` | State dict. Needs `--model-class pkg.module:Class`. Also `.pth`, `.bin`, `.ckpt`. |
| `org/repo` | Hugging Face hub id. Needs the `[hf]` extra. |
| `path/to/repo/dir/` | Locally downloaded Hugging Face repo: a directory containing `config.json`. Needs the `[hf]` extra. |

Checkpoints are loaded with `torch.load(weights_only=True)`. A file that holds a pickled full module will not load that way; `--unsafe-load` switches to `weights_only=False`, which means running arbitrary code from the file. Only use it on files you would run as a script.

## Compatibility matrix

Generated by `scripts/gen_matrix.py` from the fixture corpus in `tests/models/`, each fixture isolating one export hazard. CI regenerates it weekly against current torch and onnxruntime and opens a PR when it changes. Full file, with versions and legend: [docs/compatibility.md](docs/compatibility.md).

<!-- matrix:start -->
| Model | Hazard | Family | Export | Capture | Numerics | Shape-general | Backend |
|---|---|---|---|---|---|---|---|
| `bf16_weights` | bfloat16 weights: ONNX Runtime CPU has no bf16 Gemm kernel | generic-torch | FAILED | strict=False | — | — | torch |
| `broken_factory` | Not an export hazard fixture: raises as soon as it's instantiated | — | skipped (RuntimeError) | — | — | — | — |
| `clean_mlp` | Control fixture: no export hazards | generic-torch | CLEAN | strict=False | 8.9e-08 | ✓ | onnxruntime |
| `custom_autograd` | custom autograd.Function with no symbolic override | generic-torch | CLEAN | strict=False | 8.9e-08 | ✓ | onnxruntime |
| `data_dependent_branch` | data-dependent control flow | generic-torch | FAILED | — | — | — | torch |
| `dict_input` | dataclass container input | generic-torch | CLEAN | strict=False | 1.2e-07 | ✓ | onnxruntime |
| `dropout_model` | stochastic layer | generic-torch | CLEAN | strict=False | 2.4e-07 | ✓ | onnxruntime |
| `dynamic_batch_cnn` | batch-dim generalization | generic-torch | CLEAN | strict=False | 3.0e-08 | ✓ | onnxruntime |
| `gnn_gat` | GNN fixture: 3-layer GAT node classifier | pyg | CLEAN | strict=False | 1.5e-07 | ✓ | onnxruntime |
| `gnn_gcn` | GNN fixture: 2-layer GCN node classifier | pyg | CLEAN | strict=False | 3.6e-07 | ✓ | onnxruntime |
| `gnn_sage` | GNN fixture: 2-layer GraphSAGE node classifier | pyg | CLEAN | strict=False | 1.2e-07 | ✓ | onnxruntime |
| `scatter_include_self_false` | scatter_reduce(include_self=False) has no faithful ONNX translation | generic-torch | DEGRADED | strict=False | 1.3e+00 | — | torch |
| `tied_weights` | tied embedding/output weight (GPT-2/OPT-style) | generic-torch | CLEAN | strict=False | 1.9e-06 | ✓ | onnxruntime |
| `tiny_bert` | HF fixture: a randomly initialised two-layer BERT encoder | hf-transformers | CLEAN | strict=False | 6.0e-07 | ✓ | onnxruntime |
<!-- matrix:end -->

Two rows worth reading twice. `custom_autograd` was expected to fail and is CLEAN, because `torch.export` traces straight through a `Function.forward` made of ordinary ops. `scatter_include_self_false` was expected to fail loudly and instead exports with zero errors and returns the wrong numbers; the only thing standing between that graph and production is the numerics check. A third: `bf16_weights` exports cleanly and ONNX Runtime can't run it, since bfloat16 has no CPU Gemm kernel; that used to crash the tool outright and is now a FAILED verdict like any other.

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
    def example_inputs(self, model) -> tuple | None: ...  # None if you can't guess
    def prepare(self, model, example_inputs) -> Prepared: ...


ADAPTER = MyAdapter()
```

`Prepared` carries the export-ready module, the flat example inputs, their names, the per-input `dynamic_shapes` spec, an optional `vary_fn(i) -> inputs` that generates verification samples, and the family string. `--seed` reproduces those samples for free if `vary_fn` draws its randomness from torch's global RNG (as the built-in `hf` and `pyg` adapters do, inside the `torch.random.fork_rng()` `verify()` already runs every sample in); an adapter that keeps its own `random.Random` won't pick up the seed. Register it under the `downshift.adapters` entry-point group in your own package:

```toml
[project.entry-points."downshift.adapters"]
myfamily = "my_pkg.adapter:ADAPTER"
```

Adapters are tried most-specific first; `generic` always goes last. An adapter whose optional dependency is missing is skipped silently.

**The plugin contract:** keep the entry-point module cheap to import — downshift imports every registered entry point's module just to build the adapter list (an `ImportError` there is treated as "optional dependency not installed" and skipped silently). Do the heavy import (your model library, a large parser, ...) inside `prepare()`, which only runs once an adapter has actually matched, not inside the module `ADAPTER` is defined in. The built-in `hf` and `pyg` adapters aren't entry points at all precisely because they can't follow that rule (`transformers`/`torch_geometric` have to be imported to define `HFAdapter`/`PyGAdapter` in the first place); `downshift.adapters.registry` loads them directly instead, gated on the family's module already being in `sys.modules`, so discovering adapters for a plain PyTorch model never imports either.

For a one-off adapter that isn't worth packaging, `--adapter` (and `check()`'s `adapter=`) also accepts a bare `.py` file directly, no install or entry point required:

```bash
downshift check my_model.py:model --adapter path/to/pointcloud_adapter.py
```

The file needs a module-level `ADAPTER = MyAdapter()`, or point at the class directly with `--adapter path/to/pointcloud_adapter.py:MyAdapter` and it's instantiated with no arguments.

## Development

```bash
pip install -e ".[dev,all]"
ruff check .
ruff format --check .
mypy src
pytest
python scripts/gen_matrix.py     # regenerates docs/compatibility.md and this README's table
```

Plain `pytest` enforces no coverage gate locally; CI runs it with `--cov-fail-under=95`.

## Changelog

See [CHANGELOG.md](CHANGELOG.md).

## License

MIT.
