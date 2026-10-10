# downshift

Serve a model you have already downloaded over HTTP with one command. Before the first request, downshift exports the model to ONNX, verifies the graph against PyTorch on inputs the exporter never saw, and serves eager PyTorch instead when the ONNX graph would be lying to you.

Everything downshift serves is already on the machine you run it on: a PyTorch checkpoint, an ONNX file, or a downloaded Hugging Face repo directory (one holding a `config.json`). Nothing is fetched — a hub id is rejected, not downloaded.

```
$ downshift serve examples.scatter_include_self_false:make_model

2026-10-10 19:10:19 WARNING downshift.serve: DOWNSHIFT_SERVER_API_KEY is not set, so the endpoints (including /metadata) are unauthenticated. Set it to require a fixed API key, or add your own authentication middleware.
2026-10-10 19:10:19 INFO downshift.report: loading examples.scatter_include_self_false:make_model
2026-10-10 19:10:19 INFO downshift.report: will listen on http://127.0.0.1:8000 (not ready yet)
2026-10-10 19:10:25 INFO downshift.report: verify: sample 1/8, x [6, 8], segment_ids [6]
2026-10-10 19:10:25 INFO downshift.report: verify: sample 2/8, x [12, 8], segment_ids [12]
2026-10-10 19:10:25 INFO downshift.report: verify: sample 3/8, x [7, 8], segment_ids [7]
2026-10-10 19:10:25 INFO downshift.report: verify: sample 4/8, x [7, 8], segment_ids [7]
2026-10-10 19:10:25 INFO downshift.report: verify: sample 5/8, x [1, 8], segment_ids [1]
2026-10-10 19:10:25 INFO downshift.report: verify: sample 6/8, x [12, 8], segment_ids [12]
2026-10-10 19:10:25 INFO downshift.report: verify: sample 7/8, x [1, 8], segment_ids [1]
2026-10-10 19:10:25 INFO downshift.report: verify: sample 8/8, x [7, 8], segment_ids [7]
2026-10-10 19:10:25 INFO downshift.report: downshift v0.5.0
  Model          examples.scatter_include_self_false:make_model
  Family         generic
  Verdict        DEGRADED  (strict=False, opset 20)
  Numerics       max abs err 1.43e+00 over 8 samples  6/8 failed
                 ! numerics diverge on 6/8 samples (max abs err 1.43e+00)
  Override       --force-onnx to serve the ONNX graph anyway
  Tolerance      atol 1e-04, rtol 1e-03 (float32)
  Worst          output_0[1, 5]: torch 0.3096, onnxruntime -1.1223  (sample 1, x (12,8), segment_ids (12))
  Samples        x: (6,8) (12,8) (7,8) (7,8) (1,8) (12,8) (1,8) (7,8)
                 segment_ids: (6) (12) (7) (7) (1) (12) (1) (7)
  Warmup         3 inferences, 0.14 ms each
  Boot           4.5 s: load 0.0, export 4.5, verify 0.0, session 0.0, warmup 0.0
  Backend        torch (eager) | cpu  <- auto-selected
  Verified on    CPUExecutionProvider
  Dynamic dims   `dim0` (x[0])  sampled 1-12, serves 1-65536  ! unverified above 12
                 `dim0` (segment_ids[0])  sampled 1-12, serves 1-65536  ! unverified above 12
  Encoding       json  (clients override with output_encoding)
  Capacity       4 inferences at a time, 4 prep threads, 64 queued, 30 s timeout  (--max-concurrency, --prep-threads, --max-queue, --request-timeout)
  Execution      threadpool  (--execution)
  Endpoint       http://127.0.0.1:8000  (GET /schema for the input format)
2026-10-10 19:10:25 INFO downshift.report: ready in 4.5 s
```

This graph exported without a single error and produces wrong numbers on 6 of 8 inputs. downshift caught it before the first request and is serving PyTorch instead. The model is [`examples/scatter_include_self_false.py`](examples/scatter_include_self_false.py); run the command from a clone of this repo.

Everything downshift prints, the banner included, is plain text through Python `logging` on stdout: no colours, no box drawing, nothing a log shipper has to strip. See [Logging](#logging).

## Install

```bash
pip install downshift-server            # core: any nn.Module, any .onnx
pip install "downshift-server[gnn]"     # + PyTorch Geometric adapter
pip install "downshift-server[hf]"      # + Hugging Face encoder adapter
pip install "downshift-server[fast]"    # + pybase64, ~12x faster binary tensor I/O
pip install "downshift-server[all]"
```

For development, install editable from a checkout instead: `pip install -e ".[dev,all]"`.

`downshift` is also runnable as `python -m downshift` if you'd rather not rely on the console script being on `PATH`.

Python 3.12 to 3.14. The test suite and the numerics gate run on the CPU. `--device cuda` was smoke-tested once, by hand, on an RTX 4050 with torch 2.14 (CUDA 13.0) and `onnxruntime-gpu` 1.30: both backends served and agreed with the CPU. It needs a matching pair, so install `onnxruntime-gpu` in place of `onnxruntime`, and a torch build whose CUDA major version is the one your `onnxruntime-gpu` wheel was built for (1.30 wants CUDA 13; with a mismatch ONNX Runtime cannot load its CUDA libraries). ONNX Runtime's CUDA provider uses TF32 matrix multiplies by default, which moved a small MLP's output by 4e-4 against the CPU, past the gate's float32 tolerance (atol 1e-4, rtol 1e-3); the gate never sees this because it only runs on the CPU. `--backend torch` on CUDA matched the CPU to 5e-8.

## Serve

```bash
downshift serve my_pkg.models:build --port 8000
```

`serve` binds the port first, then loads the model, runs the export-and-verify gate described in [The gate](#the-gate-export-and-verify-before-serving), picks a backend from the verdict, warms it up and prints the banner above, all on a background thread. `--workers N` is the exception: the parent still does that work before uvicorn binds, since every worker needs the export it produces. The routes are the same whichever backend is behind them.

Running this behind a real deployment (probes, sizing, Kubernetes, the optional API key, no built-in TLS) is its own page: [docs/production.md](docs/production.md). Set `DOWNSHIFT_SERVER_API_KEY` and every route but `/health` and `/ready` needs `Authorization: Bearer <key>`; without it the server logs one warning at startup that its endpoints are unauthenticated.

| Route | What it does |
|---|---|
| `POST /predict` | Named tensor inputs, any model; `503` (with `Retry-After`) until the model is ready |
| `POST /predict/graph` | One graph: `x`, `edge_index`, optional `edge_attr`; same `503` until ready |
| `GET /health` | Liveness: `200` as soon as the process is up, even mid-load |
| `GET /ready` | `503` (`{"ready": false, "phase": "export"}`) until the model has loaded, exported, verified and warmed up; `200` after, and it never goes back to `503` without a restart |
| `GET /metadata` | Family, backend, full verdict, input names, limits, boot timings, warmup stats; `503` until ready |
| `GET /schema` | What to POST: every input's name, dtype and shape, plus an example body you can send back unchanged; `503` until ready |

### What to POST

You do not have to guess the input format, and you do not have to have seen the model. `GET /schema` reads it off the graph that is actually running:

```bash
curl -s localhost:8000/schema | python -m json.tool
```

```json
{"model": "bert-base-uncased",
 "source": {"spec": "bert-base-uncased", "kind": "hf-repo-dir",
            "description": "a downloaded Hugging Face repo directory on this machine (has config.json)",
            "fetched_at_runtime": false},
 "family": "hf", "backend": "onnxruntime", "endpoint": "/predict",
 "inputs": [{"name": "input_ids", "dtype": "int64", "shape": ["batch", "seq"],
             "required": true, "example_shape": [1, 1],
             "bounds": [{"min": 1, "max": 4096}, {"min": 1, "max": 512}]},
            {"name": "attention_mask", "dtype": "int64", "shape": ["batch", "seq"],
             "required": true, "example_shape": [1, 1],
             "bounds": [{"min": 1, "max": 4096}, {"min": 1, "max": 512}]}],
 "outputs": [{"name": "output_0", "dtype": "float32", "shape": ["dynamic", "dynamic", 16]}],
 "example_request": {"inputs": {"input_ids": [[0]], "attention_mask": [[1]]}},
 "example_curl": "curl -s http://localhost:8000/predict -H 'content-type: application/json' -d '...'"}
```

An `int` in `shape` is a fixed size and a string is a dynamic axis, either a name the adapter chose (`"batch"`, `"seq"`) or the plain word `"dynamic"`. For a model downshift exported itself, each input also carries `bounds`, one `{"min", "max"}` per dynamic axis (`null` on a fixed one): the range the export was traced for. A request outside it is a readable `400` (`input_ids axis 1 is 513; this model accepts 1 to 512`) instead of an ONNX Runtime error. A bare `.onnx` with no `--reference` has no bounds, and its axes read `"dynamic"`. `example_request` is a complete, well-formed body: post it back to `/predict` unchanged and it returns `200`. The values in it are filler; the names, dtypes and shapes are the part to copy. On a model whose example would run past 256 numbers it is `null` instead, and `notes` gives the shape to build yourself. Full field reference: [docs/code-docs/http-api.md](docs/code-docs/http-api.md#get-schema).

```bash
curl -s localhost:8000/predict -H 'content-type: application/json' \
  -d '{"inputs": {"x": [[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]]}}'
```

```json
{"outputs": {"output_0": [[0.199, -0.206, 0.561, 0.405]]},
 "shapes": {"output_0": [1, 4]},
 "dtypes": {"output_0": "float32"}}
```

Integer lists become `int64`, everything else `float32`. To be explicit, pass `{"data": [...], "dtype": "float16", "shape": [1, 16]}` instead of a bare list. On the torch backend a bfloat16 or float16 model takes and returns float32 (the cast happens inside the backend, since neither has a numpy dtype), and `/schema` reports float32 for it.

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

Send large tensors as safetensors (below) or base64: the server parses base64 about 4x cheaper than the same tensor as nested lists, and a JSON float list holds the GIL while it is parsed, which slows everything else in that process. `downshift serve --output-encoding base64` (or `DOWNSHIFT_OUTPUT_ENCODING=base64`) makes base64 the default for every response without clients changing their requests; a per-request `output_encoding` still wins. The banner's `Encoding` row shows which default is in effect. Base64 input is validated, and each failure is a `400`:

- `dtype` and `shape` are mandatory. Shape is not inferable from bytes.
- The decoded length must equal `prod(shape) * itemsize`. The error says what was expected and what arrived.
- Little-endian only. Big-endian dtype strings such as `>f4` are rejected.
- The decoded size is capped by `--max-input-bytes` / `DOWNSHIFT_MAX_INPUT_BYTES` (default 256 MiB).

Install `downshift-server[fast]` to get `pybase64`, a SIMD base64 codec about 12x faster than the standard library's on both directions. Without it the server works the same; the banner prints a tip. Request bodies on every route are parsed with `orjson`, which is several times faster than the standard library on MiB-scale bodies, so clients that keep sending nested lists get a smaller win for free.

Do not expect this to help on small payloads: under roughly 100 KiB the JSON codec is not where the time goes, and base64 gains nothing. The win scales with payload width. Measured over HTTP at concurrency 8 (`bench/REPORT_v0.4.0.md`), the best case is `dynamic_batch_cnn` at batch 32 - a `(32, 3, 16, 16)` float32 batch, 96 KiB of tensor arriving as 498 KiB of nested-list JSON or 128 KiB of base64 - at 6.65x throughput and 6.33x p50; `cnn_large` and `mlp_large` at batch 32 are smaller but still real, at 2.65x and 2.10x throughput. At batch 1 on a narrow input the win can go slightly negative: `mlp_large` is 0.72x throughput, 0.75x p50, because base64's framing overhead is not free when there is little data to save. `bert_small`'s hidden-state outputs, once estimated here at ~3.5x, measure 1.05x on p50 at batch 32.

The cheapest body of all is a safetensors file: send `Content-Type: application/vnd.safetensors` with one tensor per input name, and ask for a safetensors response with `Accept: application/vnd.safetensors`. The arrays are read straight out of the request bytes, with no JSON and no base64 framing (`bert_small`: p50 11.2 ms as JSON, 9.1 ms as safetensors). The dtype must be the model's own (no silent cast, no BF16), and `/predict/graph` takes a batch this way too. Full rules in [`docs/code-docs/http-api.md`](docs/code-docs/http-api.md#wire-formats).

```python
from safetensors.numpy import load, save

req = urllib.request.Request(
    "http://localhost:8000/predict",
    save({"x": x}),
    {"content-type": "application/vnd.safetensors", "accept": "application/vnd.safetensors"},
)
y = load(urllib.request.urlopen(req).read())["output_0"]
```

See CHANGELOG.md for wire-format changes in 0.4 and 0.5.

### Mount it in your own app

Already have a FastAPI service? `downshift.serve.app_for(model, example_inputs)` runs the same export-and-verify gate and warmup as `downshift serve`, synchronously, and hands back a plain `FastAPI` app - no subprocess, no second port:

```python
from fastapi import FastAPI
from downshift.serve import app_for

app = FastAPI()
app.mount("/model", app_for(my_model, example_inputs))
```

`/model/predict`, `/model/health` and the rest behave exactly like a standalone `downshift serve`. `app_for` takes `ServeOptions` fields as keyword arguments (`backend=`, `warmup=`, `max_concurrency=`, ...), a `reference=` model for verifying a pre-built ONNX graph, and `api_key=` (default: `DOWNSHIFT_SERVER_API_KEY`). Every `DOWNSHIFT_*` environment variable below is read by `app_for()` and `ServeOptions()` too, not only by the CLI; they are read once, when `downshift` is imported. See [`app_for`](docs/code-docs/python-api.md#app_for) for the full signature, including the `LoadedModel`/`prepare_serving`/`build_app` pieces it wraps.

### Options

Options that change what gets served:

- `--backend auto|onnxruntime|torch` (env `DOWNSHIFT_BACKEND`). `auto` follows the verdict. `torch` skips the export entirely. `--backend onnxruntime` on a DEGRADED verdict is an error unless `--force-onnx` is also given.
- `--force-onnx` serves a DEGRADED graph through ONNX Runtime anyway. The banner says so.
- `--reference model` verifies a pre-built `.onnx` against a PyTorch model; without it the verdict is UNVERIFIED. Purely numeric: pass `--tokenizer-from` too if `.onnx` also needs a tokenizer.
- `--tokenizer-from dir/` loads the tokenizer, pooling recipe and label metadata for a `.onnx` or PyTorch MODEL from a Hugging Face repo directory, so `{"text": ...}` works without re-exporting. The graph is served as-is: it doesn't add pooling to a bare encoder, and `/schema` reports the recipe only when the output is already pooled. Independent of `--reference`: it never affects verification, and the two can name the same directory or different ones. See [Accepted model forms](#accepted-model-forms).
- `--middleware pkg.module:Attr` (repeatable) attaches a middleware class (pure ASGI, e.g. `starlette.middleware.gzip:GZipMiddleware`, or a `BaseHTTPMiddleware` subclass) or an `async (request, call_next)` function. No middleware means no overhead.
- `--output-encoding json|base64` (env `DOWNSHIFT_OUTPUT_ENCODING`, default `json`) sets the response encoding for requests that do not send their own `output_encoding`.
- `--max-input-bytes N` (env `DOWNSHIFT_MAX_INPUT_BYTES`, default 256 MiB) caps the decoded size of one base64 input; larger is a `400`.
- `--max-body-bytes N` (env `DOWNSHIFT_MAX_BODY_BYTES`, default 32 MiB) caps every request body, checked before it is parsed as JSON; larger is a `413`. A `Content-Length` over the cap is refused without reading the body, and a chunked body is refused as soon as its running total passes it. There is no separate limit on text length: for a `{"text": ...}` request this cap is the limit.
- `--max-concurrency N` (env `DOWNSHIFT_MAX_CONCURRENCY`, default 4) caps inferences running at once per worker process, via a dedicated thread pool of that size; requests beyond it wait in a queue rather than run inline. One inference rarely fills every core: all-MiniLM-L6-v2, batch 8, 8 concurrent clients served 52 req/s at `--max-concurrency 1` and 134 req/s at 4, and Qwen3-Embedding-0.6B on CPU 7.2 and 14.6 single queries/s. Each in-flight inference holds its own activation memory, so lower it if a large model's big requests run out of memory together. `--workers` adds processes instead: measured on 16 logical cores, `--workers 4` raises `clean_mlp` throughput at concurrency 32 by about 3.8x over `--workers 1` (`bench/REPORT_v0.4.0.md`), because each worker gets its own thread pool instead of sharing one.
- `--max-queue N` (env `DOWNSHIFT_MAX_QUEUE`, default 64) caps predicts waiting past `--max-concurrency`. Once `max-concurrency + max-queue` requests are admitted, a new one gets an immediate `503` with `Retry-After: 1` instead of joining the queue.
- `--request-timeout SECONDS` (env `DOWNSHIFT_REQUEST_TIMEOUT`, default 30, `0` turns it off) caps how long an admitted predict may wait, queued, for its turn before it gets a `503` instead of an inference. A request already running is never interrupted.
- `--workers N` (env `DOWNSHIFT_WORKERS`, default 1) starts that many uvicorn worker processes. The parent exports and verifies once; each worker builds its own ONNX Runtime session over that graph (or reloads the model, when torch serves it) and warms up, so memory and startup time scale with `N`. The parent frees its own copy of the model before the workers start.
- `--execution threadpool|inline` (env `DOWNSHIFT_EXECUTION`, default `threadpool`). `threadpool` parses and prepares each request in the prep pool and runs inference and encoding on an inference thread: two thread hops, and the event loop never blocks. `inline` runs small JSON bodies (Content-Length up to 64 KiB, no `text`) on the event loop with no hop at all; it only pays off for models that infer in well under a millisecond, and a slower model stalls `/health` and `/ready`.
- `--prep-threads N` (env `DOWNSHIFT_PREP_THREADS`, default `min(4, CPUs)`) sizes the prep pool that parses, validates and converts request bodies, apart from the inference threads.
- `--device auto|cpu|cuda`, `--warmup N` (inferences before `/ready` flips), `--intra-op-threads N` (threads inside one op, for ONNX Runtime or torch; 0 = the backend's default; more threads cut single-request latency but cost throughput under concurrent load) and `--inter-op-threads N` (ONNX Runtime threads across ops; 0 lets it choose), `--host`, `--port`, `--log-level`. `--device cuda` on a machine without CUDA is an error (exit code 4) on both backends, not a silent run on the CPU. The numerics gate only ever runs on the CPU, so the banner's `Verified on` row says so when you serve on a GPU.
- `--access-log/--no-access-log` (default on) controls the one log line per request; see [Logging](#logging).
- `--version` prints the installed version and exits.

Every response carries an `X-Request-Id` header (echoing the client's own if it sent one, otherwise a generated one) and every `/predict`/`/predict/graph` response carries `Server-Timing` with six stages, `parse`, `prep_wait`, `prep`, `infer_wait`, `infer` and `encode`, so you can see whether a slow request was waiting for a thread, converting its body or running the model. A `500` body includes the same `request_id`, and every log line written while that request was being served does too, so "see the server log" has a key to search for. A predict is admitted, or refused with a `503`, before its body is read, so an overloaded server does not buffer or parse what it is about to refuse.

`serve` runs the same gate as `check`, so it also accepts `-k/--samples`, `--dynamic`, `--adapter`, `--inputs`, `--model-class`, `--unsafe-load`, `--atol`/`--rtol`, `--seed` and `--vary`, described below.

### Logging

Everything goes through Python `logging` into one plain-text handler on stdout: the boot banner, the `check` and `export` reports, uvicorn's own startup and error lines, `warnings`, and downshift's request log line. Each line is `time LEVEL logger: message`, with ` request_id=<id>` appended when the line belongs to a request. There is no JSON format and no colour.

`--log-level` is `warning` by default for `check`, `export` and `serve`:

- **`warning`.** The banner, the reports and the `loading` / `will listen on` / `ready in X s` lines are logged on `downshift.report`, which always prints. So do warnings, errors and every `4xx`/`5xx` request line.
- **`info`.** Adds uvicorn's own lines and one line per request on `downshift.access`: `POST /predict 200 12.3 ms request_id=3f9c...`. Probes (`/health`, `/ready`) are logged at `DEBUG` only, so they never flood the output.
- **`debug`.** Adds tracebacks, and torch's own export output for a FAILED verdict.

`--no-access-log` drops the request line but keeps the request id on the response. With `check --json` or `export --json` the verdict JSON is the only thing on stdout; the log lines go to stderr.

### Errors

| Status | Cause |
|---|---|
| `400` | Client-caused input problem: bad JSON shape/dtype, an axis outside the range the model was exported for (`input_ids axis 1 is 65; this model accepts 1 to 64`), a Hugging Face repo model's `input_ids` outside `[0, vocab_size)` (the message names the value), or an input the backend rejects (message like `input 'x': ...`). The torch backend answers `400` only for client-shaped errors (an index out of range, a shape or dtype mismatch); any other exception from the model is a `500`. |
| `401` | `DOWNSHIFT_SERVER_API_KEY` is set and the request has no matching `Authorization: Bearer <key>`. Carries `WWW-Authenticate: Bearer`. `/health` and `/ready` are exempt. |
| `413` | Request body larger than `--max-body-bytes`. |
| `422` | Malformed JSON, or a required field is missing. |
| `500` | Server-side fault. The body is `{"detail": "inference failed on the server; see the server log", "request_id": "..."}`; the actual exception is logged (with the same `request_id`), not returned. |
| `503` | Server at capacity (`max-concurrency + max-queue` predicts already admitted; carries `Retry-After: 1`), a queued predict waited past `--request-timeout`, or the model is still loading (carries `Retry-After: 2`). |

## The gate: export and verify before serving

`torch.onnx.export` succeeding is not evidence that the graph computes the same function as the model. So before anything is served, downshift exports the model in memory, runs `k` random samples (default 8) through both PyTorch and ONNX Runtime, and varies the dynamic axes so some samples have shapes the exporter never saw. Nothing is written to disk. The result is one of four verdicts, and the verdict picks the backend:

- **CLEAN**: exports, matches PyTorch on every sample, survives shapes it was not traced on. Served via ONNX Runtime.
- **DEGRADED**: exports without error, but numerics drift past tolerance on at least one sample. Served via eager PyTorch; `--force-onnx` overrides.
- **FAILED**: does not export, or exports but ONNX Runtime cannot load or run the graph. Served via eager PyTorch. Not an error, a supported path.
- **UNVERIFIED**: a `.onnx` with no reference model, or `--no-verify`. Served via ONNX Runtime and labelled as never checked.

A sample passes when every output element satisfies `numpy.allclose`: `abs_err <= atol + rtol * |expected|`, the same rule numpy uses. Defaults are chosen by the narrowest floating dtype present in the model's parameters: bfloat16 or float16 win first if either appears (their tolerances are the loosest), float64 wins only when it's the *only* floating dtype present (a model mixing float32 and float64 is still bound by float32's precision), and float32 is the fallback. Overridable per dtype via `DOWNSHIFT_TOL_FLOAT32_ATOL=1e-3` / `DOWNSHIFT_TOL_FLOAT16_RTOL=0.05`, or outright with `--atol`/`--rtol` on `check`, `export` and `serve`. The `check` report's `Tolerance` row shows which dtype picked the default, or `(--atol/--rtol)` when either flag overrides it.

The gate also runs on its own, to gate CI and to write artifacts.

### `check`: is the export trustworthy?

```
$ downshift check examples.scatter_include_self_false:make_model

2026-10-10 19:10:15 INFO downshift.report: verify: sample 1/8, x [6, 8], segment_ids [6]
2026-10-10 19:10:15 INFO downshift.report: verify: sample 2/8, x [12, 8], segment_ids [12]
2026-10-10 19:10:15 INFO downshift.report: verify: sample 3/8, x [7, 8], segment_ids [7]
2026-10-10 19:10:15 INFO downshift.report: verify: sample 4/8, x [7, 8], segment_ids [7]
2026-10-10 19:10:15 INFO downshift.report: verify: sample 5/8, x [1, 8], segment_ids [1]
2026-10-10 19:10:15 INFO downshift.report: verify: sample 6/8, x [12, 8], segment_ids [12]
2026-10-10 19:10:15 INFO downshift.report: verify: sample 7/8, x [1, 8], segment_ids [1]
2026-10-10 19:10:15 INFO downshift.report: verify: sample 8/8, x [7, 8], segment_ids [7]
2026-10-10 19:10:15 INFO downshift.report: downshift v0.5.0
  Model           examples.scatter_include_self_false:make_model
  Family          generic
  Export          DEGRADED  (strict=False, opset 20)
  Numerics        max abs err 1.48e+00 over 8 samples  6/8 failed
  Tolerance       atol 1e-04, rtol 1e-03 (float32)
  Worst           output_0[0, 5]: torch 0.0240, onnxruntime -1.4570  (sample 2, x (7,8), segment_ids (7))
  Samples         x: (6,8) (12,8) (7,8) (7,8) (1,8) (12,8) (1,8) (7,8)
                  segment_ids: (6) (12) (7) (7) (1) (12) (1) (7)
  Shape-general   n/a (baseline fails)
  Dynamic dims    `dim0` (x[0])  sampled 1-12, serves 1-65536  ! unverified above 12
                  `dim0` (segment_ids[0])  sampled 1-12, serves 1-65536  ! unverified above 12
  Backend         torch
  Reason          exported via strict=False but numerics diverge on 6/8 samples (max abs err 1.48e+00)
```

The `Tolerance` row shows which dtype picked the default (or `--atol/--rtol` when either overrides it); `Worst` (DEGRADED only) is the single largest-error output element across every sample tried; `Samples` lists every sample's input shapes, so shape generalization has visible content; `Shape-general` is `yes`, `no`, or `n/a (baseline fails)` when the un-varied example itself didn't pass (shape generalization was never evaluated in that case).

The exit code is the verdict, so it can gate CI: `0` CLEAN, `1` FAILED, `2` DEGRADED, `3` UNVERIFIED. (`4` is a usage error such as an unloadable model; `5` is a crash.) `--json` prints the full verdict as JSON and nothing else on stdout (log lines go to stderr):

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

Three kinds of downloaded artifact, all of them already on the machine running the server:

| Argument | Meaning |
|---|---|
| `model.onnx` | A downloaded or already-exported ONNX file, served as-is. UNVERIFIED unless `--reference` is given. |
| `weights.pt` | A downloaded PyTorch checkpoint (state dict). Needs `--model-class pkg.module:Class`. Also `.pth`, `.bin`, `.ckpt`. |
| `path/to/repo/dir/` | A downloaded Hugging Face repo: a directory with a `config.json` in it, which is how one is recognised. Needs the `[hf]` extra. |

A Hugging Face repo is served the way its own files say. A classifier (`...ForSequenceClassification`) loads with its head and answers with class probabilities. A sentence-transformers repo (`modules.json`, `1_Pooling/`) is pooled and normalised inside the exported graph, so `output_0` is one embedding per text. Either way `POST /predict` takes `{"text": ...}` when the repo has tokenizer files, up to the length the model's author declared (for the RoBERTa family that is `max_position_embeddings - (pad_token_id + 1)`, so 514 positions serve 512 tokens). A longer row is refused with a 400, never cut; the only cap on how much you send is the request body limit. A repo that declares no pooling is not pooled unless you pass `--pooling`. See [`docs/code-docs/http-api.md`](docs/code-docs/http-api.md#embedding-models).

`downshift serve model.onnx --tokenizer-from path/to/repo/dir/` combines the first two rows: the `.onnx` is served as-is (optimized, UNVERIFIED unless `--reference` is also given), and the repo directory supplies the tokenizer, pooling recipe and label metadata `POST /predict` needs for `{"text": ...}`. `--tokenizer-from` and `--reference` are independent — pass one, the other, both (naming the same directory or different ones), or neither. This is how to serve a model someone already exported to ONNX (with `optimum`, Olive, or by hand) without re-tracing it through `torch.export`, while keeping text input and embedding pooling.

Plus one form that names a model already importable in the server process rather than a file on disk:

| Argument | Meaning |
|---|---|
| `pkg.module:attr` | Import spec. `attr` is an `nn.Module` instance or a zero-argument factory. A sibling `make_inputs` in the same module is picked up automatically; otherwise pass `--inputs pkg.module:fn`. The module is found on `sys.path` or in the current directory, so `downshift serve my_model:build` works next to `my_model.py`; name the module, not the file (`my_model`, not `my_model.py`). |

`downshift` never contacts the Hugging Face hub: a bare repo id like `bert-base-uncased` is rejected, and the `hf` adapter loads with `local_files_only=True`, so a repo has to be downloaded first (`huggingface-cli download bert-base-uncased --local-dir ./bert-base-uncased`) and passed as a path. That keeps a `serve` in an air-gapped or egress-restricted environment from silently depending on the network at startup. A running server reports which form it was given under `source` on [`GET /schema`](#what-to-post).

Checkpoints are loaded with `torch.load(weights_only=True)`. A file that holds a pickled full module will not load that way; `--unsafe-load` switches to `weights_only=False`, which means running arbitrary code from the file. Only use it on files you would run as a script.

## Compatibility matrix

Generated by `scripts/gen_matrix.py` from the fixture corpus in `tests/models/`, each fixture isolating one export hazard. CI regenerates it weekly against current torch and onnxruntime and opens a PR when it changes. Full file, with versions and legend: [docs/compatibility.md](docs/compatibility.md).

<!-- matrix:start -->
| Model | Hazard | Family | Export | Capture | Numerics | Shape-general | Backend |
|---|---|---|---|---|---|---|---|
| `bf16_weights` | bfloat16 weights: ONNX Runtime CPU has no bf16 Gemm kernel | generic | FAILED | strict=False | — | — | torch |
| `broken_factory` | Not an export hazard fixture: raises as soon as it's instantiated | — | skipped (RuntimeError) | — | — | — | — |
| `clean_mlp` | Control fixture: no export hazards | generic | CLEAN | strict=False | 1.2e-07 | ✓ | onnxruntime |
| `custom_autograd` | custom autograd.Function with no symbolic override | generic | CLEAN | strict=False | 1.8e-07 | ✓ | onnxruntime |
| `data_dependent_branch` | data-dependent control flow | generic | FAILED | — | — | — | torch |
| `dict_input` | dataclass container input | generic | CLEAN | strict=False | 2.4e-07 | ✓ | onnxruntime |
| `dropout_model` | stochastic layer | generic | CLEAN | strict=False | 1.2e-07 | ✓ | onnxruntime |
| `dynamic_batch_cnn` | batch-dim generalization | generic | CLEAN | strict=False | 3.0e-08 | ✓ | onnxruntime |
| `gnn_gat` | GNN fixture: 3-layer GAT node classifier | pyg | CLEAN | strict=False | 1.0e-07 | ✓ | onnxruntime |
| `gnn_gcn` | GNN fixture: 2-layer GCN node classifier | pyg | CLEAN | strict=False | 2.1e-07 | ✓ | onnxruntime |
| `gnn_sage` | GNN fixture: 2-layer GraphSAGE node classifier | pyg | CLEAN | strict=False | 8.9e-08 | ✓ | onnxruntime |
| `scatter_include_self_false` | scatter_reduce(include_self=False) has no faithful ONNX translation | generic | DEGRADED | strict=False | 1.2e+00 | — | torch |
| `tied_weights` | tied embedding/output weight (GPT-2/OPT-style) | generic | CLEAN | strict=False | 1.9e-06 | ✓ | onnxruntime |
| `tiny_bert` | HF fixture: a randomly initialised two-layer BERT encoder | hf | CLEAN | strict=False | 7.2e-07 | ✓ | onnxruntime |
<!-- matrix:end -->

Two rows worth reading twice. `custom_autograd` was expected to fail and is CLEAN, because `torch.export` traces straight through a `Function.forward` made of ordinary ops. `scatter_include_self_false` was expected to fail loudly and instead exports with zero errors and returns the wrong numbers; the only thing standing between that graph and production is the numerics check. A third: `bf16_weights` exports cleanly and ONNX Runtime can't run it, since bfloat16 has no CPU Gemm kernel; that used to crash the tool outright and is now a FAILED verdict like any other.

## Scope and non-goals

downshift serves models whose answer is one forward pass: tensors in, tensors out. That is what makes the gate possible, because a single pass can be checked numerically between the ONNX graph and PyTorch. GNNs, encoders, classifiers, embedders and your own `nn.Module`s fit. Anything that needs a loop around the model does not.

- **No generation.** No `onnxruntime-genai` backend, no OpenAI-compatible endpoints, no KV cache, no sampling loop, no chat templates. Use vLLM, TGI or llama.cpp for that. A decoder-only Hugging Face repo (`...ForCausalLM`) is refused unless it ships an embedding recipe (Qwen3-Embedding does: last-token pooling, left padding), in which case it is served as an embedder.
- **No remote code.** `trust_remote_code` is never set, so a Hugging Face repo whose `config.json` has an `auto_map` (Alibaba GTE v1.5 and other custom architectures) fails to load. Running Python shipped inside a model repo would break the promise that serving is offline and does nothing you did not ask for. Convert the model to a native architecture, or export it to ONNX yourself and serve that with `--tokenizer-from`.
- **No multimodal Hugging Face models.** Text in, one tensor out is the supported shape.
- **No quantization or graph optimization, ever.** Not deferred, cut. Run Olive, `onnxruntime.quantization`, or your own script, then hand the result to `downshift serve model.onnx --reference model.pt` and it gets verified against the original weights like any other export. `--fp16` is a cast before tracing, nothing lower exists here.
- **No continuous batching, no PagedAttention.** The boot banner is a visual homage to vLLM. That is the full extent of the resemblance.
- **No dynamic request batching yet.** One request, one inference.
- **No graph-level batching yet.** `/predict/graph` takes a batch of graphs when the model's outputs are per node or per edge; a model with a pooled, fixed-size output still takes one graph per request.
- **No Docker image.** pip plus version pins is the path.
- **No DGL adapter yet.** PyG only.

**vs. anydeploy.** `anydeploy` also does export, validate, and serve, with a pass/fail validation step and an edge/mobile focus. downshift differs in three places: the verdict is tiered, with DEGRADED as a real middle state between "works" and "crashes"; the eager PyTorch fallback sits behind the same endpoint so a FAILED or DEGRADED model still serves; and GNNs (PyTorch Geometric) are a supported family with independent node and edge dynamic dims.

## Writing your own adapter

An adapter knows one model family well enough to build example inputs when the user gave none, and to turn the model plus inputs into something `torch.export` can trace: a module with a flat tensor signature. Implement the `Adapter` protocol from `downshift.adapters.base`:

```python
from downshift.adapters.base import Prepared


class MyAdapter:
    name = "myfamily"

    def matches(self, model, example_inputs) -> bool: ...
    def example_inputs(self, model) -> tuple | None: ...  # None if you can't guess
    def prepare(self, model, example_inputs, axis_max=None) -> Prepared: ...
```

`Prepared` carries the export-ready module, the flat example inputs, their names, the per-input `dynamic_shapes` spec, an optional `vary_fn(i) -> inputs` that generates verification samples, and the family string. `--seed` reproduces those samples for free if `vary_fn` draws its randomness from torch's global RNG (as the built-in `hf` and `pyg` adapters do, inside the `torch.random.fork_rng()` `verify()` already runs every sample in); an adapter that keeps its own `random.Random` won't pick up the seed. Register it under the `downshift.adapters` entry-point group in your own package:

```toml
[project.entry-points."downshift.adapters"]
myfamily = "my_pkg.adapter:MyAdapter"
```

The entry point names the class; downshift instantiates it with no arguments. (An entry point that names a ready-made instance still works.)

Adapters are tried most-specific first; `generic` always goes last. An adapter whose optional dependency is missing is skipped silently.

**The plugin contract:** keep the entry-point module cheap to import — downshift imports every registered entry point's module just to build the adapter list (an `ImportError` there is treated as "optional dependency not installed" and skipped silently). Do the heavy import (your model library, a large parser, ...) inside `prepare()`, which only runs once an adapter has actually matched, not at the top of the module that defines your adapter class. The built-in `hf` and `pyg` adapters aren't entry points at all precisely because they can't follow that rule (`transformers`/`torch_geometric` have to be imported to define `HFAdapter`/`PyGAdapter` in the first place); `downshift.adapters.registry` loads them directly instead, gated on the family's module already being in `sys.modules`, so discovering adapters for a plain PyTorch model never imports either.

For a one-off adapter that isn't worth packaging, `--adapter` (and `check()`'s `adapter=`) also accepts a bare `.py` file directly, no install or entry point required:

```bash
downshift check my_model:model --adapter path/to/pointcloud_adapter.py
```

Point at the class with `--adapter path/to/pointcloud_adapter.py:MyAdapter` and it's instantiated with no arguments. A bare `path/to/pointcloud_adapter.py` looks for a module-level `ADAPTER` naming the class (`ADAPTER = MyAdapter`) or an instance of it.

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
