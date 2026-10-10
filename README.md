# downshift

Serve a downloaded model over HTTP with one command. Before the first request, downshift exports the model to ONNX. It then compares the ONNX graph with PyTorch on inputs that the exporter did not see. If the graph gives wrong numbers, downshift serves eager PyTorch instead.

Everything that downshift serves is already on the machine. The model can be a PyTorch checkpoint, an ONNX file, or a downloaded Hugging Face repo directory (a directory with a `config.json`). Downshift fetches nothing. It rejects a hub id and does not download it.

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

This graph exported without an error, but it gives wrong numbers on 6 of 8 inputs. Downshift found the fault before the first request and serves PyTorch instead. The model is [`examples/scatter_include_self_false.py`](examples/scatter_include_self_false.py). Run the command from a clone of this repo.

Downshift prints everything, the banner included, as plain text through Python `logging` on stdout. There are no colours and no box drawing. A log shipper does not need to remove anything. Refer to [Logging](#logging).

## Install

```bash
pip install downshift-server            # core: any nn.Module, any .onnx
pip install "downshift-server[gnn]"     # + PyTorch Geometric adapter
pip install "downshift-server[hf]"      # + Hugging Face encoder adapter
pip install "downshift-server[fast]"    # + pybase64, ~12x faster binary tensor I/O
pip install "downshift-server[all]"
```

To develop, install from a checkout in editable mode: `pip install -e ".[dev,all]"`.

You can also run `downshift` as `python -m downshift`. Use this form if the console script is not on `PATH`.

Downshift needs Python 3.12 to 3.14. The test suite and the numerics gate run on the CPU.

### GPU

A person tested `--device cuda` once, by hand. The test used an RTX 4050, torch 2.14 (CUDA 13.0) and `onnxruntime-gpu` 1.30. Both backends served and agreed with the CPU. Use these rules:

- Install `onnxruntime-gpu` in place of `onnxruntime`.
- Install a torch build with the same CUDA major version as your `onnxruntime-gpu` wheel. Version 1.30 needs CUDA 13. If the versions differ, ONNX Runtime cannot load its CUDA libraries.

The CUDA provider of ONNX Runtime uses TF32 matrix multiplies by default. This moved the output of a small MLP by 4e-4 against the CPU. That error is more than the float32 tolerance of the gate (atol 1e-4, rtol 1e-3). The gate does not see this error, because it runs only on the CPU. With `--backend torch` on CUDA, the output matched the CPU to 5e-8.

## Serve

```bash
downshift serve my_pkg.models:build --port 8000
```

`serve` binds the port first. It then does these steps on a background thread:

1. Load the model.
2. Run the export-and-verify gate. Refer to [The gate](#the-gate-export-and-verify-before-serving).
3. Pick a backend from the verdict.
4. Warm up the backend.
5. Print the banner.

With `--workers N`, the parent process does these steps before uvicorn binds the port. Each worker needs the export that the parent makes. The routes are the same for all backends.

[docs/production.md](docs/production.md) describes how to run downshift in a real deployment. It covers probes, sizing, Kubernetes, the optional API key and the absence of built-in TLS. If you set `DOWNSHIFT_SERVER_API_KEY`, every route except `/health` and `/ready` needs the header `Authorization: Bearer <key>`. If you do not set it, the server logs one warning at startup. The warning says that the endpoints are unauthenticated.

| Route | What it does |
|---|---|
| `POST /predict` | Takes named tensor inputs for any model. Returns `503` (with `Retry-After`) until the model is ready. |
| `POST /predict/graph` | Takes one graph: `x`, `edge_index` and optional `edge_attr`. Returns the same `503` until ready. |
| `GET /health` | Liveness. Returns `200` when the process is up, also during the load. |
| `GET /ready` | Returns `503` (`{"ready": false, "phase": "export"}`) until the model is loaded, exported, verified and warmed up. Returns `200` after that. It does not return to `503` until a restart. |
| `GET /metadata` | Returns the family, backend, full verdict, input names, limits, boot timings and warmup statistics. Returns `503` until ready. |
| `GET /schema` | Returns the name, dtype and shape of each input, and an example body that you can send back unchanged. Returns `503` until ready. |

### What to POST

You do not need to guess the input format. You do not need to know the model. `GET /schema` reads the format from the graph that is running:

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

In `shape`, an integer is a fixed size. A string is a dynamic axis. The string is a name that the adapter chose (`"batch"`, `"seq"`) or the word `"dynamic"`.

For a model that downshift exported, each input also has `bounds`. This is one `{"min", "max"}` pair for each dynamic axis (`null` for a fixed axis). The pair gives the range that the export was traced for. If a request is outside this range, the server returns a readable `400`, for example `input_ids axis 1 is 513; this model accepts 1 to 512`. It does not return an ONNX Runtime error. A bare `.onnx` file with no `--reference` has no bounds. Its axes show `"dynamic"`.

`example_request` is a complete and correct body. If you post it to `/predict` unchanged, the server returns `200`. The values are filler. Copy the names, dtypes and shapes. If the example has more than 256 numbers, `example_request` is `null`, and `notes` gives the shape to build. The full field reference is in [docs/code-docs/http-api.md](docs/code-docs/http-api.md#get-schema).

```bash
curl -s localhost:8000/predict -H 'content-type: application/json' \
  -d '{"inputs": {"x": [[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]]}}'
```

```json
{"outputs": {"output_0": [[0.199, -0.206, 0.561, 0.405]]},
 "shapes": {"output_0": [1, 4]},
 "dtypes": {"output_0": "float32"}}
```

Integer lists become `int64`. All other lists become `float32`. To set the dtype, send `{"data": [...], "dtype": "float16", "shape": [1, 16]}` instead of a bare list.

A bfloat16 or float16 model on the torch backend takes and returns float32. Neither type has a numpy dtype, so the backend does the cast. `/schema` reports float32 for such a model.

For graph models, use this request:

```bash
curl -s localhost:8000/predict/graph -H 'content-type: application/json' \
  -d '{"x": [[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
             [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
             [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]],
       "edge_index": [[0, 1, 2], [1, 2, 0]]}'
```

The response has the same shape as the `/predict` response. It has one output row for each node.

### Binary tensors

The `{"data", "dtype", "shape"}` form also accepts a base64 string in `data`. The string holds the raw bytes of the array: little-endian, C-contiguous, standard alphabet, with padding. The server finds the form from the type. A string is base64. A list is the JSON path above. Existing clients continue to work.

- Any input on `/predict` accepts base64.
- On `/predict/graph`, `x`, `edge_index` and `edge_attr` accept base64.
- Outputs are JSON unless you ask for base64. Add `"output_encoding": "base64"` next to `inputs`, or next to `x` on `/predict/graph`. Each `outputs[name]` then becomes the same `{"data", "dtype", "shape"}` dict.
- `shapes` and `dtypes` are always filled.

A client needs only the standard library and numpy:

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

Send large tensors as safetensors (see below) or as base64. The server parses base64 at about one quarter of the cost of the same tensor as nested lists. A JSON float list holds the GIL while the server parses it. This slows all other work in that process.

To make base64 the default for all responses, start the server with `--output-encoding base64` or set `DOWNSHIFT_OUTPUT_ENCODING=base64`. Clients do not change their requests. An `output_encoding` in a request overrides the default. The `Encoding` row of the banner shows the default in use.

The server validates base64 input. Each failure is a `400`:

- `dtype` and `shape` are mandatory. The server cannot find the shape from the bytes.
- The decoded length must equal `prod(shape) * itemsize`. The error message gives the expected length and the length that arrived.
- Only little-endian data is accepted. The server rejects big-endian dtype strings such as `>f4`.
- `--max-input-bytes` and `DOWNSHIFT_MAX_INPUT_BYTES` set the maximum decoded size (default 256 MiB).

Install `downshift-server[fast]` to get `pybase64`. This SIMD base64 codec is about 12x faster than the standard library in both directions. Without it, the server works the same way, and the banner prints a tip. The server parses request bodies on every route with `orjson`. This is several times faster than the standard library on bodies of MiB size. Clients that send nested lists get a small gain at no cost.

Base64 does not help on small payloads. Below about 100 KiB, the JSON codec is not where the time goes, and base64 gains nothing. The gain increases with the width of the payload. The following numbers come from HTTP tests at concurrency 8 (`bench/REPORT_v0.4.0.md`):

- The best case is `dynamic_batch_cnn` at batch 32. The input is a `(32, 3, 16, 16)` float32 batch. It is 96 KiB of tensor, sent as 498 KiB of nested-list JSON or 128 KiB of base64. Base64 gives 6.65x the throughput and 6.33x the p50.
- `cnn_large` and `mlp_large` at batch 32 gain less. Throughput is 2.65x and 2.10x.
- At batch 1 on a narrow input, the gain can be negative. `mlp_large` gives 0.72x the throughput and 0.75x the p50. The framing overhead of base64 is not free when there is little data to save.
- The hidden-state outputs of `bert_small` were estimated at about 3.5x. The measurement is 1.05x on p50 at batch 32.

The cheapest body is a safetensors file. Send `Content-Type: application/vnd.safetensors` with one tensor for each input name. To get a safetensors response, send `Accept: application/vnd.safetensors`. The server reads the arrays directly from the request bytes. There is no JSON and no base64 framing. For `bert_small`, the p50 is 11.2 ms as JSON and 9.1 ms as safetensors.

The dtype must be the dtype of the model. The server does not cast silently, and it does not accept BF16. `/predict/graph` also accepts a batch in this format. The full rules are in [`docs/code-docs/http-api.md`](docs/code-docs/http-api.md#wire-formats).

```python
from safetensors.numpy import load, save

req = urllib.request.Request(
    "http://localhost:8000/predict",
    save({"x": x}),
    {"content-type": "application/vnd.safetensors", "accept": "application/vnd.safetensors"},
)
y = load(urllib.request.urlopen(req).read())["output_0"]
```

CHANGELOG.md lists the wire-format changes in 0.4 and 0.5.

### Mount it in your own app

You can add downshift to an existing FastAPI service. `downshift.serve.app_for(model, example_inputs)` runs the same export-and-verify gate and warmup as `downshift serve`. It runs them synchronously and returns a plain `FastAPI` app. It starts no subprocess and uses no second port.

```python
from fastapi import FastAPI
from downshift.serve import app_for

app = FastAPI()
app.mount("/model", app_for(my_model, example_inputs))
```

`/model/predict`, `/model/health` and the other routes work the same way as in a standalone `downshift serve`. `app_for` accepts these arguments:

- `ServeOptions` fields as keyword arguments (`backend=`, `warmup=`, `max_concurrency=`, and others).
- `reference=`: a model to verify a pre-built ONNX graph.
- `api_key=`: the default is `DOWNSHIFT_SERVER_API_KEY`.

`app_for()` and `ServeOptions()` read every `DOWNSHIFT_*` environment variable below, not only the CLI. They read the variables once, when `downshift` is imported. For the full signature, refer to [`app_for`](docs/code-docs/python-api.md#app_for). It also describes the `LoadedModel`, `prepare_serving` and `build_app` parts that `app_for` wraps.

### Options

These options change what downshift serves:

- `--backend auto|onnxruntime|torch` (env `DOWNSHIFT_BACKEND`). `auto` follows the verdict. `torch` skips the export. `--backend onnxruntime` on a DEGRADED verdict is an error, unless you also give `--force-onnx`.
- `--force-onnx` serves a DEGRADED graph through ONNX Runtime. The banner shows this.
- `--reference model` verifies a pre-built `.onnx` file against a PyTorch model. Without it, the verdict is UNVERIFIED. This option is only numeric. If the `.onnx` file also needs a tokenizer, add `--tokenizer-from`.
- `--tokenizer-from dir/` loads the tokenizer, the pooling recipe and the label metadata from a Hugging Face repo directory. It works for a `.onnx` file or a PyTorch model. With it, `{"text": ...}` works without a new export. The graph is served as it is. The option does not add pooling to a bare encoder. `/schema` reports the recipe only if the output is already pooled. The option does not depend on `--reference`, and it does not change verification. Both options can name the same directory or different directories. Refer to [Accepted model forms](#accepted-model-forms).
- `--middleware pkg.module:Attr` (repeatable) adds a middleware. It can be a class (pure ASGI, for example `starlette.middleware.gzip:GZipMiddleware`, or a `BaseHTTPMiddleware` subclass) or an `async (request, call_next)` function. If you add no middleware, there is no overhead.
- `--output-encoding json|base64` (env `DOWNSHIFT_OUTPUT_ENCODING`, default `json`) sets the response encoding for requests that do not send `output_encoding`.
- `--max-input-bytes N` (env `DOWNSHIFT_MAX_INPUT_BYTES`, default 256 MiB) sets the maximum decoded size of one base64 input. A larger input gets a `400`.
- `--max-body-bytes N` (env `DOWNSHIFT_MAX_BODY_BYTES`, default 32 MiB) sets the maximum size of a request body. The server checks it before it parses the JSON. A larger body gets a `413`. If `Content-Length` is more than the limit, the server refuses the request and does not read the body. For a chunked body, the server refuses it when the running total passes the limit. There is no separate limit on text length. For a `{"text": ...}` request, this limit is the limit on the text.
- `--max-concurrency N` (env `DOWNSHIFT_MAX_CONCURRENCY`, default 4) sets the number of inferences that run at the same time in each worker process. A dedicated thread pool of this size runs them. Other requests wait in a queue. They do not run inline.
- `--max-queue N` (env `DOWNSHIFT_MAX_QUEUE`, default 64) sets the number of predicts that can wait beyond `--max-concurrency`. When `max-concurrency + max-queue` requests are admitted, a new request gets an immediate `503` with `Retry-After: 1`. It does not join the queue.
- `--request-timeout SECONDS` (env `DOWNSHIFT_REQUEST_TIMEOUT`, default 30, `0` turns it off) sets how long an admitted predict can wait in the queue for its turn. After this time, it gets a `503` and no inference. The server never interrupts a request that is already running.
- `--workers N` (env `DOWNSHIFT_WORKERS`, default 1) starts N uvicorn worker processes. The parent exports and verifies once. Each worker builds its own ONNX Runtime session over that graph, or reloads the model if torch serves it. Each worker then warms up. Memory use and startup time increase with `N`. The parent frees its copy of the model before the workers start.
- `--execution threadpool|inline` (env `DOWNSHIFT_EXECUTION`, default `threadpool`). `threadpool` parses and prepares each request in the prep pool. It runs inference and encoding on an inference thread. This makes two thread hops, and the event loop never blocks. `inline` runs small JSON bodies (Content-Length up to 64 KiB, no `text`) on the event loop, with no hop. Use `inline` only for models that infer in much less than one millisecond. A slower model stalls `/health` and `/ready`.
- `--prep-threads N` (env `DOWNSHIFT_PREP_THREADS`, default `min(4, CPUs)`) sets the size of the prep pool. The prep pool parses, validates and converts request bodies. It is separate from the inference threads.
- `--device auto|cpu|cuda` sets the device. `--device cuda` on a machine without CUDA is an error (exit code 4) on both backends. It does not silently run on the CPU. The numerics gate runs only on the CPU. When you serve on a GPU, the `Verified on` row of the banner shows this.
- `--warmup N` sets the number of inferences before `/ready` changes to `200`.
- `--intra-op-threads N` sets the number of threads inside one operation, for ONNX Runtime or torch. The value 0 uses the default of the backend. More threads decrease the latency of one request, but they decrease throughput under concurrent load.
- `--inter-op-threads N` sets the number of ONNX Runtime threads across operations. The value 0 lets ONNX Runtime choose.
- `--host`, `--port` and `--log-level` set the address and the log level.
- `--access-log/--no-access-log` (default on) controls the log line for each request. Refer to [Logging](#logging).
- `--version` prints the installed version and exits.

#### Choose a concurrency

One inference does not usually fill all the cores. These measurements show the effect of `--max-concurrency`:

- all-MiniLM-L6-v2, batch 8, 8 concurrent clients: 52 req/s at `--max-concurrency 1` and 134 req/s at 4.
- Qwen3-Embedding-0.6B on CPU: 7.2 and 14.6 single queries/s for the same two values.

Each inference in progress holds its own activation memory. If the large requests of a large model run out of memory together, decrease the value.

`--workers` adds processes instead of threads. On 16 logical cores, `--workers 4` increases `clean_mlp` throughput at concurrency 32 by about 3.8x over `--workers 1` (`bench/REPORT_v0.4.0.md`). Each worker has its own thread pool. They do not share one.

#### Request headers

Every response has an `X-Request-Id` header. If the client sent an ID, the server returns it. Otherwise, the server makes one.

Every `/predict` and `/predict/graph` response has a `Server-Timing` header with six stages: `parse`, `prep_wait`, `prep`, `infer_wait`, `infer` and `encode`. Use these to find the cause of a slow request: a wait for a thread, the conversion of the body, or the model run.

A `500` body includes the same `request_id`. Every log line written while the server handled that request includes it too. Use it as a search key when the message says to see the server log.

The server admits a predict, or refuses it with a `503`, before it reads the body. An overloaded server does not buffer or parse a request that it will refuse.

`serve` runs the same gate as `check`. It also accepts `-k/--samples`, `--dynamic`, `--adapter`, `--inputs`, `--model-class`, `--unsafe-load`, `--atol`/`--rtol`, `--seed` and `--vary`. The sections below describe them.

### Logging

All output goes through Python `logging` to one plain-text handler on stdout. This includes the boot banner, the `check` and `export` reports, the startup and error lines of uvicorn, `warnings`, and the request log line of downshift. Each line has the form `time LEVEL logger: message`. If the line belongs to a request, the server adds ` request_id=<id>`. There is no JSON format and no colour.

`--log-level` is `warning` by default for `check`, `export` and `serve`:

- **`warning`.** The banner, the reports, and the `loading`, `will listen on` and `ready in X s` lines go to the `downshift.report` logger, which always prints. Warnings, errors and all `4xx` and `5xx` request lines also print.
- **`info`.** Adds the lines of uvicorn. Adds one line for each request on `downshift.access`, for example `POST /predict 200 12.3 ms request_id=3f9c...`. The server logs probes (`/health`, `/ready`) only at `DEBUG`, so they do not flood the output.
- **`debug`.** Adds tracebacks. For a FAILED verdict, it also adds the export output of torch.

`--no-access-log` removes the request line. The response keeps the request ID. With `check --json` or `export --json`, the verdict JSON is the only output on stdout. The log lines go to stderr.

### Errors

| Status | Cause |
|---|---|
| `400` | The client caused an input problem. Examples: a wrong JSON shape or dtype. An axis outside the range that the model was exported for (`input_ids axis 1 is 65; this model accepts 1 to 64`). An `input_ids` value outside `[0, vocab_size)` for a Hugging Face repo model (the message names the value). An input that the backend rejects (a message such as `input 'x': ...`). The torch backend returns `400` only for errors that the client caused (an index out of range, a shape or dtype mismatch). Any other exception from the model is a `500`. |
| `401` | `DOWNSHIFT_SERVER_API_KEY` is set, and the request has no matching `Authorization: Bearer <key>`. The response has `WWW-Authenticate: Bearer`. `/health` and `/ready` are exempt. |
| `413` | The request body is larger than `--max-body-bytes`. |
| `422` | The JSON is malformed, or a required field is missing. |
| `500` | A server fault. The body is `{"detail": "inference failed on the server; see the server log", "request_id": "..."}`. The server logs the exception with the same `request_id`. It does not return the exception. |
| `503` | The server is at capacity (`max-concurrency + max-queue` predicts are admitted, with `Retry-After: 1`). Or a queued predict waited longer than `--request-timeout`. Or the model is still loading (with `Retry-After: 2`). |

## The gate: export and verify before serving

A successful `torch.onnx.export` does not prove that the graph computes the same function as the model. Before downshift serves anything, it exports the model in memory. It runs `k` random samples (default 8) through PyTorch and through ONNX Runtime. It also varies the dynamic axes, so some samples have shapes that the exporter did not see. Nothing is written to disk. The result is one of four verdicts. The verdict selects the backend:

- **CLEAN**: The model exports, matches PyTorch on every sample, and works on shapes that the trace did not include. Served by ONNX Runtime.
- **DEGRADED**: The model exports without error, but the numbers differ by more than the tolerance on at least one sample. Served by eager PyTorch. `--force-onnx` overrides this.
- **FAILED**: The model does not export. Or it exports, but ONNX Runtime cannot load or run the graph. Served by eager PyTorch. This is a supported path and not an error.
- **UNVERIFIED**: A `.onnx` file with no reference model, or a run with `--no-verify`. Served by ONNX Runtime and labelled as never checked.

A sample passes when every output element satisfies `numpy.allclose`: `abs_err <= atol + rtol * |expected|`. This is the rule that numpy uses.

The default tolerances depend on the narrowest floating dtype in the parameters of the model:

1. If bfloat16 or float16 is present, it wins. These dtypes have the loosest tolerances.
2. float64 wins only if it is the only floating dtype. A model that mixes float32 and float64 has the precision of float32.
3. Otherwise, float32 is the default.

To change the tolerance for one dtype, set a variable such as `DOWNSHIFT_TOL_FLOAT32_ATOL=1e-3` or `DOWNSHIFT_TOL_FLOAT16_RTOL=0.05`. To change it for one run, use `--atol` and `--rtol` on `check`, `export` and `serve`. The `Tolerance` row of the `check` report shows the dtype that selected the default. If a flag overrides it, the row shows `(--atol/--rtol)`.

The gate can also run alone. Use it to gate CI and to write artifacts.

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

These rows need an explanation:

- `Tolerance` shows the dtype that selected the default, or `--atol/--rtol` if a flag overrides it.
- `Worst` (DEGRADED only) is the output element with the largest error across all samples.
- `Samples` lists the input shapes of each sample. This shows the shapes that the check covered.
- `Shape-general` is `yes`, `no`, or `n/a (baseline fails)`. The last value means that the unvaried example did not pass. The check did not evaluate shape generalization in that case.

The exit code is the verdict, so it can gate CI:

| Code | Meaning |
|---|---|
| `0` | CLEAN |
| `1` | FAILED |
| `2` | DEGRADED |
| `3` | UNVERIFIED |
| `4` | Usage error, for example a model that cannot load |
| `5` | Crash |

`--json` prints the full verdict as JSON and nothing else on stdout. The log lines go to stderr:

```bash
downshift check my_pkg.models:build --json -k 16 > verdict.json
```

Useful options:

- `-k/--samples` sets the number of samples.
- `--dynamic "x:0,edge_index:1"` sets the dynamic axes. The default is axis 0 of every input.
- `--adapter generic|pyg|hf` skips the detection of the adapter.
- `--inputs pkg.module:fn` supplies example inputs.
- `--atol` and `--rtol` override the tolerance.
- `--seed N` (default 0) makes the verification samples reproducible.
- `--vary pkg.module:fn` supplies your own `fn(i) -> inputs` instead of the sampler of downshift. `fn(0)` must return the example inputs.

### `export`: write the artifact

```bash
downshift export my_pkg.models:build -o artifacts/ --name classifier
```

This command writes `artifacts/classifier.onnx` and `artifacts/classifier.manifest.json`. The manifest records:

- The SHA-256 of the artifact, and of the source checkpoint (if the model came from a file).
- The versions of torch, onnx and onnxruntime.
- The opset and the observed weight dtype.
- The full verdict, with the numerics report.

Downshift writes a DEGRADED export, because the manifest records how far the numbers differ. It writes nothing for a FAILED export.

`--fp16` casts the model to half precision before the export. This is a plain `.half()` and not quantization. It works on a deep copy, so the model object that you passed in does not change. `--no-verify` skips the numerics check and marks the verdict UNVERIFIED, with a warning.

## Accepted model forms

Downshift accepts three kinds of downloaded artifact. All of them must be on the machine that runs the server:

| Argument | Meaning |
|---|---|
| `model.onnx` | A downloaded or already exported ONNX file. Served as it is. UNVERIFIED unless you give `--reference`. |
| `weights.pt` | A downloaded PyTorch checkpoint (state dict). Needs `--model-class pkg.module:Class`. The extensions `.pth`, `.bin` and `.ckpt` also work. |
| `path/to/repo/dir/` | A downloaded Hugging Face repo: a directory that has a `config.json`. This file identifies the repo. Needs the `[hf]` extra. |

Downshift serves a Hugging Face repo the way its own files say:

- A classifier (`...ForSequenceClassification`) loads with its head. It returns class probabilities.
- A sentence-transformers repo (`modules.json`, `1_Pooling/`) is pooled and normalised inside the exported graph. `output_0` is then one embedding for each text.
- If the repo has tokenizer files, `POST /predict` takes `{"text": ...}`. The maximum length is the length that the model author declared. For the RoBERTa family, this is `max_position_embeddings - (pad_token_id + 1)`. For example, 514 positions serve 512 tokens.
- The server refuses a longer row with a `400`. It never cuts the row. The only other limit on the size of the request is the request body limit.
- If a repo declares no pooling, downshift does not pool, unless you pass `--pooling`.

Refer to [`docs/code-docs/http-api.md`](docs/code-docs/http-api.md#embedding-models).

`downshift serve model.onnx --tokenizer-from path/to/repo/dir/` combines the first two rows. The server serves the `.onnx` file as it is (optimized, and UNVERIFIED unless you also give `--reference`). The repo directory supplies the tokenizer, the pooling recipe and the label metadata that `POST /predict` needs for `{"text": ...}`.

`--tokenizer-from` and `--reference` are independent. You can pass one, the other, both (for the same directory or different directories), or neither. Use this form to serve a model that someone already exported to ONNX (with `optimum`, Olive, or by hand). You do not need to trace it again through `torch.export`, and you keep text input and embedding pooling.

There is one more form. It names a model that the server process can already import. It is not a file on disk:

| Argument | Meaning |
|---|---|
| `pkg.module:attr` | An import spec. `attr` is an `nn.Module` instance or a factory with no arguments. If a `make_inputs` function is in the same module, downshift uses it automatically. Otherwise, pass `--inputs pkg.module:fn`. Downshift finds the module on `sys.path` or in the current directory. For example, `downshift serve my_model:build` works next to `my_model.py`. Name the module and not the file (`my_model`, not `my_model.py`). |

Downshift never contacts the Hugging Face hub. It rejects a bare repo ID such as `bert-base-uncased`. The `hf` adapter loads with `local_files_only=True`. You must download the repo first and pass it as a path:

```bash
huggingface-cli download bert-base-uncased --local-dir ./bert-base-uncased
```

With this rule, a `serve` in an air-gapped environment, or an environment with restricted egress, does not depend on the network at startup. A running server reports the form that it received under `source` on [`GET /schema`](#what-to-post).

Downshift loads checkpoints with `torch.load(weights_only=True)`. A file that holds a pickled full module does not load this way. `--unsafe-load` switches to `weights_only=False`. This runs arbitrary code from the file. Use it only on files that you would run as a script.

## Compatibility matrix

`scripts/gen_matrix.py` generates the matrix from the fixture corpus in `tests/models/`. Each fixture isolates one export hazard. CI regenerates the matrix every week against the current torch and onnxruntime. It opens a PR if the matrix changes. The full file, with versions and a legend, is [docs/compatibility.md](docs/compatibility.md).

<!-- matrix:start -->
| Model | Hazard | Family | Export | Capture | Numerics | Shape-general | Backend |
|---|---|---|---|---|---|---|---|
| `bf16_weights` | bfloat16 weights: ONNX Runtime CPU has no bf16 Gemm kernel | generic | FAILED | strict=False | — | — | torch |
| `broken_factory` | Not an export hazard fixture: raises as soon as it is instantiated | — | skipped (RuntimeError) | — | — | — | — |
| `clean_mlp` | Control fixture: no export hazards | generic | CLEAN | strict=False | 1.8e-07 | ✓ | onnxruntime |
| `custom_autograd` | custom autograd.Function with no symbolic override | generic | CLEAN | strict=False | 1.5e-07 | ✓ | onnxruntime |
| `data_dependent_branch` | data-dependent control flow | generic | FAILED | — | — | — | torch |
| `dict_input` | dataclass container input | generic | CLEAN | strict=False | 1.2e-07 | ✓ | onnxruntime |
| `dropout_model` | stochastic layer | generic | CLEAN | strict=False | 2.4e-07 | ✓ | onnxruntime |
| `dynamic_batch_cnn` | batch-dim generalization | generic | CLEAN | strict=False | 3.0e-08 | ✓ | onnxruntime |
| `gnn_gat` | GNN fixture: 3-layer GAT node classifier | pyg | CLEAN | strict=False | 2.4e-07 | ✓ | onnxruntime |
| `gnn_gcn` | GNN fixture: 2-layer GCN node classifier | pyg | CLEAN | strict=False | 3.6e-07 | ✓ | onnxruntime |
| `gnn_sage` | GNN fixture: 2-layer GraphSAGE node classifier | pyg | CLEAN | strict=False | 2.4e-07 | ✓ | onnxruntime |
| `scatter_include_self_false` | scatter_reduce(include_self=False) has no faithful ONNX translation | generic | DEGRADED | strict=False | 2.2e+00 | — | torch |
| `tied_weights` | tied embedding/output weight (GPT-2/OPT-style) | generic | CLEAN | strict=False | 1.9e-06 | ✓ | onnxruntime |
| `tiny_bert` | HF fixture: a BERT encoder with two layers and random initialization | hf | CLEAN | strict=False | 7.2e-07 | ✓ | onnxruntime |
<!-- matrix:end -->

Read these rows twice:

- `custom_autograd` was expected to fail, but it is CLEAN. `torch.export` traces through a `Function.forward` that has ordinary operations.
- `scatter_include_self_false` was expected to fail with an error. Instead, it exports without an error and returns wrong numbers. Only the numerics check stops this graph from reaching production.
- `bf16_weights` exports without an error, but ONNX Runtime cannot run it, because bfloat16 has no CPU Gemm kernel. This case crashed the tool in the past. It is now a FAILED verdict like all others.

## Scope and non-goals

Downshift serves models that give their answer in one forward pass: tensors in, tensors out. This makes the gate possible, because the gate can compare one pass between the ONNX graph and PyTorch. GNNs, encoders, classifiers, embedders and your own `nn.Module` classes fit. A model that needs a loop around it does not fit.

- **No generation.** There is no `onnxruntime-genai` backend, no OpenAI-compatible endpoint, no KV cache, no sampling loop and no chat template. Use vLLM, TGI or llama.cpp for these. Downshift refuses a decoder-only Hugging Face repo (`...ForCausalLM`), unless the repo has an embedding recipe. Qwen3-Embedding has one (last-token pooling, left padding). Downshift then serves it as an embedder.
- **No remote code.** Downshift never sets `trust_remote_code`. A Hugging Face repo with an `auto_map` in `config.json` does not load. Alibaba GTE v1.5 and other custom architectures are examples. Code that ships in a model repo would break the promise that serving is offline and does only what you asked. Convert the model to a native architecture. Or export it to ONNX yourself and serve it with `--tokenizer-from`.
- **No multimodal Hugging Face models.** Downshift supports text in and one tensor out.
- **No quantization or graph optimization.** This is a decision and not a delay. Use Olive, `onnxruntime.quantization`, or your own script. Then give the result to `downshift serve model.onnx --reference model.pt`. Downshift verifies it against the original weights like any other export. `--fp16` is a cast before the trace. Downshift has no lower precision.
- **No continuous batching and no PagedAttention.** The boot banner is a visual homage to vLLM. This is the only similarity.
- **No dynamic request batching yet.** One request makes one inference.
- **No graph-level batching yet.** `/predict/graph` takes a batch of graphs when the outputs of the model are for each node or each edge. A model with a pooled output of fixed size takes one graph for each request.
- **No Docker image.** Use pip and version pins.
- **No DGL adapter yet.** Downshift supports PyG only.

**Comparison with anydeploy.** `anydeploy` also exports, validates and serves. It has a pass-or-fail validation step and focuses on edge and mobile devices. Downshift is different in three ways:

- The verdict has tiers. DEGRADED is a real state between "works" and "crashes".
- An eager PyTorch fallback is behind the same endpoint. A FAILED or DEGRADED model still serves.
- GNNs (PyTorch Geometric) are a supported family. They have separate dynamic dimensions for nodes and edges.

## Writing your own adapter

An adapter knows one model family. It builds example inputs when the user gave none. It also turns the model and the inputs into a module that `torch.export` can trace, with a flat tensor signature. Implement the `Adapter` protocol from `downshift.adapters.base`:

```python
from downshift.adapters.base import Prepared


class MyAdapter:
    name = "myfamily"

    def matches(self, model, example_inputs) -> bool: ...
    def example_inputs(self, model) -> tuple | None: ...  # None if you can't guess
    def prepare(self, model, example_inputs, axis_max=None) -> Prepared: ...
```

`Prepared` holds these items:

- The module that is ready for export.
- The flat example inputs and their names.
- The `dynamic_shapes` spec for each input.
- An optional `vary_fn(i) -> inputs` that makes the verification samples.
- The family string.

`--seed` reproduces the samples at no cost if `vary_fn` takes its random numbers from the global RNG of torch. The built-in `hf` and `pyg` adapters do this. They run inside the `torch.random.fork_rng()` that `verify()` already uses for each sample. An adapter with its own `random.Random` does not use the seed.

Register the adapter under the `downshift.adapters` entry-point group in your own package:

```toml
[project.entry-points."downshift.adapters"]
myfamily = "my_pkg.adapter:MyAdapter"
```

The entry point names the class. Downshift creates an instance with no arguments. An entry point that names a ready-made instance also works.

Downshift tries adapters from the most specific to the least specific. `generic` is always last. If the optional dependency of an adapter is missing, downshift skips the adapter without a message.

**The plugin contract:** Keep the import of the entry-point module cheap. Downshift imports the module of each registered entry point to build the adapter list. An `ImportError` there means "optional dependency not installed", and downshift skips the adapter without a message. Do the heavy import (your model library, a large parser, and so on) inside `prepare()`. `prepare()` runs only after an adapter matched. Do not put the heavy import at the top of the module that defines your adapter class.

The built-in `hf` and `pyg` adapters are not entry points, because they cannot follow this rule. `transformers` and `torch_geometric` must be imported to define `HFAdapter` and `PyGAdapter`. `downshift.adapters.registry` loads them directly. It loads one only if the module of its family is already in `sys.modules`. Because of this, the search for adapters for a plain PyTorch model never imports either library.

For a one-off adapter that is not worth a package, `--adapter` (and `adapter=` of `check()`) also accepts a bare `.py` file. You do not need an installation or an entry point:

```bash
downshift check my_model:model --adapter path/to/pointcloud_adapter.py
```

To name the class, use `--adapter path/to/pointcloud_adapter.py:MyAdapter`. Downshift creates an instance with no arguments. For a bare `path/to/pointcloud_adapter.py`, downshift looks for a module-level `ADAPTER`. It must name the class (`ADAPTER = MyAdapter`) or be an instance of the class.

## Development

```bash
pip install -e ".[dev,all]"
ruff check .
ruff format --check .
mypy src
pytest
python scripts/gen_matrix.py     # regenerates docs/compatibility.md and this README's table
```

A plain `pytest` run has no coverage gate on your machine. CI runs it with `--cov-fail-under=95`.

## Changelog

Refer to [CHANGELOG.md](CHANGELOG.md).

## License

MIT.
