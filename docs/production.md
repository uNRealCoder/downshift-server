# Running downshift in production

`downshift serve` is a FastAPI app behind uvicorn. It does these jobs:

- It runs the export-and-verify gate.
- It selects a backend.
- It warms up the backend.
- It answers `/predict`.

It has one fixed API key for authentication. It has no TLS, no rate limiting and no process supervision. A real deployment adds these around downshift. This page describes how to do that.

## Authentication: one API key, no TLS, no rate limiting

Set `DOWNSHIFT_SERVER_API_KEY`. Every route except `/health` and `/ready` then needs `Authorization: Bearer <key>`. The comparison takes constant time. Any other request gets a `401` with `WWW-Authenticate: Bearer` and this body: `{"detail": "Authorization header is not set or incorrect"}`. The two probes stay open, so Kubernetes never needs the key.

```bash
export DOWNSHIFT_SERVER_API_KEY="$(openssl rand -hex 32)"
downshift serve my_pkg.models:build
curl -s localhost:8000/predict -H "Authorization: Bearer $DOWNSHIFT_SERVER_API_KEY" \
  -H 'content-type: application/json' -d '{"inputs": {"x": [[0.0, 0.1]]}}'
```

If the variable is unset or empty, the server logs one `WARNING` at startup. The warning says that the endpoints are unauthenticated. To fix this, set the key or add your own authentication middleware. The server then serves all routes without a check. An empty string does not make `Bearer ` a valid credential.

`app_for()` and `build_app(..., api_key=...)` read the same variable by default. To override it for one app, pass `api_key=`. To turn it off, pass `api_key=None`.

One key is not a user system. It is a shared secret that travels in a header. Use it only behind TLS. A proxy layer does the other jobs better than downshift can. Put a reverse proxy in front of downshift (nginx, Envoy, the load balancer of your cloud, or a service mesh sidecar). Configure these items in the proxy:

- TLS termination.
- Authentication that is more than one shared key (mTLS, keys for each client, OAuth).
- Rate limiting and IP allow-lists.
- Limits on request size that are lower than `--max-body-bytes`. Use them to reject oversized bodies before they reach this process.

If you need something in the process, use `--middleware pkg.module:Attr` (repeatable). It accepts these forms:

- A pure ASGI class. This is the cheapest form. `starlette.middleware.gzip:GZipMiddleware` of Starlette works as it is.
- A `BaseHTTPMiddleware` subclass.
- An `async (request, call_next)` function.

Downshift attaches them in the order that you give. Your middleware runs inside the key check. It sees only authenticated requests. A `401` never reaches it.

## `/health` and `/ready`

- **`/health`** is the liveness probe. It returns `200` when the process is up and the event loop runs, with or without a loaded model. It never shows readiness or load. It never waits for predicts that are running. `/health` and `/ready` are `async def` routes. They run directly on the event loop and do not share a thread pool with predicts. Use it to decide "restart this container". Do not use it to decide "send traffic to it".
- **`/ready`** returns `503` with `{"ready": false, "phase": "export"}` until the model is loaded, exported, verified and warmed up. It then returns `200` with `{"ready": true}`. It does not return to `503` by itself. `phase` is the step that the loader is in now: `load`, `export`, `verify`, `session` or `warmup`. Use `/ready` to decide "send traffic to this pod".

The boot is different for different values of `--workers`:

- **One worker (default).** `serve` binds the port before it loads anything.
  - `/health` answers `200` immediately.
  - `/ready`, `/metadata`, `/schema` and both predict routes answer `503` while a background thread runs the gate. Predicts, `/metadata` and `/schema` have `Retry-After: 2`.
  - `/ready` changes to `200` in place, without a restart, when the thread has a verdict.
  - If the load fails, the process exits with the same code that `check` or `export` uses for the same error. Your process supervisor sees a crash. It does not see a server that serves `503` forever.
- **`--workers N`.** The parent process runs the export-and-verify gate once, before any worker binds a socket (see "What `--workers` costs" below). Every worker needs the artifact that the gate makes. After that, uvicorn forks the workers. Each worker loads its own copy and binds immediately. There is no `503` window. When the port is open, every worker is ready. Set `initialDelaySeconds` to the boot time of the parent for this mode. Do not set it to zero.

## What `--workers` costs

Each worker is a separate process with its own memory. In `--workers N` mode, each worker also has its own copy of the model. The parent exports and verifies once. It sends the result to every worker. The result is the ONNX graph, or only the verdict for a torch-backed verdict. `capture()` and `verify()` therefore do not repeat for each worker.

- A worker that serves ONNX builds its own `InferenceSession` over the graph of the parent, and it warms up.
- A worker that serves torch reloads the model itself.

Memory and startup time both increase with `N`.

### Thread budgeting

Assume that you use `--workers N` and leave `--intra-op-threads` at the default (`0`, "let ONNX Runtime choose"). Downshift then sets it to `max(1, logical_cores // N)` before it starts the workers. The `N` processes share the machine. They do not each claim all cores. If you set `--intra-op-threads` explicitly, your value always has priority. The torch backend gets the same budget through `torch.set_num_threads`. The banner shows it:

```
Threads     4 intra-op per worker  (16 logical / 4 workers)
```

With one worker, `--intra-op-threads 0` keeps the default of ONNX Runtime (usually the full number of cores). This is the best choice, unless other CPU-bound work runs in the same container.

This split has a measured cost. The test machine had 16 logical cores (`bench/REPORT_v0.4.0.md`, `--intra-op-threads` at the default). Changing from 1 worker to 4 workers increases the throughput at concurrency 32. It decreases the throughput at concurrency 1:

| model         | workers | rps c=1 | rps c=32 |
|---------------|---------|---------|----------|
| `clean_mlp`   | 1       | 1377.8  | 1693.4   |
| `clean_mlp`   | 4       | 1257.8  | 6366.4   |
| `mlp_large`   | 1       | 546.6   | 1311.3   |
| `mlp_large`   | 4       | 503.2   | 2766.2   |
| `bert_small`  | 1       | 84.3    | 97.9     |
| `bert_small`  | 4       | 67.5    | 244.0    |

At concurrency 32, `clean_mlp` gains about 3.8x, `bert_small` about 2.5x and `mlp_large` about 2.1x. The 4 workers run 4 requests at the same time and not 1. Each worker has its own budget of 4 threads. The workers do not compete for threads.

At concurrency 1, each model is slightly worse with 4 workers. One request now gets 4 intra-op threads (16 logical cores / 4 workers) and not 16. The only request that runs has fewer cores. `--workers` gives more concurrency. It does not give faster single requests. Choose the number of workers from the number of predicts that you expect at the same time. Do not choose it from the speed that you want for one predict.

## Serving first: where the time of a request goes

Downshift uses boot time to make serving fast. Before `/ready` changes to `200`, it exports, verifies, builds an optimised ONNX Runtime session and warms up. None of this repeats for each request. Each predict pays only for its own work, in two thread hops:

1. **Event loop:** Admit the request (or return `503`). Check the body size (`413`). Read the body.
2. **Prep pool** (`--prep-threads`, default `min(4, CPUs)`): Parse and validate the body. Convert it to arrays (or tokenize text). Check the shapes and the bounds.
3. **Inference pool** (`--max-concurrency`, default 4): Run the model. Then encode the response on the same thread.

`Server-Timing` on each predict response shows each stage: `parse`, `prep_wait`, `prep`, `infer_wait`, `infer` and `encode`. You can see the slow part of each request. The access log line has the same split as `timings_ms`.

These items change the numbers. They are in the order of the gain, approximately. The test machine had 16 logical cores. `bench/results/v0.5.0/REPORT.md` has every row.

| Lever | When it helps | Measured |
|---|---|---|
| `--max-concurrency` (default 4) | It is on by default. Decrease it to 1 or 2 for a large model whose concurrent activations run out of memory. | all-MiniLM-L6-v2: 282 req/s at 1, 483 at 4 |
| `--workers N` | Many requests run at the same time. Each worker has its own pools and event loop. | `clean_mlp` at concurrency 32: about 3.8x with 4 workers |
| safetensors or base64 bodies | Inputs or outputs of more than about 100 KiB. `parse` or `encode` is the largest part. | `bert_small` p50 11.2 ms as JSON, 9.1 ms as safetensors. `mlp_large` batch 32: 6.6k to 14.4k rows/s |
| `--execution inline` | Models that infer in much less than one millisecond. Nothing else helps them. | `clean_mlp`: 1498 req/s peak inline against 1030 by default. It stalls `/health` for the time of one inference. |
| `--intra-op-threads` | Keep it at 0 with one worker. `--workers` already splits the cores. | `1` thread: `bert_small` 98 to 48 req/s, `clean_mlp` 903 to 919 (spot check) |

Below about one millisecond of inference, the HTTP stack (uvicorn, routing and the response) is most of the latency of a request. For `clean_mlp` on that machine, it was 0.63 of 1.00 ms. Only `--workers` and `--execution inline` decrease it. On Linux, uvicorn uses `uvloop` and `httptools` (installed with `uvicorn[standard]`). They are faster than the Windows event loop that was used for the numbers above.

## Concurrency, queueing and the meaning of a `503`

- **`--max-concurrency N`** (default 4) is the number of inferences that run at the same time in each worker process. A dedicated thread pool of this size runs them. Each inference also encodes its response before it gives the slot back. One inference does not usually fill all cores. The default therefore serves more requests for each second than 1. Measurements:
  - all-MiniLM-L6-v2, batch 8, 8 clients: 52 req/s at 1, 134 req/s at 4.
  - Qwen3-Embedding-0.6B on CPU, one query, 4 clients: 7.2 at 1, 14.6 at 4.

  Each inference in progress holds its own activation memory. For a large model whose big requests run out of memory together, decrease the value.
- **`--max-queue N`** (default 64) is the number of predicts that can wait beyond `--max-concurrency`. When `max-concurrency + max-queue` requests are admitted, the next request gets an immediate `503` with `Retry-After: 1`. The body says how many requests run and how many wait. The request does not join an unbounded queue. Such a queue would time out on the client side later.
- **`--request-timeout SECONDS`** (default `30`, `0` turns it off) sets how long an admitted predict can wait in the queue for its turn. The clock starts when the request is admitted. It therefore counts the time in the queue. After this time, the request gets a `503` and does not start. This limit never interrupts a request that already runs.

The `Capacity` row of the banner shows all three:

```
Capacity    4 inferences at a time, 64 queued, 30 s timeout  (--max-concurrency, --max-queue, --request-timeout)
```

A `503` from `/predict` or `/predict/graph` always has one of these three meanings:

- The server is at capacity.
- The request waited too long in the queue.
- The server is still loading.

It never means a model error. A model error is a `400` or a `500`. The server admits or refuses a predict before it reads the body. An overloaded server therefore answers `503` without buffering or parsing the body that it refuses. The capacity case and the loading case have `Retry-After`. A client that behaves well waits and does not retry immediately into the same queue.

Two other limits guard the body:

- `--max-body-bytes` (default 32 MiB) gives a `413` when a body is larger. If `Content-Length` is more than the limit, the server refuses the request without a read. For a chunked body, it refuses the request when the running total passes the limit. There is no separate limit on text length. For a `{"text": ...}` request, this limit is the limit.
- `--max-input-bytes` (default 256 MiB) limits one decoded base64 or safetensors tensor. The error is a `400`.

## Sizing

Use this rough memory budget:

- The size of the model (the parameters, and what the ONNX Runtime session or the eager module needs at inference time), multiplied by `--workers`.
- Plus a temporary peak during the export. `torch.export` and the ONNX translation hold an extra copy of the graph in memory while they work. This peak occurs one time for each boot and each worker. It does not occur for each request.

The boot is deliberately the slow part. Downshift verifies before it serves. It uses seconds at startup to save microseconds on each request. If the boot time is important (frequent restarts, autoscaling), use `--export-cache-dir DIR`. The directory keeps verified exports. A repeat boot of the same model then skips the export and the verification.

The `Boot` row of the banner shows the parts of a boot. You do not need to guess which phase is slow:

```
Boot        6.4 s: load 0.3, export 2.8, verify 0.2, session 0.1, warmup 0.1
```

- `load` reads the model from disk or imports it.
- `export` is `torch.export` plus the ONNX translation.
- `verify` runs the samples of the numerics gate through both backends.
- `session` builds the `InferenceSession`. For eager torch, it adds nothing.
- `warmup` runs the `--warmup N` inferences before `/ready` changes.

The `boot` field of `/metadata` has the same numbers as JSON.

With `--workers N`, the parent pays `export` and `verify` one time. Each worker pays its own `load`, `session` and `warmup`. The parent frees its own copy of the model before the workers start. It is therefore not held in memory `N+1` times.

## Logs, request IDs and `Server-Timing`

All output goes through Python `logging` to one plain-text handler on stdout. This includes the boot banner, the `check` and `export` reports, the startup and error lines of uvicorn, `warnings`, and the request log line of downshift. There is no JSON format, no colour and no box drawing. Point your log shipper at stdout, and parse `time LEVEL logger: message`.

- **`--log-level`** is `warning` by default. The banner, the reports, and the `loading`, `will listen on` and `ready in X s` lines go to `downshift.report`. This logger always prints. A default `serve` therefore still shows its banner. At `warning`, you also get warnings, errors, and one line for each `4xx` and `5xx` request. You can see a `401`, `413` or `503` without a change in the settings. `info` adds the lines of uvicorn and one line for each request. `debug` adds tracebacks.
- **The request line** goes to `downshift.access`. Example: `POST /predict 200 12.3 ms request_id=3f9c1a7b2e4d5c60`. Downshift logs the probes (`/health`, `/ready`) only at `DEBUG`. A kubelet that polls them does not fill the log. A `/ready` that answers `503` while the model loads is not a warning. `--access-log`/`--no-access-log` (default on) turns this line on or off. Turn it off if your proxy already logs each request and you want only the warnings and errors of downshift.
- Each request has an `X-Request-Id`. If the client sent one, downshift returns it. Otherwise, downshift makes one. It is in the response, on the request line, and on each other log line that is written while the request is served. The JSON body of a `500` has the same ID under `"request_id"`. "See the server log" then has a search key. You do not need to match a plain timestamp by hand.
- The responses of `/predict` and `/predict/graph` have `Server-Timing` with six stages: `parse`, `prep_wait`, `prep`, `infer_wait`, `infer` and `encode` (see [Serving first](#serving-first-where-the-time-of-a-request-goes)). You can see it in the network tab of each browser, or with `curl -sD - -o /dev/null`. You do not need extra client code. The request log line has the same split as `timings_ms`. Send large tensors as safetensors or base64 and not as nested lists. This decreases `parse` and `prep` by a large factor. A JSON float list holds the GIL while it is parsed. It can also delay other requests in that process.
- With `check --json` or `export --json`, the verdict JSON is the only output on stdout. The log lines go to stderr.

## Configuration through the environment

Downshift reads each `DOWNSHIFT_*` variable once, when `downshift` is imported. The variables are:

- `DOWNSHIFT_HOST`, `DOWNSHIFT_PORT`, `DOWNSHIFT_DEVICE`, `DOWNSHIFT_BACKEND`
- `DOWNSHIFT_WARMUP`, `DOWNSHIFT_SAMPLES`
- `DOWNSHIFT_INTRA_OP_THREADS`, `DOWNSHIFT_INTER_OP_THREADS`
- `DOWNSHIFT_OUTPUT_ENCODING`
- `DOWNSHIFT_MAX_INPUT_BYTES`, `DOWNSHIFT_MAX_BODY_BYTES`
- `DOWNSHIFT_MAX_CONCURRENCY`, `DOWNSHIFT_MAX_QUEUE`, `DOWNSHIFT_REQUEST_TIMEOUT`
- `DOWNSHIFT_WORKERS`, `DOWNSHIFT_SERVER_API_KEY`
- The `DOWNSHIFT_TOL_*` variables for each dtype.

They apply to library use (`app_for()` and `ServeOptions()`) and to the CLI. A flag or a keyword argument has priority over the variable. The variable has priority over the default. To change a variable, restart the process that reads it.

## No egress at startup

`serve` runs only a model that is already downloaded to this machine. It is one of these:

- A `.onnx` file.
- A `weights.pt` checkpoint.
- A directory with a Hugging Face repo. Downshift identifies it by the `config.json` in it.
- An import spec. It names a model that the process can already import.

Downshift rejects a Hugging Face hub id as a `MODEL` argument. The `hf` adapter calls `AutoModel.from_pretrained(..., local_files_only=True)`. A load therefore never resolves a repo ID or fetches a file.

This is important for a deployment behind an egress policy, or in an air-gapped cluster:

- Startup cannot depend on the network without a message.
- A missing or half-copied file fails with a clear error during the load. A download that works on one node and not on another does not hide it.
- If the load of a single-worker `serve` fails, it logs the error and exits with the same code that `check` uses. The code is `4` for a bad model spec or option, and `5` for a crash. A supervisor sees a failed start. It does not see a pod that is stuck at `503`.

Put the repo directory in the image, or mount it. Pass its path.

## Temporary files

`--workers N` writes the exported ONNX graph to a temporary directory, so that each worker can load it. If you gave real example inputs, it also writes a `.npz` file of them. The directory comes from `tempfile.mkdtemp(prefix="downshift-")`.

Downshift removes the directory when `serve` exits normally (`finally`). It also removes it through `atexit` on most other exits. A container orchestrator that sends `SIGKILL` skips both. This is the same for the temporary files of all other processes. If you mount `/tmp` from persistent storage, look for a stray `downshift-*` directory after a hard kill, and remove it.

## Kubernetes

For one worker:

```yaml
livenessProbe:
  httpGet: {path: /health, port: 8000}
  initialDelaySeconds: 1
readinessProbe:
  httpGet: {path: /ready, port: 8000}
  initialDelaySeconds: 1
  periodSeconds: 2
resources:
  requests: {cpu: "1", memory: 512Mi}
```

`initialDelaySeconds` can stay small in this mode. The port is open immediately. `/ready` reports `503` (and not connection-refused) for the time that the gate needs. The probe therefore does not need to guess the boot time.

With `--workers N`, the export-and-verify gate of the parent runs before the port opens. Give the liveness probe time for this:

```yaml
livenessProbe:
  httpGet: {path: /health, port: 8000}
  initialDelaySeconds: 15  # >= the banner's Boot row for the slowest model you serve
readinessProbe:
  httpGet: {path: /ready, port: 8000}
  initialDelaySeconds: 15
  periodSeconds: 2
resources:
  requests: {cpu: "4", memory: 2Gi}  # scale with --workers; see Sizing above
```

Read the `Boot` row from a real run of the model that you deploy. Set `initialDelaySeconds` a little above it, on both probes in this mode. The pod does not listen at all until the export of the parent finishes.
