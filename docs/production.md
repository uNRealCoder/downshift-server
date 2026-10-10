# Running downshift in production

`downshift serve` is a FastAPI app behind uvicorn. It does the export-and-verify gate,
picks a backend, warms it up, and answers `/predict`. It has one fixed API key for
authentication, and no TLS, rate limiting, or process supervision - those are what a real
deployment wraps around it, and this page is about that wrapping.

## Authentication: one API key, no TLS, no rate limiting

Set `DOWNSHIFT_SERVER_API_KEY` and every route except `/health` and `/ready` requires
`Authorization: Bearer <key>`. The comparison is constant-time. Anything else gets a `401`
with `WWW-Authenticate: Bearer` and the body
`{"detail": "Authorization header is not set or incorrect"}`. The two probes stay open so
that Kubernetes never needs the key.

```bash
export DOWNSHIFT_SERVER_API_KEY="$(openssl rand -hex 32)"
downshift serve my_pkg.models:build
curl -s localhost:8000/predict -H "Authorization: Bearer $DOWNSHIFT_SERVER_API_KEY" \
  -H 'content-type: application/json' -d '{"inputs": {"x": [[0.0, 0.1]]}}'
```

With the variable unset, or set to an empty string, the server logs one `WARNING` at
startup that its endpoints are unauthenticated - set the key, or add your own
authentication middleware - and serves everything without a check. An empty string does
not make `Bearer ` a valid credential. `app_for()` and `build_app(..., api_key=...)` read
the same variable by default; pass `api_key=` to override it for one app, or `api_key=None`
to force it off.

One key is not a user system. It is a shared secret, and it travels in a header, so it
belongs behind TLS. The rest is solved at the proxy layer and downshift would only do a
worse version of it. Put a reverse proxy in front (nginx, Envoy, your cloud load balancer,
or a service mesh sidecar) and configure there:

- TLS termination.
- Anything richer than one shared key (mTLS, per-client keys, OAuth).
- Rate limiting and IP allow-lists.
- Request size limits ahead of `--max-body-bytes`, if you want to reject oversized bodies
  before they reach this process at all.

`--middleware pkg.module:Attr` (repeatable) is the escape hatch if you need something
in-process instead - a middleware class, either pure ASGI (cheapest; Starlette's own
`starlette.middleware.gzip:GZipMiddleware` works as is) or a `BaseHTTPMiddleware` subclass,
or an `async (request, call_next)` function, attached in the order given. Middleware you attach runs inside the key check, so
it only sees authenticated requests; a `401` never reaches it.

## `/health` vs `/ready`

- **`/health`** is liveness: `200` as soon as the process is up and the event loop is
  running, whether or not a model has loaded. It never reflects readiness or load, and
  it never blocks behind in-flight predicts (`/health` and `/ready` are `async def`
  routes, so they run on the event loop directly instead of sharing a thread pool with
  predicts). Use it to decide "restart this container", not "send it traffic".
- **`/ready`** is `{"ready": false, "phase": "export"}` with a `503` until the model has
  loaded, exported, verified and warmed up, then `{"ready": true}` with a `200` - and it
  never goes back to `503` on its own. `phase` is the step the loader is in right now: one
  of `load`, `export`, `verify`, `session` or `warmup`. Use `/ready` to decide "send this
  pod traffic".

Boot behaves differently depending on `--workers`:

- **Single worker (default).** `serve` binds the port *before* it loads anything: `/health`
  answers `200` immediately, and `/ready`, `/metadata`, `/schema` and both predict routes
  answer `503` (predicts, `/metadata` and `/schema` with `Retry-After: 2`) while a background thread runs the
  gate. `/ready` flips to `200` in place, with no restart, once that thread lands a
  verdict. If loading fails, the process exits with the same code `check`/`export` would
  use for the same error - your process supervisor sees a crash, not a server stuck
  serving `503` forever.
- **`--workers N`.** The parent process still runs the export-and-verify gate once,
  *before* any worker binds a socket (see "What `--workers` costs" below), because every
  worker needs the artifact it produces. Only after that does uvicorn fork the workers,
  each of which loads its own copy and binds immediately. There is no `503` window here:
  by the time the port is listening, every worker is ready. Size `initialDelaySeconds`
  for the parent's boot time in this mode, not zero.

## What `--workers` costs

Each worker is a separate process with its own memory and, in `--workers N` mode, its own
copy of the model. The parent exports and verifies exactly once and ships the result
(the ONNX graph, or the verdict alone for a torch-backed verdict) to every worker, so
`capture()`/`verify()` do not repeat per worker. An ONNX-served worker builds its own
`InferenceSession` over the parent's graph and warms up; a torch-served worker reloads the
model itself. Both memory and startup time scale with `N`.

### Thread budgeting

With `--workers N` and `--intra-op-threads` left at its default (`0`, "let ONNX Runtime
choose"), downshift resolves it to `max(1, logical_cores // N)` before starting the
workers, so `N` processes split the machine instead of each one claiming every core.
Passing `--intra-op-threads` explicitly always wins over that split. The torch backend
gets the same budget via `torch.set_num_threads`. The banner shows it:

```
Threads     4 intra-op per worker  (16 logical / 4 workers)
```

With a single worker, `--intra-op-threads 0` leaves ONNX Runtime's own default in place
(usually the full core count), which is what you want unless you're also running other
CPU-bound work in the same container.

That split has a measured cost. On a 16-logical-core machine (`bench/REPORT_v0.4.0.md`,
`--intra-op-threads` left at its default), going from 1 worker to 4 raises throughput at
concurrency 32 but lowers it at concurrency 1:

| model         | workers | rps c=1 | rps c=32 |
|---------------|---------|---------|----------|
| `clean_mlp`   | 1       | 1377.8  | 1693.4   |
| `clean_mlp`   | 4       | 1257.8  | 6366.4   |
| `mlp_large`   | 1       | 546.6   | 1311.3   |
| `mlp_large`   | 4       | 503.2   | 2766.2   |
| `bert_small`  | 1       | 84.3    | 97.9     |
| `bert_small`  | 4       | 67.5    | 244.0    |

At concurrency 32, `clean_mlp` gains about 3.8x, `bert_small` about 2.5x, `mlp_large`
about 2.1x - the 4 workers are running 4 requests at once instead of 1, each with its own
4-thread budget instead of contending for it. At concurrency 1, every model is slightly
*worse* with 4 workers: a single request now gets 4 intra-op threads (16 logical / 4
workers) instead of 16, so the one thing running has fewer cores. `--workers` buys
concurrency, not single-request speed - size it against how many predicts you expect in
flight at once, not how fast you want any one of them to return.

## Serving first: where a request's time goes

downshift spends boot time to make serving fast: it exports, verifies, builds an optimised
ONNX Runtime session and warms up before `/ready` turns `200`, and none of that is repeated
per request. Each predict then costs only its own work, in two thread hops:

1. **Event loop:** admit the request (or `503`), check the body size (`413`), read the body.
2. **Prep pool** (`--prep-threads`, default `min(4, CPUs)`): parse and validate the body,
   convert it to arrays (or tokenize text), check shapes and bounds.
3. **Inference pool** (`--max-concurrency`, default 4): run the model, then encode the
   response on the same thread.

`Server-Timing` on every predict response shows each stage (`parse`, `prep_wait`, `prep`,
`infer_wait`, `infer`, `encode`), so the slow part is visible per request, and the access log
line carries the same split as `timings_ms`.

What moves the numbers, roughly in order of payoff (measured on a 16-logical-core machine;
`bench/results/v0.5.0/REPORT.md` has every row):

| Lever | When it helps | Measured |
|---|---|---|
| `--max-concurrency` (default 4) | On by default; lower it to 1-2 for a large model whose concurrent activations run out of memory | all-MiniLM-L6-v2: 282 req/s at 1, 483 at 4 |
| `--workers N` | Many requests in flight; each worker gets its own pools and event loop | `clean_mlp` at concurrency 32: about 3.8x with 4 workers |
| safetensors or base64 bodies | Inputs or outputs past about 100 KiB; `parse` or `encode` dominates | `bert_small` p50 11.2 ms as JSON, 9.1 ms as safetensors; `mlp_large` batch 32: 6.6k to 14.4k rows/s |
| `--execution inline` | Models that infer in well under a millisecond; nothing else helps them | `clean_mlp`: 1498 req/s peak inline against 1030 by default; stalls `/health` for one inference |
| `--intra-op-threads` | Leave it at 0 on one worker. `--workers` already splits the cores | `1` thread: `bert_small` 98 to 48 req/s, `clean_mlp` 903 to 919 (spot check) |

Below about one millisecond of inference, the HTTP stack itself (uvicorn, routing, the
response) is most of a request's latency, 0.63 of 1.00 ms for `clean_mlp` on that machine; `--workers` and
`--execution inline` are the only levers that reduce it. On Linux, uvicorn uses `uvloop` and
`httptools` (installed with `uvicorn[standard]`), which are faster than the Windows event
loop the numbers above were measured on.

## Concurrency, queueing, and what a `503` means

- **`--max-concurrency N`** (default 4) is inferences allowed to run at once *per worker
  process*, via a dedicated thread pool of that size; each one also encodes its response
  before giving the slot back. One inference rarely fills every core, so the default serves
  more requests per second than 1 (all-MiniLM-L6-v2, batch 8, 8 clients: 52 req/s at 1, 134
  at 4; Qwen3-Embedding-0.6B on CPU, one query, 4 clients: 7.2 at 1, 14.6 at 4). Each
  in-flight inference holds its own activation memory, so lower it for a large model whose
  big requests run out of memory together.
- **`--max-queue N`** (default 64) is how many more predicts may wait past
  `--max-concurrency` before a new one is refused outright. Once
  `max-concurrency + max-queue` requests are admitted, the next one gets an immediate
  `503` with `Retry-After: 1` and a body naming how many are running and queued, instead
  of joining an unbounded queue that would eventually time out on the client side anyway.
- **`--request-timeout SECONDS`** (default `30`; `0` turns it off) bounds how long an
  admitted predict may sit in the queue before its turn comes. The clock starts when the
  request is admitted, so it counts time spent queued. Once it has waited that long, it
  gets a `503` instead of starting - a request that has *already started* running is never
  interrupted by this.

The banner's `Capacity` row shows all three:

```
Capacity    4 inferences at a time, 64 queued, 30 s timeout  (--max-concurrency, --max-queue, --request-timeout)
```

A `503` from `/predict` or `/predict/graph` always means one of "server is at capacity",
"waited too long in the queue" or "still loading", never a model error (those are `400` or
`500`). A predict is admitted, or refused, before its body is read, so an overloaded server
answers `503` without buffering or parsing what it is turning away. The capacity and
loading cases carry `Retry-After`, so a well-behaved client backs off instead of retrying
immediately into the same queue.

Two other limits guard the body itself. `--max-body-bytes` (default 32 MiB) is a `413`
once a body passes it: a `Content-Length` over the cap is refused without reading a byte,
and a chunked body is refused as soon as its running total crosses the cap. There is no
separate limit on text length, so for a `{"text": ...}` request this cap is the limit.
`--max-input-bytes` (default 256 MiB) caps one decoded base64 or safetensors tensor and is
a `400`.

## Sizing

Rough memory budget: model size (parameters plus whatever the ONNX Runtime session or
the eager module needs at inference time) times `--workers`, plus a temporary peak during
export - `torch.export` and the ONNX translation hold an extra copy of the graph in
memory while they work. Size for that peak once per boot per worker, not per request.

Boot is deliberately the slow part: downshift verifies before it serves, and it would rather
spend seconds at startup than microseconds on every request. If boot time matters (frequent
restarts, autoscaling), `--export-cache-dir DIR` keeps verified exports so a repeat boot of
the same model skips export and verify. The banner's `Boot` row breaks a boot down so you
don't have to guess which phase is slow:

```
Boot        6.4 s: load 0.3, export 2.8, verify 0.2, session 0.1, warmup 0.1
```

`load` is reading the model off disk or importing it; `export` is `torch.export` plus the
ONNX translation; `verify` is running the numerics gate's samples through both backends;
`session` is building the `InferenceSession` (or nothing extra for eager torch); `warmup`
is the `--warmup N` inferences before `/ready` flips. `/metadata`'s `boot` field carries
the same numbers as JSON. With `--workers N`, the parent pays `export`+`verify` once and
every worker pays its own `load`+`session`+`warmup`. The parent frees its own copy of the
model before the workers start, so it is not held in memory `N+1` times.

## Logs, request ids, and `Server-Timing`

Everything goes through Python `logging` into one plain-text handler on stdout: the boot
banner, the `check` and `export` reports, uvicorn's own startup and error lines, `warnings`,
and downshift's request log line. There is no JSON format, no colour and no box drawing;
point your log shipper at stdout and parse `time LEVEL logger: message`.

- **`--log-level`** defaults to `warning`. The banner, the reports and the `loading` /
  `will listen on` / `ready in X s` lines are logged on `downshift.report`, which always
  prints, so a default `serve` still shows its banner. At `warning` you also get warnings,
  errors and one line for every `4xx` and `5xx` request (a `401`, `413` or `503` is
  visible without turning anything on). `info` adds uvicorn's own lines and one line for
  every request; `debug` adds tracebacks.
- **The request line** is logged on `downshift.access`:
  `POST /predict 200 12.3 ms request_id=3f9c1a7b2e4d5c60`. Probes (`/health`, `/ready`)
  are logged at `DEBUG` only, so a kubelet polling them never fills the log, and a
  `/ready` that answers `503` while the model loads is not a warning.
  `--access-log`/`--no-access-log` (default on) toggles this line; turn it off if your
  proxy already logs every request and you only want downshift's own warnings and errors.
- Every request has an `X-Request-Id`: echoed back if the client sent one, otherwise a
  generated one. It appears on the response, on the request line, and on every other log
  line written while that request was being served. A `500`'s JSON body includes the same
  id under `"request_id"` - "see the server log" has something to search for instead of a
  bare timestamp to correlate by hand.
- `/predict` and `/predict/graph` responses carry `Server-Timing` with six stages: `parse`,
  `prep_wait`, `prep`, `infer_wait`, `infer`, `encode` (see
  [Serving first](#serving-first-where-a-requests-time-goes)) - visible in any browser's
  network tab or with `curl -sD - -o /dev/null`, no extra client code needed. The request log
  line carries the same split as `timings_ms`. Large tensors sent as safetensors or base64
  instead of nested lists cut `parse` and `prep` several-fold, and a JSON float list holds the
  GIL while it is parsed, so it can delay other requests in that process as well.
- With `check --json` or `export --json`, the verdict JSON is the only thing on stdout and
  the log lines go to stderr.

## Configuration through the environment

Every `DOWNSHIFT_*` variable (`DOWNSHIFT_HOST`, `DOWNSHIFT_PORT`, `DOWNSHIFT_DEVICE`,
`DOWNSHIFT_BACKEND`, `DOWNSHIFT_WARMUP`, `DOWNSHIFT_SAMPLES`, `DOWNSHIFT_INTRA_OP_THREADS`,
`DOWNSHIFT_INTER_OP_THREADS`, `DOWNSHIFT_OUTPUT_ENCODING`, `DOWNSHIFT_MAX_INPUT_BYTES`,
`DOWNSHIFT_MAX_BODY_BYTES`, `DOWNSHIFT_MAX_CONCURRENCY`, `DOWNSHIFT_MAX_QUEUE`,
`DOWNSHIFT_REQUEST_TIMEOUT`, `DOWNSHIFT_WORKERS`, `DOWNSHIFT_SERVER_API_KEY`, and the
per-dtype `DOWNSHIFT_TOL_*`) is read once, when `downshift` is imported, and applies to
library use (`app_for()`, `ServeOptions()`) as well as to the CLI. A flag or a keyword
argument wins over the variable, and the variable wins over the default. Setting one means
restarting the process that reads it.

## No egress at startup

`serve` only runs a model that is already downloaded onto this machine: a `.onnx` file, a
`weights.pt` checkpoint, or a directory holding a Hugging Face repo (recognised by the
`config.json` in it) - plus an import spec, which names a model already importable in the
process. A Hugging Face hub id is rejected as a `MODEL` argument, and the `hf` adapter calls
`AutoModel.from_pretrained(..., local_files_only=True)`, so a load never resolves a repo
id or fetches a file.

That matters for a deployment behind an egress policy or in an air-gapped cluster:
startup cannot silently depend on the network, and a missing or half-copied file fails
loudly during load rather than being papered over by a download that works on one node
and not another. A single-worker `serve` whose load fails logs the error and exits with
the same code `check` would use (`4` for a bad model spec or option, `5` for a crash), so
a supervisor sees a failed start instead of a pod stuck at `503`. Bake the repo directory into
the image, or mount it, and pass its path.

## Temporary files

`--workers N` writes the exported ONNX graph (and, if given real example inputs, a
`.npz` of them) to a temp directory so every worker can load it, created with
`tempfile.mkdtemp(prefix="downshift-")`. It's removed when `serve` exits normally
(`finally`) and via `atexit` on most other exits; a container orchestrator that sends
`SIGKILL` skips both, same as it would for any other process's temp files. If you mount
`/tmp` from persistent storage, an occasional stray `downshift-*` directory after a hard
kill is what to look for and clean up.

## Kubernetes

Single worker:

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

`initialDelaySeconds` can stay small in this mode: the port is open immediately, and
`/ready` reports `503` (not connection-refused) for as long as the gate takes, so the
probe doesn't need to guess the boot time.

With `--workers N`, the parent's export-and-verify gate runs before the port opens at
all, so give the liveness probe room for that instead:

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

Read the `Boot` row off a real run of the model you're deploying and set
`initialDelaySeconds` a little above it, on both probes in this mode - the pod isn't
listening at all until the parent's export finishes.
