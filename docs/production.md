# Running downshift in production

`downshift serve` is a FastAPI app behind uvicorn. It does the export-and-verify gate,
picks a backend, warms it up, and answers `/predict`. It does not do auth, TLS, rate
limiting, or process supervision - those are what a real deployment wraps around it, and
this page is about that wrapping.

## No auth, no TLS, no rate limiting

None of those are in scope, on purpose: they are solved problems at the proxy layer and
downshift would only do a worse version of them. Put a reverse proxy in front
(nginx, Envoy, your cloud load balancer, or a service mesh sidecar) and configure there:

- TLS termination.
- Authentication (mTLS, an API key header, OAuth) - downshift trusts whatever reaches it.
- Rate limiting and IP allow-lists.
- Request size limits ahead of `--max-body-bytes`, if you want to reject oversized bodies
  before they reach this process at all.

`--middleware pkg.module:Attr` (repeatable) is the escape hatch if you need something
in-process instead - a `BaseHTTPMiddleware` subclass or an `async (request, call_next)`
function, attached in the order given.

## `/health` vs `/ready`

- **`/health`** is liveness: `200` as soon as the process is up and the event loop is
  running, whether or not a model has loaded. It never reflects readiness or load, and
  it never blocks behind in-flight predicts (`/health` and `/ready` are `async def`
  routes, so they run on the event loop directly instead of sharing a thread pool with
  predicts). Use it to decide "restart this container", not "send it traffic".
- **`/ready`** is `{"ready": false, "phase": "export"}` with a `503` until the model has
  loaded, exported, verified and warmed up, then `{"ready": true}` with a `200` - and it
  never goes back to `503` on its own. Use it to decide "send this pod traffic".

Boot behaves differently depending on `--workers`:

- **Single worker (default).** `serve` binds the port *before* it loads anything: `/health`
  answers `200` immediately, and `/ready`, `/metadata` and both predict routes answer
  `503` (predicts and `/metadata` with `Retry-After: 2`) while a background thread runs the
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
`capture()`/`verify()` do not repeat per worker - but loading the model itself, building
an `InferenceSession` (or an eager `nn.Module`), and warmup all still happen once per
worker, so both memory and startup time scale with `N`.

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

## Concurrency, queueing, and what a `503` means

- **`--max-concurrency N`** (default 1) is inferences allowed to run at once *per worker
  process*, via a dedicated thread pool of that size. One inference already uses every
  core it's given through ONNX Runtime's intra-op threads, so on CPU raising this rarely
  adds throughput - it mostly adds contention between inferences sharing the same cores.
  Use `--workers` for more processes instead, not a bigger `--max-concurrency`.
- **`--max-queue N`** (default 64) is how many more predicts may wait past
  `--max-concurrency` before a new one is refused outright. Once
  `max-concurrency + max-queue` requests are admitted, the next one gets an immediate
  `503` with `Retry-After: 1` and a body naming how many are running and queued, instead
  of joining an unbounded queue that would eventually time out on the client side anyway.
- **`--request-timeout SECONDS`** (default `0`, off) bounds how long an admitted predict
  may sit in the queue before its turn comes. Once it has waited that long, it gets a
  `503` instead of starting - a request that has *already started* running is never
  interrupted by this.

The banner's `Queue` row shows both:

```
Queue       64 waiting max, no timeout  (--max-queue, --request-timeout)
```

A `503` from `/predict` or `/predict/graph` always means one of "server is at capacity"
or "waited too long in the queue", never a model error (those are `400` or `500`). Both
carry `Retry-After`, so a well-behaved client backs off instead of retrying immediately
into the same queue.

## Sizing

Rough memory budget: model size (parameters plus whatever the ONNX Runtime session or
the eager module needs at inference time) times `--workers`, plus a temporary peak during
export - `torch.export` and the ONNX translation hold an extra copy of the graph in
memory while they work. Size for that peak once per boot per worker, not per request.

Boot time is the number people ask about first; the banner's `Boot` row breaks it down so
you don't have to guess which phase is slow:

```
Boot        6.4 s: load 0.3, export 2.8, verify 0.2, session 0.1, warmup 0.1
```

`load` is reading the model off disk or importing it; `export` is `torch.export` plus the
ONNX translation; `verify` is running the numerics gate's samples through both backends;
`session` is building the `InferenceSession` (or nothing extra for eager torch); `warmup`
is the `--warmup N` inferences before `/ready` flips. `/metadata`'s `boot` field carries
the same numbers as JSON. With `--workers N`, the parent pays `export`+`verify` once and
every worker pays its own `load`+`session`+`warmup`.

## Logs, request ids, and `Server-Timing`

- `--log-format json` writes one JSON object per line to stderr (`time`, `level`,
  `logger`, `message`, and `exc_info` on an exception) instead of the default plain-text
  formatter; point your log shipper at stderr either way.
- Every response carries an `X-Request-Id` header: echoed back if the client sent one,
  otherwise a generated one. A `500`'s JSON body includes the same id under
  `"request_id"`, and the server log line for that failure names it too - "see the server
  log" now has something to search for instead of a bare timestamp to correlate by hand.
- `/predict` and `/predict/graph` responses carry
  `Server-Timing: codec;dur=<ms>, infer;dur=<ms>`, splitting request/response conversion
  time from the backend call itself - visible in any browser's network tab or with
  `curl -sD - -o /dev/null`, no extra client code needed.
- `--access-log`/`--no-access-log` (default on) toggles uvicorn's own per-request access
  log line; turn it off if your proxy already logs every request and you only want
  downshift's own warnings and errors.

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
