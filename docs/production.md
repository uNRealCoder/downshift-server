# Running downshift in production

`downshift serve` is a FastAPI app behind uvicorn. It runs the export-and-verify gate, selects a backend, warms it up and answers `/predict`. It has one fixed API key for authentication and optional HTTPS with your certificate. It has no rate limiting and no process supervision. A real deployment adds these around downshift. This page tells you how.

Each flag on this page is in [`cli.md`](cli.md#serve), with its environment variable and default. A flag has priority over its variable. The variable has priority over the default. Downshift reads the variables one time, at start.

## Authentication

Set `DOWNSHIFT_SERVER_API_KEY`. Each route except `/health` and `/ready` then needs `Authorization: Bearer <key>`. The comparison takes constant time. Any other request gets a `401` with `WWW-Authenticate: Bearer`. The probes stay open, so Kubernetes never needs the key.

```bash
export DOWNSHIFT_SERVER_API_KEY="$(openssl rand -hex 32)"
downshift serve my_pkg.models:build
curl -s localhost:8000/predict -H "Authorization: Bearer $DOWNSHIFT_SERVER_API_KEY" \
  -H 'content-type: application/json' -d '{"inputs": {"x": [[0.0, 0.1]]}}'
```

If the variable is unset or empty, the server logs one `WARNING` at start and checks nothing. An empty string does not make `Bearer ` a valid credential. `app_for()` and `build_app()` read the same variable. To override it for one app, pass `api_key=`. To turn it off, pass `api_key=None`.

One key is a shared secret in a header. Use it only with TLS: the `--ssl-certfile` of downshift, or a proxy. Put a reverse proxy (nginx, Envoy, the load balancer of your cloud, a service mesh) in front of downshift for:

- TLS termination with automatic renewal of the certificate.
- Authentication that is more than one key (mTLS, a key for each client, OAuth).
- Rate limiting and IP allow-lists.
- Size limits that are lower than `--max-body-bytes`.

To add code in the process, give `--middleware pkg.module:Attr` (repeatable). It takes a pure ASGI class (the cheapest form, for example `starlette.middleware.gzip:GZipMiddleware`), a `BaseHTTPMiddleware` subclass, or an `async (request, call_next)` function. Downshift attaches them in the order that you give. They run inside the key check, so a `401` never reaches them.

## HTTPS

1. Get a PEM certificate and its PEM private key.
2. Give the two files to `serve`:

   ```bash
   downshift serve my_pkg.models:build --host 0.0.0.0 --port 8443 \
     --ssl-certfile /etc/downshift/cert.pem --ssl-keyfile /etc/downshift/key.pem
   ```

3. If the key is encrypted, set `DOWNSHIFT_SSL_KEYFILE_PASSWORD`. There is no flag, because a flag shows in the process list.
4. Make sure that the `Endpoint` row of the banner shows `https://`.

Downshift loads the files before the model. A missing file, a key that does not agree with the certificate, or a wrong password stops `serve` with exit code `4` and a message that gives the next step. Downshift never asks for a password on the terminal.

Downshift uses the TLS settings of `ssl.create_default_context`: TLS 1.2 or later, and the cipher list of the Python standard library. It does not use the default cipher string of uvicorn, which also enables NULL and anonymous ciphers. `--workers N` works with HTTPS.

Downshift does not renew certificates, does not do ACME requests or client certificates, and does not redirect HTTP to HTTPS. It reads the files one time, so restart it after a renewal. If a proxy terminates TLS, do not give `--ssl-certfile`. Keep downshift on plain HTTP on `127.0.0.1` or a private network.

## `/health` and `/ready`

- **`/health`** is the liveness probe. It gives `200` when the process is up, also while the model loads. It runs on the event loop and never waits for predicts. Use it to decide "restart this container".
- **`/ready`** gives `503` with `{"ready": false, "phase": "export"}` until the model is loaded, exported, verified and warmed up. It then gives `200` and does not change back. `phase` is `load`, `export`, `verify`, `session` or `warmup`. Use it to decide "send traffic to this pod".

With one worker, `serve` binds the port before it loads anything. `/ready`, `/metadata`, `/schema` and the predict routes give `503` while a background thread runs the gate. If the load fails, the process exits with the code that `check` gives for the same error. A supervisor sees a failed start, not a server that gives `503` forever.

With `--workers N`, the parent runs the gate one time before the port opens. Each worker then binds, loads the export of the parent and warms up. `/ready` of a worker gives `503` for this short time. If a worker cannot load, the whole server stops.

## Workers and threads

Each worker is a process with its own copy of the model, its own pools and its own result cache. The parent exports and verifies one time and sends the result to each worker: the ONNX graph, or only the verdict if torch serves the model. A worker that serves ONNX builds its own session over that graph. A worker that serves torch loads the model again. Memory and start time increase with `N`.

If you leave `--intra-op-threads` at `0`, `--workers N` sets it to `max(1, CPUs // N)`, so the workers do not all claim all cores. A value that you set has priority. The banner shows the split:

```
Threads     4 intra-op per worker  (16 logical / 4 workers)
```

`--workers` gives more requests at the same time. It does not make one request faster. In the 0.4.0 benchmark on 16 logical cores (`bench/REPORT_v0.4.0.md`), 4 workers against 1 worker:

| model | workers | rps c=1 | rps c=32 |
|---|---|---|---|
| `clean_mlp` | 1 | 1377.8 | 1693.4 |
| `clean_mlp` | 4 | 1257.8 | 6366.4 |
| `mlp_large` | 1 | 546.6 | 1311.3 |
| `mlp_large` | 4 | 503.2 | 2766.2 |
| `bert_small` | 1 | 84.3 | 97.9 |
| `bert_small` | 4 | 67.5 | 244.0 |

At concurrency 1, each request gets 4 threads and not 16, so it is a little slower. Choose `N` from the number of predicts that you expect at the same time.

`--workers N` writes the graph (and a `.npz` file of the example inputs) to a `downshift-*` directory from `tempfile.mkdtemp`. Downshift removes it when `serve` exits, and through `atexit` on most other exits. A `SIGKILL` skips both. If `/tmp` is on persistent storage, remove a stray `downshift-*` directory after a hard kill.

## Where the time of a request goes

Downshift uses the boot to make requests fast. Each predict pays only for its own work, in two thread hops:

1. **Event loop:** admit the request or give `503`, check the body size (`413`), read the body.
2. **Prep pool** (`--prep-threads`): parse and validate the body, convert it to arrays or tokenize the text, check the shapes and bounds.
3. **Inference pool** (`--max-concurrency`): run the model, then encode the response on the same thread.

The `Server-Timing` header of each predict response gives each stage: `parse`, `prep_wait`, `prep`, `infer_wait`, `infer` and `encode`, and `cache` if the result cache is on. You can see it with `curl -sD - -o /dev/null` or in the network tab of a browser. The access log line has the same values as `timings_ms`.

These levers change the numbers, approximately in the order of the gain (16 logical cores, `bench/results/v0.5.0/REPORT.md`):

| Lever | When it helps | Measured |
|---|---|---|
| `--max-concurrency` (default 4) | On by default. Decrease it to 1 or 2 for a large model whose concurrent activations run out of memory. | all-MiniLM-L6-v2: 282 req/s at 1, 483 at 4 |
| `--workers N` | Many requests at the same time. | `clean_mlp` at concurrency 32: about 3.8x with 4 workers |
| safetensors or base64 bodies | Inputs or outputs of more than about 100 KiB, where `parse` or `encode` is the largest stage. A JSON float list also holds the GIL while it is parsed. | `bert_small` p50 11.2 ms as JSON, 9.1 ms as safetensors |
| `--execution inline` | Models that infer in much less than 1 ms. It stalls `/health` for the time of one inference. | `clean_mlp`: 1498 req/s inline against 1030 by default |
| `--intra-op-threads` | Keep it at 0 with one worker. | `1` thread: `bert_small` 98 to 48 req/s |

Below about 1 ms of inference, the HTTP stack is most of the latency. Only `--workers` and `--execution inline` decrease it.

## Capacity and the meaning of a `503`

- **`--max-concurrency N`** (default 4) is the number of inferences at the same time in each worker. One inference does not usually fill all cores: all-MiniLM-L6-v2 (batch 8, 8 clients) gave 52 req/s at 1 and 134 req/s at 4. Each inference holds its own activation memory.
- **`--max-queue N`** (default 64) is the number of predicts that can wait beyond `--max-concurrency`. The next one gets an immediate `503` with `Retry-After: 1`. It counts requests, not rows.
- **`--request-timeout SECONDS`** (default 30) is how long an admitted predict can wait before its inference starts, also for a row that another request computes. After that, it gets a `503`. A running inference is never stopped.

```
Capacity    4 inferences at a time, 4 prep threads, 64 queued, 30 s timeout  (--max-concurrency, --prep-threads, --max-queue, --request-timeout)
```

A `503` from a predict route means one of three things: the server is at capacity, the request waited too long, or the model still loads. It never means a model error, which is a `400` or a `500`. The server admits or refuses a predict before it reads the body, so an overloaded server does not buffer what it refuses.

`--max-body-bytes` (default 32 MiB) gives a `413` for a larger body, before a read if `Content-Length` shows the size. `--max-input-bytes` (default 256 MiB) limits one decoded base64 or safetensors tensor (`400`).

## Result cache

The result cache keeps the output of single rows. A row that arrives again skips the model. This helps a server that gets the same query many times, for example the same RAG question.

- It is on by default and keeps 100 rows in each worker. `--result-cache-rows N` changes the number. `0` turns it off.
- A row enters the cache on its second arrival. A bulk load of rows that arrive one time does not push out the frequent rows.
- It needs a model with independent rows: each row gives the same output with any other rows in its batch. Downshift checks this at boot with two different rows, alone and together, on both backends. The check finds rows that change each other by more than the tolerance. It cannot prove that a small effect is absent. The `Rows` line of the banner and `verdict.row_independent` in `/metadata` give the result. For a model such as `x - x.mean(0)`, or a model with an output for each token, the cache is off and the banner says why. If you set `--result-cache-rows` for such a model, `serve` stops with exit code `4`.
- If two requests ask for the same new row at the same time, the model runs one time and the second request waits. If the first request fails or leaves, the second one computes the row.
- It counts rows, not bytes. Its memory is the number of rows times the size of one output row.
- `/predict/graph` does not use it.
- Each response has `X-Downshift-Cache: hit`, `partial` or `miss`. `Cache-Control: no-cache` skips the lookup for one request. The server still stores the rows.

**Warning for a server with more than one tenant.** A hit is faster than a miss. A client can use the time to learn that another client sent the same text. If your clients must not learn this about each other, set `--result-cache-rows 0`.

A hit gives exactly the values of the request that stored the row. A miss for the same row in another batch can differ in the last bits, inside the tolerance of the verdict.

## Export cache

The boot is the slow part on purpose. If the boot time is important (frequent restarts, autoscaling), give `--export-cache-dir DIR` (or `DOWNSHIFT_EXPORT_CACHE_DIR`). The directory must exist and be writable.

- Downshift saves each CLEAN or DEGRADED export in `DIR/<key>/`: the `.onnx` file, its manifest, the verdict and the example inputs. A later boot of the same model with the same options skips the export and the verification. The `Boot` row of the banner then says `reused from --export-cache-dir`.
- The key covers the weights, the source code of the model classes, the options of the gate, and the versions of downshift, torch, onnx, onnxscript, onnxruntime, transformers and torch_geometric. A change to any of them misses.
- For a Hugging Face repo directory, the key comes from the content of the files, so pods that share a volume share the cache. A hit that ONNX Runtime serves does not load the weights.
- A read checks the hashes of the graph against the manifest. A bad entry gives one warning and counts as a miss. A write goes to a temporary directory first, so a reader never sees half an entry.
- Downshift never removes entries. To clear the cache, delete the directory.

Without the flag, nothing goes to disk.

## Sizing

Use this rough memory budget: the size of the model (the parameters, plus what the session or the eager module needs at inference time), times `--workers`, plus the result cache. The export holds an extra copy of the graph for a short time, one time for each boot.

The `Boot` row of the banner shows the phases of a boot:

```
Boot        6.4 s: load 0.3, export 2.8, verify 0.2, session 0.1, warmup 0.1
```

`load` reads or imports the model. `export` is `torch.export` plus the ONNX translation. `verify` runs the samples through both backends. `session` builds the backend. `warmup` runs `--warmup N` inferences. The `boot` field of `/metadata` has the same numbers. With `--workers N`, the parent pays `export` and `verify` one time, and each worker pays its own `load`, `session` and `warmup`.

## Logs and request IDs

All output goes through Python `logging` to stdout, as plain text in the form `time LEVEL logger: message`. Point your log shipper at stdout. [`cli.md`](cli.md#logging) gives what each `--log-level` prints.

- The request line goes to `downshift.access`, for example `POST /predict 200 12.3 ms request_id=3f9c1a7b2e4d5c60`. A `4xx` or `5xx` line prints at the default level. The probes log only at `debug`. Give `--no-access-log` if your proxy already logs each request.
- Each response has `X-Request-Id`. If the client sent one, downshift returns it. The ID is on each log line that the request writes. A `500` body has the same `request_id`, so "see the server log" has a search key.

## No egress at start

`serve` runs only a model that is already on this machine. It rejects a Hugging Face hub id, and the `hf` adapter loads with `local_files_only=True`. Behind an egress policy or in an air-gapped cluster:

- Start does not depend on the network.
- A missing or half-copied file fails with a clear error during the load.
- A failed load exits with code `4` (a bad model or option) or `5` (a crash).

Put the model in the image, or mount it, and give its path.

## GPU

A person tested `--device cuda` one time, by hand: an RTX 4050, torch 2.14 (CUDA 13.0) and `onnxruntime-gpu` 1.30. Both backends served and agreed with the CPU.

- Install `onnxruntime-gpu` in place of `onnxruntime`.
- Install a torch build with the same CUDA major version as the `onnxruntime-gpu` wheel (1.30 needs CUDA 13). If the versions differ, ONNX Runtime cannot load its CUDA libraries, and `serve --device cuda` stops with exit code `4`.

The gate runs only on the CPU. The CUDA provider of ONNX Runtime uses TF32 matrix multiplies by default. In the test, this moved the output of a small MLP by 4e-4 against the CPU, which is more than the float32 tolerance. The gate does not see this. With `--backend torch` on CUDA, the output matched the CPU to 5e-8. The `Verified on` row of the banner says when the numerics did not cover the serving device.

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

The port opens at once, and `/ready` gives `503` (not connection-refused) while the gate runs. The probe does not need to guess the boot time.

With `--workers N`, the gate of the parent runs before the port opens. Read the `Boot` row from a real run of your model, and set `initialDelaySeconds` a little above it on both probes:

```yaml
livenessProbe:
  httpGet: {path: /health, port: 8000}
  initialDelaySeconds: 15  # >= the Boot row of the slowest model that you serve
readinessProbe:
  httpGet: {path: /ready, port: 8000}
  initialDelaySeconds: 15
  periodSeconds: 2
resources:
  requests: {cpu: "4", memory: 2Gi}  # scale with --workers
```

If `serve` uses `--ssl-certfile`, add `scheme: HTTPS` to the `httpGet` of both probes. The kubelet does not check the certificate.
