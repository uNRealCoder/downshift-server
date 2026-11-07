# Running downshift in production

`downshift serve` is a FastAPI app behind uvicorn. It runs the export-and-verify gate, selects a backend, warms it up and answers `/predict`. It has one fixed API key for authentication and optional HTTPS with your certificate. It has no rate limiting and no process supervision. A real deployment adds these around downshift. This page tells you how.

Each flag on this page is in [`cli.md`](cli.md#serve), with its environment variable and default. A flag has priority over its variable. The variable has priority over the default. Downshift reads the variables one time, at start.

## Authentication

Set `DOWNSHIFT_API_KEY`. Each route except `/health` and `/ready` then needs `Authorization: Bearer <key>`. The comparison takes constant time. Any other request gets a `401` with `WWW-Authenticate: Bearer`. The probes stay open, so Kubernetes never needs the key.

```bash
export DOWNSHIFT_API_KEY="$(openssl rand -hex 32)"
downshift serve my_pkg.models:build
curl -s localhost:8000/predict -H "Authorization: Bearer $DOWNSHIFT_API_KEY" \
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

## Behind a proxy

If a reverse proxy sends requests to downshift, tell downshift what the proxy does:

- **Root path.** If the proxy removes a path prefix (it sends `/models/bert/predict` to downshift as `/predict`), give `--root-path /models/bert`. The links of `GET /` and the example URL and `curl` of `GET /schema` then include the prefix.
- **Forwarded IPs.** Downshift trusts `X-Forwarded-For` and `X-Forwarded-Proto` only from `127.0.0.1`. If the proxy has another address, give `--forwarded-allow-ips 10.0.0.5` (comma-separated IPs, or `*` if only the proxy can reach the port). The example URL of `GET /schema` then uses `https` when the proxy gets the request on HTTPS, and the access log has the IP of the client.
- **HTTPS at the proxy.** Let the proxy end TLS. Do not give `--ssl-certfile`. Keep downshift on plain HTTP on a private network.

## Shut down

On `SIGTERM` (or `SIGINT`), the server does these steps:

1. `/ready` gives `503` with `{"ready": false, "phase": "stopping"}` at once. The load balancer sends no new requests.
2. The server stops new connections. A request on an open connection still runs, `/predict` too.
3. The server waits up to `--timeout-graceful-shutdown SECONDS` (default `30`) for the running requests. `0` means wait without a limit.
4. The server stops.

With `--workers N`, each worker does the same steps for itself.

In Kubernetes, set `terminationGracePeriodSeconds` above `--timeout-graceful-shutdown`. Also set it above `--request-timeout`, so a slow request can finish. If the grace period is shorter, the kubelet sends `SIGKILL` and the running requests fail.

## `/health` and `/ready`

- **`/health`** is the liveness probe. It gives `200` when the process is up, also while the model loads. It runs on the event loop and never waits for predicts. Use it to decide "restart this container".
- **`/ready`** gives `503` with `{"ready": false, "phase": "export"}` until the model is loaded, exported, verified and warmed up. It then gives `200`. It gives `503` again with `"phase": "stopping"` when the server shuts down (refer to "Shut down"). `phase` is `load`, `export`, `verify`, `session` or `warmup`. Use it to decide "send traffic to this pod".

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

With `--server-timing` (or `DOWNSHIFT_SERVER_TIMING=1`), the `Server-Timing` header of each predict response gives each stage: `parse`, `prep_wait`, `prep`, `infer_wait`, `infer` and `encode`, and `cache` if the result cache is on. You can see it with `curl -sD - -o /dev/null` or in the network tab of a browser. The header is off by default, because the timings and a fast cache hit tell a client about the server and about the requests of other clients. The access log line has the same values as `timings_ms`, with the header on or off.

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
- **`--request-timeout SECONDS`** (default 30) is the time limit for the whole request, from its admission. The limit includes the body read, the queue, the inference and the encoding. If the request is not complete at the limit, the client gets a `503` with the code `request_timeout`, and the slot is free. The thread of an inference that already runs is not stopped. It ends by itself. `0` means no limit.

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
- Each response has `X-Downshift-Cache: hit`, `partial` or `miss`. To use a different name, set `DOWNSHIFT_CACHE_HEADER`. `Cache-Control: no-cache` skips the lookup for one request. The server still stores the rows.

**Warning for a server with more than one tenant.** A hit is faster than a miss. A client can use the time to learn that another client sent the same text. If your clients must not learn this about each other, set `--result-cache-rows 0`.

A hit gives exactly the values of the request that stored the row. A miss for the same row in another batch can differ in the last bits, inside the tolerance of the verdict.

## Export cache

The boot is the slow part on purpose. If the boot time is important (frequent restarts, autoscaling), give `--export-cache-dir DIR` (or `DOWNSHIFT_EXPORT_CACHE_DIR`). The directory must exist and be writable.

- Downshift saves each CLEAN or DEGRADED export in `DIR/<key>/`: the `.onnx` file, its manifest, the verdict and the example inputs. A later boot of the same model with the same options skips the export and the verification. The `Boot` row of the banner then says `reused from --export-cache-dir`.
- The key covers the weights, the source code of the model classes, the options of the gate, and the versions of downshift, torch, onnx, onnxscript, onnxruntime, transformers and torch_geometric. A change to any of them misses.
- For a Hugging Face repo directory, the key comes from the content of the files, so pods that share a volume share the cache. A hit that ONNX Runtime serves does not load the weights.
- A read checks the hashes of the graph against the manifest. A bad entry, or an entry with another `format_version`, gives one warning and counts as a miss. A write goes to a temporary directory first, so a reader never sees half an entry.
- Downshift never removes entries. To clear the cache, delete the directory.

Without the flag, nothing goes to disk.

## Sizing

Use this rough memory budget: the size of the model (the parameters, plus what the session or the eager module needs at inference time), times `--workers`, plus the result cache. The export holds an extra copy of the graph for a short time, one time for each boot.

The `Boot` row of the banner shows the phases of a boot:

```
Boot        6.4 s: load 0.3, export 2.8, verify 0.2, session 0.1, warmup 0.1
```

`load` reads or imports the model. `export` is `torch.export` plus the ONNX translation. `verify` runs the samples through both backends. `session` builds the backend. `warmup` runs `--warmup N` inferences. The `boot` field of `/metadata` has the same numbers. With `--workers N`, the parent pays `export` and `verify` one time, and each worker pays its own `load`, `session` and `warmup`.

## Model name

`GET /v1/models` lists one model. `POST /v1/embeddings` accepts only that name in `model`. Without `--served-model-name NAME` (or `DOWNSHIFT_SERVED_MODEL_NAME`), the name is the last part of the model path. For example, the model `./models/bge-small` has the name `bge-small`. For an import spec such as `pkg.mod:make_model`, the name is the whole spec. Set the option if a client must use a fixed name that does not depend on the path.

## Matryoshka cuts

If the model was trained for Matryoshka cuts, start the server with `--matryoshka` (or `DOWNSHIFT_MATRYOSHKA=1`). `POST /v1/embeddings` then accepts `dimensions`. The server keeps the first `dimensions` values of each vector and scales the vector to unit length. The result cache keeps the full vector, so one cache entry serves all sizes. Without the option, a request with `dimensions` gets the error `dimensions_not_supported`.

## Metrics

To count the requests, start the server with `--metrics` (or `DOWNSHIFT_METRICS=1`). The server then serves the counts on `GET /metrics` as Prometheus text (format 0.0.4). If you do not set the option, the route does not exist (`404`), and the request path counts nothing. `/metrics` follows the API key rule of the other routes: send `Authorization: Bearer <key>`. The probes `/health` and `/ready` are the only routes without a key.

Each series has the label `worker`. Its value is `0`, because `--metrics` needs one worker.

| Name | Type | Labels | Meaning |
|---|---|---|---|
| `downshift_requests_total` | counter | `route`, `code` | Requests. `route` is the route template (`/predict`), or `other` for a path that has no route. `code` is the HTTP status. |
| `downshift_request_seconds` | histogram | `route` | Seconds from the start of a request to the end of its response. |
| `downshift_stage_seconds` | histogram | `stage` | Seconds of one stage of a predict: `queue`, `prep`, `infer` or `encode`. |
| `downshift_inflight` | gauge | none | Admitted predicts that have not finished. |
| `downshift_queue_depth` | gauge | none | Admitted predicts that wait for an inference slot. |
| `downshift_queue_limit` | gauge | none | The value of `--max-queue`. |
| `downshift_rejected_total` | counter | `reason` | Requests that got a `503`: `capacity`, `timeout` or `not_ready`. |
| `downshift_cache_rows_total` | counter | `result` | Rows that the result cache found (`hit`) or did not find (`miss`). Only with the cache on. |
| `downshift_cache_requests_total` | counter | `result` | Predicts by the `X-Downshift-Cache` value: `hit`, `partial` or `miss`. |
| `downshift_model_info` | gauge | `downshift_version`, `verdict`, `backend`, `family` | The value is always `1`. |
| `downshift_ready` | gauge | none | `1` when the model is ready. `0` while the model loads. |
| `downshift_boot_seconds` | gauge | `phase` | Seconds of each boot phase. |

The histogram buckets are fixed, in seconds: 0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5 and 10. The `queue` stage is `prep_wait`, `infer_wait` and `encode_wait` of the access log added together. The `prep` stage is `parse` and `prep` added together.

A scrape reaches one worker only, so `--metrics` with `--workers N` (N above 1) is a usage error (exit code `4`). Use one worker for each container. Scale with replicas.

Scrape config:

```yaml
scrape_configs:
  - job_name: downshift
    metrics_path: /metrics
    authorization:
      credentials_file: /etc/prometheus/downshift-key
    static_configs:
      - targets: ["downshift:8000"]
```

Alert rules:

```yaml
groups:
  - name: downshift
    rules:
      - alert: DownshiftQueueFull
        expr: downshift_queue_limit > 0 and downshift_queue_depth >= downshift_queue_limit
        for: 1m
        annotations:
          summary: The queue of {{ $labels.instance }} is full. New predicts get 503.
      - alert: DownshiftErrorRate
        expr: |
          sum(rate(downshift_requests_total{code=~"5..", route!~"/health|/ready"}[5m]))
            / sum(rate(downshift_requests_total{route!~"/health|/ready"}[5m])) > 0.01
        for: 5m
        annotations:
          summary: More than 1% of the requests get a 5xx status.
      - alert: DownshiftSlowPredict
        expr: |
          histogram_quantile(0.99,
            sum by (le) (rate(downshift_request_seconds_bucket{route="/predict"}[5m]))) > 0.5
        for: 10m
        annotations:
          summary: The p99 latency of /predict is more than 500 ms.
```

`examples/grafana-dashboard.json` is a Grafana dashboard for these series: the request rate, the error rate, the p50 and p99 latency, the latency of each stage, the queue depth and the cache hit rate. Import it in Grafana, and select your Prometheus data source.

## Logs and request IDs

All output goes through Python `logging` to stdout, as plain text in the form `time LEVEL logger: message`. Point your log shipper at stdout. [`cli.md`](cli.md#logging) gives what each `--log-level` prints.

- The request line goes to `downshift.access`, for example `POST /predict 200 12.3 ms request_id=3f9c1a7b2e4d5c60`. A `4xx` or `5xx` line prints at the default level. The probes log only at `debug`. Give `--no-access-log` if your proxy already logs each request.
- Each response has `X-Request-Id`. If your proxy or tracing uses another name, set `DOWNSHIFT_REQUEST_ID_HEADER`, for example `X-Correlation-Id`. If the client sent one, downshift returns it. The ID is on each log line that the request writes. A `500` body has the same `request_id`, so "see the server log" has a search key.

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

The verdict is made on the CPU. When the server serves on a GPU, it runs a serving check at boot. It is not on the request path.

1. The server builds the backend that serves (ONNX Runtime with the CUDA provider, or torch on `cuda`).
2. It runs the inputs of the verification samples (the 8 samples, with the edge shapes) one time each through that backend and through the CPU. Downshift keeps the sample inputs in memory, in the export cache (`feeds.npz`) and in the hand-off to `--workers`. If the samples are larger than 64 MiB together, it keeps the smallest and the largest first, then as many others as fit. An old cache entry or a torch model with no verdict has only the example input. For ONNX Runtime, the CPU run is the same graph on the CPU provider. For torch, it is the same model on the CPU.
3. It compares the two results with the tolerance of the verdict. For an embedding model, each row must also reach `DOWNSHIFT_TOL_COSINE`.

The `GPU check` row of the banner shows the result, for example `CUDAExecutionProvider cuda  max abs err 3.1e-06 over 8 samples  OK` (`over 3 of 8 samples` if the size limit cut the set). `GET /metadata` has the same data in `serving_numerics`. On the CPU, the check does not run and `serving_numerics` is `null`. If the model has no example input (a `.onnx` file with no reference), the check does not run.

The CUDA provider of ONNX Runtime uses TF32 matrix multiplies by default on Ampere and newer GPUs. TF32 moves the result by about 1e-3 relative, which is more than the float32 tolerance. For this reason, downshift turns TF32 off for ONNX Runtime (`use_tf32` is `0`) and for torch (`torch.backends.cuda.matmul.allow_tf32` and `torch.backends.cudnn.allow_tf32` are `False`). To allow TF32, pass `--tf32` (or set `DOWNSHIFT_TF32=1`). The serving check then uses the larger of the TF32 tolerance (`DOWNSHIFT_TOL_TF32_ATOL` and `DOWNSHIFT_TOL_TF32_RTOL`, default `1e-3` and `1e-2`) and the tolerance of the verdict.

If the serving check fails, the server logs one `WARNING` that names the provider, the error and the next step.

- With `--backend auto` and no `--force-onnx`, the server serves torch instead (as for a DEGRADED verdict). The torch backend then gets the same check.
- With `--backend onnxruntime` or `--force-onnx`, the server serves ONNX Runtime as you asked. `serving_numerics.ok` is `false`.
- If torch fails the check, the server keeps serving, because there is no other backend. `serving_numerics.ok` is `false`.

To remove the failure, do one of these steps: remove `--tf32`, pass `--device cpu`, or pass `--force-onnx` to keep ONNX Runtime.

### Half precision

To serve in half precision on a GPU, export with `downshift export --fp16`, then serve the file with the PyTorch model as the reference:

```bash
downshift export ./model -o ./out --fp16
downshift serve ./out/model.onnx --reference ./model --device cuda
```

`--fp16` exports the fp32 graph and then changes the weights to fp16. The inputs and outputs of the graph stay fp32, so clients do not change. Downshift verifies the graph against the fp32 torch model with the `float16` tolerances. The serving check on the GPU uses the same tolerances for a graph with fp16 weights. A person measured an RTX 4050 with all-MiniLM-L6-v2: the maximum absolute error was 3.7e-4, the minimum cosine was 0.999998, and a request took 3.7 ms in place of 4.0 ms with the fp32 graph. On a small model, the gain is small. Measure your model before you rely on it.

If ONNX Runtime has no CPU kernel for an fp16 operation, the check of the export gives `FAILED`. Use a newer ONNX Runtime, or export the model without `--fp16`.

The check adds a few inferences to the boot time. The `session` value of `boot` includes them.

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
