# Stability

From version 1.0.0, the items on this page do not change in a way that breaks a client. This page lists the items. It does not repeat the reference pages. Each section points to the page that describes the item.

Version numbers follow [Semantic Versioning](https://semver.org/). A change that breaks a listed item needs a new major version.

## What a breaking change is

These changes are breaking:

- Remove or rename a listed item.
- Change the type of a field, an option or a variable.
- Change a default so that a response, an exit code or a file is different.
- Add a field that a client must send (a new required field).
- Change the meaning of a status code, an error `code`, an exit code or a metric.

These changes are not breaking:

- Add an optional field to a request.
- Add a field to a response. A client must ignore the fields that it does not know.
- Add a route, an option, an environment variable, an error code, a metric, a label value or a header.
- Change the text of a log line.
- Change the text of `detail` in an error body. To handle an error, branch on `code`, not on `detail`.
- Change a number that depends on the machine (timings, CPU counts, sizes of the boot).
- Change anything that is not listed on this page. This includes the modules of the package that are not in `downshift.__all__`, and the files of the export cache.

## The deprecation rule

A deprecated item stays for one minor release before removal. The item is removed in the next minor release at the earliest. In the release that deprecates it:

- The server writes one warning to the log when it sees the item.
- `CHANGELOG.md` has an entry that names the item and the replacement.

Downshift removes the item in the next minor release or later. A removal that breaks a frozen item needs a new major version.

## HTTP routes

Refer to [`http-api.md`](http-api.md) for the bodies, the status codes and the wire formats.

| Route | Notes |
|---|---|
| `GET /` | Fields: `service`, `version`, `ready`, `model`, `routes`, `next_step`. |
| `GET /health` | Field: `status`. |
| `GET /ready` | Fields: `ready`, and `phase` while the model loads. The status is `200` or `503`. |
| `GET /metadata` | The fields of `MetadataResponse` in `serve/schemas.py`. |
| `GET /schema` | The fields of `SchemaResponse` in `serve/schemas.py`. |
| `GET /version` | Field: `version`. |
| `POST /predict` | Request: `inputs` or `text`, `prompt_name`, `output_encoding`. Response: `outputs`, `shapes`, `dtypes`, and `predictions` for a classifier. |
| `POST /predict/graph` | Request: `x`, `edge_index`, `edge_attr`, or `graphs`, and `output_encoding`. |
| `GET /metrics` | Only with `--metrics`. Refer to "Metrics" below. |

The JSON wire formats and the safetensors wire format are part of the contract. The `Content-Type` of a body selects the format.

## Headers

| Header | Name can change | Notes |
|---|---|---|
| Request ID | Yes (experimental): `DOWNSHIFT_REQUEST_ID_HEADER` (default `X-Request-Id`) | Read from the request. Set on each response. The variable is experimental. The default name is frozen. |
| Result cache | Yes (experimental): `DOWNSHIFT_CACHE_HEADER` (default `X-Downshift-Cache`) | The values are `hit`, `partial` and `miss`. The variable is experimental. The default name is frozen. |
| `Server-Timing` | No | Only with `--server-timing`. The header is frozen. The stage names are experimental. |
| `Retry-After` | No | On a `503` for "not ready" and "at capacity". |
| `WWW-Authenticate` | No | On a `401`. |
| `Authorization` | No | `Bearer <key>`, if `DOWNSHIFT_API_KEY` is set. |
| `Cache-Control` | No | `no-cache` on a request skips the lookup in the result cache. |

## Result cache

The result cache is on when the row check finds the rows independent. The row check uses two rows, so it cannot prove that a small effect is absent. On a server with more than one tenant, a hit is faster than a miss, and a client can see the difference. To turn the cache off, set `--result-cache-rows 0`. Refer to [Result cache](production.md#result-cache).

## OpenAI routes

The routes `GET /v1/models`, `GET /v1/models/{id}` and `POST /v1/embeddings` are frozen with their fields. Refer to [`openai.md`](openai.md). On these routes, an error body has the OpenAI shape.

## Errors

Refer to "Error codes" in [`http-api.md`](http-api.md) for the status and the meaning of each code. An error code stays the same for the whole major version. These codes are frozen:

`invalid_request`, `invalid_input`, `missing_input`, `text_too_long`, `unknown_prompt`, `text_not_supported`, `not_graph_model`, `unauthorized`, `body_too_large`, `unsupported_media_type`, `not_found`, `method_not_allowed`, `http_error`, `not_ready`, `at_capacity`, `request_timeout`, `model_not_found`, `not_embedding_model`, `dimensions_not_supported`, `internal_error`.

There are two error body shapes:

- Other routes: `{"detail": ..., "code": "<code>"}`. A `500` also has `request_id`. A `422` has a list in `detail`.
- Routes under `/v1/`: `{"error": {"message", "type", "param", "code"}}`.

## Metrics

Refer to "Metrics" in [`production.md`](production.md). Each series has the label `worker`. The label is frozen. Its value is experimental. These names and labels are frozen:

| Name | Type | Labels |
|---|---|---|
| `downshift_requests_total` | counter | `route`, `code` |
| `downshift_request_seconds` | histogram | `route`, `le` |
| `downshift_stage_seconds` | histogram | `stage`, `le` |
| `downshift_rejected_total` | counter | `reason` |
| `downshift_cache_requests_total` | counter | `result` |
| `downshift_cache_rows_total` | counter | `result` |
| `downshift_ready` | gauge | none |
| `downshift_inflight` | gauge | none |
| `downshift_queue_depth` | gauge | none |
| `downshift_queue_limit` | gauge | none |
| `downshift_model_info` | gauge | `downshift_version`, `verdict`, `backend`, `family` |
| `downshift_boot_seconds` | gauge | `phase` |

## Command line

Refer to [`cli.md`](cli.md) for every option.

- The commands are `check`, `export` and `serve`. The option `--version` and the command `python -m downshift` are frozen too.
- The long names of the options (for example `--max-queue`) are frozen. The short names `-k` and `-o` are frozen. The options `--execution`, `--axis-max` and `--middleware` are experimental.
- The exit codes of `check` and `export` are `0` CLEAN, `1` FAILED, `2` DEGRADED, `3` UNVERIFIED, `4` usage error and `5` crash. `serve` exits with `4` or `5` only before the server is ready.
- The output of `check --json` and `export --json` is the verdict. Its fields follow the rules of a response body above.
- The text of the boot banner and of the reports (not `--json`) is not frozen.

## Environment variables

Refer to "Environment variables" in [`cli.md`](cli.md). These variables are frozen, with their types and defaults:

- Server: `DOWNSHIFT_HOST`, `DOWNSHIFT_PORT`, `DOWNSHIFT_WORKERS`, `DOWNSHIFT_DEVICE`, `DOWNSHIFT_BACKEND`, `DOWNSHIFT_WARMUP`, `DOWNSHIFT_SAMPLES`.
- Threads and limits: `DOWNSHIFT_INTRA_OP_THREADS`, `DOWNSHIFT_INTER_OP_THREADS`, `DOWNSHIFT_PREP_THREADS`, `DOWNSHIFT_MAX_INPUT_BYTES`, `DOWNSHIFT_MAX_BODY_BYTES`, `DOWNSHIFT_MAX_CONCURRENCY`, `DOWNSHIFT_MAX_QUEUE`, `DOWNSHIFT_REQUEST_TIMEOUT`.
- Request and response: `DOWNSHIFT_OUTPUT_ENCODING`, `DOWNSHIFT_RESULT_CACHE_ROWS`, `DOWNSHIFT_SERVER_TIMING`, `DOWNSHIFT_METRICS`, `DOWNSHIFT_SERVED_MODEL_NAME`, `DOWNSHIFT_MATRYOSHKA`.
- Security: `DOWNSHIFT_API_KEY`, `DOWNSHIFT_SSL_CERTFILE`, `DOWNSHIFT_SSL_KEYFILE`, `DOWNSHIFT_SSL_KEYFILE_PASSWORD`.
- Export cache: `DOWNSHIFT_EXPORT_CACHE_DIR`.
- Deprecated, removed in 1.1 at the earliest: `DOWNSHIFT_SERVER_API_KEY`. It is the old name of `DOWNSHIFT_API_KEY`. If only the old name is set, downshift uses it and logs one warning. If both are set, `DOWNSHIFT_API_KEY` wins.

## Python API

Refer to [`python-api.md`](python-api.md). The names in `downshift.__all__` are frozen, with their signatures: `Adapter`, `ExportVerdict`, `NumericsReport`, `OnnxRuntimeError`, `Prepared`, `app_for`, `build_app`, `build_verdict`, `check`, `export`, `intake`, `prepare_model` and `__version__`.

Code in other modules of `downshift` is internal, also if it has no leading underscore. `ServeOptions` and `ServingState` are internal. Refer to [`serving.md`](serving.md).

The names `Adapter` and `Prepared` are frozen. The protocol and the fields are experimental. Refer to "Experimental" below and to [`adapters.md`](adapters.md).

## Experimental

The items in this section can change in a 1.x minor release. A change has a note in `CHANGELOG.md`. They are not part of the breaking-change rule above. They are listed here so that you know which items to pin.

- The `Adapter` protocol (the attribute `name`, and the methods `matches`, `example_inputs` and `prepare`) and the fields of `Prepared`. Refer to [`adapters.md`](adapters.md).
- The tolerance variables: `DOWNSHIFT_TOL_FLOAT32_ATOL`, `DOWNSHIFT_TOL_FLOAT32_RTOL`, `DOWNSHIFT_TOL_FLOAT64_ATOL`, `DOWNSHIFT_TOL_FLOAT64_RTOL`, `DOWNSHIFT_TOL_FLOAT16_ATOL`, `DOWNSHIFT_TOL_FLOAT16_RTOL`, `DOWNSHIFT_TOL_BFLOAT16_ATOL`, `DOWNSHIFT_TOL_BFLOAT16_RTOL`.
- `--execution` and `DOWNSHIFT_EXECUTION`.
- `--axis-max` and `DOWNSHIFT_AXIS_MAX`.
- `--middleware`.
- The variables that set a header name: `DOWNSHIFT_CACHE_HEADER` and `DOWNSHIFT_REQUEST_ID_HEADER`.
- The stage names of `Server-Timing`.
- The value of the metrics label `worker`.

## Files

The manifest `<name>.manifest.json` that `export` writes is frozen. Its fields stay, and `format_version` is `1`: `format_version`, `downshift_version`, `created_utc`, `onnx_file`, `onnx_sha256`, `external_data`, `source`, `source_sha256`, `versions`, `opset`, `observed_dtype`, `execution_providers_available` and `verdict`.

The export cache (`--export-cache-dir`) is internal. Its entries and `index.json` have `format_version`. If a file has another version, downshift ignores it, writes one warning and exports again. You can delete the directory at any time.
