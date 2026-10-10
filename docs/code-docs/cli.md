# CLI reference

Every `downshift` command and flag, read from `src/downshift/cli/main.py` (the commands),
`cli/options.py` (the typer option types) and `cli/runtime.py` (`ServeArgs` and the
`--workers` machinery), and cross-checked against `--help` output. Flags with a
`DOWNSHIFT_*` environment variable get their default from it (`src/downshift/settings.py`);
a flag on the command line always wins over the environment, and the environment always
wins over the hardcoded default. The same variables apply to library use (`ServeOptions()`,
`app_for()`), not only to the CLI: see "Environment variables" below.

`downshift` is also runnable as `python -m downshift`, for when the console script isn't on
`PATH`. Both forms accept `--version` (prints just the bare `<version>` and exits 0) at the top
level, before any subcommand.

## Commands

```
downshift check MODEL [OPTIONS]
downshift export MODEL -o DIR [OPTIONS]
downshift serve MODEL [OPTIONS]
```

`intake` (a `.onnx` file with `--reference`) is not a separate subcommand - `check` and
`serve` both detect a `MODEL` argument that points at an existing `.onnx` file and route to
`downshift.intake()` internally instead of `downshift.check()`.

`MODEL` accepts the same forms on every command, and every one of them is already on this
machine. Three downloaded artifacts:

| Form | Meaning |
|---|---|
| `model.onnx` | A downloaded or already-exported ONNX file, served/checked as-is. `UNVERIFIED` unless `--reference` is given. |
| `weights.pt` (also `.pth`, `.bin`, `.ckpt`) | A downloaded PyTorch checkpoint (state dict); needs `--model-class`. |
| `path/to/repo/dir/` | A downloaded Hugging Face repo directory, recognised by the `config.json` in it (needs `[hf]`). |

and one form that names a model importable in the process rather than a file:

| Form | Meaning |
|---|---|
| `pkg.module:attr` | Import spec; `attr` is an `nn.Module` instance or a zero-argument factory. Resolved on `sys.path`, with the current directory appended by the CLI (as `python -m` would), so a `my_model.py` next to you is `my_model:attr`. |

Model loading is local-only: a Hugging Face hub id (`org/repo`) is rejected with an error
pointing at `huggingface-cli download`, and the `hf` adapter passes `local_files_only=True`,
so no command here reaches the network to find a model. A running `serve` reports which of
these forms it was given under `source` on `GET /schema`
([`http-api.md`](http-api.md#get-schema)).

## `check`

Export in memory and verify numerics; writes nothing to disk. Exit code is the verdict.

```
downshift check MODEL [OPTIONS]
```

| Flag | Env var | Default | Meaning |
|---|---|---|---|
| `--json` | - | off | Print the verdict as JSON and nothing else on stdout; log lines go to stderr. |
| `--reference MODEL` | - | none | PyTorch model to verify a `.onnx` `MODEL` against. |
| `--inputs pkg.module:fn` | - | none | Example inputs: a tuple, or a zero-arg factory for one. |
| `--model-class pkg.module:Class` | - | none | Class to load a state-dict `MODEL` into. |
| `--unsafe-load` | - | off | Allow `torch.load(weights_only=False)`; runs code from the file. |
| `--adapter NAME\|path/to/adapter.py[:attr]` | - | detect | `generic`, `pyg`, `hf`, or a custom adapter. |
| `-k`, `--samples` | `DOWNSHIFT_SAMPLES` | `8` | Number of verification samples. |
| `--dynamic NAME:AXIS[,...]` | - | axis 0 of every input | Dynamic axes, e.g. `"x:0,edge_index:1"`. |
| `--atol FLOAT` | - | by output dtype | Absolute tolerance override. |
| `--rtol FLOAT` | - | by output dtype | Relative tolerance override. |
| `--seed INT` | - | `0` | Seed for verification sample generation. |
| `--vary pkg.module:fn` | - | none | `fn(i) -> inputs` overriding the adapter's own sampler; `fn(0)` must return the example inputs. |
| `--pooling mean\|cls\|max\|mean_sqrt_len\|none` | - | what the repo declares | Hugging Face encoder repos: override the pooling in the repo's `modules.json`, or set one when it declares none. `none` serves the token vectors as they are. See "Embedding models" in `http-api.md`. |
| `--normalize`/`--no-normalize` | - | what the repo declares | Hugging Face encoder repos: L2-normalise the embedding, or not. Needs a pooling, from the repo or `--pooling`. |
| `--log-level debug\|info\|warning\|error` | - | `warning` | See "Logging" below. |

`--atol`/`--rtol` have no environment variable of their own; the *default* they override
comes from `settings.TOLERANCES`, keyed by the model's floating dtype and overridable per
dtype via `DOWNSHIFT_TOL_FLOAT32_ATOL`, `DOWNSHIFT_TOL_FLOAT32_RTOL`,
`DOWNSHIFT_TOL_FLOAT16_ATOL`, `DOWNSHIFT_TOL_FLOAT16_RTOL`,
`DOWNSHIFT_TOL_BFLOAT16_ATOL`, `DOWNSHIFT_TOL_BFLOAT16_RTOL`, `DOWNSHIFT_TOL_FLOAT64_ATOL`,
`DOWNSHIFT_TOL_FLOAT64_RTOL` (defaults `1e-4`/`1e-3`, `1e-2`/`1e-2`, `5e-2`/`5e-2`,
`1e-6`/`1e-5` respectively). `--atol`/`--rtol` win over those env vars when given.

Exit codes: `0` CLEAN, `1` FAILED, `2` DEGRADED, `3` UNVERIFIED, `4` usage error, `5` crash
(see "Exit codes" below).

```bash
downshift check my_pkg.models:build --json -k 16 > verdict.json
```

## `export`

Export to `DIR/NAME.onnx` with a `NAME.manifest.json` sidecar. Nothing is written if
`FAILED`.

```
downshift export MODEL -o DIR [OPTIONS]
```

All of `check`'s flags apply (same table above), plus:

| Flag | Env var | Default | Meaning |
|---|---|---|---|
| `-o`, `--output PATH` | - | *(required)* | Output directory. |
| `--name STR` | - | model slug | Artifact stem. |
| `--fp16` | - | off | Cast the model to fp16 before export (a plain `.half()`, on a deep copy). |
| `--no-verify` | - | off | Skip numerics; the verdict is `UNVERIFIED`. |

Same exit codes as `check`.

```bash
downshift export my_pkg.models:build -o artifacts/ --name classifier
```

## `serve`

Check the model, pick a backend from the verdict, and serve it over HTTP.

```
downshift serve MODEL [OPTIONS]
```

`check`'s adapter/verification flags (`-k`/`--samples`, `--dynamic`, `--adapter`,
`--inputs`, `--model-class`, `--unsafe-load`, `--atol`, `--rtol`, `--seed`, `--vary`,
`--pooling`, `--normalize`) apply
here too, with the same defaults, since `serve` runs the same gate before it picks a
backend. The serving-specific flags:

| Flag | Env var | Default | Meaning |
|---|---|---|---|
| `--host STR` | `DOWNSHIFT_HOST` | `127.0.0.1` | |
| `--port INT` | `DOWNSHIFT_PORT` | `8000` | |
| `--backend auto\|onnxruntime\|torch` | `DOWNSHIFT_BACKEND` | `auto` | `auto` follows the verdict; `onnxruntime` on a `DEGRADED` verdict is an error unless `--force-onnx` is also given. |
| `--force-onnx` | - | off | Serve a `DEGRADED` graph via ONNX Runtime anyway. |
| `--device auto\|cpu\|cuda` | `DOWNSHIFT_DEVICE` | `auto` | `cuda` without CUDA is a usage error (exit `4`) on both backends, not a silent run on the CPU. The numerics gate only runs on the CPU; the banner's `Verified on` row says so when the server runs elsewhere. |
| `--warmup INT` | `DOWNSHIFT_WARMUP` | `3` | Warm-up inferences before `/ready` flips. |
| `--reference MODEL` | - | none | PyTorch model to verify a `.onnx` `MODEL` against. |
| `--middleware pkg.module:Attr` (repeatable) | - | none | Middleware to attach, in the order given. |
| `--intra-op-threads INT` | `DOWNSHIFT_INTRA_OP_THREADS` | `0` | Threads inside one op, for whichever backend serves (ONNX Runtime session option, or `torch.set_num_threads`); `0` = the backend's default. More threads cut single-request latency but cost throughput under concurrent load. |
| `--inter-op-threads INT` | `DOWNSHIFT_INTER_OP_THREADS` | `0` | ORT threads across ops; `0` = let ONNX Runtime choose. |
| `--output-encoding json\|base64` | `DOWNSHIFT_OUTPUT_ENCODING` | `json` | Default encoding of response tensors; a request's own `output_encoding` overrides it. |
| `--max-input-bytes INT` | `DOWNSHIFT_MAX_INPUT_BYTES` | `268435456` (256 MiB) | Reject base64 tensor inputs larger than this once decoded. |
| `--max-body-bytes INT` | `DOWNSHIFT_MAX_BODY_BYTES` | `33554432` (32 MiB) | Reject request bodies larger than this (`413`), before they are parsed. A chunked body is refused as soon as its running total passes the limit. There is no separate text-length limit; this covers `{"text": ...}` too. |
| `--max-concurrency INT` | `DOWNSHIFT_MAX_CONCURRENCY` | `4` | Inference threads per worker process: inferences (each followed by its response encoding) allowed to run at once. Each holds its own activation memory; lower it if large requests run out of memory. |
| `--execution threadpool\|inline` | `DOWNSHIFT_EXECUTION` | `threadpool` | `threadpool`: parse and prep in the prep pool, inference and encode on an inference thread. `inline`: a JSON body up to 64 KiB with a Content-Length, and no `text`, runs on the event loop; for models under about 1 ms per inference. |
| `--prep-threads INT` | `DOWNSHIFT_PREP_THREADS` | `min(4, usable CPUs)` | Threads per worker process that parse and convert request bodies, apart from the inference threads. |
| `--max-queue INT` | `DOWNSHIFT_MAX_QUEUE` | `64` | Predicts allowed to wait past `--max-concurrency` before a new one gets a fast `503`. |
| `--request-timeout FLOAT` | `DOWNSHIFT_REQUEST_TIMEOUT` | `30.0` | Seconds a predict may wait, unstarted, before a `503` instead of an inference; counts time spent queued. `0` = no limit. |
| `--workers INT` | `DOWNSHIFT_WORKERS` | `1` | Uvicorn worker processes. The parent exports and verifies once; each worker builds its own session over that graph (or reloads the model for torch) and warms up. The parent frees its own copy of the model before the workers start. A worker that fails to load stops the whole server (uvicorn's startup-failure exit code) instead of being respawned. |
| `--log-level debug\|info\|warning\|error` | - | `warning` | See "Logging" below. |
| `--access-log` / `--no-access-log` | - | on | The one log line per request (`downshift.access`). Uvicorn's own access log is always off. |

```bash
downshift serve my_pkg.models:build --port 8000 --output-encoding base64
```

`serve` does not exit on a verdict the way `check`/`export` do - it keeps running and
serving `/predict` regardless of `CLEAN`/`DEGRADED`/`FAILED`/`UNVERIFIED` (that is the whole
point of the eager-PyTorch fallback). It only exits early on a usage error (`4`), a crash
during boot (`5`), or normal process shutdown.

## Logging

Everything the CLI and the server say goes through Python `logging` into one plain-text
handler on stdout (`downshift/logs.py`, `setup_logging`): the boot banner, the `check`
and `export` reports, uvicorn's own startup and error lines (re-routed into the same
handler), `warnings`, and the request log line. Each line is
`time LEVEL logger: message`, with ` request_id=<id>` appended when the line belongs to a
request. There is no JSON format, no format switch, no colour and no box drawing.

`--log-level` defaults to `warning` on `check`, `export` and `serve`:

| Level | What prints |
|---|---|
| `warning` (default) | The banner, the reports, and the `loading` / `will listen on` / `ready in X s` lines: they are logged on `downshift.report`, which prints at every level. Plus warnings, errors, and every `4xx`/`5xx` request line (`downshift.access` at `WARNING`). |
| `info` | Adds uvicorn's own lines and one line per request on `downshift.access`: `METHOD PATH STATUS N ms request_id=...`. The `/health` and `/ready` probes are `DEBUG` only. |
| `debug` | Adds tracebacks, and for a `FAILED` verdict torch's own export stderr and every strategy's traceback. |
| `error` | Only errors, plus the always-printing `downshift.report` lines. |

`--access-log/--no-access-log` (default on) controls the request line; the
`X-Request-Id` (echoed from the client or generated) is set either way. With
`check --json` or `export --json` the verdict JSON is the only thing on stdout and the log
lines go to stderr.

## Environment variables

Read once, at import time, by `downshift/settings.py`. Every one applies to `ServeOptions()`
and `app_for()` as well as to the CLI flags above:

| Variable | Default | Sets |
|---|---|---|
| `DOWNSHIFT_HOST`, `DOWNSHIFT_PORT` | `127.0.0.1`, `8000` | `--host`, `--port` (CLI only: `ServeOptions` has no bind address) |
| `DOWNSHIFT_DEVICE`, `DOWNSHIFT_BACKEND` | `auto`, `auto` | `--device`, `--backend` |
| `DOWNSHIFT_WARMUP`, `DOWNSHIFT_SAMPLES` | `3`, `8` | `--warmup`, `-k/--samples` (`ServeOptions.k`) |
| `DOWNSHIFT_INTRA_OP_THREADS`, `DOWNSHIFT_INTER_OP_THREADS` | `0`, `0` | `--intra-op-threads`, `--inter-op-threads` |
| `DOWNSHIFT_OUTPUT_ENCODING` | `json` | `--output-encoding` |
| `DOWNSHIFT_MAX_INPUT_BYTES`, `DOWNSHIFT_MAX_BODY_BYTES` | 256 MiB, 32 MiB | `--max-input-bytes`, `--max-body-bytes` |
| `DOWNSHIFT_MAX_CONCURRENCY`, `DOWNSHIFT_MAX_QUEUE`, `DOWNSHIFT_REQUEST_TIMEOUT` | `4`, `64`, `30` | `--max-concurrency`, `--max-queue`, `--request-timeout` |
| `DOWNSHIFT_EXECUTION`, `DOWNSHIFT_PREP_THREADS` | `threadpool`, `min(4, usable CPUs)` | `--execution`, `--prep-threads` |
| `DOWNSHIFT_WORKERS` | `1` | `--workers` (CLI only) |
| `DOWNSHIFT_SERVER_API_KEY` | unset | Requires `Authorization: Bearer <key>` on every route except `/health` and `/ready`; unset or empty logs one startup warning that the endpoints are unauthenticated. No flag: it is a secret. Read by `build_app(api_key=...)` and `app_for(api_key=...)` by default. |
| `DOWNSHIFT_TOL_{FLOAT32,FLOAT64,FLOAT16,BFLOAT16}_{ATOL,RTOL}` | see `check` above | The default tolerances. |

The library functions `check()`, `export()` and `intake()` default `k` to the constant `8`,
not to `DOWNSHIFT_SAMPLES`; only `ServeOptions.k` (and so `app_for()`) and the CLI flag read
that variable.

## Adding a `serve` option

The typer option types live in `cli/options.py` (`Annotated` aliases such as
`MaxQueueOpt`), `ServeArgs` in `cli/runtime.py`. `ServeArgs` is
`{load: LoadSpec, options: ServeOptions, reference, middleware, log_level, artifact,
access_log}`: it round-trips through JSON to hand a `--workers N` run to each worker
process. A new option is three edits: a typer parameter on `serve_cmd`, a field on
`ServeOptions` (default from `settings.py` if it has an environment variable), and a
parameter on `_collect_serve_args`, which builds the `LoadSpec`/`ServeOptions` pair.
Nothing in between needs to change.

## Exit codes

From `cli/main.py`'s `_exit_on_error` context manager, which wraps every command body:

| Code | Meaning |
|---|---|
| `0` | `check`/`export`: verdict is `CLEAN`. |
| `1` | `check`/`export`: verdict is `FAILED`. |
| `2` | `check`/`export`: verdict is `DEGRADED`. |
| `3` | `check`/`export`: verdict is `UNVERIFIED`. |
| `4` | Usage error: a `ValueError` escaped the command body - an unloadable model spec, a bad `--dynamic` string, an unknown adapter name, `--backend onnxruntime` on a `DEGRADED` verdict without `--force-onnx`, `--device cuda` without CUDA, and so on. |
| `5` | Anything else: an unexpected exception. `--log-level debug` also prints the traceback in this case. |

Codes `0`-`3` come from `ExportVerdict.exit_code` (`EXIT_CODES = {"CLEAN": 0, "FAILED": 1,
"DEGRADED": 2, "UNVERIFIED": 3}` in `core/verdict.py`); `4` and `5` are CLI-only
(`EXIT_USAGE`, `EXIT_CRASH` in `cli/main.py`) and apply to all three commands, including
`serve` during its boot phase.
