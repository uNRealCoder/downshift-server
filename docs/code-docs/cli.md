# CLI reference

This page describes every `downshift` command and flag. The information comes from `src/downshift/cli/main.py` (the commands), `cli/options.py` (the typer option types) and `cli/runtime.py` (`ServeArgs` and the `--workers` code). It was checked against the output of `--help`.

A flag that has a `DOWNSHIFT_*` environment variable gets its default from that variable (`src/downshift/settings.py`). The order of priority is:

1. A flag on the command line.
2. The environment variable.
3. The default in the code.

The same variables apply to library use (`ServeOptions()` and `app_for()`), not only to the CLI. Refer to "Environment variables" below.

You can also run `downshift` as `python -m downshift`. Use this form if the console script is not on `PATH`. Both forms accept `--version` at the top level, before a subcommand. It prints only the version number and exits with code 0.

## Commands

```
downshift check MODEL [OPTIONS]
downshift export MODEL -o DIR [OPTIONS]
downshift serve MODEL [OPTIONS]
```

`intake` (a `.onnx` file with `--reference`) is not a separate subcommand. `check` and `serve` detect a `MODEL` argument that points to an existing `.onnx` file. They then call `downshift.intake()` and not `downshift.check()`.

`MODEL` accepts the same forms on every command. Every form is already on this machine. Three forms are downloaded artifacts:

| Form | Meaning |
|---|---|
| `model.onnx` | A downloaded or already exported ONNX file. Served or checked as it is. `UNVERIFIED` unless you give `--reference`. |
| `weights.pt` (also `.pth`, `.bin`, `.ckpt`) | A downloaded PyTorch checkpoint (state dict). Needs `--model-class`. |
| `path/to/repo/dir/` | A downloaded Hugging Face repo directory. Downshift identifies it by the `config.json` in it. Needs `[hf]`. |

One form names a model that the process can import. It is not a file:

| Form | Meaning |
|---|---|
| `pkg.module:attr` | An import spec. `attr` is an `nn.Module` instance or a factory with no arguments. Downshift resolves it on `sys.path`. The CLI adds the current directory, as `python -m` does. A `my_model.py` in the current directory is `my_model:attr`. |

Model loading is local only. A Hugging Face hub id (`org/repo`) gives an error that refers to `huggingface-cli download`. The `hf` adapter passes `local_files_only=True`. No command on this page uses the network to find a model. A running `serve` reports the form that it received under `source` on `GET /schema` ([`http-api.md`](http-api.md#get-schema)).

## `check`

This command exports the model in memory and verifies the numbers. It writes nothing to disk. The exit code is the verdict.

```
downshift check MODEL [OPTIONS]
```

| Flag | Env var | Default | Meaning |
|---|---|---|---|
| `--json` | - | off | Print the verdict as JSON and nothing else on stdout. Log lines go to stderr. |
| `--reference MODEL` | - | none | The PyTorch model to verify a `.onnx` `MODEL` against. |
| `--inputs pkg.module:fn` | - | none | Example inputs: a tuple, or a factory with no arguments that returns one. |
| `--model-class pkg.module:Class` | - | none | The class to load a state-dict `MODEL` into. |
| `--unsafe-load` | - | off | Allow `torch.load(weights_only=False)`. This runs code from the file. |
| `--adapter NAME\|path/to/adapter.py[:attr]` | - | detect | `generic`, `pyg`, `hf`, or a custom adapter. |
| `-k`, `--samples` | `DOWNSHIFT_SAMPLES` | `8` | The number of verification samples. |
| `--dynamic NAME:AXIS[,...]` | - | axis 0 of every input | The dynamic axes, for example `"x:0,edge_index:1"`. |
| `--atol FLOAT` | - | by output dtype | Overrides the absolute tolerance. |
| `--rtol FLOAT` | - | by output dtype | Overrides the relative tolerance. |
| `--seed INT` | - | `0` | The seed for the generation of verification samples. |
| `--vary pkg.module:fn` | - | none | `fn(i) -> inputs`. It replaces the sampler of the adapter. `fn(0)` must return the example inputs. |
| `--pooling mean\|cls\|max\|mean_sqrt_len\|lasttoken\|weightedmean\|none` | - | what the repo declares | For Hugging Face encoder repos. It overrides the pooling in `modules.json` of the repo. It also sets a pooling if the repo declares none. `none` serves the token vectors as they are. Refer to "Embedding models" in `http-api.md`. |
| `--normalize`/`--no-normalize` | - | what the repo declares | For Hugging Face encoder repos. It sets the L2 normalisation of the embedding. It needs a pooling from the repo or from `--pooling`. |
| `--log-level debug\|info\|warning\|error` | - | `warning` | Refer to "Logging" below. |

`--atol` and `--rtol` have no environment variable. The default that they override comes from `settings.TOLERANCES`. The key is the floating dtype of the model. You can change the default for each dtype with these variables:

| Dtype | atol variable | rtol variable | Default atol / rtol |
|---|---|---|---|
| float32 | `DOWNSHIFT_TOL_FLOAT32_ATOL` | `DOWNSHIFT_TOL_FLOAT32_RTOL` | `1e-4` / `1e-3` |
| float16 | `DOWNSHIFT_TOL_FLOAT16_ATOL` | `DOWNSHIFT_TOL_FLOAT16_RTOL` | `1e-2` / `1e-2` |
| bfloat16 | `DOWNSHIFT_TOL_BFLOAT16_ATOL` | `DOWNSHIFT_TOL_BFLOAT16_RTOL` | `5e-2` / `5e-2` |
| float64 | `DOWNSHIFT_TOL_FLOAT64_ATOL` | `DOWNSHIFT_TOL_FLOAT64_RTOL` | `1e-6` / `1e-5` |

If you give `--atol` or `--rtol`, they have priority over these variables.

Exit codes: `0` CLEAN, `1` FAILED, `2` DEGRADED, `3` UNVERIFIED, `4` usage error, `5` crash. Refer to "Exit codes" below.

```bash
downshift check my_pkg.models:build --json -k 16 > verdict.json
```

## `export`

This command exports the model to `DIR/NAME.onnx` and writes a `NAME.manifest.json` file next to it. If the verdict is `FAILED`, it writes nothing.

```
downshift export MODEL -o DIR [OPTIONS]
```

All flags of `check` apply (see the table above). There are also these flags:

| Flag | Env var | Default | Meaning |
|---|---|---|---|
| `-o`, `--output PATH` | - | *(required)* | The output directory. |
| `--name STR` | - | model slug | The name of the artifact, without the extension. |
| `--fp16` | - | off | Cast the model to fp16 before the export. This is a plain `.half()` on a deep copy. |
| `--no-verify` | - | off | Skip the numerics check. The verdict is `UNVERIFIED`. |

The exit codes are the same as for `check`.

```bash
downshift export my_pkg.models:build -o artifacts/ --name classifier
```

## `serve`

This command checks the model, selects a backend from the verdict, and serves the model over HTTP.

```
downshift serve MODEL [OPTIONS]
```

The adapter and verification flags of `check` also apply to `serve`, with the same defaults. These flags are `-k`/`--samples`, `--dynamic`, `--adapter`, `--inputs`, `--model-class`, `--unsafe-load`, `--atol`, `--rtol`, `--seed`, `--vary`, `--pooling` and `--normalize`. `serve` runs the same gate before it selects a backend. These flags are only for `serve`:

| Flag | Env var | Default | Meaning |
|---|---|---|---|
| `--host STR` | `DOWNSHIFT_HOST` | `127.0.0.1` | The address to bind. |
| `--port INT` | `DOWNSHIFT_PORT` | `8000` | The port to bind. |
| `--backend auto\|onnxruntime\|torch` | `DOWNSHIFT_BACKEND` | `auto` | `auto` follows the verdict. `onnxruntime` on a `DEGRADED` verdict is an error, unless you also give `--force-onnx`. |
| `--force-onnx` | - | off | Serve a `DEGRADED` graph through ONNX Runtime. |
| `--device auto\|cpu\|cuda` | `DOWNSHIFT_DEVICE` | `auto` | `cuda` without CUDA is a usage error (exit code `4`) on both backends. It does not run silently on the CPU. The numerics gate runs only on the CPU. When the server runs on another device, the `Verified on` row of the banner shows this. |
| `--warmup INT` | `DOWNSHIFT_WARMUP` | `3` | The number of warm-up inferences before `/ready` changes. |
| `--reference MODEL` | - | none | The PyTorch model to verify a `.onnx` `MODEL` against. |
| `--middleware pkg.module:Attr` (repeatable) | - | none | A middleware to attach. Downshift attaches them in the order that you give. |
| `--intra-op-threads INT` | `DOWNSHIFT_INTRA_OP_THREADS` | `0` | The number of threads inside one operation, for the backend that serves. This is an ONNX Runtime session option or `torch.set_num_threads`. `0` is the default of the backend. More threads decrease the latency of one request. They decrease throughput under concurrent load. |
| `--inter-op-threads INT` | `DOWNSHIFT_INTER_OP_THREADS` | `0` | The number of ONNX Runtime threads across operations. `0` lets ONNX Runtime choose. |
| `--output-encoding json\|base64` | `DOWNSHIFT_OUTPUT_ENCODING` | `json` | The default encoding of response tensors. An `output_encoding` in a request overrides it. |
| `--max-input-bytes INT` | `DOWNSHIFT_MAX_INPUT_BYTES` | `268435456` (256 MiB) | The server rejects a base64 tensor input if it is larger than this value after decoding. |
| `--max-body-bytes INT` | `DOWNSHIFT_MAX_BODY_BYTES` | `33554432` (32 MiB) | The server rejects a request body that is larger than this value (`413`), before it parses the body. It refuses a chunked body when the running total passes the limit. There is no separate limit on text length. This limit also applies to `{"text": ...}`. |
| `--max-concurrency INT` | `DOWNSHIFT_MAX_CONCURRENCY` | `4` | The number of inference threads for each worker process. This is the number of inferences that can run at the same time. Each inference includes the encoding of its response. Each one holds its own activation memory. If large requests run out of memory, decrease the value. |
| `--execution threadpool\|inline` | `DOWNSHIFT_EXECUTION` | `threadpool` | `threadpool`: the prep pool parses and prepares the request. An inference thread runs the inference and the encoding. `inline`: the event loop runs a JSON body of up to 64 KiB (with a Content-Length and no `text`). Use it for models that take less than about 1 ms for one inference. |
| `--prep-threads INT` | `DOWNSHIFT_PREP_THREADS` | `min(4, usable CPUs)` | The number of threads in each worker process that parse and convert request bodies. They are separate from the inference threads. |
| `--max-queue INT` | `DOWNSHIFT_MAX_QUEUE` | `64` | The number of predicts that can wait beyond `--max-concurrency`. After that, a new predict gets a fast `503`. |
| `--request-timeout FLOAT` | `DOWNSHIFT_REQUEST_TIMEOUT` | `30.0` | The number of seconds that a predict can wait without a start. After this time, it gets a `503` and no inference. The time in the queue counts. `0` means no limit. |
| `--workers INT` | `DOWNSHIFT_WORKERS` | `1` | The number of uvicorn worker processes. The parent exports and verifies once. Each worker builds its own session over that graph (or reloads the model for torch) and warms up. The parent frees its copy of the model before the workers start. If a worker fails to load, the whole server stops with the startup-failure exit code of uvicorn. Downshift does not start the worker again. |
| `--log-level debug\|info\|warning\|error` | - | `warning` | Refer to "Logging" below. |
| `--access-log` / `--no-access-log` | - | on | The log line for each request (`downshift.access`). The access log of uvicorn is always off. |

```bash
downshift serve my_pkg.models:build --port 8000 --output-encoding base64
```

`serve` does not exit on a verdict, as `check` and `export` do. It continues to serve `/predict` for all of `CLEAN`, `DEGRADED`, `FAILED` and `UNVERIFIED`. This is the purpose of the eager-PyTorch fallback. `serve` exits early only in these cases:

- A usage error (`4`).
- A crash during the boot (`5`).
- A normal shutdown of the process.

## Logging

All output of the CLI and the server goes through Python `logging` to one plain-text handler on stdout (`downshift/logs.py`, `setup_logging`). This includes:

- The boot banner.
- The `check` and `export` reports.
- The startup and error lines of uvicorn. Downshift sends them to the same handler.
- `warnings`.
- The request log line.

Each line has the form `time LEVEL logger: message`. If the line belongs to a request, downshift adds ` request_id=<id>`. There is no JSON format, no format switch, no colour and no box drawing.

`--log-level` is `warning` by default on `check`, `export` and `serve`:

| Level | What prints |
|---|---|
| `warning` (default) | The banner, the reports, and the `loading`, `will listen on` and `ready in X s` lines. These go to `downshift.report`, which prints at every level. Also warnings, errors, and every `4xx` and `5xx` request line (`downshift.access` at `WARNING`). |
| `info` | Adds the lines of uvicorn. Adds one line for each request on `downshift.access`: `METHOD PATH STATUS N ms request_id=...`. The probes `/health` and `/ready` print only at `DEBUG`. |
| `debug` | Adds tracebacks. For a `FAILED` verdict, it also adds the stderr of the torch export and the traceback of each strategy. |
| `error` | Only errors, and the lines of `downshift.report`, which always print. |

`--access-log/--no-access-log` (default on) controls the request line. The `X-Request-Id` is set in both cases. The client can send it, or downshift generates it. With `check --json` or `export --json`, the verdict JSON is the only output on stdout. The log lines go to stderr.

## Environment variables

`downshift/settings.py` reads these variables once, at import time. Every variable applies to `ServeOptions()` and `app_for()`, and also to the CLI flags above:

| Variable | Default | Sets |
|---|---|---|
| `DOWNSHIFT_HOST`, `DOWNSHIFT_PORT` | `127.0.0.1`, `8000` | `--host`, `--port` (CLI only. `ServeOptions` has no bind address.) |
| `DOWNSHIFT_DEVICE`, `DOWNSHIFT_BACKEND` | `auto`, `auto` | `--device`, `--backend` |
| `DOWNSHIFT_WARMUP`, `DOWNSHIFT_SAMPLES` | `3`, `8` | `--warmup`, `-k/--samples` (`ServeOptions.k`) |
| `DOWNSHIFT_INTRA_OP_THREADS`, `DOWNSHIFT_INTER_OP_THREADS` | `0`, `0` | `--intra-op-threads`, `--inter-op-threads` |
| `DOWNSHIFT_OUTPUT_ENCODING` | `json` | `--output-encoding` |
| `DOWNSHIFT_MAX_INPUT_BYTES`, `DOWNSHIFT_MAX_BODY_BYTES` | 256 MiB, 32 MiB | `--max-input-bytes`, `--max-body-bytes` |
| `DOWNSHIFT_MAX_CONCURRENCY`, `DOWNSHIFT_MAX_QUEUE`, `DOWNSHIFT_REQUEST_TIMEOUT` | `4`, `64`, `30` | `--max-concurrency`, `--max-queue`, `--request-timeout` |
| `DOWNSHIFT_EXECUTION`, `DOWNSHIFT_PREP_THREADS` | `threadpool`, `min(4, usable CPUs)` | `--execution`, `--prep-threads` |
| `DOWNSHIFT_WORKERS` | `1` | `--workers` (CLI only) |
| `DOWNSHIFT_SERVER_API_KEY` | unset | Requires `Authorization: Bearer <key>` on every route except `/health` and `/ready`. If it is unset or empty, downshift logs one startup warning that the endpoints are unauthenticated. There is no flag, because the key is a secret. `build_app(api_key=...)` and `app_for(api_key=...)` read it by default. |
| `DOWNSHIFT_TOL_{FLOAT32,FLOAT64,FLOAT16,BFLOAT16}_{ATOL,RTOL}` | see `check` above | The default tolerances. |

The library functions `check()`, `export()` and `intake()` use the constant `8` as the default for `k`. They do not use `DOWNSHIFT_SAMPLES`. Only `ServeOptions.k` (and so `app_for()`) and the CLI flag read that variable.

## Add a `serve` option

The typer option types are in `cli/options.py` (`Annotated` aliases such as `MaxQueueOpt`). `ServeArgs` is in `cli/runtime.py`. `ServeArgs` is `{load: LoadSpec, options: ServeOptions, reference, middleware, log_level, artifact, access_log}`. It goes through JSON and back, to give a `--workers N` run to each worker process.

A new option needs three edits:

1. A typer parameter on `serve_cmd`.
2. A field on `ServeOptions`. The default comes from `settings.py` if the option has an environment variable.
3. A parameter on `_collect_serve_args`. This function builds the `LoadSpec` and `ServeOptions` pair.

Nothing between these parts needs a change.

## Exit codes

These codes come from the `_exit_on_error` context manager in `cli/main.py`. It wraps the body of every command:

| Code | Meaning |
|---|---|
| `0` | `check` and `export`: the verdict is `CLEAN`. |
| `1` | `check` and `export`: the verdict is `FAILED`. |
| `2` | `check` and `export`: the verdict is `DEGRADED`. |
| `3` | `check` and `export`: the verdict is `UNVERIFIED`. |
| `4` | Usage error. A `ValueError` left the command body. Examples: a model spec that cannot load, a bad `--dynamic` string, an unknown adapter name, `--backend onnxruntime` on a `DEGRADED` verdict without `--force-onnx`, and `--device cuda` without CUDA. |
| `5` | Any other unexpected exception. `--log-level debug` also prints the traceback. |

The codes `0` to `3` come from `ExportVerdict.exit_code`. The mapping is `EXIT_CODES = {"CLEAN": 0, "FAILED": 1, "DEGRADED": 2, "UNVERIFIED": 3}` in `core/verdict.py`. Only the CLI uses `4` and `5` (`EXIT_USAGE` and `EXIT_CRASH` in `cli/main.py`). They apply to all three commands, including `serve` during the boot phase.
