# CLI reference

This page describes every `downshift` command and flag. The commands are in `src/downshift/cli/main.py`. The option types are in `cli/options.py`.

You can also run `downshift` as `python -m downshift`. Use this form if the console script is not on `PATH`. `downshift --version` prints the version and exits with code 0.

A flag that has a `DOWNSHIFT_*` environment variable takes its default from that variable. The order of priority is:

1. The flag on the command line.
2. The environment variable.
3. The default in `src/downshift/settings.py`.

## Commands

```
downshift check MODEL [OPTIONS]
downshift export MODEL -o DIR [OPTIONS]
downshift serve MODEL [OPTIONS]
```

`MODEL` is always on this machine. It has one of these forms:

| Form | Meaning |
|---|---|
| `model.onnx` | A downloaded or already exported ONNX file. Served or checked as it is. `UNVERIFIED` unless you give `--reference`. |
| `weights.pt` (also `.pth`, `.bin`, `.ckpt`) | A downloaded PyTorch checkpoint (state dict). Needs `--model-class`. |
| `path/to/repo/dir/` | A downloaded Hugging Face repo directory. Downshift identifies it by its `config.json`. Needs the `[hf]` extra. |
| `pkg.module:attr` | An import spec. `attr` is an `nn.Module` or a function with no arguments that returns one. If the module has a `make_inputs` function, downshift uses it as the example inputs. The CLI adds the current directory to `sys.path`, so `my_model:build` works next to `my_model.py`. |

Downshift fetches nothing. A Hugging Face hub id (`org/repo`) gives an error that tells you to run `huggingface-cli download`. For a `.onnx` `MODEL`, `check` and `serve` call `downshift.intake()` and not `downshift.check()`.

## Options of the gate

`check`, `export` and `serve` all run the export-and-verify gate. These options control it:

| Flag | Env var | Default | Meaning |
|---|---|---|---|
| `--inputs pkg.module:fn` | - | none | Example inputs: a tuple, or a function that returns one. |
| `--model-class pkg.module:Class` | - | none | The class to load a state-dict `MODEL` into. |
| `--unsafe-load` | - | off | Allow `torch.load(weights_only=False)`. This runs code from the file. Use it only on a file that you trust. |
| `--adapter NAME\|path/to/adapter.py[:attr]` | - | detect | `generic`, `pyg`, `hf`, or your own adapter. Refer to [`adapters.md`](adapters.md). |
| `-k`, `--samples INT` | `DOWNSHIFT_SAMPLES` | `8` | The number of verification samples. |
| `--dynamic NAME:AXIS[,...]` | - | axis 0 of each input | The dynamic axes, for example `"x:0,edge_index:1"`. Each axis gets its own dimension, with the name `<input>_<axis>`. |
| `--axis-max NAME=N` (repeatable) | `DOWNSHIFT_AXIS_MAX` | the bound of the export | Serve at most `N` along the named axis, and verify one sample at exactly `N`. The banner and `/schema` list the names. Generic models use `dim0` (or `<input>_<axis>` with `--dynamic`), Hugging Face models use `batch` and `seq`, PyG models use `num_nodes` and `num_edges`. The value can only lower a bound. The variable takes `NAME=N,NAME=N`. |
| `--atol FLOAT`, `--rtol FLOAT` | - | by dtype | Override the tolerance. Refer to "Tolerances" below. |
| `--seed INT` | - | `0` | The seed of the verification samples. |
| `--vary pkg.module:fn` | - | none | `fn(i) -> inputs`. It replaces the sampler of the adapter. `fn(0)` must return the example inputs. The default samples are synthetic: random values at the edge shapes. To verify on inputs like your real traffic, give `--vary`. |
| `--pooling mean\|cls\|max\|mean_sqrt_len\|lasttoken\|weightedmean\|none` | - | what the repo declares | For a Hugging Face encoder repo. It overrides the pooling in `modules.json`, or sets one if the repo declares none. `none` serves the token vectors. Refer to "Embedding models" in [`http-api.md`](http-api.md#embedding-models). |
| `--normalize` / `--no-normalize` | - | what the repo declares | For a Hugging Face encoder repo. L2-normalise the embedding. It needs a pooling. |
| `--log-level debug\|info\|warning\|error` | - | `warning` | Refer to "Logging" below. |

### Tolerances

A sample passes when each output element satisfies `abs(got - expected) <= atol + rtol * abs(expected)`. This is the rule of `numpy.allclose`. The default tolerance comes from the narrowest floating dtype in the parameters of the model:

1. If bfloat16 or float16 is present, it wins.
2. float64 wins only if it is the only floating dtype.
3. Otherwise, float32 applies.

| Dtype | atol variable | rtol variable | Default atol / rtol |
|---|---|---|---|
| float32 | `DOWNSHIFT_TOL_FLOAT32_ATOL` | `DOWNSHIFT_TOL_FLOAT32_RTOL` | `1e-4` / `1e-3` |
| float16 | `DOWNSHIFT_TOL_FLOAT16_ATOL` | `DOWNSHIFT_TOL_FLOAT16_RTOL` | `1e-2` / `1e-2` |
| bfloat16 | `DOWNSHIFT_TOL_BFLOAT16_ATOL` | `DOWNSHIFT_TOL_BFLOAT16_RTOL` | `5e-2` / `5e-2` |
| float64 | `DOWNSHIFT_TOL_FLOAT64_ATOL` | `DOWNSHIFT_TOL_FLOAT64_RTOL` | `1e-6` / `1e-5` |

`--atol` and `--rtol` have priority over these variables.

## `check`

`check` exports the model in memory and verifies the numbers. It writes nothing to disk. The exit code is the verdict, so `check` can gate CI.

| Flag | Meaning |
|---|---|
| `--json` | Print the verdict as JSON and nothing else on stdout. The log lines go to stderr. |
| `--reference MODEL` | The PyTorch model to verify a `.onnx` `MODEL` against. It takes the same forms as `MODEL`, and it shares `--pooling` and `--normalize` with it. |

All options of the gate also apply.

```
$ downshift check examples.scatter_include_self_false:make_model
...
2026-10-20 16:20:24 INFO downshift.report: downshift v0.6.0
  Model           examples.scatter_include_self_false:make_model
  Family          generic
  Export          DEGRADED  (strict=False, opset 20)
  Numerics        max abs err 1.02e+00 over 8 samples  6/8 failed
  Tolerance       atol 1e-04, rtol 1e-03 (float32)
  Worst           output_0[3, 6]: torch -0.2082, onnxruntime 0.8119  (sample 5, x (12,8), segment_ids (12))
  Samples         x: (6,8) (12,8) (7,8) (7,8) (1,8) (12,8) (1,8) (7,8)
                  segment_ids: (6) (12) (7) (7) (1) (12) (1) (7)
  Rows            not checked (the numerics check failed)
  Shape-general   n/a (baseline fails)
  Dynamic dims    `dim0` (x[0])  sampled 1-12, serves 1-65536  ! unverified above 12
                  `dim0` (segment_ids[0])  sampled 1-12, serves 1-65536  ! unverified above 12
  Backend         torch
  Reason          exported via strict=False but numerics diverge on 6/8 samples (max abs err 1.02e+00)
```

- `Tolerance` shows the dtype that selected the default, or `--atol/--rtol` if a flag set it.
- `Worst` (DEGRADED only) is the output element with the largest error across all samples.
- `Samples` lists the input shapes of each sample.
- `Rows` says if each row of a batch gives the same output when it runs with other rows. The result cache needs `independent`.
- `Shape-general` is `yes`, `no`, or `n/a (baseline fails)`. The last value means that the unvaried example failed, so the check did not test other shapes.
- `Dynamic dims` gives the sizes that the check ran and the sizes that a server accepts. `! unverified above N` marks an axis that serves much more than the check ran.

```bash
downshift check my_pkg.models:build --json -k 16 > verdict.json
```

## `export`

`export` writes `DIR/NAME.onnx` and `DIR/NAME.manifest.json`. A DEGRADED export is written, because the manifest records how far the numbers differ. A FAILED export writes nothing.

| Flag | Env var | Default | Meaning |
|---|---|---|---|
| `-o`, `--output DIR` | - | *(required)* | The output directory. |
| `--name STR` | - | the model slug | The name of the files, without the extension. |
| `--fp16` | - | off | Cast a copy of the model to fp16 before the export. This is a plain `.half()` and not quantization. |
| `--no-verify` | - | off | Skip the numerics check. The verdict is `UNVERIFIED`. |
| `--json` | - | off | As for `check`. The JSON also has `manifest_path`. |
| `--export-cache-dir DIR` | `DOWNSHIFT_EXPORT_CACHE_DIR` | none | An existing directory. Downshift reuses a verified export from it, and saves new ones in it. Refer to "Export cache" in [`production.md`](production.md#export-cache). |

All options of the gate also apply, except `--reference`.

The manifest records:

- The SHA-256 of the `.onnx` file, of its external data file (if any), and of the source checkpoint (if the model came from a file). The manifest names files. It never records a directory.
- The versions of downshift, torch, onnx and onnxruntime.
- The opset and the observed weight dtype.
- The full verdict, with the numerics report.

```bash
downshift export my_pkg.models:build -o artifacts/ --name classifier
```

## `serve`

`serve` runs the gate, selects a backend from the verdict, and serves the model over HTTP. All options of the gate also apply. [`production.md`](production.md) explains how to choose the values.

| Flag | Env var | Default | Meaning |
|---|---|---|---|
| `--host STR` | `DOWNSHIFT_HOST` | `127.0.0.1` | The address to bind. |
| `--port INT` | `DOWNSHIFT_PORT` | `8000` | The port to bind. |
| `--backend auto\|onnxruntime\|torch` | `DOWNSHIFT_BACKEND` | `auto` | `auto` follows the verdict. `torch` skips the export. `onnxruntime` on a DEGRADED verdict is an error, unless you also give `--force-onnx`. |
| `--force-onnx` | - | off | Serve a DEGRADED graph through ONNX Runtime. The banner says so. |
| `--device auto\|cpu\|cuda` | `DOWNSHIFT_DEVICE` | `auto` | `cuda` without CUDA is a usage error on both backends. The gate runs only on the CPU. The `Verified on` row of the banner says this when you serve on a GPU. |
| `--warmup INT` | `DOWNSHIFT_WARMUP` | `3` | The number of inferences before `/ready` changes to `200`. |
| `--reference MODEL` | - | none | The PyTorch model to verify a `.onnx` `MODEL` against. |
| `--tokenizer-from DIR` | - | none | A downloaded Hugging Face repo directory. It gives the tokenizer, the pooling recipe and the labels to a `.onnx` or PyTorch `MODEL` that has none. Then `/predict` takes `{"text": ...}`. It does not change the graph or the verification, and it does not depend on `--reference`. |
| `--middleware pkg.module:Attr` (repeatable) | - | none | A middleware to attach, in the order that you give. Refer to [`production.md`](production.md#authentication). |
| `--intra-op-threads INT` | `DOWNSHIFT_INTRA_OP_THREADS` | `0` | The threads inside one operation, for ONNX Runtime or torch. `0` is the default of the backend. With `--workers N`, `0` becomes `max(1, CPUs // N)`. |
| `--inter-op-threads INT` | `DOWNSHIFT_INTER_OP_THREADS` | `0` | The ONNX Runtime threads across operations. `0` lets ONNX Runtime choose. |
| `--output-encoding json\|base64` | `DOWNSHIFT_OUTPUT_ENCODING` | `json` | The encoding of response tensors when a request does not set `output_encoding`. |
| `--max-input-bytes INT` | `DOWNSHIFT_MAX_INPUT_BYTES` | 256 MiB | The largest decoded size of one base64 or safetensors tensor. A larger one gets a `400`. |
| `--max-body-bytes INT` | `DOWNSHIFT_MAX_BODY_BYTES` | 32 MiB | The largest request body. A larger one gets a `413` before the server parses it. This is also the only limit on the size of a `text` request. |
| `--max-concurrency INT` | `DOWNSHIFT_MAX_CONCURRENCY` | `4` | The inferences that run at the same time in each worker process. Each one holds its own activation memory. |
| `--max-queue INT` | `DOWNSHIFT_MAX_QUEUE` | `64` | The predicts that can wait beyond `--max-concurrency`. After that, a new predict gets a fast `503`. |
| `--request-timeout FLOAT` | `DOWNSHIFT_REQUEST_TIMEOUT` | `30` | The seconds that an admitted predict can wait before its inference starts. After that, it gets a `503`. `0` means no limit. |
| `--prep-threads INT` | `DOWNSHIFT_PREP_THREADS` | `min(4, CPUs)` | The threads in each worker process that parse and convert request bodies. |
| `--execution threadpool\|inline` | `DOWNSHIFT_EXECUTION` | `threadpool` | `inline` runs a JSON body of up to 64 KiB (with a `Content-Length` and no `text`) on the event loop. Use it only for a model that infers in less than about 1 ms. |
| `--result-cache-rows INT` | `DOWNSHIFT_RESULT_CACHE_ROWS` | `100`, or off | The rows that the result cache keeps in each worker. `0` turns it off. The default is on only for a model with independent rows. A number for another model is a usage error. Refer to [`production.md`](production.md#result-cache). |
| `--workers INT` | `DOWNSHIFT_WORKERS` | `1` | The uvicorn worker processes. The parent exports and verifies one time. Each worker loads that result and warms up. If a worker cannot load, the whole server stops. |
| `--export-cache-dir DIR` | `DOWNSHIFT_EXPORT_CACHE_DIR` | none | As for `export`. A repeat boot of the same model skips the export and the verification. |
| `--ssl-certfile PATH` | `DOWNSHIFT_SSL_CERTFILE` | none | A PEM certificate or chain. If you give it, the server uses HTTPS. |
| `--ssl-keyfile PATH` | `DOWNSHIFT_SSL_KEYFILE` | none | The PEM private key. Not necessary if the certificate file contains the key. For an encrypted key, set `DOWNSHIFT_SSL_KEYFILE_PASSWORD`. |
| `--access-log` / `--no-access-log` | - | on | One log line for each request. |

`serve` binds the port first. It then loads, exports, verifies and warms up on a background thread. `/ready` gives `503` until the model is ready. `serve` serves all four verdicts. It exits early only for a usage error (`4`) or a crash (`5`) before the server is ready. Downshift loads the TLS files before the model, so a bad TLS file stops `serve` with exit code `4`.

```bash
downshift serve my_pkg.models:build --port 8000 --output-encoding base64
```

## Logging

All output goes through Python `logging` to one plain-text handler on stdout: the banner, the reports, the lines of uvicorn, `warnings` and the request lines. Each line has the form `time LEVEL logger: message`. A line that belongs to a request ends with ` request_id=<id>`. There is no JSON format and no colour.

| `--log-level` | What prints |
|---|---|
| `warning` (default) | The banner, the reports, and the `loading`, `will listen on` and `ready in N s` lines (logger `downshift.report`, which always prints). Also warnings, errors, and one line for each `4xx` and `5xx` request. |
| `info` | Adds the lines of uvicorn, the `serving with:` list of all settings, and one line for each request on `downshift.access`, for example `POST /predict 200 12.3 ms request_id=3f9c...`. The probes `/health` and `/ready` log only at `debug`. |
| `debug` | Adds tracebacks. For a FAILED verdict, it also adds the output of the torch export. |
| `error` | Only errors and the lines of `downshift.report`. |

`--no-access-log` removes the request line. The response keeps its `X-Request-Id`. With `--json`, the verdict is the only output on stdout, and the log lines go to stderr.

## Environment variables

Downshift reads these variables one time, when `downshift` is imported. To change one, restart the process. The variables also apply to library use (`ServeOptions()` and `app_for()`).

| Variable | Default | Sets |
|---|---|---|
| `DOWNSHIFT_HOST`, `DOWNSHIFT_PORT` | `127.0.0.1`, `8000` | `--host`, `--port` (CLI only) |
| `DOWNSHIFT_DEVICE`, `DOWNSHIFT_BACKEND` | `auto`, `auto` | `--device`, `--backend` |
| `DOWNSHIFT_WARMUP`, `DOWNSHIFT_SAMPLES` | `3`, `8` | `--warmup`, `-k` |
| `DOWNSHIFT_INTRA_OP_THREADS`, `DOWNSHIFT_INTER_OP_THREADS` | `0`, `0` | `--intra-op-threads`, `--inter-op-threads` |
| `DOWNSHIFT_OUTPUT_ENCODING` | `json` | `--output-encoding` |
| `DOWNSHIFT_MAX_INPUT_BYTES`, `DOWNSHIFT_MAX_BODY_BYTES` | 256 MiB, 32 MiB | `--max-input-bytes`, `--max-body-bytes` |
| `DOWNSHIFT_MAX_CONCURRENCY`, `DOWNSHIFT_MAX_QUEUE`, `DOWNSHIFT_REQUEST_TIMEOUT` | `4`, `64`, `30` | `--max-concurrency`, `--max-queue`, `--request-timeout` |
| `DOWNSHIFT_EXECUTION`, `DOWNSHIFT_PREP_THREADS` | `threadpool`, `min(4, CPUs)` | `--execution`, `--prep-threads` |
| `DOWNSHIFT_RESULT_CACHE_ROWS` | unset (`100`, or off) | `--result-cache-rows` |
| `DOWNSHIFT_AXIS_MAX` | unset | `--axis-max` (`NAME=N,NAME=N`) |
| `DOWNSHIFT_EXPORT_CACHE_DIR` | unset | `--export-cache-dir` |
| `DOWNSHIFT_WORKERS` | `1` | `--workers` (CLI only) |
| `DOWNSHIFT_SSL_CERTFILE`, `DOWNSHIFT_SSL_KEYFILE` | unset | `--ssl-certfile`, `--ssl-keyfile` (CLI only) |
| `DOWNSHIFT_SSL_KEYFILE_PASSWORD` | unset | The password of an encrypted key. There is no flag, because a flag shows in the process list. |
| `DOWNSHIFT_SERVER_API_KEY` | unset | Requires `Authorization: Bearer <key>` on each route except `/health` and `/ready`. There is no flag, because the key is a secret. |
| `DOWNSHIFT_TOL_{FLOAT32,FLOAT64,FLOAT16,BFLOAT16}_{ATOL,RTOL}` | refer to "Tolerances" | The default tolerances. |

A bad value (for example `DOWNSHIFT_PORT=abc`) stops the process at start with a message that names the variable. The library functions `check()`, `export()` and `intake()` use `8` as the default of `k`. They do not read `DOWNSHIFT_SAMPLES`.

## Exit codes

| Code | Meaning |
|---|---|
| `0` | `check` and `export`: the verdict is CLEAN. |
| `1` | `check` and `export`: the verdict is FAILED. |
| `2` | `check` and `export`: the verdict is DEGRADED. |
| `3` | `check` and `export`: the verdict is UNVERIFIED. |
| `4` | Usage error, on all commands. A `ValueError` left the command. Examples: a model that cannot load, a bad `--dynamic`, an unknown adapter, `--backend onnxruntime` on a DEGRADED verdict, `--device cuda` without CUDA, a bad TLS file. The message says what to change. |
| `5` | Any other error. `--log-level debug` also prints the traceback. |

## Add a `serve` option

1. If the option has an environment variable, add its default to `settings.py`.
2. Add a field to `ServeOptions` in `serve/options.py`. Take the default from `settings.py`.
3. Add an `Annotated` option type to `cli/options.py`.
4. Add a parameter to `serve_cmd` in `cli/main.py`, and pass it to `ServeOptions(...)`.
5. If the field is an enum, convert it in `ServeArgs.from_json` (`cli/runtime.py`). `--workers N` sends `ServeOptions` to each worker as JSON.
6. Add a row to the `serve` table and to "Environment variables" on this page, and describe the option in [`production.md`](production.md).
7. Add a test, and record the option in `CHANGELOG.md`.

The banner line `serving with:` (at `--log-level info`) lists all `ServeOptions` fields. A new field shows there without a change.
