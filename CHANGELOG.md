# Changelog

All notable changes to this project are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## 0.4.0 - Unreleased

### Fixed

- `downshift serve mymodel:build` now finds `mymodel.py` in the current directory, as
  `python -m downshift` always did; the `downshift` console script used to fail with
  "No module named 'mymodel'". An import spec that names a file (`mymodel.py:build`) is a
  usage error that gives the module form, and the docs that showed the file form are fixed.
- A Hugging Face model's numerics check now always includes one full-length sample at the
  longest sequence the model declares (512 tokens for all-MiniLM-L6-v2). Before, every
  sample stayed within twice the example's length (16 tokens), so a graph that diverged
  only on long inputs could be verdicted CLEAN.
- `GET /metadata` and `GET /schema` no longer send the server's directory layout to whoever
  can reach the port: `model` and `source.spec` are the file or directory name only
  (`/srv/models/bert` reports `bert`; an import spec is unchanged), `verdict.onnx_path` is
  gone from `/metadata`, and paths quoted in `verdict.reason`, `verdict.warnings` and `notes`
  (ONNX Runtime's and transformers' load errors) are cut to their file name.
- The export manifest records `source_model` and `verdict.onnx_path` as file names, not the
  absolute paths of the machine that exported it, since the manifest travels with the `.onnx`.
- `--adapter <typo>` on `check`/`export`/`serve` is now a usage error (exit code 4), not an
  unhandled `KeyError` reported as exit code 5.
- The pinned dependency floors (`torch>=2.0`, `onnx>=1.14`, `onnxscript>=0.1`) didn't
  actually install or export together: `onnxscript>=0.1` requires `onnx>=1.16`, and
  `onnxscript` below `0.5` fails to translate ops torch 2.5's exporter emits for a
  plain dynamic-batch `Conv2d`. Raised the floors; a new `floor` CI job pins and tests them.
- A model whose parameters are `.double()` (float64) now gets float64 tolerances
  (`1e-6, 1e-5`) instead of silently checking against float32's looser default.
- A Hugging Face model whose example sits at `max_position_embeddings` (the model's own
  sequence-length ceiling) used to crash `check`/`export`/`serve` with a torch export error
  once the sampler tried a longer sequence; the sampler now clamps varied sequence lengths
  to that ceiling.
- `/health` and `/ready` no longer share a thread pool with in-flight predicts, so they stay
  fast (well under a millisecond) no matter how many predicts are running or queued; before,
  every route was a sync `def` on starlette's default 40-token thread pool, and enough
  predicts parked on it made `/health` wait behind them (measured: 5.7s at 60 in-flight).
- Overload used to mean a growing, unbounded queue of predicts until clients timed out; a
  predict past `--max-concurrency + --max-queue` admitted requests now gets an immediate
  `503` instead.
- `/ready` now means something: `serve` (single worker) binds the port first and loads,
  exports, verifies and warms up on a background thread; `/ready` is `503` with
  `{"ready": false, "phase": "export"}` (`phase` is `load`, `export`, `verify`, `session`
  or `warmup`, whichever the loader is in) until that lands, then `200`, with no restart in
  between. `/metadata` and both predict routes are `503` with `Retry-After` in the same
  window. Before, `/ready` was already `200` by the time uvicorn started serving, so it
  never carried information a client could act on. A failed load still exits the process
  with the same code `check`/`export` would use for the same error, instead of leaving the
  server up and permanently unready. With `--workers N`, a worker that fails to load stops
  the whole server instead of being respawned by uvicorn into the same failure forever.
- `downshift --help` (and every other subcommand's `--help`) no longer imports torch,
  onnxruntime, fastapi or uvicorn, and returns in well under a second instead of about
  5s. `downshift/__init__.py` is lazy (PEP 562): `check`, `intake`, `ExportVerdict` and
  the rest of `downshift.__all__` resolve on first access instead of at import time, and
  `downshift.cli.main` defers its own heavy imports into each command's body.
- A Hugging Face repo whose `config.json` declares a sequence- or token-classification
  architecture was loaded with `AutoModel`, which drops the head: `serve` returned the
  encoder's hidden states (the load report listed `classifier.weight` as `UNEXPECTED`), not
  the class logits. It now loads with its head. On Llama Prompt Guard 86M this also cleared a
  DEGRADED verdict that had sent the server to eager torch: the hidden states differed from
  ONNX Runtime by 5.6e-4 on values near 7, and the logits differ by 4e-05, so it is CLEAN and
  served on ONNX Runtime.
- A bfloat16 model on the torch fallback now boots (numpy has no bfloat16). bf16 and fp16
  models are served on a float32 wire: floating inputs are cast to the module's own dtype
  inside the torch backend and outputs are widened back, and `GET /schema` reports float32.
  On ONNX Runtime, a bfloat16 output needs an onnxruntime whose `OrtValue` supports DLPack;
  an older one gets an error that points at `--backend torch`.
- A negative token id sent to a Hugging Face repo model was wrapped around by ONNX Runtime
  and came back as a confident `200`. `input_ids` outside `[0, vocab_size)` is now a `400`
  naming the value, before inference.
- The torch backend answered every `RuntimeError`/`ValueError` from the model with a `400`,
  blaming the client for bugs and unsupported ops. Server-side faults are now a `500`: out of
  memory (CPU or GPU), a CUDA/cuDNN/cuBLAS error, an internal assert, a missing kernel, or
  tensors on different devices. The rest stay a `400`, since torch reports the client's
  shape, dtype and index mistakes in too many phrasings to list.
- The usable sequence length of a RoBERTa-family Hugging Face model is
  `max_position_embeddings - (pad_token_id + 1)` (514 positions serve 512 tokens), not
  `max_position_embeddings`.
- `--device cuda` without CUDA silently ran on the CPU. It is now an error (exit code 4) on
  both backends.
- A predict is admitted, or refused with a `503`, before its body is read, and a chunked
  body over `--max-body-bytes` is refused with a `413` as soon as its running total passes
  the limit, instead of after the whole body was buffered.
- A graph that exported but that ONNX Runtime could not load or run (for example, bfloat16
  weights, which have no CPU Gemm kernel) used to crash `check`/`export`/`serve` with exit
  code 5. It is now a FAILED verdict whose reason carries ONNX Runtime's own error message,
  and the model is served via PyTorch instead.
- The numerics pass/fail rule is now the same one `numpy.allclose` uses, applied per element:
  `abs_err <= atol + rtol * |expected|`. Before, a sample failed only when its per-sample
  maximum absolute error *and* maximum relative error both exceeded tolerance, and those two
  maxima could come from different elements; models whose outputs mix large logits with
  near-zero values could be marked DEGRADED wrongly, or pass wrongly. Output shape or count
  mismatches between the torch and ONNX Runtime outputs now fail the sample with a note in
  `numerics.notes` instead of raising.
- bfloat16 and float16 model outputs are compared correctly (widened to float32 first)
  instead of hitting numpy's missing bfloat16 dtype.
- `check(model, fp16=True)` (and `downshift export --fp16`) no longer converts the caller's
  model to float16 in place; it now works on a deep copy.
- Importing `downshift` no longer changes global logging levels.
- On the torch fallback, `GET /schema` treated every input's first axis as the batch axis,
  so a graph model's `edge_index` (`[2, E]`) got an `example_request` of shape `[1, E]` that
  `/predict` then rejected. The torch backend now reports the axes the adapter actually made
  dynamic and keeps every other axis at its real size, like ONNX Runtime does.
- CI was red on every job: two tests built ONNX graphs stamped with the installed onnx's
  newest IR version (14), which the installed onnxruntime can't read (it stops at 13), and a
  failing worker load ended the whole `floor` pytest run through `os._exit`. The test graphs
  now pin their IR version, a test calling `os._exit` fails just that test, and the tests
  that export a Hugging Face encoder, which torch 2.5's exporter can't handle, carry
  `needs_torch_26` so the `floor` job skips them. Checked locally on Python 3.12 (floor pins
  and current releases) and Python 3.14.
- `docs/compatibility.md` and the README's matrix are regenerated (torch 2.14, onnx 1.22,
  onnxruntime 1.30): same verdict and backend for every fixture.

### Added

- **An optional API key.** Set `DOWNSHIFT_SERVER_API_KEY` and every route except `/health`
  and `/ready` requires `Authorization: Bearer <key>` (constant-time compare); anything else
  is a `401` with `WWW-Authenticate: Bearer` and `{"detail": "Authorization header is not
  set or incorrect"}`. Unset or empty logs one `WARNING` at startup that the endpoints are
  unauthenticated (set the key or add your own middleware). `build_app(..., api_key=)` and
  `app_for(..., api_key=)` read the variable by default.
- `GET /schema` inputs carry the adapter's axis names and per-axis `bounds` (`min`, `max`)
  for a model downshift exported itself, and a request outside them is a readable `400`
  (`input_ids axis 1 is 65; this model accepts 1 to 64`) instead of an ONNX Runtime or torch
  error. A `--workers N` worker answers the same way. A bare `.onnx` with no `--reference`
  has no bounds.
- `Server-Timing` includes `parse`, the time spent parsing the request's JSON.
- The boot banner has a `Verified on` row (the numerics check runs on the CPU only, and
  the row says so when the server runs elsewhere), and one `Capacity` row (`--max-concurrency`,
  `--max-queue`, `--request-timeout`) in place of the separate `Queue` and `Concurrency`
  rows. Its endpoint reads `localhost`, plus the bind address when it is `0.0.0.0`.
- `.gitattributes` (`* text=auto eol=lf`).
- `POST /predict` takes `{"text": ...}` for a Hugging Face repo directory that has tokenizer
  files: the server tokenizes and pads the batch. A row longer than the model reads is refused
  with a 400 rather than cut; the only cap on input size is the request body limit.
  `GET /schema` gains a `text_input` block.
- A sequence classifier answers with `predictions` (label, score and per-label probabilities:
  softmax, or sigmoid for a multi-label config) next to the raw logits.
- **Embedding models.** A sentence-transformers repo (`modules.json`, `1_Pooling/config.json`)
  is served as an embedding model: the pooling (mean, cls, max, mean-sqrt-len) and optional
  L2 normalisation are read from the repo and put inside the exported graph, so `output_0` is
  `[batch, dim]` and ONNX Runtime runs the whole thing. Checked on all-MiniLM-L6-v2: CLEAN
  (max abs err 4e-07), and within 1.5e-07 of a reference computed the sentence-transformers
  way. Text longer than the repo's own `max_seq_length` (256 for MiniLM, not its 512 position
  embeddings) is refused with a 400; `GET /schema` gains an `embedding` block.
- `--pooling mean|cls|max|mean_sqrt_len|none` and `--normalize/--no-normalize` on `check`,
  `export` and `serve`, for a repo that declares no recipe or to override one. `none` serves
  token vectors. A recipe downshift cannot apply faithfully (a Dense module, last-token or
  weighted pooling, several poolings at once) is refused at load rather than approximated.
- `serve model.onnx --tokenizer-from path/to/repo/dir/` loads the repo directory's tokenizer,
  pooling recipe and label metadata, so `{"text": ...}` and embedding/classifier output work
  on an already-exported `.onnx` the same way they do when the repo itself is MODEL. The
  served graph is used as-is: `GET /schema` reports the repo's pooling recipe only when the
  graph's output is already one vector per row, and `--pooling`/`--normalize` are ignored
  with a banner note, since they can't change a graph that is already built.
  Independent of `--reference`, which stays purely numeric: the two flags can name the same
  directory or different ones, and neither implies the other. A `--workers N` worker gets the
  resolved path from the parent rather than re-validating `--tokenizer-from` itself.
- **`GET /schema`**: what to POST, without having to have seen the model. Every input's
  name, dtype and shape (read off the graph that is actually running, not off the source
  model), the outputs, the three accepted wire formats, the body-size limits, and an
  `example_request` that can be posted straight back to `/predict` unchanged - plus
  `example_curl`, the same body as a runnable command against the server's own URL. A
  dynamic axis is reported by the name its adapter chose (`batch`, `seq`) or as the plain
  word `dynamic`, never as `torch.export`'s internal `s77`. On a model whose example would
  run past 256 elements, `example_request` is `null` and `notes` gives the shape to build.
  `/metadata` stays the operator's view (verdict, boot timings, warmup); `/schema` is the
  caller's.
- `/schema`'s `source` block says which accepted form a server was given
  (`onnx-file`, `torch-checkpoint`, `hf-repo-dir`, `import-spec`, `in-process-module`),
  with `fetched_at_runtime: false` - a client can see the server is pinned to one artifact
  already on that machine. The `serve` boot banner labels the same thing on its `Model`
  row, and `downshift.loading.source_kind()` exposes it to library callers.
- `python -m downshift` works as an alternative to the `downshift` script.
- CI: a `floor` job (the oldest torch/onnx/onnxruntime/onnxscript the pins allow) and a
  `windows` job.
- `--atol`/`--rtol` on `check`, `export` and `serve` override the tolerance `verify()`
  would otherwise pick from the model's floating dtype. The `check` table gets a
  `Tolerance` row, and `NumericsReport.tolerance_dtype` records which dtype chose it.
- `--seed` (default 0) on `check`, `export` and `serve` makes verification samples
  reproducible; recorded on `NumericsReport.seed` and so in `--json`/the manifest.
- `--vary pkg.module:fn` on `check`, `export` and `serve` (and `vary=` on the library
  `check()`/`prepare_model()`) supplies your own `fn(i) -> inputs` for verification samples
  in place of the adapter's own; `fn(0)` must return the example inputs.
- `downshift.core.shapes.dim_bounds(spec, axis)` reads a dynamic axis's `(min, max)` off
  its `torch.export.Dim`, falling back to `(1, 1 << 16)` if the attributes aren't there.
  The default sampler and the `pyg`/`hf` adapters now clamp their varied sizes to it, so a
  generated sample never exceeds what the model was actually declared to support.
- `NumericsReport.worst`, the single largest-error output element across every sample tried
  (output index, unravelled element index, expected value, got value, input shapes);
  rendered as a `Worst` row in the `check` table on DEGRADED.
- `NumericsReport.sample_shapes`, the input shapes of every verification sample tried;
  rendered as a `Samples` row.
- A FAILED `Reason` row now points at `--log-level debug`, which logs torch's export
  stderr and every strategy's traceback (`ExportVerdict.capture_stderr`,
  `.capture_exceptions`).
- The `Tolerance` row says `(--atol/--rtol)` instead of naming a dtype that didn't choose
  the values, when either flag overrides the default (`NumericsReport.tolerance_overridden`).
- `serve` now warms up a bare `.onnx` served without `--reference` too:
  `engine.synthesize_feeds()` builds one dummy input per input the graph declares (dynamic
  or unknown axes at 1, floats from `randn`, integers/bools zero), so first-call costs no
  longer land on the first real request. `warmup()` returns `WarmupStats` (count, mean ms,
  whether inputs were synthesized), stored on `ServingState.warmup_stats`.
- `--max-queue N` (env `DOWNSHIFT_MAX_QUEUE`, default 64) caps predicts admitted past
  `--max-concurrency`; past `max-concurrency + max-queue` a predict gets an immediate `503`
  with `Retry-After: 1` and a body naming how many are running and queued, instead of
  joining an unbounded queue. `/metadata`'s `limits` and the banner's `Capacity` row
  report it.
- `--request-timeout SECONDS` (env `DOWNSHIFT_REQUEST_TIMEOUT`, default 30; `0` turns it
  off) fails an admitted predict with a `503` if it is still waiting for its turn once the
  limit passes, so it counts time queued; a request already running is never interrupted.
  Reported in `/metadata`'s `limits`.
- Every response carries an `X-Request-Id` header: echoed from the client's own if it sent
  one, otherwise a generated `uuid4().hex[:16]`.
- `/predict` and `/predict/graph` responses carry `Server-Timing: parse;dur=<ms>,
  codec;dur=<ms>, infer;dur=<ms>`, splitting JSON parsing and request/response conversion
  from the backend call.
- `--access-log`/`--no-access-log` (default on) controls one log line per request on the
  `downshift.access` logger: `METHOD PATH STATUS N ms request_id=...`. See "Logging" under
  Changed.
- The `serve` banner explains its own verdict: `Tolerance`, `Worst` (DEGRADED only) and
  `Samples` rows (from V4/V5's `NumericsReport` fields), a `Warmup` row (from `WarmupStats`),
  a `Capacity` row (`--max-concurrency`, `--max-queue`, `--request-timeout`), and a `Boot`
  row breaking the time to ready down into `load`, `export`, `verify`, `session` and
  `warmup` seconds (`ServingState.timings`).
- `/metadata` gains `boot` (the same per-phase timings as the banner's `Boot` row) and
  `warmup` (`count`, `mean_ms`, `synthesized`).
- `downshift.serve.app_for(model, example_inputs=None, *, source=, reference=,
  middleware=, api_key=, **options)`: `LoadedModel` + `prepare_serving()` + `build_app(state=...)`
  in one synchronous call, for mounting downshift's routes inside an existing FastAPI app
  (`app.mount("/model", app_for(model))`) instead of running a whole process. `model` is
  a `torch.nn.Module` or a pre-built `.onnx` path (`reference` then verifies it, like
  `--reference`). Exported from `downshift.__all__` too.
- [docs/production.md](docs/production.md): the API key, probe semantics per `--workers`
  mode, admission control and what a `503` means, thread budgeting, sizing off the banner's
  `Boot` row, logging, request ids and `Server-Timing`, and a Kubernetes snippet. Linked
  from the README's Serve section.
- [docs/code-docs/](docs/code-docs/README.md): reference pages for the CLI, the HTTP API,
  the Python API, the serving layer and adapters, with a module map.
- Binary tensor I/O: any input on `/predict` (and `x`, `edge_index`, `edge_attr` on
  `/predict/graph`) accepts `{"data": <base64>, "dtype": ..., "shape": [...]}`, and
  `"output_encoding": "base64"` (or `--output-encoding base64` /
  `DOWNSHIFT_OUTPUT_ENCODING`) returns outputs the same way. Decoded inputs are validated and
  capped by `--max-input-bytes` / `DOWNSHIFT_MAX_INPUT_BYTES` (default 256 MiB). Request
  bodies on every route are parsed with orjson. The `[fast]` extra installs `pybase64` for a
  SIMD codec; without it the stdlib codec is used and the banner prints a tip.
- `--max-body-bytes` / `DOWNSHIFT_MAX_BODY_BYTES` (default 64 MiB) caps every request body.
  Over the limit is a `413` whose body says the observed size and the limit. There is no
  text-length limit; the body cap covers a `{"text": ...}` request.
- `--max-concurrency` / `DOWNSHIFT_MAX_CONCURRENCY` (default 1): inferences allowed to run at
  once per worker process. ONNX Runtime's intra-op threads still parallelise inside one
  inference. Shown in the `serve` banner.
- `/metadata` includes `limits` (`max_body_bytes`, `max_input_bytes`, `max_concurrency`).
- `downshift --version`.
- `downshift.demo` models (`clean_mlp`, `scatter_include_self_false`, `data_dependent_branch`)
  shipped inside the package, so the README's own commands work right after
  `pip install downshift-server`.
- `downshift.OnnxRuntimeError`, `NumericsReport.notes`.
- `py.typed`, project URLs on PyPI, Python 3.14 in CI.

### Changed

- **Adapters are classes; the registry creates their instances.** The built-in adapter
  modules no longer export module-level singletons (`downshift.adapters.generic.ADAPTER`,
  `downshift.adapters.hf.HF_ADAPTER` and `downshift.adapters.pyg.PYG_ADAPTER` are gone). Use
  `registry.get("generic")`, or construct `GenericAdapter()`, `HFAdapter()` or `PyGAdapter()`
  directly. The Hugging Face repo readers (`load_pretrained`, `load_text_io`, ...) moved from
  `downshift.adapters.hf` to `downshift.hf_repo`, so loading and serving no longer import
  the adapter. The documented way to register a plugin is now to
  point the entry point at the class (`myfamily = "my_pkg.adapter:MyAdapter"`), and the
  registry instantiates it with no arguments. Entry points and `--adapter file.py` specs that
  point at an instance still work.
- **Python 3.12 is now the minimum** (`requires-python = ">=3.12,<3.15"`); 3.11 is no longer
  supported. CI drops the 3.11 leg from the test matrix, and the `floor` job (oldest
  torch/onnx/onnxruntime/onnxscript pins) now runs on 3.12. Ruff targets `py312`.
- **`downshift` only serves models already downloaded onto this machine, and says so
  everywhere.** The accepted set is three kinds of artifact on disk - a `.onnx` file, a
  PyTorch checkpoint (`.pt`/`.pth`/`.bin`/`.ckpt`), and a Hugging Face repo directory,
  recognised by the `config.json` in it - plus an import spec naming a model already
  importable in the process. A bare repo id (`bert-base-uncased`, `org/repo`) is not an
  accepted `MODEL` argument and is rejected with an error pointing at
  `huggingface-cli download`; the `hf` adapter loads with `local_files_only=True`, so no
  model load reaches the network. A `serve` in an air-gapped or egress-restricted
  environment can no longer silently depend on a download at startup. `--help`, the load
  errors, the boot banner and `GET /schema` are all framed in those terms.
- `downshift.export` (the subpackage) is renamed to `downshift.core`, with no compatibility
  shim. `downshift.export(...)` the function (write a `.onnx` and its manifest) is unchanged.
- `torch` floor raised to `2.5` (the dynamo exporter accepting an `ExportedProgram` with
  `report=` is a 2.5 feature); `onnx` floor raised to `1.16`; `onnxscript` floor raised to
  `0.5`. See Fixed above.
- The default tolerance policy now checks bfloat16, then float16, then float64 (only when
  it's the sole floating dtype present), then falls back to float32 - the narrowest
  floating dtype present picks the tolerance, not the widest as the README used to say.
- Verification samples after the first now vary a floating input by tiling or slicing the
  example's own rows and adding noise scaled to the example's own spread, instead of
  drawing pure `torch.randn`. Samples look like plausible inputs instead of unrelated
  noise. Checked this against the fixture corpus: every fixture's verdict, shape
  generalization, and backend stayed the same; only the reported `max_abs_err` moved
  (still comfortably inside tolerance everywhere it was CLEAN before).
- The built-in `hf` and `pyg` adapters' verification samples now draw their sizes from
  torch's global RNG (already seeded and forked per sample by `verify()`) instead of a
  `random.Random(0)` fixed at `prepare()` time, so `--seed` actually reproduces them.
- Verification samples after the first Hugging Face one now vary per-row sequence length
  and zero `attention_mask` beyond it (every row keeps at least one attended position), so
  padding is actually exercised instead of every sample using a full mask.
- When a downshift-generated verification sample (not the caller's own example) makes the
  model raise, the error now says the sample came from downshift's sampler, gives the
  shapes and bounds it was drawn from, and points at `--vary` or a custom adapter, instead
  of suggesting `--dynamic`/`--inputs` as if it were the user's own input.
- `op_types` is a count-descending histogram now, not a flat list: `["Gemm", "Relu", "Gemm"]`
  is `{"Gemm": 2, "Relu": 1}`. Affects `CaptureResult.op_types`, `ExportVerdict.op_types`,
  and so `check --json`, the manifest, and `/metadata`.
- `shape_generalization` is `null` (not `false`) when the baseline sample itself failed,
  since shape generalization was never evaluated in that case; a new
  `shape_generalization_reason` string in `to_dict()` says why. The `check` table's
  `Shape-general` row shows `n/a (baseline fails)`, `no`, or `yes` accordingly
  (`NumericsReport.baseline_failed`).
- A FAILED verdict's `reason` now quotes the first export strategy's exception
  (`strict=False`) instead of the last (`strict=True`), and `unsupported_ops` is mined from
  every strategy tried, not just the one whose message happened to survive.
- `capture()` serialises the ONNX graph to bytes exactly once (`CaptureResult.onnx_bytes`,
  `ExportVerdict.onnx_bytes`); `verify()` and `serve`'s backend selection both consume it,
  and `serve` reuses `verify()`'s own `InferenceSession` when the serving options mean the
  same thing (CPU, default thread counts), instead of building a second session from a
  second serialisation. One `InferenceSession` per default boot instead of two.
- `serve --workers N` now captures and verifies the model exactly once, in the parent
  process, instead of once per worker. When the chosen backend is ONNX Runtime, the parent
  writes the exported graph (and, if it had real example inputs, a `.npz` of them) to a
  temp file and ships them plus the verdict to every worker (`ExportVerdict.from_dict()`,
  `NumericsReport.from_dict()`, `engine.serving_state_from_artifact()`); workers just load
  the graph and warm up. When the backend is torch (DEGRADED, FAILED, `--backend torch`),
  workers still load the model (there is no way around N eager copies) but skip
  capture/verify and take the verdict as given. The startup warning about
  per-worker cost is now printed only in that torch case, since the ONNX Runtime path no
  longer pays it.
- `serve --workers N` with `--intra-op-threads` left at its default now splits the logical
  CPU count across workers (`max(1, cpu_count // N)`) instead of leaving every worker free
  to claim every core, N-fold oversubscription on the flag the README recommends for more
  throughput. Passing `--intra-op-threads` explicitly still wins. The torch backend calls
  `torch.set_num_threads` with the same budget. The banner gets a `Threads` row when
  `--workers` is more than 1.
- `predict`/`predict/graph` are `async def` now. The missing-input check and admission
  happen on the event loop; conversion of the request body, the inference itself, and
  response encoding run in a per-`ServingState` `ThreadPoolExecutor(max_workers=
  max_concurrency)` instead of inline behind a semaphore, so a large JSON body's conversion
  never blocks the loop either. `ServingState.inference_semaphore` is gone, replaced by
  `.executor` and an `.in_flight` counter.
- A `500`'s body now carries `"request_id"` alongside `"detail"`, and every log line
  written while that request was served carries the same id, so "see the server log" has a
  key to search for.
- `serve --backend onnxruntime` on a DEGRADED verdict is now a usage error (exit code 4)
  naming `--force-onnx`, instead of a second warning next to the one `--force-onnx` already
  prints. `--backend onnxruntime --force-onnx` still serves the ONNX graph.
- `build_app(state, ...)` becomes `build_app(state=None, *, loader=None, middleware=(),
  api_key=<DOWNSHIFT_SERVER_API_KEY>, access_log=True)`: pass `state` for the old
  synchronous behaviour, or `loader` for the bind-first behaviour above.
  `run_predict`/`_predict_body` are unchanged.
- **Logging.** Everything goes through Python `logging` into one plain-text sink on stdout:
  the boot banner, the `check`/`export` reports, uvicorn's own startup and error lines,
  `warnings`, and the request line. There are no colours, no box drawing and no JSON format.
  `--log-level` defaults to `warning` for `check`, `export` and `serve`; the banner, the
  reports and the `loading` / `will listen on` / `ready in X s` lines are logged on
  `downshift.report`, which always prints. `--log-level info` adds uvicorn's lines and one
  line per request (`downshift.access`; `/health` and `/ready` are `DEBUG` only, and `4xx`
  and `5xx` are `WARNING`, so they show at the default level); `debug` adds tracebacks.
  With `check --json` or `export --json` the verdict is the only thing on stdout and log
  lines go to stderr. Every line written while serving a request carries its
  `request_id`.
- Every `DOWNSHIFT_*` environment variable now also applies to library use (`app_for()`,
  `ServeOptions()`), not only to the CLI. They are read once, at import time.
- With `--workers N` the parent frees its copy of the model before the workers start.
- **Internals.** The Hugging Face text and embedding pieces live in `adapters/text.py`,
  `adapters/pooling.py` and `adapters/embedding.py`. The CLI is split into `cli/main.py` (commands), `cli/options.py` (typer option types) and
  `cli/runtime.py` (`ServeArgs`, the `--workers` handoff); `ServeArgs` is
  `{load: LoadSpec, options: ServeOptions, ...}`, so a new `serve` option is a typer
  parameter, a `ServeOptions` field and a `_collect_serve_args` parameter. The `serve`
  package is split into `predict.py` and `describe.py` beside `app.py`. New stdlib-only leaf
  modules: `logs.py`, `_imports.py` (`import_object`; `loading.py` re-exports it) and
  `sources.py`. A `Phase` `StrEnum` (`load`, `export`, `verify`, `session`, `warmup`, in
  `core/phase.py`) drives `Boot` timings, `/ready`'s `phase` and the banner.
- `/predict` and `/predict/graph` return `400` only for client-caused problems (bad JSON
  shape/dtype, an input the backend rejects). Server-side faults are now a `500` with the
  fixed body `{"detail": "inference failed on the server; see the server log"}`; the
  exception text is logged, not returned to the client.
- Responses from `/predict` and `/predict/graph` are now serialized by orjson straight from
  the numpy buffers. The API is unchanged; two things on the wire are not:
  - `NaN` and `Inf` outputs serialize as `null`, which is valid JSON, instead of the
    `NaN`/`Infinity` tokens the stdlib encoder emitted. A client that parsed those tokens
    must handle `null`.
  - float32 values print with the shortest decimal that round-trips as float32 (`0.1`, not
    `0.10000000149011612`). Cast back to float32 and the values are bit-identical to before.

### Removed

- The `downshift version` subcommand. Use `downshift --version`.
- The built-in `generic`/`pyg`/`hf` adapters are no longer registered as `downshift.adapters`
  entry points. `downshift.adapters.registry` loads them directly, gated on
  `transformers`/`torch_geometric` already being imported, so discovering adapters for a
  plain PyTorch model no longer imports either.
- The `examples/` directory (seven tutorials as scripts and notebooks, and the notebook
  builder) and the CI `examples` job that ran them. The README, `docs/production.md` and
  `docs/code-docs/` carry inline snippets instead.
- The `rich` dependency: the banner and reports are plain text now.

## 0.2.0 - 2026-09-13

### Fixed

- `requires-python = ">=3.11,<=3.14"` was a PEP 440 trap: `<=3.14` matches only the exact
  `3.14.0` release, so pip refused to install on every real 3.14 patch version. Changed to
  `<3.15`.
- `__version__` had drifted from `pyproject.toml`'s version after a bump, so `downshift
  version`, the `serve` banner, `/metadata`, and export manifests all recorded the wrong
  version.
- `BackendName(str, Enum)` printed as `BackendName.onnxruntime` in the `serve` banner on
  Python 3.11+; it now prints the plain string.
- Dropped the deprecated `typer[all]` extra, which printed an install-time warning.
- The default `--host` is `127.0.0.1`, not a wildcard bind.

### Added

- Custom adapters can be loaded directly from a standalone `.py` file with `--adapter
  path/to/adapter.py[:Class]`, no install or entry point required.
- Loading a locally downloaded Hugging Face repo directory (one containing `config.json`).
- Python 3.13 exercised in the CI and release test matrices (already declared in
  classifiers and `requires-python`).

### Changed

- Manifest dtype names moved from a plain dict to an enum internally; no user-facing change.

## 0.1.0 - 2026-09-12

Initial release.

### Added

- `downshift.check()` / `downshift check`: export a PyTorch model to ONNX in memory via
  `torch.export`, verify its numerics against `k` random samples (varying dynamic axes so
  some samples have shapes the exporter never saw), and return one of four verdicts: CLEAN,
  DEGRADED, FAILED, UNVERIFIED.
  `--json`, `-k/--samples`, `--dynamic`, `--adapter`, `--inputs`.
- `downshift.export()` / `downshift export`: write the `.onnx` artifact plus a
  `.manifest.json` sidecar recording source/artifact checksums, library versions, opset,
  observed weight dtype, and the full verdict. `--fp16`, `--no-verify`.
- `downshift serve`: a FastAPI app that picks a backend (ONNX Runtime or eager PyTorch) from
  the verdict and serves `/predict`, `/predict/graph`, `/health`, `/ready`, and `/metadata`.
  `--backend`, `--force-onnx`, `--reference`, `--middleware`, `--device`, `--warmup`,
  `--workers`.
- Model-family adapters: `generic` (any `nn.Module`), `pyg` (PyTorch Geometric), `hf`
  (Hugging Face encoders), plus a documented `Adapter` protocol for third-party adapters
  registered under the `downshift.adapters` entry-point group.
- Model loading from an import spec (`pkg.module:attr`), a pre-built `.onnx` file, a
  `weights.pt` state dict with `--model-class`, or a Hugging Face hub id.
- `scripts/gen_matrix.py`: a compatibility matrix generated from a fixture corpus in
  `tests/models/`, one fixture per export hazard, published to `docs/compatibility.md` and
  refreshed weekly by CI.
- Seven runnable tutorials under `examples/`, each as both a plain script and a Jupyter
  notebook.
