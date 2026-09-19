# Changelog

All notable changes to this project are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## 0.4.0 - Unreleased

### Fixed

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

### Changed

- `downshift.export` (the subpackage) is `downshift.core` now; `downshift.export` still
  works for one release as a deprecated alias that warns `DeprecationWarning` on import.
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

### Added

- `python -m downshift` works as an alternative to the `downshift` script.
- CI: a `floor` job (Python 3.11, the oldest torch/onnx/onnxruntime/onnxscript the pins
  allow), a `windows` job, and an `examples` job that runs every tutorial script.
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

### Removed

- The `downshift version` subcommand. `downshift --version` still works.
- The built-in `generic`/`pyg`/`hf` adapters are no longer registered as `downshift.adapters`
  entry points. `downshift.adapters.registry` loads them directly, gated on
  `transformers`/`torch_geometric` already being imported, so discovering adapters for a
  plain PyTorch model no longer imports either.

## 0.3.0 - Unreleased

### Fixed

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

### Changed

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

### Added

- Binary tensor I/O: any input on `/predict` (and `x`, `edge_index`, `edge_attr` on
  `/predict/graph`) accepts `{"data": <base64>, "dtype": ..., "shape": [...]}`, and
  `"output_encoding": "base64"` (or `--output-encoding base64` /
  `DOWNSHIFT_OUTPUT_ENCODING`) returns outputs the same way. Decoded inputs are validated and
  capped by `--max-input-bytes` / `DOWNSHIFT_MAX_INPUT_BYTES` (default 256 MiB). Request
  bodies on every route are parsed with orjson. The `[fast]` extra installs `pybase64` for a
  SIMD codec; without it the stdlib codec is used and the banner prints a tip.
- `--max-body-bytes` / `DOWNSHIFT_MAX_BODY_BYTES` (default 256 MiB) caps every request body.
  Over the limit is a `413` whose body says the observed size and the limit.
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
