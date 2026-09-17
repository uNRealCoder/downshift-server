# Changelog

All notable changes to this project are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

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
