# Python API

This page describes every name in `downshift.__all__`: the functions that run the export-and-verify gate, the types that the gate returns, and how `import downshift` avoids the import of torch.

## Lazy import (PEP 562)

`downshift/__init__.py` defines `__all__` and a `_LAZY` map from a name to `(module, attribute)`. A module-level `__getattr__` resolves each name on its first access. It imports the module that defines the name:

- `downshift.core.verdict`
- `downshift.core.verify`
- `downshift.adapters.base`
- `downshift.core.prevalidated`
- `downshift.serve`

It then caches the result in `globals()` of `downshift`. Each later access is a plain attribute lookup. There is no more call to `__getattr__`.

The effect is that `import downshift` does not import torch, onnx or onnxruntime. Downshift imports nothing until you first use `downshift.check`, `downshift.Adapter` or another lazy name.

Two names are different. `__version__` is a plain string. Downshift imports it eagerly from `downshift._version`. `export` is defined directly in `__init__.py`. Its heavy imports (`downshift.core.manifest` and `downshift.core.verdict`) are inside its function body.

For this reason, `downshift.cli.main` can answer `--help` and `--version` without the import cost of torch. At module load, it imports only `downshift` and `downshift.settings`. It does the other `downshift.*` imports inside the command bodies that need them.

## Functions

### `check`

```python
def check(
    model: torch.nn.Module,
    example_inputs: tuple | None = None,
    k: int = 8,
    adapter: Adapter | str | None = None,
    dynamic: dict[str, list[int]] | None = None,
    fp16: bool = False,
    verify_numerics: bool = True,
    atol: float | None = None,
    rtol: float | None = None,
    seed: int = 0,
    vary: VaryFn | str | None = None,
    axis_max: dict[str, int] | None = None,
    cache: bool = True,
) -> ExportVerdict
```

`check` exports `model` to ONNX in memory and verifies it against PyTorch. It writes nothing to disk. `export()` is `check()` plus the writing of the artifact.

- `example_inputs`: a tuple of forward arguments. If it is `None`, the adapter that downshift resolves makes a guess (see `prepare_model`).
- `k`: the number of verification samples.
- `adapter`: an `Adapter` instance, an adapter name (`"generic"`, `"pyg"`, `"hf"`), a `path/to/adapter.py[:attr]` spec, or `None` to detect the adapter.
- `dynamic`: `{input_name: [axis, ...]}`. It overrides the dynamic axes. The default is axis 0 of every input.
- `fp16`: casts a deep copy of `model` to `float16` before the export. The model of the caller and the tensors in `example_inputs` do not change. Downshift sets the model to `eval()` in place if it was in training mode. When `fp16=True`, downshift does this on the copy. The model of the caller then stays in its own mode.
- `verify_numerics`: `False` skips the numerics check. This is the `--no-verify` behaviour. The graph is still exported, but the verdict can be only `UNVERIFIED` and never `CLEAN`.
- `atol` and `rtol`: override the absolute and relative tolerance. `None` (the default for both) selects the tolerance from the floating dtype of the model. Refer to `downshift.core.verify.default_tolerances` and `settings.TOLERANCES`.
- `seed`: seeds the generator of verification samples. `verify()` runs under `torch.random.fork_rng`, so the global RNG state of the caller does not change.
- `vary`: a `fn(i) -> inputs` that replaces the sampler of the adapter. Downshift imports a string in the same way as for `--vary`. `fn(0)` must return the example inputs.
- `axis_max`: `{axis_name: N}`. It lowers a named dynamic axis to `N` and uses one verification sample at exactly `N`. Refer to `--axis-max`.
- `cache`: if `True` (the default), downshift keeps the last two verified exports in memory. A repeat call for the same model in the same process then skips the export and the verification. Set `False` to turn this off.

`check` returns an `ExportVerdict`. It raises `ValueError` in these cases:

- No example inputs are available, and downshift could not synthesize any.
- The model raises an error on the first (baseline) verification sample.
- `adapter` names an adapter that downshift cannot find.

It raises `RuntimeError` if no adapter matches. This can happen only if the built-in `generic` adapter is also not available.

A model that exports, but that ONNX Runtime cannot load or run, is not an error here. The result is a `FAILED` verdict.

### `export`

```python
def export(
    model,
    output: str | Path,
    example_inputs: tuple | None = None,
    k: int = 8,
    adapter: Adapter | str | None = None,
    dynamic: dict[str, list[int]] | None = None,
    fp16: bool = False,
    source_path: Path | None = None,
    verify_numerics: bool = True,
    atol: float | None = None,
    rtol: float | None = None,
    seed: int = 0,
    vary: VaryFn | str | None = None,
    axis_max: dict[str, int] | None = None,
    cache: bool = True,
    export_cache_dir: str | Path | None = None,
) -> ExportVerdict
```

`export` is `check()` plus the writing of `output` (the `.onnx` path) and a `.manifest.json` file next to it (`downshift.core.manifest.write_manifest`). Each parameter that is not in this list has the same meaning as on `check()`.

- `output`: the path of the `.onnx` file. Downshift creates missing parent directories.
- `export_cache_dir`: a directory for the disk tier of the export cache. The key is the content of the weights. A restart then reuses a verified export.
- `source_path`: if it points to a real file, downshift records it in the manifest as the source checkpoint. The manifest has the file name only, never the directory. It also has the SHA-256 of the file.

A `FAILED` verdict writes nothing. There is no `.onnx` file and no manifest. A `DEGRADED` verdict writes the artifact, because the manifest records how far the numbers differ. In that case, `verdict.onnx_path` is set on the returned verdict. `export` raises the same errors as `check()`.

### `intake`

```python
def intake(
    onnx_path: str | Path,
    reference: torch.nn.Module | None = None,
    example_inputs: tuple | None = None,
    adapter: Adapter | str | None = None,
    k: int = 8,
    dynamic: dict[str, list[int]] | None = None,
    atol: float | None = None,
    rtol: float | None = None,
    seed: int = 0,
    vary: VaryFn | str | None = None,
    axis_max: dict[str, int] | None = None,
    timings: dict[str, float] | None = None,
) -> ExportVerdict
```

`intake` is the equivalent of `check()` for a `.onnx` file that someone else made (Olive, a notebook, or other tools).

- Without `reference`, `intake` returns an `UNVERIFIED` verdict immediately. The model is served as it is. `reason` says that the numbers were never checked.
- With `reference`, `intake` runs the same numerics gate as `check()` and `export()`. The verdict is `CLEAN`, `DEGRADED` or `FAILED`, with the same rules.
- `timings`: if you give it, downshift adds a `"verify"` entry in seconds of wall-clock time. A pre-built graph has no `"export"` phase.

The command `downshift check some.onnx --reference some_model.py:build` runs this function. The `check` command of the CLI calls `intake` directly when its `MODEL` argument is an existing `.onnx` file.

### `build_verdict`

```python
def build_verdict(
    prepared: Prepared,
    k: int = 8,
    verify_numerics: bool = True,
    atol: float | None = None,
    rtol: float | None = None,
    seed: int = 0,
    timings: dict[str, float] | None = None,
) -> ExportVerdict
```

`build_verdict` is the half of `check()` that runs after the resolution of the adapter and the synthesis of the inputs. It does the capture (`torch.export` plus the ONNX translation) and then the numerics verification. `check()` is `build_verdict(prepare_model(...))`, with `fp16` handled between the two calls.

Use it directly when you already have a `Prepared`. For example, you can run a custom adapter once and inspect the result before you decide to export.

- `timings`: if you give it, downshift adds `"export"` (the wall-clock time of the capture) and `"verify"`. `ServingState.timings` and the `boot` field of `/metadata` show the same dict.

It raises `ValueError` if the model raises an error on the baseline (unvaried) verification sample.

### `prepare_model`

```python
def prepare_model(
    model: torch.nn.Module,
    example_inputs: tuple | None = None,
    adapter: Adapter | str | None = None,
    dynamic: dict[str, list[int]] | None = None,
    vary: VaryFn | str | None = None,
    axis_max: dict[str, int] | None = None,
) -> Prepared
```

`prepare_model` does three things:

1. It resolves an adapter: an explicit instance or name, or `registry.detect()` against the model and the inputs.
2. It synthesizes example inputs if you gave none.
3. It asks the adapter to flatten the model into the form that is ready for export.

If you give `axis_max`, downshift passes it to `adapter.prepare()`. If you give `dynamic`, it replaces the `dynamic_shapes` of the adapter through `downshift.core.shapes.apply_dynamic_override`. If you give `vary`, it replaces the `vary_fn` of the adapter. Downshift imports a string in the same way as the `pkg.module:fn` form of `--vary`.

It raises `ValueError` if `example_inputs` is `None` and `example_inputs()` of the adapter also returns `None`. It raises `RuntimeError` if `matches()` of no registered adapter returns `True`.

### `app_for`

```python
def app_for(
    model: torch.nn.Module | str | Path,
    example_inputs: tuple | None = None,
    *,
    source: str | None = None,
    reference: torch.nn.Module | None = None,
    tokenizer_from: str | Path | None = None,
    middleware: Sequence[str] = (),
    api_key: str | None = None,
    cache: bool = True,
    **options: Any,
) -> FastAPI
```

`app_for` does three things in one call: it builds the `LoadedModel`, it runs `prepare_serving()`, and it runs `build_app(state=...)`. It is the one-line path from a model to a mounted FastAPI app. [`serving.md`](serving.md) and the section "Mount it in your own app" of the repo README describe it.

`app_for` runs synchronously. The export-and-verify gate and the warmup finish before it returns. The app is ready to serve immediately. There is no loader thread and no `503` window on `/ready`. This is different from the `loader=` form of `build_app`.

- `model`: usually an `nn.Module` that this process already built. A `str` or `Path` is a `.onnx` file that is already on this machine. `app_for` downloads nothing. The path must exist before the call.
- `source`: the label that `/metadata` and `/schema` report. For a `.onnx` path, the default is the path (reported as the file name only). For an `nn.Module`, the default is `"model"`, because a module has no path. `source.kind` on `/schema` is `onnx-file` or `in-process-module` in these two cases.
- `reference`: a PyTorch model to verify a `.onnx` `model` against. It has the same role as `--reference` on the CLI. Without it, downshift serves a `.onnx` `model` as `UNVERIFIED`.
- `tokenizer_from`: a downloaded Hugging Face repo directory. It supplies the tokenizer, the pooling recipe and the label metadata for a `.onnx` `model`. It has the same role as `--tokenizer-from` on the CLI.
- `cache`: if `True` (the default), the export cache of `check()` applies.
- `api_key`: if you set it, every route except `/health` and `/ready` needs `Authorization: Bearer <api_key>`. Without it, the route answers `401`. If you do not pass it, downshift uses the `DOWNSHIFT_SERVER_API_KEY` environment variable. Pass a value to override it for this app. If the key is unset or empty, downshift logs one `WARNING` that the endpoints are unauthenticated.
- `options`: downshift forwards these as `ServeOptions` fields (`backend=`, `warmup=`, `max_concurrency=`, and others). [`serving.md`](serving.md) has the full list of fields. Every default comes from `downshift.settings`. The `DOWNSHIFT_*` environment variables therefore also apply here, not only to the CLI. Downshift reads them once, when `downshift` is imported. A keyword argument has priority. `check()`, `export()` and `intake()` are an exception for `k`. They use the constant `8` as the default and not `DOWNSHIFT_SAMPLES`.

## Types

### `ExportVerdict`

```python
@dataclass
class ExportVerdict:
    status: Status  # "CLEAN" | "DEGRADED" | "FAILED" | "UNVERIFIED"
    model_family: str
    capture_strategy: str | None
    opset: int | None
    op_types: dict[str, int]
    numerics: NumericsReport | None
    recommended_backend: BackendName  # "onnxruntime" | "torch"
    reason: str
    input_names: tuple[str, ...] = ()
    dynamic_dims: dict[str, list[int]] = {}
    axes: list[AxisFact] = []
    output_axes: list[str] = []
    unsupported_ops: list[str] = []
    warnings: list[str] = []
    onnx_path: Path | None = None
    onnx_program: object | None = None  # torch.onnx.ONNXProgram
    onnx_bytes: bytes = b""
    prepared: Prepared | None = None
    capture_stderr: str = ""
    capture_exceptions: list[tuple[str, Exception]] = []
```

All other code reads this one object. `check`, `export`, `intake` and `build_verdict` return it. `serve` reads it to select a backend.

| Field | Type | Meaning |
|---|---|---|
| `status` | `"CLEAN" \| "DEGRADED" \| "FAILED" \| "UNVERIFIED"` | Refer to "The gate" in the repo README. |
| `model_family` | `str` | The `family` of the adapter that resolved: `"generic"`, `"pyg"`, `"hf"`, or `"onnx"` for an `intake()` without a reference. |
| `capture_strategy` | `str \| None` | The `torch.export` strategy that captured the graph (for example `"strict=False"`). `None` if the capture did not run (`intake()` without a reference, or `--backend torch`). |
| `opset` | `int \| None` | The ONNX opset of the exported graph. |
| `op_types` | `dict[str, int]` | A histogram of operation types, in descending order of count. |
| `numerics` | `NumericsReport \| None` | `None` if the numerics did not run (`verify_numerics=False`, a `FAILED` capture, or `intake()` with no reference). |
| `recommended_backend` | `BackendName` | The backend that the gate recommends. `--backend auto` of `serve` follows it. |
| `reason` | `str` | A summary for people. The `Reason` row of the `check` report and the `Verdict` row of the banner print it. |
| `input_names` | `tuple[str, ...]` | The flat input names, in the order of the forward arguments. |
| `dynamic_dims` | `dict[str, list[int]]` | `{input_name: [dynamic_axis, ...]}`. |
| `axes` | `list[AxisFact]` | The served bounds and the sampled range of each dynamic axis (`input`, `axis`, `name`, `served_min`, `served_max`, `sampled_min`, `sampled_max`). |
| `output_axes` | `list[str]` | For the `pyg` adapter, the class of axis 0 of each output: `node`, `edge`, `fixed` or `unknown`. |
| `unsupported_ops` | `list[str]` | The `aten::*` operation names that downshift found in the exception messages of a `FAILED` capture. |
| `warnings` | `list[str]` | Notices that are not fatal: tied weights, the switch to training mode, and the message of an ONNX Runtime load failure. |
| `onnx_path` | `Path \| None` | `export()` sets it after it writes the artifact. `intake()` or the serving of a pre-built `.onnx` file also sets it. |
| `onnx_program` | `object \| None` | The live `torch.onnx.ONNXProgram`, if the capture made one. It is not JSON-able. |
| `onnx_bytes` | `bytes` | The serialized graph. `capture()` builds it once. It is not JSON-able. |
| `prepared` | `Prepared \| None` | The `Prepared` behind this verdict, if a torch model took part. It is `None` for an `intake()` without a reference. It is not JSON-able. |
| `capture_stderr` | `str` | The own stderr of torch and onnx during the capture. It is for debugging only. Downshift logs it at `--log-level debug` on a `FAILED` verdict. |
| `capture_exceptions` | `list[tuple[str, Exception]]` | One `(strategy_name, exception)` pair for each capture strategy that downshift tried. It is for debugging only. |

Some properties are not dataclass fields. They are derived:

- `shape_generalization` is the same as `numerics.shape_generalization`. It is `None` if there is no numerics report.
- `shape_generalization_reason` is a string if the baseline sample failed. Otherwise it is `None`.
- `exit_code` is `EXIT_CODES[status]`: `CLEAN` 0, `FAILED` 1, `DEGRADED` 2, `UNVERIFIED` 3. `check` and `export` exit with this code.

`to_dict()` gives these keys:

- `status`, `model_family`, `capture_strategy`, `opset`, `op_types`
- `numerics` (the nested `NumericsReport.to_dict()` or `None`)
- `shape_generalization`, `shape_generalization_reason`
- `recommended_backend`, `reason`
- `input_names` (as a `list`), `dynamic_dims`, `axes` (as a list of dicts), `output_axes`, `unsupported_ops`, `warnings`
- `onnx_path` (as a `str` or `None`)

It does not include `onnx_program`, `onnx_bytes` and `prepared`, because they are not JSON-able. It does not include the debug-only `capture_stderr` and `capture_exceptions`, because machine consumers do not need them. `exit_code` is a derived property, and `to_dict()` does not include it.

`from_dict()` rebuilds an `ExportVerdict` from the output of `to_dict()`. `serve --workers N` uses it. It sends a verdict from the parent process to each worker in an environment variable. `prepared` and `onnx_program` are `None` in the result, because downshift never serializes them. If the graph is now at a temporary path in the worker, the caller sets `onnx_path` itself afterward.

### `NumericsReport`

```python
@dataclass
class NumericsReport:
    samples_tested: int
    max_abs_err: float
    max_rel_err: float
    failures: int
    shape_generalization: bool | None
    tolerance_abs: float
    tolerance_rel: float
    tolerance_dtype: str = "float32"
    tolerance_overridden: bool = False
    baseline_failed: bool = False
    worst: WorstMismatch | None = None
    sample_shapes: list[list[tuple[int, ...]]] = []
    output_shapes: list[list[tuple[int, ...]]] = []
    seed: int = 0
    notes: list[str] = []
```

| Field | Type | Meaning |
|---|---|---|
| `samples_tested` | `int` | `k`, the number of samples that ran. |
| `max_abs_err` | `float` | The largest absolute error across all output elements and samples. |
| `max_rel_err` | `float` | The largest relative error, with the same scope. |
| `failures` | `int` | The number of samples that failed the comparison in the style of `numpy.allclose`. `NumericsReport.passed` is `failures == 0`. |
| `shape_generalization` | `bool \| None` | `True` if all samples with a varied shape (all samples except the baseline) passed. `False` if the baseline passed but a varied sample failed. `None` if the baseline failed. In that case, downshift did not evaluate the generalization. |
| `tolerance_abs` / `tolerance_rel` | `float` | The `atol` and `rtol` that downshift used. |
| `tolerance_dtype` | `str` | The floating dtype whose default tolerance applied (`"float32"`, `"float16"`, `"bfloat16"`, `"float64"`). Refer to `default_tolerances`. |
| `tolerance_overridden` | `bool` | `True` if `--atol` or `--rtol` (or the `atol=` or `rtol=` keyword arguments) set the values, and not the default of `tolerance_dtype`. |
| `baseline_failed` | `bool` | `True` if sample 0 (the example without a variation) failed. |
| `worst` | `WorstMismatch \| None` | The element with the largest error across all samples. |
| `sample_shapes` | `list[list[tuple[int, ...]]]` | The shapes of each input in each sample. The `Samples` row of the CLI prints them. |
| `output_shapes` | `list[list[tuple[int, ...]]]` | The shapes of each output in each sample, as `[sample][output]`. |
| `seed` | `int` | The seed that generated the samples. |
| `notes` | `list[str]` | Notes on a mismatch in shape or count between the outputs of torch and ONNX Runtime. There is one note for each affected sample. |

`passed` is a derived property (`failures == 0`) and not a field.

`to_dict()` gives every field above. It also adds `passed`. If `worst` is present, it is a nested `WorstMismatch` dict (`sample`, `output`, `index`, `expected`, `got`, `input_shapes`).

`from_dict()` rebuilds the object from the output of `to_dict()`. It ignores `passed`, which is derived.

### `WorstMismatch`

```python
@dataclass
class WorstMismatch:
    sample: int
    output: int
    index: tuple[int, ...]
    expected: float
    got: float
    input_shapes: list[tuple[int, ...]]
```

`WorstMismatch` is not in `downshift.__all__`. You can reach it only through `NumericsReport.worst`. It is part of the shape of `NumericsReport`:

- `sample` and `output` give the verification sample and the model output of the mismatch.
- `index` is the unravelled element index in that output.
- `expected` and `got` are the values of torch and ONNX Runtime.
- `input_shapes` are the shapes of each input in that sample.

### `Adapter`

```python
@runtime_checkable
class Adapter(Protocol):
    name: str

    def matches(self, model: nn.Module, example_inputs: tuple | None) -> bool: ...
    def example_inputs(self, model: nn.Module) -> tuple | None: ...
    def prepare(
        self, model: nn.Module, example_inputs: tuple, axis_max: dict[str, int] | None = None
    ) -> Prepared: ...
```

`Adapter` is a `typing.Protocol` with `@runtime_checkable`, so `isinstance(obj, Adapter)` works. `registry.load_from_file` uses this to validate a custom adapter. [`adapters.md`](adapters.md) has the full contract, the resolution order and a worked example.

### `Prepared`

```python
@dataclass
class Prepared:
    model: nn.Module
    inputs: tuple
    input_names: tuple[str, ...]
    dynamic_shapes: tuple
    vary_fn: VaryFn | None
    family: str
```

`Prepared` is the result of `prepare()` of an adapter. It holds:

- A module that is ready for export. Its `forward` takes flat tensors.
- The matching flat example inputs and their names.
- The dynamic-shape spec for each input: `{axis: torch.export.Dim}` or `None`, one entry for each input.
- An optional `vary_fn(i) -> inputs` that generates verification samples. `None` means "use the shared-axis-0 default", `make_shared_axis0_vary_fn`.
- The name of the adapter. It becomes `ExportVerdict.model_family`.

`dynamic_dims` is a derived property: `{input_name: sorted(axes)}` for the inputs that have a non-empty `dynamic_shapes` entry.

### `OnnxRuntimeError`

```python
class OnnxRuntimeError(RuntimeError): ...
```

`downshift.core.verify.verify` raises this error internally when ONNX Runtime fails. ONNX Runtime fails when it cannot load the exported graph, or when it raises an error while it runs a sample.

It is different from a numeric mismatch. A mismatch is a `NumericsReport` with `failures > 0` and not an exception. It is also different from an error raised by the torch model on a sample. That error is a `ValueError`, because it means that the caller gave the model a shape that it does not support.

`check()`, `export()`, `intake()` and `build_verdict()` catch `OnnxRuntimeError` internally and make a `FAILED` verdict. Downshift exports it mainly for callers that use `downshift.core.verify.verify` directly. They can catch it too.

### `__version__`

```python
__version__: str  # "0.5.0"
```

Downshift reads it from `downshift._version`. It imports it eagerly, and not through the lazy `_LAZY` map. `downshift.__version__` and `downshift --version` therefore never cause the import of torch.
