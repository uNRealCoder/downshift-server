# Python API

Every name in `downshift.__all__`: the functions that run the export-and-verify gate, the
types the gate returns, and how `import downshift` avoids importing torch.

## Lazy import (PEP 562)

`downshift/__init__.py` defines `__all__` and a `_LAZY` map from name to
`(module, attribute)`. A module-level `__getattr__` resolves each name on first access by
importing the module that actually defines it (`downshift.core.verdict`,
`downshift.core.verify`, `downshift.adapters.base`, `downshift.core.prevalidated`,
`downshift.serve`) and caches the result on `downshift`'s own `globals()`, so every access
after the first is a plain attribute lookup with no further `__getattr__` call.

The practical effect: `import downshift` does not import torch, onnx or onnxruntime.
Nothing is pulled in until you touch `downshift.check`, `downshift.Adapter`, or another
lazy name for the first time. `downshift.__version__` and `downshift.export` are the two
exceptions - `__version__` is a plain string imported eagerly from `downshift._version`,
and `export` is defined directly in `__init__.py` (its own heavy imports,
`downshift.core.manifest` and `downshift.core.verdict`, are deferred inside its function
body instead). This is also why `downshift.cli.main` can answer `--help` and `--version`
without paying torch's import cost: it imports `downshift` and `downshift.settings` at
module load time, and defers every other `downshift.*` import into the command bodies
that need them.

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
) -> ExportVerdict
```

Exports `model` to ONNX in memory and verifies it against PyTorch. Writes nothing to disk;
`export()` is `check()` plus writing the artifact.

- `example_inputs`: a tuple of forward-args. `None` lets the resolved adapter guess (see
  `prepare_model`).
- `k`: number of verification samples.
- `adapter`: an `Adapter` instance, an adapter name (`"generic"`, `"pyg"`, `"hf"`), a
  `path/to/adapter.py[:attr]` spec, or `None` to auto-detect.
- `dynamic`: `{input_name: [axis, ...]}` overriding which axes are dynamic; default is
  axis 0 of every input.
- `fp16`: casts a deep copy of `model` to `float16` before export. The caller's model
  (and any tensors in `example_inputs`) are left untouched; a copy is cast instead. The
  model is still switched to `eval()` in place if it was in training mode - on the copy,
  when `fp16=True`, so the caller's own model keeps whichever mode it was already in.
- `verify_numerics`: `False` skips the numerics check entirely (the `--no-verify`
  behaviour); the graph still exports, but the verdict can only be `UNVERIFIED`, never
  `CLEAN`.
- `atol`/`rtol`: absolute/relative tolerance overrides. `None` (the default for both)
  picks by the model's floating dtype - see `downshift.core.verify.default_tolerances`
  and `settings.TOLERANCES`.
- `seed`: seeds the verification sample generator, inside a forked RNG (`verify()` runs
  under `torch.random.fork_rng`), so it never disturbs the caller's own global RNG state.
- `vary`: a `fn(i) -> inputs` overriding the adapter's own sampler; a string is imported
  the same way as `--vary`. `fn(0)` must return the example inputs.

Returns an `ExportVerdict`. Raises `ValueError` if no example inputs are available and
none could be synthesized, if the model itself raises on the first (baseline) verification
sample, or if `adapter` names an adapter that can't be found; raises `RuntimeError` if no
adapter matches at all (only possible if even the built-in `generic` adapter is
unavailable). A model that exports but that ONNX Runtime cannot load or run is not an
error here - it comes back as a `FAILED` verdict instead.

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
) -> ExportVerdict
```

`check()` plus writing `output` (the `.onnx` path) and a `.manifest.json` sidecar next to
it (`downshift.core.manifest.write_manifest`). Every parameter not listed below means the
same as on `check()`.

- `output`: the `.onnx` file path. Parent directories are created if missing.
- `source_path`: recorded in the manifest as the source checkpoint (file name only, never the directory), with its SHA-256, when
  it points at a real file.

A `FAILED` verdict writes nothing - no `.onnx`, no manifest. A `DEGRADED` verdict still
writes the artifact, because the manifest records exactly how far off the numerics are;
`verdict.onnx_path` is set on the returned verdict in that case. Raises the same errors as
`check()`.

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
    timings: dict[str, float] | None = None,
) -> ExportVerdict
```

The `check()` equivalent for a `.onnx` file someone else already produced (Olive, a
notebook, whatever). Without `reference`, returns an `UNVERIFIED` verdict immediately -
served as-is, with `reason` explaining numerics were never checked. With `reference`, runs
the same numerics gate as `check()`/`export()` against it, and the verdict is `CLEAN`,
`DEGRADED` or `FAILED` on the same rules.

- `timings`: when given, gets a `"verify"` wall-clock-seconds entry added (there is no
  `"export"` phase for a pre-built graph).

This is what `downshift check some.onnx --reference some_model.py:build` runs under the
hood; the CLI's `check` command calls `intake` directly when its `MODEL` argument resolves
to an existing `.onnx` file.

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

The half of `check()` that runs after adapter resolution and input synthesis: capture
(`torch.export` plus ONNX translation), then numeric verification. `check()` is
`build_verdict(prepare_model(...))` with `fp16` handled in between. Useful directly when
you already have a `Prepared` (for example, from a custom adapter you want to run once and
inspect before deciding whether to export).

- `timings`: when given, gets `"export"` (the capture wall-clock time) and `"verify"`
  added to it. This is the same dict `ServingState.timings` and `/metadata`'s `boot` field
  surface.

Raises `ValueError` if the model raises on the baseline (unvaried) verification sample.

### `prepare_model`

```python
def prepare_model(
    model: torch.nn.Module,
    example_inputs: tuple | None = None,
    adapter: Adapter | str | None = None,
    dynamic: dict[str, list[int]] | None = None,
    vary: VaryFn | str | None = None,
) -> Prepared
```

Resolves an adapter (an explicit instance or name, or `registry.detect()` against the
model and inputs), synthesizes example inputs if none were given, and asks the adapter to
flatten the model into export-ready form. `dynamic`, if given, overrides the adapter's own
`dynamic_shapes` via `downshift.core.shapes.apply_dynamic_override`. `vary`, if given,
overrides the adapter's own `vary_fn` (a string is imported like `--vary`'s
`pkg.module:fn` form).

Raises `ValueError` if `example_inputs` is `None` and the adapter's `example_inputs()`
also returns `None`; raises `RuntimeError` if no registered adapter's `matches()` returns
`True`.

### `app_for`

```python
def app_for(
    model: torch.nn.Module | str | Path,
    example_inputs: tuple | None = None,
    *,
    source: str | None = None,
    reference: torch.nn.Module | None = None,
    middleware: Sequence[str] = (),
    api_key: str | None = settings.API_KEY,
    **options: Any,
) -> FastAPI
```

`LoadedModel` construction, `prepare_serving()`, and `build_app(state=...)` in one call -
the one-line path from a model to a mounted FastAPI app, documented in
[`serving.md`](serving.md) and the repo README's "Mount it in your own app" section. Runs
synchronously: the export-and-verify gate and warmup complete before this returns, so the
app is ready to serve immediately (no loader thread, no `/ready` `503` window - contrast
`build_app`'s `loader=` form).

- `model`: usually an `nn.Module` this process already built; a `str`/`Path` is an `.onnx`
  file already on this machine. Nothing is downloaded here either - the path has to exist
  before the call.
- `source`: the label `/metadata` and `/schema` report. Defaults to the `.onnx` path (reported as its file name only), or to
  `"model"` for an `nn.Module`, which has no path to name. `/schema`'s `source.kind` is
  `onnx-file` or `in-process-module` accordingly.
- `reference`: a PyTorch model verifying a `.onnx` `model` against, same role as
  `--reference` on the CLI. Without it, a `.onnx` `model` is served `UNVERIFIED`.
- `api_key`: when set, every route except `/health` and `/ready` requires
  `Authorization: Bearer <api_key>` and answers `401` otherwise. Defaults to the
  `DOWNSHIFT_SERVER_API_KEY` environment variable; pass a value, or `None` to force it off
  for this app. Unset or empty logs one `WARNING` that the endpoints are unauthenticated.
- `options`: forwarded as `ServeOptions` fields (`backend=`, `warmup=`,
  `max_concurrency=`, ...) - see [`serving.md`](serving.md) for the full field list. Every
  default is read from `downshift.settings`, so the `DOWNSHIFT_*` environment variables
  apply here too, not only to the CLI (read once, when `downshift` is imported; a keyword
  argument wins). `check()`, `export()` and `intake()` are the exception for `k`: they
  default to the constant `8`, not to `DOWNSHIFT_SAMPLES`.

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
    unsupported_ops: list[str] = []
    warnings: list[str] = []
    onnx_path: Path | None = None
    onnx_program: object | None = None  # torch.onnx.ONNXProgram
    onnx_bytes: bytes = b""
    prepared: Prepared | None = None
    capture_stderr: str = ""
    capture_exceptions: list[tuple[str, Exception]] = []
```

The one object everything else reads: what `check`/`export`/`intake`/`build_verdict`
return, and what `serve` reads to pick a backend.

| Field | Type | Meaning |
|---|---|---|
| `status` | `"CLEAN" \| "DEGRADED" \| "FAILED" \| "UNVERIFIED"` | See "The gate" in the repo README. |
| `model_family` | `str` | The resolving adapter's `family` (`"generic-torch"`, `"pyg"`, `"hf-transformers"`, `"onnx"` for an unreferenced `intake()`, ...). |
| `capture_strategy` | `str \| None` | Which `torch.export` strategy captured the graph (e.g. `"strict=False"`), or `None` when capture never ran (`intake()` without a reference, or `--backend torch`). |
| `opset` | `int \| None` | The ONNX opset the graph was exported at. |
| `op_types` | `dict[str, int]` | Op-type histogram, count-descending. |
| `numerics` | `NumericsReport \| None` | `None` when numerics were never run (`verify_numerics=False`, a `FAILED` capture, or `intake()` with no reference). |
| `recommended_backend` | `BackendName` | What the gate recommends; `serve`'s `--backend auto` follows it. |
| `reason` | `str` | Human-readable summary; what the `check` report's `Reason` row and the banner's `Verdict` row print. |
| `input_names` | `tuple[str, ...]` | Flat input names, in forward-argument order. |
| `dynamic_dims` | `dict[str, list[int]]` | `{input_name: [dynamic_axis, ...]}`. |
| `unsupported_ops` | `list[str]` | `aten::*` op names mined from a `FAILED` capture's exception messages. |
| `warnings` | `list[str]` | Non-fatal notices (tied weights, training-mode switch, an ONNX Runtime load failure's message). |
| `onnx_path` | `Path \| None` | Set by `export()` after writing the artifact, or by `intake()`/a pre-built `.onnx` serve. |
| `onnx_program` | `object \| None` | The live `torch.onnx.ONNXProgram`, when capture produced one. Not JSON-able. |
| `onnx_bytes` | `bytes` | The serialized graph, built once by `capture()`. Not JSON-able. |
| `prepared` | `Prepared \| None` | The `Prepared` behind this verdict, when a torch model was involved (not for a referenceless `intake()`). Not JSON-able. |
| `capture_stderr` | `str` | torch/onnx's own stderr during capture. Debug-only; logged at `--log-level debug` on a `FAILED` verdict. |
| `capture_exceptions` | `list[tuple[str, Exception]]` | `(strategy_name, exception)` per capture strategy tried. Debug-only. |

Derived properties (not dataclass fields): `shape_generalization` (proxies
`numerics.shape_generalization`, `None` if there's no numerics report),
`shape_generalization_reason` (a string when the baseline sample itself failed, else
`None`), and `exit_code` (`EXIT_CODES[status]`: `CLEAN` 0, `FAILED` 1, `DEGRADED` 2,
`UNVERIFIED` 3 - what `check`/`export` exit with).

`to_dict()` emits: `status`, `model_family`, `capture_strategy`, `opset`, `op_types`,
`numerics` (nested `NumericsReport.to_dict()` or `None`), `shape_generalization`,
`shape_generalization_reason`, `recommended_backend`, `reason`, `input_names` (as a
`list`), `dynamic_dims`, `unsupported_ops`, `warnings`, `onnx_path` (as a `str` or
`None`). It excludes `onnx_program`, `onnx_bytes`, `prepared` (none are JSON-able), and
the debug-only `capture_stderr`/`capture_exceptions` (not meant for machine consumers).
`exit_code` is a derived property and is also not included.

`from_dict()` rebuilds an `ExportVerdict` from `to_dict()`'s output - used by `serve
--workers N` to ship a verdict from the parent process to each worker over an environment
variable. `prepared` and `onnx_program` come back `None` (never serialized); the caller
sets `onnx_path` itself afterwards if the graph now lives at a worker-local temp path.

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
    seed: int = 0
    notes: list[str] = []
    session: ort.InferenceSession | None = None
```

| Field | Type | Meaning |
|---|---|---|
| `samples_tested` | `int` | `k`, the number of samples run. |
| `max_abs_err` | `float` | Largest absolute error seen across every output element and sample. |
| `max_rel_err` | `float` | Largest relative error, same scope. |
| `failures` | `int` | Samples that failed the `numpy.allclose`-style comparison. `NumericsReport.passed` is `failures == 0`. |
| `shape_generalization` | `bool \| None` | `True` if every varied-shape sample (all but the baseline) passed, `False` if the baseline passed but a varied one failed, `None` if the baseline itself failed (generalization was never evaluated). |
| `tolerance_abs` / `tolerance_rel` | `float` | The `atol`/`rtol` actually used. |
| `tolerance_dtype` | `str` | Which floating dtype's default tolerance applied (`"float32"`, `"float16"`, `"bfloat16"`, `"float64"`) - see `default_tolerances`. |
| `tolerance_overridden` | `bool` | `True` when `--atol`/`--rtol` (or the `atol=`/`rtol=` kwargs) picked the values instead of `tolerance_dtype`'s default. |
| `baseline_failed` | `bool` | Whether sample 0 (the un-varied example) itself failed. |
| `worst` | `WorstMismatch \| None` | The single largest-error element across every sample. |
| `sample_shapes` | `list[list[tuple[int, ...]]]` | Per-sample, per-input shapes actually used - what the CLI's `Samples` row prints. |
| `seed` | `int` | The seed the samples were generated with. |
| `notes` | `list[str]` | Shape/count-mismatch notes between torch and ONNX Runtime outputs, one per affected sample. |
| `session` | `ort.InferenceSession \| None` | The ONNX Runtime session `verify()` built (or was given) to run the samples. Not JSON-able; `serve/engine.py` reuses this session directly instead of building a second one when the serving options match (CPU, default thread counts). |

`passed` is a derived property (`failures == 0`), not a field.

`to_dict()` emits every field above except `session` (dropped, not JSON-able), and adds
`passed`. `worst`, when present, is a nested `WorstMismatch` dict (`sample`, `output`,
`index`, `expected`, `got`, `input_shapes`).

`from_dict()` rebuilds from `to_dict()`'s output; `passed` (derived) and `session` (never
serialized) are ignored on the way in.

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

Not in `downshift.__all__` (reached only via `NumericsReport.worst`), but part of the
`NumericsReport` shape: `sample`/`output` locate which verification sample and which
model output the mismatch was in; `index` is the unravelled element index within that
output; `expected`/`got` are the torch and ONNX Runtime values; `input_shapes` are that
sample's per-input shapes.

### `Adapter`

```python
@runtime_checkable
class Adapter(Protocol):
    name: str
    family: str

    def matches(self, model: nn.Module, example_inputs: tuple | None) -> bool: ...
    def example_inputs(self, model: nn.Module) -> tuple | None: ...
    def prepare(
        self, model: nn.Module, example_inputs: tuple, axis_max: dict[str, int] | None = None
    ) -> Prepared: ...
```

A `typing.Protocol`, `@runtime_checkable` so `isinstance(obj, Adapter)` works (used by
`registry.load_from_file` to validate a custom adapter). Full contract, resolution order,
and a worked example: [`adapters.md`](adapters.md).

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

What an adapter's `prepare()` returns: an export-ready module whose `forward` takes flat
tensors, the matching flat example inputs, their names, the per-input dynamic-shape spec
(`{axis: torch.export.Dim}` or `None`, one entry per input), an optional
`vary_fn(i) -> inputs` for generating verification samples (`None` means "use the
shared-axis-0 default", `make_shared_axis0_vary_fn`), and the family string that ends up
in `ExportVerdict.model_family`. `dynamic_dims` is a derived property:
`{input_name: sorted(axes)}` for inputs whose `dynamic_shapes` entry is non-empty.

### `OnnxRuntimeError`

```python
class OnnxRuntimeError(RuntimeError): ...
```

Raised internally (by `downshift.core.verify.verify`) when ONNX Runtime itself fails - it
can't load the exported graph, or it raises while running a sample. Distinct from a
numeric mismatch (still a `NumericsReport` with `failures > 0`, not an exception) and from
the torch model itself raising on a sample (a `ValueError`, since that means the caller
gave the model a shape it doesn't support). `check()`/`export()`/`intake()`/
`build_verdict()` catch it internally and turn it into a `FAILED` verdict; it is exported
mainly so callers using `downshift.core.verify.verify` directly can catch it too.

### `__version__`

```python
__version__: str  # "0.4.0"
```

Read from `downshift._version`, imported eagerly (not through the lazy `_LAZY` map) so
`downshift.__version__` and `downshift --version` never trigger torch's import.
