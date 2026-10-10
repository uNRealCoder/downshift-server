# Serving layer

The programmatic pieces beneath `downshift serve` and `app_for`: the options dataclass,
the state object a running server holds, how a `ServingState` gets built, the app that
serves it, and the two backends it can wrap. The code is split by job: `serve/options.py`
(`ServeOptions`, `BackendChoice`), `serve/engine.py` (`ServingState`, `prepare_serving`),
`serve/app.py` (`build_app`, routing, the request-id and API-key middleware),
`serve/predict.py` (the `/predict` internals), `serve/describe.py` (`GET /schema`),
`serve/backends.py`, `serve/schemas.py` and `serve/codec.py`. Operational guidance (probes, sizing, Kubernetes, thread budgeting)
lives in [`docs/production.md`](../production.md) and isn't repeated here.

## `ServeOptions`

```python
@dataclass
class ServeOptions:
    backend: BackendChoice = BackendChoice.auto
    force_onnx: bool = False
    device: str = "auto"
    warmup: int = 3
    k: int = 8
    adapter: str | None = None
    dynamic: dict[str, list[int]] | None = None
    intra_op_threads: int = 0
    inter_op_threads: int = 0
    output_encoding: OutputEncoding = OutputEncoding.json
    max_input_bytes: int = DEFAULT_MAX_INPUT_BYTES  # 256 MiB
    max_body_bytes: int = DEFAULT_MAX_BODY_BYTES  # 32 MiB
    max_concurrency: int = 4
    max_queue: int = DEFAULT_MAX_QUEUE  # 64
    request_timeout: float = DEFAULT_REQUEST_TIMEOUT  # 30.0
    atol: float | None = None
    rtol: float | None = None
    seed: int = 0
    vary: str | None = None
    pooling: str | None = None
    normalize: bool | None = None
```

Defined in `downshift.serve.options` with no torch import at module scope, so the CLI can
parse `--backend`/`--output-encoding` and print `--help` without paying torch's import
cost. Every default is read from `downshift.settings`, so the `DOWNSHIFT_*` environment
variables reach `ServeOptions()` and `app_for()` as well as the CLI (the values shown are
what you get with none set; they are read once, when `downshift` is imported, and a keyword
argument wins). `serve_cmd` builds a `ServeArgs` -
`{load: LoadSpec, options: ServeOptions, reference, middleware, log_level, artifact,
access_log}`, a plain-JSON-able dataclass in `cli/runtime.py` used to hand a run to `--workers N`
worker processes over an environment variable - and `_collect_serve_args` turns `serve`'s
typer parameters into its `LoadSpec`/`ServeOptions` pair, so adding an option is a typer
parameter, a `ServeOptions` field and a `_collect_serve_args` parameter. See
[`cli.md`](cli.md#serve) for what each field's CLI flag/env var does.

| Field | Type | Default | Controls |
|---|---|---|---|
| `backend` | `BackendChoice` | `auto` | Which backend to serve: `auto` follows the verdict's recommendation, or force `onnxruntime`/`torch`. |
| `force_onnx` | `bool` | `False` | Serve a `DEGRADED` graph through ONNX Runtime anyway. |
| `device` | `str` | `"auto"` | `"auto"` / `"cpu"` / `"cuda"`, resolved by `resolve_device`. `"cuda"` without CUDA raises `ValueError` (exit `4` on the CLI) on both backends. |
| `warmup` | `int` | `3` | Inferences run before `ServingState.ready` flips. |
| `k` | `int` | `8` | Verification sample count, forwarded to the gate. |
| `adapter` | `str \| None` | `None` | Adapter name/spec, or `None` to auto-detect. |
| `dynamic` | `dict[str, list[int]] \| None` | `None` | Dynamic-axis override, forwarded to the gate. |
| `intra_op_threads` | `int` | `0` | ONNX Runtime `SessionOptions.intra_op_num_threads`; `0` lets ORT choose. Also sets `torch.set_num_threads` for the torch backend when `> 0`. |
| `inter_op_threads` | `int` | `0` | ONNX Runtime `SessionOptions.inter_op_num_threads`; `0` lets ORT choose. |
| `output_encoding` | `OutputEncoding` | `json` | Default response tensor encoding; a request's own `output_encoding` field overrides it. |
| `max_input_bytes` | `int` | 256 MiB | Cap on one decoded base64 tensor input. |
| `max_body_bytes` | `int` | 32 MiB | Cap on the whole request body (`413`), checked before it is parsed. No separate text-length limit exists; this covers `text` requests. |
| `max_concurrency` | `int` | `4` | Size of the inference thread pool (`ServingState.executor`), per worker process: how many requests infer and encode at once. |
| `execution` | `ExecutionChoice` | `threadpool` | `threadpool` runs every request's work on the two pools; `inline` runs a small JSON body (Content-Length up to 64 KiB, no `text`) on the event loop. See [Request path](#request-path-servepredictpy). |
| `prep_threads` | `int` | `min(4, usable CPUs)` | Size of the prep pool (`ServingState.prep_executor`): parsing, validating and converting request bodies, and tokenizing. |
| `axis_max` | `dict[str, int] \| None` | `None` | `--axis-max NAME=N`: narrow a dynamic axis's served maximum; verify pins one sample at exactly `N`. |
| `export_cache_dir` | `str \| None` | `None` | `--export-cache-dir`: where verified exports are kept so the next boot skips export and verify. |
| `max_queue` | `int` | `64` | Additional admitted-but-waiting predicts allowed past `max_concurrency`. |
| `request_timeout` | `float` | `30.0` | Seconds an admitted predict may wait, queued, before a `503` instead of running; `0` disables the check. |
| `atol` / `rtol` | `float \| None` | `None` | Tolerance override, forwarded to the gate; `None` means "by dtype". |
| `seed` | `int` | `0` | Verification sample seed, forwarded to the gate. |
| `vary` | `str \| None` | `None` | `pkg.module:fn` overriding the adapter's own verification sampler. |
| `pooling` | `str \| None` | `None` | A `PoolingChoice` value (`mean`, `cls`, `max`, `mean_sqrt_len`, `none`) overriding a Hugging Face repo's embedding recipe. |
| `normalize` | `bool \| None` | `None` | L2-normalise the embedding; `None` means whatever the recipe says. |

## `BackendChoice`

```python
class BackendChoice(StrEnum):
    auto = "auto"
    onnxruntime = "onnxruntime"
    torch = "torch"
```

What the caller asked for. `"auto"` is a CLI/options-only sentinel - it is never itself a
running backend; `choose_backend` resolves it to a concrete `BackendName`
(`"onnxruntime"` or `"torch"`, from `core.verdict`) before anything is built.

## `choose_backend`

```python
def choose_backend(verdict: ExportVerdict, opts: ServeOptions) -> tuple[BackendName, list[str]]
```

Returns `(backend_name, notes)`. Rules, in order:

1. Start from `verdict.recommended_backend` if `opts.backend == BackendChoice.auto`,
   otherwise from `opts.backend` directly.
2. `--backend onnxruntime` on a `DEGRADED` verdict without `--force-onnx` raises
   `ValueError` - the one way `serve` refuses to boot rather than falling back silently.
3. `--force-onnx` on a `DEGRADED` verdict forces `onnxruntime` and appends a banner note
   ("serving a DEGRADED graph; outputs may be wrong").
4. If `onnxruntime` was chosen but the verdict has no ONNX graph at all
   (`onnx_program is None and onnx_path is None` - a `FAILED` capture, or `--backend
   torch` skipped export outright), silently falls back to `torch` with a note.
5. If `torch` was chosen but the verdict has no `prepared` (a referenceless `intake()` of
   a bare `.onnx`, which never ran an adapter), raises `ValueError` - there is no
   `nn.Module` to run eagerly.

## `ServingState`

```python
@dataclass
class ServingState:
    source: str
    verdict: ExportVerdict
    backend: Backend
    input_names: tuple[str, ...]
    options: ServeOptions
    example_inputs: tuple | None = None
    ready: bool = False
    notes: list[str] = field(default_factory=list)
    source_kind: str = UNKNOWN_SOURCE
    text: TextIO | None = None
    embedding: EmbeddingRecipe | None = None
    vocab_size: int | None = None
    warmup_stats: WarmupStats | None = None
    timings: dict[str, float] = field(default_factory=dict)
    axis_bounds: dict[str, dict[int, DimBound]] = field(default_factory=dict, repr=False)
    executor: ThreadPoolExecutor = field(init=False, repr=False, compare=False)
    prep_executor: ThreadPoolExecutor = field(init=False, repr=False, compare=False)
    in_flight: int = field(init=False, repr=False, compare=False, default=0)
```

The object one running server (or worker process) holds: what `/metadata` reads, and what
the predict routes admit requests against (`PredictRoute` in `serve/app.py`, `run_predict`
in `serve/predict.py`).

| Field | Meaning |
|---|---|
| `source` | The model spec `serve`/`app_for` was given - always something already on this machine. Echoed in `/metadata`'s `model` field. |
| `verdict` | The `ExportVerdict` the gate produced. |
| `backend` | The concrete `Backend` instance (`OnnxRuntimeBackend` or `TorchBackend`) serving requests. |
| `input_names` | Flat input names, in forward-argument order. |
| `options` | The `ServeOptions` this state was built from. |
| `example_inputs` | Real example inputs, when any were available - used to warm up with realistic shapes rather than synthesized ones, and to build `--workers N`'s `.npz` sidecar. |
| `ready` | Whether the state is warmed up; set `True` as the last step of `warmup()`, even with `--warmup 0`. |
| `notes` | Strings the CLI banner should print (e.g. a `--force-onnx` warning, or a backend fallback note from `choose_backend`). |
| `warmup_stats` | A `WarmupStats` once `warmup()` has run, else `None`. |
| `source_kind` | Which accepted form `source` named (`onnx-file`, `torch-checkpoint`, `hf-repo-dir`, `import-spec`, `in-process-module`, `unknown`; constants in `downshift/sources.py`). Reported under `source` on `/schema` and labelled on the banner's `Model` row. |
| `text` | A `TextIO` (`adapters/text.py`: the tokenizer, the model's token limit, and labels for a classifier) when the source is a Hugging Face repo directory with tokenizer files; what lets `/predict` take `{"text": ...}`. |
| `embedding` | An `EmbeddingRecipe` (`adapters/embedding.py`) when the repo declares a pooling recipe, or `--pooling` gives one. Already part of the exported graph; kept so `/schema` can say what it is. |
| `vocab_size` | The Hugging Face model's vocabulary size, set even when the tokenizer or recipe fails to load. `input_ids` outside `[0, vocab_size)` is a `400` before inference. |
| `timings` | Wall-clock seconds per boot phase, keyed by `Phase` (`downshift/core/phase.py`, a `StrEnum` of `load`, `export`, `verify`, `session`, `warmup`) - whichever actually ran. `load` is added by the caller after `prepare_serving` returns. Read by the banner's `Boot` row and `/metadata`'s `boot` field. |
| `axis_bounds` | Per input, per dynamic axis: a `DimBound(name, min, max)`, the range the adapter's export was traced for. Empty for a bare `.onnx` with no reference model. Drives the `bounds` in `/schema` and the readable `400` on an out-of-range axis. |
| `executor` | The inference pool: a `ThreadPoolExecutor` sized to `options.max_concurrency`, not part of the dataclass's identity (`compare=False`). Each request's backend call and response encoding run here, never on the event loop (unless `--execution inline`), so a slow request never blocks `/health`/`/ready`. |
| `prep_executor` | The prep pool, sized to `options.prep_threads`: each request's parse, validation, conversion to arrays and checks. |
| `in_flight` | Admitted-but-unfinished predicts (in either pool or waiting for one), guarded by an internal lock. |

Methods: `try_admit()` atomically claims one of `max_concurrency + max_queue` slots
(returns `False` if none are free); `release()` gives one back. Properties:
`backend_auto_selected` (`True` when `options.backend == BackendChoice.auto` and not
`forced_onnx`), `forced_onnx` (`True` when `--force-onnx` and the verdict is `DEGRADED`),
and `declared_dtypes` (a `cached_property`: `{input_name: dtype}` read once from
`backend.metadata().inputs`, used to interpret ambiguous JSON input on every request
without re-deriving it each time). A `--workers N` worker that rebuilt from the parent's
ONNX artifact re-derives `source_kind` from the spec (`loading.source_kind`), so it still
reports the original model's kind.

`WarmupStats`:

```python
@dataclass
class WarmupStats:
    count: int
    mean_ms: float
    synthesized: bool  # True when there was no example input, so warmup() made one up
```

## `prepare_serving`

```python
def prepare_serving(
    loaded: LoadedModel, opts: ServeOptions | None = None, reference: LoadedModel | None = None
) -> ServingState
```

Runs the gate (`intake()` for a `.onnx` `LoadedModel`, otherwise `prepare_model()` +
`build_verdict()`, or a `torch`-only `UNVERIFIED` verdict when `opts.backend ==
BackendChoice.torch` skips export outright), picks a backend via `choose_backend`, builds
it, and warms it up (`opts.warmup` inferences). This is what `serve_cmd` and `app_for` both
call; `serve_cmd` adds a `Phase.load` timing entry itself (reading/importing the model,
before `prepare_serving` is even called) since that step happens outside this function.
While it runs, `core.phase.report(Phase.x)` tells `/ready` which phase the loader is in (a
no-op outside a loader-built app).

## `serving_state_from_artifact`

```python
def serving_state_from_artifact(
    source: str,
    onnx_path: Path,
    verdict: ExportVerdict,
    opts: ServeOptions,
    input_names: tuple[str, ...],
    notes: list[str] | None = None,
    example_inputs: tuple | None = None,
    axis_bounds: dict[str, dict[int, DimBound]] | None = None,
) -> ServingState
```

Rebuilds a `ServingState` in a `serve --workers N` worker from the ONNX graph and verdict
the parent process already exported and verified: no capture, no verify, just a fresh
`OnnxRuntimeBackend` session over `onnx_path` and a warmup pass. `input_names`/`notes` come
from the parent (the shipped `verdict` has no `prepared` to derive them from), and so do
the adapter's axis names and bounds (`axis_bounds`, shipped as JSON by
`axis_bounds_to_json`), which `/schema` and the out-of-range `400` need.
`example_inputs`, when the parent had any, come from the `.npz` sidecar it wrote -
otherwise `warmup()` synthesizes them.

`serving_state_from_torch_artifact(loaded, verdict, opts, notes=None)` is the torch-backed
counterpart: the worker reloads the model itself (torch weights are not shipped between
processes) and prepares it, but takes the parent's already-verified verdict as given.

## `warmup` and `synthesize_feeds`

```python
def warmup(state: ServingState, n: int) -> WarmupStats
def synthesize_feeds(backend: Backend) -> dict[str, np.ndarray]
```

`warmup` runs `n` inferences (using `state.example_inputs` if present, else
`synthesize_feeds`), records `WarmupStats`, and sets `state.ready = True` regardless of
`n` (including `n=0`) - `ready` is really "the state finished booting", not "at least one
warmup inference ran". `synthesize_feeds` builds one dummy array per input the backend
declares: dynamic or unknown axes become size 1; floats are `np.random.randn`, integers
and bools are zero.

## `build_app`

```python
def build_app(
    state: ServingState | None = None,
    *,
    loader: Callable[[], ServingState] | None = None,
    middleware: Sequence[str] = (),
    api_key: str | None = API_KEY,
    access_log: bool = True,
) -> FastAPI
```

Exactly one of `state`/`loader` must be given (`ValueError` otherwise).

- **`state=`** (library use, `app_for`, tests): the app serves `state` immediately. No
  `/ready` `503` window - by the time this returns, the gate and warmup already ran,
  synchronously, before `build_app` was even called.
- **`loader=`** (`downshift serve`, single worker): the app binds with no `ServingState`
  at all. `/health` is `200` right away; `/ready`, `/metadata`, `/schema` and both predict routes are
  `503` until `loader` (run on a background thread, started by the app's lifespan once
  uvicorn actually begins serving - not when `build_app` is called) returns a
  `ServingState`, which lands on `app.state.serving`. Meanwhile `/ready`'s `phase` says
  which `Phase` the loader is in. A raising `loader` leaves
  `app.state.serving` `None` forever as far as this function is concerned; `serve_cmd`
  closes over the `uvicorn.Server` and sets `should_exit` itself on failure, so the
  process still exits instead of serving `503` indefinitely.

`middleware` is a sequence of `pkg.module:Attr` specs (`--middleware`, repeatable),
attached in order via `load_middleware` (`serve/middleware.py`): each spec must resolve to
a middleware class built as `cls(app)` (pure ASGI, or a
`starlette.middleware.base.BaseHTTPMiddleware` subclass) or an `async (request,
call_next)` coroutine function.

`api_key` defaults to `DOWNSHIFT_SERVER_API_KEY` (`settings.API_KEY`), so the variable
reaches `app_for()` and any other library caller as well as the CLI. When it is set, every
route except `/health` and `/ready` requires `Authorization: Bearer <api_key>` (compared
in constant time) or answers `401` with `WWW-Authenticate: Bearer`. `None` or an empty
string means unauthenticated, and `build_app` then logs one `WARNING` (logger
`downshift.serve`) that the endpoints are unauthenticated: set the key or add your own
authentication middleware. Pass `api_key=None` to override the environment for one app.

`access_log` (default `True`; `--access-log` on the CLI) controls the one log line per
request that `RequestIdMiddleware` writes on logger `downshift.access`: `METHOD PATH STATUS
N ms`, at `INFO`, at `WARNING` from status `400` up, and at `DEBUG` for the `/health` and
`/ready` probes. The request id is set and echoed either way.

Every app is built with, outermost first: `RequestIdMiddleware` (assigns/echoes
`X-Request-Id`, binds it to `downshift.logs.request_id_var` so every log line written while
serving the request carries `request_id=...`, and writes the request line),
`ApiKeyMiddleware`, then your `middleware`, so yours only ever sees authenticated traffic.
`/predict` and `/predict/graph` use the `PredictRoute` route class, described next; every
other route is a plain FastAPI route. See [`http-api.md`](http-api.md) for the wire-level
detail.

## Request path (`serve/predict.py`)

Serving speed comes first in downshift's design: boot may be slow (export, verify, warmup
all happen before `/ready`), but every request pays only for its own work. One predict
request is two thread hops:

```
event loop     PredictRoute: admit or 503, Content-Type (415), --max-body-bytes (413),
               read the body
prep pool      _parse: orjson + the route's pydantic model (422), per-route checks (400)
               _prepare_feeds: arrays, tokenizing, graph batching, vocab/edge/shape/bound
               checks (400)
inference pool _infer: the backend call only
               _encode_response: JSON, base64 or safetensors, split per graph
```

- `PredictRoute` serves the two routes itself instead of letting FastAPI resolve the body:
  `run_predict` gets the raw bytes. The `predict` and `predict_graph` endpoint functions
  exist only so the OpenAPI schema documents `PredictRequest` and `GraphPredictRequest`;
  `_validate_json` reproduces FastAPI's `422` shapes (missing body, `json_invalid`,
  validation errors under `"body"`).
- Admission happens before a byte is read, and the slot is released in `finally` on every
  exit. `--request-timeout` is checked after prep and again when inference starts; a running
  inference is never interrupted.
- Encoding runs on the inference thread right after the backend call, so it holds the
  inference slot: one hop fewer, at the cost of throughput when outputs are large and
  `--max-concurrency` is the bottleneck.
- `--execution inline` runs both hops on the event loop for a JSON body with a
  Content-Length up to 64 KiB (`INLINE_MAX_BODY_BYTES`); a `text` request still tokenizes in
  the prep pool. It only pays off for models that infer in well under a millisecond, and
  stalls `/health` and `/ready` for as long as one inference takes.
- `_run_in` carries the request's contextvars into the pool thread and records the wait as
  `prep_wait` / `infer_wait`. `Server-Timing` (and the access log's `timings_ms`) lists
  `parse`, `prep_wait`, `prep`, `infer_wait`, `infer`, `encode`.

## Backends (`serve/backends.py`)

```python
class Backend(Protocol):
    name: BackendName
    input_names: list[str]
    verified_provider: str | None

    def infer(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]: ...
    def metadata(self) -> BackendMeta: ...
```

`verified_provider` is the execution provider the numerics gate ran against
(`"CPUExecutionProvider"` when numerics ran at all, else `None`), set by `engine._build_backend`
right after a backend is constructed. The banner's `Verified on` row prints it, and says so
when the server runs on a device the numerics never covered.

Both concrete backends take and return dicts of numpy arrays keyed by input/output name,
so `serve/app.py` doesn't need to know which one it's talking to.

### `OnnxRuntimeBackend`

```python
OnnxRuntimeBackend(
    model: bytes | str | Path | None = None,
    device: str = "auto",
    intra_op_threads: int = 0,
    inter_op_threads: int = 0,
    *,
    session: ort.InferenceSession | None = None,
)
```

Wraps an `onnxruntime.InferenceSession`. Either pass `model` (bytes, or a path to a
`.onnx` file) or an already-built `session` directly - `prepare_serving` passes `session=`
when it can reuse the exact session `verify()` already built (device resolves to `cpu` and
both thread counts are `0`, matching ORT's own default session), avoiding a second load of
the same graph. `device="cuda"` selects `CUDAExecutionProvider` (with `CPUExecutionProvider` behind it)
when it is in `onnxruntime.get_available_providers()`, and raises `ValueError` when it is
not, rather than running on the CPU without saying so.
Graph optimization is always `ORT_ENABLE_ALL`. Outputs are keyed positionally
(`output_0`, `output_1`, ...) regardless of what the ONNX graph itself calls them, so
responses look the same from either backend. ORT `InvalidArgument`/`InvalidGraph`
exceptions, and any `Fail` exception whose message mentions "shape", are translated to
`InferenceInputError` (a `400`, not a `500`); anything else propagates.

### `TorchBackend`

```python
TorchBackend(
    module: nn.Module,
    input_names: tuple[str, ...],
    device: str = "auto",
    example_inputs: tuple | None = None,
    intra_op_threads: int = 0,
)
```

Wraps an `nn.Module` in eager mode: `module.eval().to(device)`; `device="cuda"` without
`torch.cuda.is_available()` raises `ValueError`. `intra_op_threads > 0`
calls `torch.set_num_threads` (`0` leaves torch's own default alone). A bfloat16 or float16
module is served on a float32 wire: floating inputs are cast to the module's own dtype on
the way in and outputs widened to float32 on the way out (numpy has no bfloat16), so such a
model boots on the torch fallback and `/schema` reports its dtypes as `float32`. When
`example_inputs` is given, one warmup-free pass through the model at construction time
fills in real dtypes/shapes for `/metadata` (`_spec_from_tensor`/`_spec_from_array`);
without it, input specs have `dtype=None, shape=None` and there are no output specs at
all. `infer()` converts numpy inputs to tensors with `torch.from_numpy` (copying only when
the array isn't C-contiguous/writable, which base64-decoded arrays sometimes aren't),
runs under `torch.inference_mode()`, and translates only client-shaped errors to
`InferenceInputError` (a `400`): an `IndexError`, and a `RuntimeError`/`ValueError` whose
message names a shape, size, dimension, dtype, broadcast or out-of-range problem (and is not
an out-of-memory error). Any other exception from the model - an unsupported op, a CUDA
failure, a bug in the module, an OOM - propagates and becomes a `500`.

### `BackendMeta` / `IOSpec`

```python
@dataclass
class IOSpec:
    name: str
    dtype: str | None
    shape: list[int | str | None] | None


@dataclass
class BackendMeta:
    name: BackendName
    device: str
    inputs: list[IOSpec]
    outputs: list[IOSpec]
```

What `backend.metadata()` returns and `/metadata`'s `backend` field serializes
(`BackendMeta.to_dict()`). `IOSpec.dtype` is ONNX Runtime's own `"tensor(float)"`-style
string for the ORT backend, or `"tensor(float32)"`-style (torch dtype name) for the torch
backend when example inputs were available. `IOSpec.shape` marks axis 0 as the string
`"batch"` for the torch backend's specs (`_dynamic_shape`) - axis 0 is dynamic for
anything downshift serves - while the ORT backend reports whatever the graph itself
declares.

### `resolve_device`

```python
def resolve_device(device: str) -> str
```

`"auto"` becomes `"cuda"` if `torch.cuda.is_available()`, else `"cpu"`; any other value
(`"cpu"`, `"cuda"`) passes through unchanged.

## `app_for`

The one-call path from a model to a mounted app: `LoadedModel` construction +
`prepare_serving()` + `build_app(state=...)`. It reads the same `DOWNSHIFT_*` environment
variables as the CLI (through `ServeOptions()` and `build_app`'s `api_key` default). Full
signature and parameter meaning: [`python-api.md`](python-api.md#app_for).
