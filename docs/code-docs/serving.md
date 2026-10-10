# Serving layer

This page describes the programmatic parts under `downshift serve` and `app_for`:

- The options dataclass.
- The state object that a running server holds.
- How downshift builds a `ServingState`.
- The app that serves it.
- The two backends that it can wrap.

The code is split by job:

- `serve/options.py`: `ServeOptions` and `BackendChoice`.
- `serve/engine.py`: `ServingState` and `prepare_serving`.
- `serve/app.py`: `build_app`, the routing, and the request-id and API-key middleware.
- `serve/predict.py`: the internals of `/predict`.
- `serve/describe.py`: `GET /schema`.
- `serve/backends.py`, `serve/schemas.py` and `serve/codec.py`.

[`docs/production.md`](../production.md) has operational guidance (probes, sizing, Kubernetes and thread budgets). This page does not repeat it.

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

`ServeOptions` is defined in `downshift.serve.options`. It has no torch import at module scope. The CLI can therefore parse `--backend` and `--output-encoding` and print `--help` without the import cost of torch.

Every default comes from `downshift.settings`. The `DOWNSHIFT_*` environment variables therefore reach `ServeOptions()` and `app_for()`, and also the CLI. The values in the code above are the values when no variable is set. Downshift reads the variables once, when `downshift` is imported. A keyword argument has priority.

`serve_cmd` builds a `ServeArgs`. This is a dataclass that can be converted to plain JSON, in `cli/runtime.py`. It is `{load: LoadSpec, options: ServeOptions, reference, middleware, log_level, artifact, access_log}`. Downshift uses it to give a run to the `--workers N` worker processes, through an environment variable. `_collect_serve_args` turns the typer parameters of `serve` into its `LoadSpec` and `ServeOptions` pair. To add an option, you add a typer parameter, a `ServeOptions` field and a `_collect_serve_args` parameter. [`cli.md`](cli.md#serve) describes the CLI flag and the environment variable of each field.

| Field | Type | Default | Controls |
|---|---|---|---|
| `backend` | `BackendChoice` | `auto` | The backend to serve. `auto` follows the recommendation of the verdict. You can also force `onnxruntime` or `torch`. |
| `force_onnx` | `bool` | `False` | Serve a `DEGRADED` graph through ONNX Runtime. |
| `device` | `str` | `"auto"` | `"auto"`, `"cpu"` or `"cuda"`. `resolve_device` resolves it. `"cuda"` without CUDA raises `ValueError` (exit code `4` on the CLI) on both backends. |
| `warmup` | `int` | `3` | The number of inferences that run before `ServingState.ready` changes. |
| `k` | `int` | `8` | The number of verification samples. Downshift forwards it to the gate. |
| `adapter` | `str \| None` | `None` | The adapter name or spec. `None` detects the adapter. |
| `dynamic` | `dict[str, list[int]] \| None` | `None` | The override of the dynamic axes. Downshift forwards it to the gate. |
| `intra_op_threads` | `int` | `0` | `SessionOptions.intra_op_num_threads` of ONNX Runtime. `0` lets ONNX Runtime choose. If the value is more than `0`, it also sets `torch.set_num_threads` for the torch backend. |
| `inter_op_threads` | `int` | `0` | `SessionOptions.inter_op_num_threads` of ONNX Runtime. `0` lets ONNX Runtime choose. |
| `output_encoding` | `OutputEncoding` | `json` | The default encoding of response tensors. The `output_encoding` field of a request overrides it. |
| `max_input_bytes` | `int` | 256 MiB | The limit on one decoded base64 tensor input. |
| `max_body_bytes` | `int` | 32 MiB | The limit on the whole request body (`413`). Downshift checks it before it parses the body. There is no separate limit on text length. This limit also applies to `text` requests. |
| `max_concurrency` | `int` | `4` | The size of the inference thread pool (`ServingState.executor`) in each worker process. It is the number of requests that infer and encode at the same time. |
| `execution` | `ExecutionChoice` | `threadpool` | `threadpool` runs the work of each request on the two pools. `inline` runs a small JSON body (Content-Length up to 64 KiB, no `text`) on the event loop. Refer to [Request path](#request-path-servepredictpy). |
| `prep_threads` | `int` | `min(4, usable CPUs)` | The size of the prep pool (`ServingState.prep_executor`). The pool parses, validates and converts request bodies, and it tokenizes. |
| `axis_max` | `dict[str, int] \| None` | `None` | `--axis-max NAME=N`. It sets a lower maximum that the server serves for a dynamic axis. Verification pins one sample at exactly `N`. |
| `export_cache_dir` | `str \| None` | `None` | `--export-cache-dir`. The directory where downshift keeps verified exports. The next boot then skips the export and the verification. |
| `max_queue` | `int` | `64` | The number of additional predicts that can be admitted and wait beyond `max_concurrency`. |
| `request_timeout` | `float` | `30.0` | The number of seconds that an admitted predict can wait in the queue. After this time, it gets a `503` and does not run. `0` turns the check off. |
| `atol` / `rtol` | `float \| None` | `None` | The override of the tolerance. Downshift forwards it to the gate. `None` means "by dtype". |
| `seed` | `int` | `0` | The seed of the verification samples. Downshift forwards it to the gate. |
| `vary` | `str \| None` | `None` | A `pkg.module:fn` that replaces the verification sampler of the adapter. |
| `pooling` | `str \| None` | `None` | A `PoolingChoice` value (`mean`, `cls`, `max`, `mean_sqrt_len`, `lasttoken`, `weightedmean`, `none`). It overrides the embedding recipe of a Hugging Face repo. |
| `normalize` | `bool \| None` | `None` | L2-normalise the embedding. `None` means the setting of the recipe. |

## `BackendChoice`

```python
class BackendChoice(StrEnum):
    auto = "auto"
    onnxruntime = "onnxruntime"
    torch = "torch"
```

`BackendChoice` is what the caller asked for. `"auto"` exists only in the CLI and the options. It is never a running backend. Before downshift builds anything, `choose_backend` resolves it to a concrete `BackendName` (`"onnxruntime"` or `"torch"`, from `core.verdict`).

## `choose_backend`

```python
def choose_backend(verdict: ExportVerdict, opts: ServeOptions) -> tuple[BackendName, list[str]]
```

`choose_backend` returns `(backend_name, notes)`. These rules apply, in this order:

1. If `opts.backend == BackendChoice.auto`, start from `verdict.recommended_backend`. Otherwise, start from `opts.backend`.
2. `--backend onnxruntime` on a `DEGRADED` verdict without `--force-onnx` raises `ValueError`. This is the one case where `serve` refuses to boot and does not fall back silently.
3. `--force-onnx` on a `DEGRADED` verdict selects `onnxruntime`. Downshift adds a banner note ("serving a DEGRADED graph; outputs may be wrong").
4. If `onnxruntime` is selected, but the verdict has no ONNX graph (`onnx_program is None and onnx_path is None`), downshift falls back to `torch` and adds a note. This occurs after a `FAILED` capture, or when `--backend torch` skipped the export.
5. If `torch` is selected, but the verdict has no `prepared`, `choose_backend` raises `ValueError`. This occurs for an `intake()` of a bare `.onnx` file without a reference. It never ran an adapter, so there is no `nn.Module` to run eagerly.

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

One running server, or one worker process, holds this object. `/metadata` reads it. The predict routes admit requests against it (`PredictRoute` in `serve/app.py` and `run_predict` in `serve/predict.py`).

| Field | Meaning |
|---|---|
| `source` | The model spec that `serve` or `app_for` received. It is always a form that is already on this machine. `/metadata` shows it in the `model` field. |
| `verdict` | The `ExportVerdict` that the gate made. |
| `backend` | The concrete `Backend` instance (`OnnxRuntimeBackend` or `TorchBackend`) that serves requests. |
| `input_names` | The flat input names, in the order of the forward arguments. |
| `options` | The `ServeOptions` that built this state. |
| `example_inputs` | Real example inputs, if there are any. Downshift uses them to warm up with realistic shapes and not with synthesized shapes. It also uses them to build the `.npz` sidecar of `--workers N`. |
| `ready` | `True` after the state is warmed up. The last step of `warmup()` sets it, also with `--warmup 0`. |
| `notes` | The strings that the CLI banner prints. Examples: a warning about `--force-onnx`, or a note about a backend fallback from `choose_backend`. |
| `warmup_stats` | A `WarmupStats` after `warmup()` ran. Otherwise `None`. |
| `source_kind` | The accepted form that `source` names: `onnx-file`, `torch-checkpoint`, `hf-repo-dir`, `import-spec`, `in-process-module` or `unknown`. The constants are in `downshift/sources.py`. `/schema` reports it under `source`. The `Model` row of the banner shows it. |
| `text` | A `TextIO` (`adapters/text.py`: the tokenizer, the token limit of the model, and the labels of a classifier). It exists when the source is a Hugging Face repo directory with tokenizer files. It lets `/predict` take `{"text": ...}`. |
| `embedding` | An `EmbeddingRecipe` (`adapters/embedding.py`). It exists when the repo declares a pooling recipe, or `--pooling` gives one. The recipe is already part of the exported graph. Downshift keeps it so that `/schema` can say what it is. |
| `vocab_size` | The vocabulary size of the Hugging Face model. Downshift sets it also when the tokenizer or the recipe fails to load. An `input_ids` value outside `[0, vocab_size)` gets a `400` before the inference. |
| `timings` | The wall-clock seconds of each boot phase that ran. The keys are `Phase` values (`downshift/core/phase.py`, a `StrEnum` of `load`, `export`, `verify`, `session` and `warmup`). The caller adds `load` after `prepare_serving` returns. The `Boot` row of the banner and the `boot` field of `/metadata` read it. |
| `axis_bounds` | For each input and each dynamic axis: a `DimBound(name, min, max)`. It is the range that the export of the adapter was traced for. It is empty for a bare `.onnx` file with no reference model. It drives the `bounds` in `/schema` and the readable `400` for an axis that is out of range. |
| `executor` | The inference pool: a `ThreadPoolExecutor` with the size `options.max_concurrency`. It is not part of the identity of the dataclass (`compare=False`). The backend call and the response encoding of each request run here. They never run on the event loop, unless `--execution inline` is set. A slow request therefore never blocks `/health` or `/ready`. |
| `prep_executor` | The prep pool, with the size `options.prep_threads`. It does the parse, the validation, the conversion to arrays and the checks of each request. |
| `in_flight` | The number of predicts that are admitted and not finished (in a pool, or waiting for one). An internal lock guards it. |

Methods:

- `try_admit()` claims one of the `max_concurrency + max_queue` slots in one atomic step. It returns `False` if no slot is free.
- `release()` gives one slot back.

Properties:

- `backend_auto_selected` is `True` when `options.backend == BackendChoice.auto` and not `forced_onnx`.
- `forced_onnx` is `True` when `--force-onnx` is set and the verdict is `DEGRADED`.
- `declared_dtypes` is a `cached_property`. It is `{input_name: dtype}`, read once from `backend.metadata().inputs`. Downshift uses it to interpret ambiguous JSON input on each request, without deriving it again.

A `--workers N` worker can rebuild from the ONNX artifact of the parent. It derives `source_kind` again from the spec (`loading.source_kind`). It therefore reports the kind of the original model.

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

`prepare_serving` does these steps:

1. It runs the gate. For a `.onnx` `LoadedModel`, it runs `intake()`. Otherwise, it runs `prepare_model()` and `build_verdict()`. If `opts.backend == BackendChoice.torch`, the export is skipped, and it makes an `UNVERIFIED` verdict for torch only.
2. It selects a backend with `choose_backend`.
3. It builds the backend.
4. It warms up the backend (`opts.warmup` inferences).

`serve_cmd` and `app_for` both call it. `serve_cmd` adds a `Phase.load` timing entry itself. The read and import of the model happen before `prepare_serving` is called. They are outside this function.

While `prepare_serving` runs, `core.phase.report(Phase.x)` tells `/ready` which phase the loader is in. Outside an app that a loader built, this call does nothing.

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

This function rebuilds a `ServingState` in a `serve --workers N` worker. It starts from the ONNX graph and the verdict that the parent process already exported and verified. There is no capture and no verification. It makes a new `OnnxRuntimeBackend` session over `onnx_path`, and it does a warmup pass.

- `input_names` and `notes` come from the parent. The verdict that the parent ships has no `prepared`, so the worker cannot derive them.
- The axis names and bounds of the adapter (`axis_bounds`) also come from the parent. The parent ships them as JSON with `axis_bounds_to_json`. `/schema` and the `400` for an axis out of range need them.
- If the parent had `example_inputs`, they come from the `.npz` sidecar that the parent wrote. Otherwise, `warmup()` synthesizes them.

`serving_state_from_torch_artifact(loaded, verdict, opts, notes=None)` is the equivalent for torch. The worker reloads the model itself, because downshift does not ship torch weights between processes. It prepares the model, but it uses the verdict that the parent already verified.

## `warmup` and `synthesize_feeds`

```python
def warmup(state: ServingState, n: int) -> WarmupStats
def synthesize_feeds(backend: Backend) -> dict[str, np.ndarray]
```

`warmup` runs `n` inferences. It uses `state.example_inputs` if they exist. Otherwise, it uses `synthesize_feeds`. It records `WarmupStats`, and it sets `state.ready = True` for all `n`, also `n=0`. `ready` means "the state finished the boot". It does not mean "at least one warmup inference ran".

`synthesize_feeds` builds one dummy array for each input that the backend declares:

- Dynamic or unknown axes get size 1.
- Floats are `np.random.randn`.
- Integers and bools are zero.

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

You must give exactly one of `state` and `loader`. Otherwise, `build_app` raises `ValueError`.

- **`state=`** is for library use, `app_for` and tests. The app serves `state` immediately. There is no `503` window on `/ready`. The gate and the warmup already ran, synchronously, before the call to `build_app`.
- **`loader=`** is for `downshift serve` with one worker. The app binds with no `ServingState`. `/health` returns `200` immediately. `/ready`, `/metadata`, `/schema` and both predict routes return `503` until `loader` returns a `ServingState`. The lifespan of the app starts `loader` on a background thread when uvicorn starts to serve. It does not start when `build_app` is called. The returned `ServingState` goes to `app.state.serving`. During this time, `phase` on `/ready` says which `Phase` the loader is in. If `loader` raises an error, `app.state.serving` stays `None` for this function. `serve_cmd` holds the `uvicorn.Server` and sets `should_exit` itself when the loader fails. The process then exits and does not serve `503` forever.

`middleware` is a sequence of `pkg.module:Attr` specs (`--middleware`, repeatable). `load_middleware` (`serve/middleware.py`) attaches them in order. Each spec must resolve to one of these:

- A middleware class that is built as `cls(app)`: pure ASGI, or a subclass of `starlette.middleware.base.BaseHTTPMiddleware`.
- An `async (request, call_next)` coroutine function.

`api_key` defaults to `DOWNSHIFT_SERVER_API_KEY` (`settings.API_KEY`). The variable therefore reaches `app_for()` and all other library callers, and also the CLI. If the key is set, every route except `/health` and `/ready` needs `Authorization: Bearer <api_key>`. Downshift compares it in constant time. Otherwise, the route answers `401` with `WWW-Authenticate: Bearer`.

`None` or an empty string means unauthenticated. `build_app` then logs one `WARNING` (logger `downshift.serve`) that the endpoints are unauthenticated. To fix this, set the key or add your own authentication middleware. To override the environment for one app, pass `api_key=None`.

`access_log` (default `True`, `--access-log` on the CLI) controls the one log line for each request that `RequestIdMiddleware` writes on the logger `downshift.access`. The format is `METHOD PATH STATUS N ms`. The level is:

- `INFO` for normal requests.
- `WARNING` from status `400`.
- `DEBUG` for the probes `/health` and `/ready`.

Downshift sets and returns the request ID in both cases.

Each app has this stack of middleware, from the outermost to the innermost:

1. `RequestIdMiddleware` assigns or echoes `X-Request-Id`. It binds the ID to `downshift.logs.request_id_var`, so each log line that is written while the request is served has `request_id=...`. It also writes the request line.
2. `ApiKeyMiddleware`.
3. Your `middleware`. It therefore sees only authenticated traffic.

`/predict` and `/predict/graph` use the `PredictRoute` route class, which the next section describes. All other routes are plain FastAPI routes. [`http-api.md`](http-api.md) has the details of the wire level.

## Request path (`serve/predict.py`)

Serving speed has the first priority in the design of downshift. The boot can be slow, because the export, the verification and the warmup all happen before `/ready`. Each request pays only for its own work. One predict request makes two thread hops:

```
event loop     PredictRoute: admit or 503, Content-Type (415), --max-body-bytes (413),
               read the body
prep pool      _parse: orjson + the route's pydantic model (422), per-route checks (400)
               _prepare_feeds: arrays, tokenizing, graph batching, vocab/edge/shape/bound
               checks (400)
inference pool _infer: the backend call only
               _encode_response: JSON, base64 or safetensors, split per graph
```

- `PredictRoute` serves the two routes itself. FastAPI does not resolve the body. `run_predict` gets the raw bytes. The functions `predict` and `predict_graph` exist only so that the OpenAPI schema documents `PredictRequest` and `GraphPredictRequest`. `_validate_json` makes the same `422` shapes as FastAPI: a missing body, `json_invalid`, and validation errors under `"body"`.
- Admission happens before downshift reads one byte. The slot is released in `finally` on every exit. `--request-timeout` is checked after the prep and again when the inference starts. Downshift never interrupts an inference that is running.
- The encoding runs on the inference thread directly after the backend call. It therefore holds the inference slot. This saves one hop. The cost is lower throughput when the outputs are large and `--max-concurrency` is the bottleneck.
- `--execution inline` runs both hops on the event loop for a JSON body with a Content-Length of up to 64 KiB (`INLINE_MAX_BODY_BYTES`). A `text` request still tokenizes in the prep pool. This mode helps only for models that infer in much less than one millisecond. It stalls `/health` and `/ready` for as long as one inference takes.
- `_run_in` carries the contextvars of the request into the pool thread. It records the wait as `prep_wait` or `infer_wait`. `Server-Timing` (and `timings_ms` of the access log) lists `parse`, `prep_wait`, `prep`, `infer_wait`, `infer` and `encode`.

## Backends (`serve/backends.py`)

```python
class Backend(Protocol):
    name: BackendName
    input_names: list[str]
    verified_provider: str | None

    def infer(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]: ...
    def metadata(self) -> BackendMeta: ...
```

`verified_provider` is the execution provider that the numerics gate ran against. It is `"CPUExecutionProvider"` if the numerics ran. Otherwise, it is `None`. `engine._build_backend` sets it directly after it builds a backend. The `Verified on` row of the banner prints it. The row also says so when the server runs on a device that the numerics never covered.

Both concrete backends take and return dicts of numpy arrays. The keys are the input or output names. `serve/app.py` therefore does not need to know which backend it uses.

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

This class wraps an `onnxruntime.InferenceSession`. Pass `model` (bytes, or a path to a `.onnx` file). Or pass an existing `session` directly. `prepare_serving` passes `session=` when it can reuse the exact session that `verify()` built. This is possible when the device resolves to `cpu` and both thread counts are `0`, which is the default session of ONNX Runtime. It avoids a second load of the same graph.

- `device="cuda"` selects `CUDAExecutionProvider`, with `CPUExecutionProvider` behind it. This requires that it is in `onnxruntime.get_available_providers()`. If it is not, the class raises `ValueError`. It does not run on the CPU without a message.
- Graph optimization is always `ORT_ENABLE_ALL`.
- The outputs have positional keys (`output_0`, `output_1`, and so on), for all names that the ONNX graph uses. The responses therefore look the same for both backends.
- ONNX Runtime `InvalidArgument` and `InvalidGraph` exceptions become `InferenceInputError` (a `400` and not a `500`). So does each `Fail` exception whose message contains "shape". Other exceptions propagate.

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

This class wraps an `nn.Module` in eager mode: `module.eval().to(device)`. `device="cuda"` without `torch.cuda.is_available()` raises `ValueError`. If `intra_op_threads > 0`, the class calls `torch.set_num_threads`. With `0`, it leaves the default of torch.

A bfloat16 or float16 module is served on a float32 wire. Numpy has no bfloat16. The backend casts floating inputs to the dtype of the module on the way in. It widens the outputs to float32 on the way out. Such a model boots on the torch fallback, and `/schema` reports its dtypes as `float32`.

If you give `example_inputs`, the class does one pass through the model at construction, with no warmup. This pass fills in the real dtypes and shapes for `/metadata` (`_spec_from_tensor` and `_spec_from_array`). Without `example_inputs`, the input specs have `dtype=None, shape=None`, and there are no output specs.

`infer()` converts numpy inputs to tensors with `torch.from_numpy`. It copies only when the array is not C-contiguous or not writable. Base64-decoded arrays are sometimes like this. It runs under `torch.inference_mode()`. It turns only errors that the client caused into `InferenceInputError` (a `400`):

- An `IndexError`.
- A `RuntimeError` or `ValueError` whose message names a problem with shape, size, dimension, dtype, broadcast or range, and that is not an out-of-memory error.

Any other exception from the model propagates and becomes a `500`. Examples: an unsupported operation, a CUDA failure, a bug in the module, and an out-of-memory error.

### `BackendMeta` and `IOSpec`

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

`backend.metadata()` returns `BackendMeta`. The `backend` field of `/metadata` serializes it (`BackendMeta.to_dict()`).

- `IOSpec.dtype` for the ONNX Runtime backend is the own string of ONNX Runtime, in the style `"tensor(float)"`. For the torch backend, it is in the style `"tensor(float32)"` (the torch dtype name). This applies when example inputs were available.
- `IOSpec.shape` marks axis 0 with the string `"batch"` for the specs of the torch backend (`_dynamic_shape`). Axis 0 is dynamic for all models that downshift serves. The ONNX Runtime backend reports what the graph declares.

### `resolve_device`

```python
def resolve_device(device: str) -> str
```

`"auto"` becomes `"cuda"` if `torch.cuda.is_available()`. Otherwise, it becomes `"cpu"`. Other values (`"cpu"` and `"cuda"`) pass through unchanged.

## `app_for`

`app_for` is the one-call path from a model to a mounted app. It does three things: it builds the `LoadedModel`, it runs `prepare_serving()`, and it runs `build_app(state=...)`. It reads the same `DOWNSHIFT_*` environment variables as the CLI, through `ServeOptions()` and the `api_key` default of `build_app`. [`python-api.md`](python-api.md#app_for) has the full signature and the meaning of the parameters.
