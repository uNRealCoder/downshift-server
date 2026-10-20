# Serving layer

This page describes the code under `downshift serve` and `app_for`, for a person who changes it. To run a server, read [`production.md`](production.md). To call one, read [`http-api.md`](http-api.md).

| Module | Contents |
|---|---|
| `serve/options.py` | `ServeOptions`, `BackendChoice`, `ExecutionChoice`. No torch import. |
| `serve/engine.py` | `ServingState`, `choose_backend`, the boot builders and the warmup. |
| `serve/reuse.py` | The boot from an earlier export: the memo, the disk cache, or the `--workers` handoff. |
| `serve/app.py` | `build_app`, the routes, `PredictRoute`, and the request-ID and API-key middleware. |
| `serve/predict.py` | The request path of `/predict` and `/predict/graph`. |
| `serve/result_cache.py` | The result cache. |
| `serve/backends.py` | `OnnxRuntimeBackend` and `TorchBackend`. |
| `serve/describe.py`, `serve/schemas.py`, `serve/codec.py`, `serve/graphs.py` | `/schema`, the pydantic models, the base64 and safetensors codecs, and graph batching. |

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
    axis_max: dict[str, int] | None = None
    export_cache_dir: str | None = None
    intra_op_threads: int = 0
    inter_op_threads: int = 0
    output_encoding: OutputEncoding = OutputEncoding.json
    max_input_bytes: int = 256 * 1024 * 1024
    max_body_bytes: int = 32 * 1024 * 1024
    max_concurrency: int = 4
    execution: ExecutionChoice = ExecutionChoice.threadpool
    prep_threads: int = min(4, usable_cpus())
    max_queue: int = 64
    request_timeout: float = 30.0
    result_cache_rows: int | None = None  # None: 100 for independent rows, else off
    atol: float | None = None
    rtol: float | None = None
    seed: int = 0
    vary: str | None = None
    pooling: str | None = None
    normalize: bool | None = None
```

The values above apply when no `DOWNSHIFT_*` variable is set. Each default comes from `downshift.settings`, so the variables reach `app_for()` too. Each field has the meaning of the `serve` flag with the same name. Refer to [`cli.md`](cli.md#serve). To add a field, follow "Add a `serve` option" in [`cli.md`](cli.md#add-a-serve-option).

`BackendChoice` is what the caller asked for: `auto`, `onnxruntime` or `torch`. `auto` is never a running backend. `choose_backend` resolves it to a `BackendName` (`onnxruntime` or `torch`, from `core/verdict.py`).

## `choose_backend`

```python
def choose_backend(
    verdict: ExportVerdict, opts: ServeOptions, *, has_torch: bool | None = None
) -> tuple[BackendName, list[str]]
```

It returns the backend and the notes for the banner. The rules, in this order:

1. With `auto`, start from `verdict.recommended_backend`. Otherwise, start from `opts.backend`.
2. `onnxruntime` on a DEGRADED verdict without `force_onnx` raises `ValueError`.
3. `force_onnx` on a DEGRADED verdict selects `onnxruntime` and adds a note.
4. If `onnxruntime` is selected but the verdict has no graph (`onnx_program`, `onnx_path` and `onnx_bytes` are all empty), fall back to `torch` and add a note. This happens after a FAILED capture or `--backend torch`.
5. If `torch` is selected but there is no PyTorch model, raise `ValueError`. By default, `has_torch` is `verdict.prepared is not None`. A reused export passes `has_torch=True`, because the caller can load the model when it needs it.

## The boot builders

Each builder makes a `ServingState` through `_new_state`. `_new_state` attaches the Hugging Face metadata (`attach_hf_metadata`), records the boot timings, warms up and sets `ready`. A builder therefore cannot skip a step.

| Builder | Used by | What it does |
|---|---|---|
| `prepare_serving(loaded, opts, reference, tokenizer_from)` | a boot without a reusable export | Runs the gate (`intake()` for a `.onnx` file, else `prepare_model()` and `build_verdict()`), selects a backend, builds it, then `_new_state`. `--backend torch` and bfloat16-only weights skip the export and give an UNVERIFIED verdict. |
| `serving_state_from_artifact(source, verdict, opts, input_names, ...)` | a reused export on ONNX Runtime | Builds a session over `verdict.onnx_path` or `verdict.onnx_bytes`. No capture, no verification. The caller gives `input_names`, `axis_bounds`, the saved example feeds and the source kind, because the verdict has no `Prepared`. |
| `serving_state_from_torch_artifact(loaded, verdict, opts, ...)` | a reused export on torch | Prepares the loaded model again with the adapter and serves it eagerly. It uses the verdict of the earlier export. |

`serve/reuse.py` chooses between them:

- `prepare_serving_reusing(load, opts, ...)` makes the export key, looks it up in the memo of the process and then in `--export-cache-dir`, and boots from a hit with `state_from_entry`. On a miss, it calls `prepare_serving` and stores a CLEAN or DEGRADED result in both tiers. For a Hugging Face repo directory, the key comes from the files of the repo, so a hit that ONNX Runtime serves never loads the weights.
- `state_from_entry(entry, load, opts, ...)` selects the backend again from the verdict of the entry. It calls `load()` only if torch serves the model. A `serve --workers N` worker uses it too: the parent sends its export as an `ArtifactHandoff` (`cli/runtime.py`).

After the selection of ONNX Runtime on a CLEAN verdict, `prepare_serving` drops each reference to the torch model before the warmup. The peak memory then does not hold the weights two or three times. If the device is `cpu` and both thread counts are `0`, the backend reuses the session that `verify()` built.

`core.phase.report(Phase.x)` tells `/ready` which phase the loader is in. Outside an app with a loader, it does nothing.

`warmup(state, n)` runs `n` inferences on the example inputs, or on `synthesize_feeds(backend)` if there are none (dynamic axes become 1, floats are random, integers are zero). It sets `state.ready = True`, also for `n = 0`.

## `ServingState`

One server process holds one `ServingState`. `/metadata`, `/schema` and the predict routes read it.

| Field | Meaning |
|---|---|
| `source`, `source_kind` | The `MODEL` argument and its form (`onnx-file`, `torch-checkpoint`, `hf-repo-dir`, `import-spec`, `in-process-module`, `unknown`, from `sources.py`). Clients see only the file or directory name. |
| `verdict` | The `ExportVerdict` of the gate. |
| `backend` | The `OnnxRuntimeBackend` or `TorchBackend` that serves requests. |
| `input_names` | The flat input names, in the order of the forward arguments. |
| `options` | The `ServeOptions` of this state. |
| `example_inputs` | Real example inputs, if any. The warmup and the `--workers` handoff use them. |
| `ready`, `warmup_stats` | Set by `warmup()`. |
| `notes` | Lines for the banner, for example a backend fallback. |
| `reused` | `"memory"` or `"disk"` if the boot reused an earlier export. Otherwise `None`. |
| `hf_source` | The Hugging Face repo directory for the tokenizer and the recipe: the model itself, or `--tokenizer-from`. |
| `text` | A `TextIO` (tokenizer, token limit, labels) if `/predict` takes `text`. |
| `embedding` | The `EmbeddingRecipe` if the graph ends in a pooling step. |
| `vocab_size` | For a Hugging Face model. An `input_ids` value outside `[0, vocab_size)` gets a `400`. |
| `timings` | The seconds of each boot phase (`Phase` values: `load`, `export`, `verify`, `session`, `warmup`). |
| `axis_bounds` | `{input: {axis: DimBound(name, min, max)}}`, the traced range of each dynamic axis. Empty for a bare `.onnx` file. It gives `bounds` in `/schema` and the `400` for an axis out of range. |
| `executor`, `prep_executor` | The inference pool (`max_concurrency` threads) and the prep pool (`prep_threads` threads). |
| `in_flight` | The admitted predicts that did not finish. `try_admit()` claims one of `max_concurrency + max_queue` slots, and `release()` gives it back. |
| `result_cache`, `cache_off_reason` | The `ResultCache`, or `None` with the reason why the default turned it off. |

## `build_app`

```python
def build_app(
    state: ServingState | None = None,
    *,
    loader: Callable[[], ServingState] | None = None,
    middleware: Sequence[str] = (),
    api_key: str | None = settings.API_KEY,
    access_log: bool = True,
) -> FastAPI
```

Give exactly one of `state` and `loader`:

- `state=` serves at once. `app_for` and the tests use it.
- `loader=` is for `downshift serve`. The app binds with no state. The lifespan of the app runs `loader` on a background thread. Until it returns, `/health` gives `200`, and `/ready`, `/metadata`, `/schema` and the predict routes give `503`. If `loader` raises an error, `serve_cmd` stops the server.

The middleware stack, from the outside in: `RequestIdMiddleware` (the `X-Request-Id` header, the request ID on each log line, and the access log line), `ApiKeyMiddleware`, then your `--middleware`. Your middleware therefore sees only authenticated requests.

## Request path (`serve/predict.py`)

Per-request speed has priority over boot time. The boot does the export, the verification and the warmup, so each request pays only for its own work. One predict makes two thread hops:

```
event loop      PredictRoute: admit or 503, Content-Type (415), --max-body-bytes (413), read the body
prep pool       _parse: orjson and the pydantic model of the route (422), route checks (400)
                _prepare_feeds: arrays, tokenizing, graph batching, then the input checks (400)
inference pool  _infer: the backend call
                _encode_response: JSON, base64 or safetensors, split for each graph
```

- `PredictRoute` serves the raw body itself, so FastAPI does not parse it. The functions `predict` and `predict_graph` exist only to document the bodies in OpenAPI. `_validate_json` makes the same `422` bodies as FastAPI.
- The slot is admitted before the server reads the body, and released in `finally`.
- `--request-timeout` is checked after the prep and again when the inference starts. Downshift never stops an inference that runs.
- The input checks (`_violation`) run before the inference: the vocabulary, the edge index, the rank and fixed axes, and the axis bounds. A shape error of the model therefore never reaches the client as a `500`.
- `--execution inline` runs both hops on the event loop for a JSON body of up to 64 KiB with a `Content-Length` and no `text`.
- `_run_in` carries the context variables (the request ID) into the pool thread and records `prep_wait` and `infer_wait` for `Server-Timing`.

With the result cache on, `/predict` takes `_run_cached`. The prep hop makes one key for each row (`text_keys` before the tokenizer, `tensor_keys` from the arrays) and calls `ResultCache.lookup`. The lookup sorts the rows into hits, rows that another request computes, and rows that this request claims. Only the claimed rows go to the tokenizer and the model. After the inference, `ResultCache.complete` gives each row to its waiters. If the owner of a row fails or leaves, a waiter computes the row itself. The prep pool joins the rows and encodes the response, so a request in which each row hits makes no inference hop. `/predict/graph` does not use the cache.

## Backends (`serve/backends.py`)

```python
class Backend(Protocol):
    name: BackendName
    input_names: list[str]
    verified_provider: str | None  # "CPUExecutionProvider" if the numerics ran, else None

    def infer(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]: ...
    def metadata(self) -> BackendMeta: ...  # name, device, inputs and outputs as IOSpec
```

Both backends take and return dicts of numpy arrays. The output keys are positional (`output_0`, `output_1`, ...), so the responses look the same for both backends. bfloat16 and float16 tensors go on the wire as float32.

- `OnnxRuntimeBackend(model=None, device="auto", intra_op_threads=0, inter_op_threads=0, *, session=None)` wraps an `InferenceSession` with all graph optimizations on. `device="cuda"` needs `CUDAExecutionProvider`. If ONNX Runtime cannot start it, the class raises `ValueError` and does not run on the CPU. An `InvalidArgument` error from ONNX Runtime becomes `InferenceInputError`, which the route gives as a `400`. Other errors are a `500`.
- `TorchBackend(module, input_names, device="auto", example_inputs=None, intra_op_threads=0, dynamic_shapes=None)` runs the module eagerly under `torch.inference_mode()`. `device="cuda"` without CUDA raises `ValueError`. With `example_inputs`, it runs one pass at construction to learn the dtypes and shapes for `/metadata`. The shape of a dynamic axis is the name of its `torch.export.Dim`, or `"batch"` for axis 0 when there are no dynamic shapes. A missing input gives `InferenceInputError`. Any other error from the model is a `500`, because the input checks of the request path ran before the call.
