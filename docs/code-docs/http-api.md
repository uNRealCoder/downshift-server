# HTTP API reference

Every route `build_app` registers, its request/response schemas, status codes, headers,
authentication, and the wire formats for tensor data. Routes and schemas are the same regardless of which
backend (`onnxruntime` or `torch`) is behind them.

## Routes

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness. |
| `GET` | `/ready` | Readiness. |
| `GET` | `/metadata` | Model, backend, verdict, limits, boot timings, warmup stats. |
| `GET` | `/schema` | What to POST: input names, dtypes, shapes, and an example body. |
| `POST` | `/predict` | Named tensor inputs or `text`, any model. JSON or safetensors body. |
| `POST` | `/predict/graph` | One graph (`x`, `edge_index`, optional `edge_attr`) or a batch of graphs. JSON or safetensors body. |

Whatever is served came from this machine - a downloaded `.onnx` file, a downloaded PyTorch
checkpoint, a downloaded Hugging Face repo directory (recognised by the `config.json` in
it), or a module importable in the server process. `/schema`'s `source` block says which of
those, per server; nothing is ever fetched to answer a request.

## Authentication

Optional, one shared key. When `DOWNSHIFT_SERVER_API_KEY` is set (or `build_app(...,
api_key=...)` / `app_for(..., api_key=...)` is given a value; both default to the
variable), every route except `/health` and `/ready` requires
`Authorization: Bearer <key>`. The scheme is matched case-insensitively and the key is
compared in constant time (`hmac.compare_digest`). A missing or wrong credential is a `401`
with the header `WWW-Authenticate: Bearer` and the body:

```json
{"detail": "Authorization header is not set or incorrect"}
```

The check runs before the route, so it applies to unknown paths and to `/docs` and
`/openapi.json` as well. When the key is unset or an empty string, nothing is checked and
one `WARNING` is logged at startup that the endpoints are unauthenticated (set the key, or
add your own authentication middleware). The rest of this page describes the routes
without repeating the `401`.

```bash
curl -s localhost:8000/schema -H "Authorization: Bearer $DOWNSHIFT_SERVER_API_KEY"
```

### `GET /health`

Always `200` once the process is up and the event loop is running - even while a model is
still loading. Runs on the event loop directly (`async def`), never queued behind a
predict. Response model `HealthResponse`:

```json
{"status": "ok"}
```

```bash
curl -s localhost:8000/health
```

### `GET /ready`

`200` with `{"ready": true}` once the model has loaded, exported, verified and warmed up;
`503` before that. Two distinct `503` shapes, both without a `Retry-After` header:

- No `ServingState` exists yet (the background loader, started when `serve` was given no
  `state=` up front, hasn't landed one): `{"ready": false, "phase": "export"}`. `phase` is
  the step the loader is in right now, one of `load`, `export`, `verify`, `session`,
  `warmup` (the `Phase` enum in `core/phase.py`, the same names as `/metadata`'s `boot`
  keys and the banner's `Boot` row). It is `null` for an app built with `state=` directly,
  which has no loading window, and absent from the `200` body.
- A `ServingState` exists but its `ready` flag is `False`: `{"ready": false}`. In practice
  `ready` is set `True` as the last step of `prepare_serving()` (inside `warmup()`, even
  with `--warmup 0`), so this second shape is not reachable in the current single-worker
  or `--workers N` boot paths - a `ServingState` object only ever appears already warmed.

Response model `ReadyResponse` (`{"ready": bool, "phase": str | None}`); the route writes
its bodies by hand, so a ready server answers `{"ready": true}` with no `phase` key.

```bash
curl -s -i localhost:8000/ready
```

### `GET /metadata`

`503` with `{"detail": "model is not ready"}` and header `Retry-After: 2` while no
`ServingState` exists yet. `200` with a `MetadataResponse` once it does:

```python
class MetadataResponse(BaseModel):
    model: str
    family: str
    verdict: dict
    backend: dict
    input_names: list[str]
    notes: list[str] = []
    version: str
    limits: Limits  # max_body_bytes, max_input_bytes, max_concurrency, max_queue, request_timeout
    execution: str  # "threadpool" or "inline"
    boot: dict[str, float] = {}
    warmup: dict | None = None
```

| Field | Meaning |
|---|---|
| `model` | The source `serve`/`app_for` was given - always something already on this machine (a `.onnx` file, a checkpoint, a downloaded Hugging Face repo directory, or an import spec). A file or directory is named by its last path component only (`/srv/models/bert` reports `bert`); the server's directory layout is never sent. An import spec is reported as given. |
| `family` | `verdict.model_family`. |
| `verdict` | `ExportVerdict.to_dict()` without `onnx_path` (a location on the server's disk) - see [`python-api.md`](python-api.md#exportverdict). Paths quoted inside `reason`, `warnings` and `notes` are cut to their file name too. |
| `backend` | `BackendMeta.to_dict()`: `{"name", "device", "inputs": [...], "outputs": [...]}`, each input/output an `IOSpec` (`name`, `dtype`, `shape`). |
| `input_names` | Flat input names, in forward-argument order. |
| `notes` | Notes for the banner (e.g. `--force-onnx` warnings). |
| `version` | `downshift.__version__`. |
| `limits` | `max_body_bytes`, `max_input_bytes`, `max_concurrency`, `max_queue`, `request_timeout`, read from `ServeOptions`; the same block as `/schema`'s `limits`. |
| `execution` | `--execution`: `"threadpool"` or `"inline"`. |
| `boot` | Wall-clock seconds per phase (`load`, `export`, `verify`, `session`, `warmup` - whichever ran); the same dict the CLI's `Boot` banner row prints. |
| `warmup` | `{"count", "mean_ms", "synthesized"}` from `WarmupStats`, or `null` if warmup hasn't run. |

```bash
curl -s localhost:8000/metadata | python -m json.tool
```

### `GET /schema`

The answer to "what do I POST?". `/metadata` is the operator's view - how the model was
judged, how long each boot phase took. `/schema` is the caller's: the name, dtype and shape
of every input, and an example body that can be posted back to `/predict` unchanged.

`503` with `{"detail": "model is not ready"}` and `Retry-After: 2` while no `ServingState`
exists yet, exactly like `/metadata`. `200` with a `SchemaResponse` once it does. Built per
request from the backend's own declared IO (`downshift/serve/describe.py`), so it always
describes the graph that is actually running, never what the source model promised.

| Field | Meaning |
|---|---|
| `model` | The source the server was given, named by its file or directory name only (an import spec is reported as given); same rule as `/metadata`'s `model`. |
| `source` | `{"spec", "kind", "description", "fetched_at_runtime": false}`. `kind` is one of `onnx-file`, `torch-checkpoint`, `hf-repo-dir`, `import-spec`, `unknown`; `description` spells the same thing out in a sentence. |
| `family` | `verdict.model_family`. |
| `backend` | `"onnxruntime"` or `"torch"`. |
| `device` | The ORT execution provider, or the torch device. |
| `endpoint` | `"/predict"` - the route these inputs go to. |
| `graph_endpoint` | `"/predict/graph"` when the model's inputs include both `x` and `edge_index`, else `null`. |
| `inputs` | One `TensorSchema` per input, in forward-argument order. All are required. For a model downshift exported itself, each carries the adapter's axis names and per-axis `bounds` (below). |
| `outputs` | One `TensorSchema` per output. Empty on a torch backend served without example inputs - nothing has been run through the model yet, so nothing is known (a `notes` entry says so). |
| `example_request` | A complete, well-formed `/predict` body, or `null` when it would run past 256 elements across all inputs (a `notes` entry then gives the shape to build yourself). The values are filler - zeros, or ones for an attention mask, where zero would be a well-formed request that hands back NaNs. Only the names, dtypes and shapes are meant to be copied. |
| `example_curl` | `example_request` as a runnable `curl` against this server's own URL, or `null` alongside a `null` example. |
| `input_formats` | The three wire forms a tensor value may take (nested list, typed object, base64) - the same rules as "Wire formats" below. |
| `output_encodings` | `["json", "base64"]`. |
| `default_output_encoding` | What this server uses when a request omits `output_encoding`. |
| `limits` | The same five limits as `/metadata`'s `limits`. |
| `notes` | Anything a caller would otherwise be surprised by; see the fields above. |

Each entry in `inputs`/`outputs` is a `TensorSchema`:

```python
class TensorSchema(BaseModel):
    name: str
    dtype: str | None = None  # numpy name: "float32", "int64" - not ORT's "tensor(float)"
    shape: list[int | str] | None = None  # int: fixed. str: a dynamic axis, named or "dynamic"
    required: bool | None = None  # True on inputs, null on outputs
    example_shape: list[int] | None = None  # `shape` with each dynamic axis pinned to 1
    bounds: list[AxisBound | None] | None = None  # parallel to `shape`; inputs only
```

A dynamic axis that an adapter named keeps its name (`"batch"`, `"seq"`); one that
`torch.export` generated for itself (`"s77"`, `"u3"`) is reported as `"dynamic"`, since
those names mean nothing outside the tracer.

`bounds` is parallel to `shape`: for each dynamic axis of a model downshift exported
itself, `{"min": int, "max": int}` is the range that export was traced for (the
`torch.export.Dim`'s own limits), and `null` stands in for every other axis. A request whose
axis falls outside the range is refused with a `400` that names the input, the axis, the
size and the range (`input_ids axis 1 is 65; this model accepts 1 to 64`), rather than ONNX
Runtime's or torch's own wording. `bounds` is `null` altogether, and the axes read
`"dynamic"`, for a bare `.onnx` served with no `--reference`: there is no adapter to name or
bound them. With `--workers N` the parent's names and bounds are shipped to the workers, so
they answer the same way.

Dtypes are the wire dtypes. A bfloat16 model reads as `float32` (numpy has no bfloat16, and
the torch backend casts inside itself, taking and returning float32); the torch backend
reports float16 as `float32` for the same reason.

```bash
curl -s localhost:8000/schema | python -m json.tool
```

```json
{"model": "bert-base-uncased",
 "source": {"spec": "bert-base-uncased", "kind": "hf-repo-dir",
            "description": "a downloaded Hugging Face repo directory on this machine (has config.json)",
            "fetched_at_runtime": false},
 "family": "hf", "backend": "onnxruntime", "device": "CPUExecutionProvider",
 "endpoint": "/predict", "graph_endpoint": null,
 "inputs": [{"name": "input_ids", "dtype": "int64", "shape": ["batch", "seq"],
             "required": true, "example_shape": [1, 1],
             "bounds": [{"min": 1, "max": 4096}, {"min": 1, "max": 512}]},
            {"name": "attention_mask", "dtype": "int64", "shape": ["batch", "seq"],
             "required": true, "example_shape": [1, 1],
             "bounds": [{"min": 1, "max": 4096}, {"min": 1, "max": 512}]}],
 "outputs": [{"name": "output_0", "dtype": "float32", "shape": ["dynamic", "dynamic", 16],
              "required": null, "example_shape": [1, 1, 16]}],
 "example_request": {"inputs": {"input_ids": [[0]], "attention_mask": [[1]]}},
 "example_curl": "curl -s http://localhost:8000/predict -H 'content-type: application/json' -d '...'"}
```

Round-tripping the example is the quickest check that a server is usable:

```bash
curl -s localhost:8000/predict -H 'content-type: application/json' \
  -d "$(curl -s localhost:8000/schema | python -c 'import json,sys; print(json.dumps(json.load(sys.stdin)["example_request"]))')"
```

### `POST /predict`

Request model `PredictRequest`:

```python
class PredictRequest(BaseModel):
    inputs: dict[str, Any] = {}  # name -> nested list, or a {"data", "dtype", "shape"} object
    text: str | list[str] | None = None  # instead of `inputs`; see "Text input" below
    prompt_name: str | None = None  # a named prompt from the repo, `text` requests only
    output_encoding: RequestOutputEncoding | None = (
        None  # "json" | "base64" | "safetensors"; omit to use the server default
    )
```

The body may instead be a safetensors file (`Content-Type: application/vnd.safetensors`, see
"Wire formats"), one tensor per input name.

Response model `PredictResponse`:

```python
class PredictResponse(BaseModel):
    outputs: dict[str, Any]
    shapes: dict[str, list[int]]
    dtypes: dict[str, str]
    predictions: list[dict] | None = None  # sequence classifiers only
```

Nested-list body:

```bash
curl -s localhost:8000/predict -H 'content-type: application/json' \
  -d '{"inputs": {"x": [[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]]}}'
```

```json
{"outputs": {"output_0": [[0.199, -0.206, 0.561, 0.405]]},
 "shapes": {"output_0": [1, 4]},
 "dtypes": {"output_0": "float32"}}
```

Base64 body (see "Wire formats" below for the encoding rules), asking for a base64
response too:

```bash
curl -s localhost:8000/predict -H 'content-type: application/json' \
  -d '{"inputs": {"x": {"data": "AACAPwAAAEA=", "dtype": "float32", "shape": [1, 2]}},
       "output_encoding": "base64"}'
```

```json
{"outputs": {"output_0": {"data": "<base64 little-endian float32 bytes>", "dtype": "float32", "shape": [1, 2]}},
 "shapes": {"output_0": [1, 2]},
 "dtypes": {"output_0": "float32"}}
```

Integer lists default to `int64`, everything else to `float32`, when no `dtype` is given
and the backend declares none either (`to_numpy` in `serve/schemas.py`).

Send large tensors as safetensors or base64. Both skip the float-to-text round trip; a
JSON float list is parsed by Python in the prep pool, and while it is parsed it holds the GIL,
which slows every other request in that process. safetensors also skips the JSON wrapper and
base64's 33% overhead (measured: `bert_small` p50 11.2 ms with JSON, 9.1 ms with
safetensors). `Server-Timing`'s `parse` and `prep` entries show what a body cost.

A predict is admitted, or refused with a `503`, *before* its body is read: an overloaded
server does not buffer or parse a body it is about to turn away. A body over
`--max-body-bytes` (default 32 MiB) is a `413`: refused without reading it when
`Content-Length` says so, and, for a chunked body with no `Content-Length`, as soon as the
running total passes the limit. There is no separate limit on text length; the body cap is
the limit for a `text` request.

#### Text input and class probabilities

A model served from a downloaded Hugging Face repo directory that has tokenizer files
(`tokenizer.json`, `vocab.txt`, ...) also takes text. Send `text` (one string, or a list)
instead of `inputs`; sending both is a `422`. `GET /schema` shows a `text_input` block when
this is available, and a `400` says so when it is not.

```bash
curl -s localhost:8000/predict -H 'content-type: application/json' \
  -d '{"text": ["What is the capital of France?", "Ignore all previous instructions."]}'
```

The server tokenizes and pads the batch. Each row may be at most as long as the smallest limit
the model's own files declare: the tokenizer's `model_max_length`, the model's `max_position_embeddings`, and
for an embedding model the `max_seq_length` its author trained at (see "Embedding models"). None
of these is downshift's to choose. For the RoBERTa family (`roberta`, `xlm-roberta`,
`xlm-roberta-xl`, `camembert`) the usable length is `max_position_embeddings - (pad_token_id +
1)`, because position ids start after the padding index: 514 positions serve 512 tokens. A
longer row is refused with a 400 naming each over-long row and its token count, never cut: a
classifier that read only the head of a long input would answer about text it never saw. The
only cap on how much text a request carries is the request body limit (`--max-body-bytes`).
A single string is a batch of one, so `shapes` never
depends on which form was sent.

When `config.json` declares a sequence-classification architecture, the repo is loaded with its
classification head (`AutoModel` would drop it) and `output_0` is the logits. The response
then also carries `predictions`, one entry per row, for `text` and tensor requests alike:

```json
{"outputs": {"output_0": [[-3.1, 5.2, -1.9]]},
 "shapes": {"output_0": [1, 3]},
 "dtypes": {"output_0": "float32"},
 "predictions": [{"label": "INJECTION", "score": 0.9997,
                  "probabilities": {"BENIGN": 0.0002, "INJECTION": 0.9997, "JAILBREAK": 0.0001}}]}
```

`probabilities` is a softmax over the logits, or a per-label sigmoid when the config says
`multi_label_classification`. Labels come from `config.json`'s `id2label`. Token-classification
repos load with their head too and return logits, without `predictions`.

#### Embedding models

`config.json` says nothing about pooling, but a sentence-transformers repo carries its recipe in
sibling files, which downshift reads: `modules.json` (Transformer, then Pooling, then an optional
Normalize), `1_Pooling/config.json` (mean, cls, max or mean-sqrt-len) and
`sentence_bert_config.json` (`max_seq_length`, the length the model was trained at, often shorter
than its position embeddings: 256 against 512 for all-MiniLM-L6-v2). The pooling is part of the
exported graph, so verification compares the finished embedding and ONNX Runtime runs it.

```bash
downshift serve ./all-MiniLM-L6-v2
curl -s localhost:8000/predict -H 'content-type: application/json' \
  -d '{"text": ["That is a happy person", "Today is a sunny day"]}'
```

`output_0` is `[batch, dim]` (`[2, 384]` here), one L2-normalised vector per text, and matches
sentence-transformers' own output to within float32 noise. `GET /schema` reports what you got:

```json
"embedding": {"pooling": "mean", "normalized": true, "dimension": 384,
              "max_seq_length": 256, "from": "modules.json"}
```

What this does not do, so a model is never served as something it is not:

- **A recipe downshift does not apply is refused at load**, not approximated: a Dense module,
  last-token or weighted-mean pooling, or several poolings enabled at once. The error says which,
  and `--pooling none` serves the encoder's token vectors instead.
- **A repo with no `modules.json` is not pooled.** `output_0` stays `[batch, seq, hidden]` and
  `/schema` notes it. To pool one, say so: `--pooling mean|cls|max|mean_sqrt_len`, with
  `--normalize` if it wants L2 normalisation. The server does not guess a pooling for you.
- `--pooling`/`--normalize` also override a recipe the repo does declare. They mean nothing for a
  repo with a classification head.
- **Prefixes are the client's job.** Models such as E5 and BGE expect a query or passage prefix
  in the text; which one depends on what you are embedding, so the server sends what it is given.
- Decoder-only embedders (last-token pooling, left padding) and models that need
  `trust_remote_code` are not supported.

Status codes:

| Status | Body | When |
|---|---|---|
| `200` | `PredictResponse` | Inference ran. |
| `400` | `{"detail": "missing inputs: [...]"}` | A declared input name is absent from `inputs`. |
| `400` | `{"detail": "input 'name': ..."}` | Bad shape/dtype/base64 in one input (from `to_numpy`). |
| `400` | `{"detail": "this model takes tensors only: ..."}` | A `text` request to a model with no tokenizer. |
| `400` | `{"detail": "input_ids axis 1 is 65; this model accepts 1 to 64"}` | An axis outside the `bounds` `GET /schema` reports for that input. |
| `400` | `{"detail": "input_ids contains -1, outside the vocabulary [0, 30522)"}` | A model served from a Hugging Face repo directory: any `input_ids` value outside `[0, vocab_size)`. Checked before inference because ONNX Runtime wraps a negative index instead of refusing it. |
| `400` | `{"detail": "x axis 1 is 5; this model takes 16"}` | Rank or a fixed axis differs from the model's declared inputs; checked before inference on both backends. |
| `400` | `{"detail": "..."}` | ONNX Runtime itself rejected the inputs (`InvalidArgument`). Any other failure inside inference, on either backend, is a `500`. |
| `401` | `{"detail": "Authorization header is not set or incorrect"}`, header `WWW-Authenticate: Bearer` | `DOWNSHIFT_SERVER_API_KEY` is set and the request has no matching bearer token. Every route but `/health` and `/ready`. |
| `413` | `{"detail": "request body is N bytes; the server limit is M bytes (--max-body-bytes)"}` | Body over `--max-body-bytes` (default 32 MiB), rejected before it is parsed. `N` is the declared `Content-Length`, or, for a chunked body, the running total at the moment it crossed the limit. |
| `415` | `{"detail": "unsupported Content-Type; ..."}` | A body that is neither JSON nor safetensors (`application/vnd.safetensors` or `application/octet-stream`). |
| `422` | FastAPI's validation error body | An empty body (`missing`), malformed JSON (`json_invalid`, with the byte offset in `loc`), or a field that fails validation (`loc` starts with `"body"`). |
| `500` | `{"detail": "inference failed on the server; see the server log", "request_id": "..."}` | Unhandled server-side exception; the real error is only in the server log. |
| `503` | `{"detail": "model is not ready"}`, header `Retry-After: 2` | No `ServingState` yet. |
| `503` | `{"detail": "server is at capacity (N running, M queued)"}`, header `Retry-After: 1` | `--max-concurrency + --max-queue` predicts already admitted. |
| `503` | `{"detail": "request waited X.Xs in queue, past the Y.Ys --request-timeout"}`, no `Retry-After` | Admitted but waited past `--request-timeout` before inference started (checked after the body is prepared, and again when inference starts). |

Response headers on `200`: `Server-Timing: parse;dur=<ms>, prep_wait;dur=<ms>,
prep;dur=<ms>, infer_wait;dur=<ms>, infer;dur=<ms>, encode;dur=<ms>` (see "Headers"). Every response, on every route and every status
code, carries `X-Request-Id` (echoing the client's own header if sent, otherwise a
generated 16-hex-char id).

### `POST /predict/graph`

Request model `GraphPredictRequest`: one graph at the top level, or a batch under `graphs`,
never both.

```python
class GraphItem(BaseModel):
    x: list | dict
    edge_index: list | dict
    edge_attr: list | dict | None = None


class GraphPredictRequest(BaseModel):
    x: list | dict | None = None
    edge_index: list | dict | None = None
    edge_attr: list | dict | None = None
    graphs: list[GraphItem] | None = None  # a batch; not empty
    output_encoding: RequestOutputEncoding | None = None
```

`400` with `"model is not graph-shaped: inputs are [...], expected at least 'x' and
'edge_index'"` if the serving model's `input_names` don't include both `x` and `edge_index`.

**One graph** answers with `PredictResponse`, the same shape as `/predict`:

```bash
curl -s localhost:8000/predict/graph -H 'content-type: application/json' \
  -d '{"x": [[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
             [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
             [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]],
       "edge_index": [[0, 1, 2], [1, 2, 0]]}'
```

**A batch** (`graphs`) runs as one inference: the graphs are joined into one disjoint graph,
each `edge_index` in its own graph's local node ids (the server offsets them), and the
outputs are split back per graph. The response is `{"graphs": [{"outputs", "shapes",
"dtypes"}, ...]}` in request order. Splitting is exact only for outputs whose axis 0 follows
nodes or edges, which the gate works out while verifying; a model with a fixed-size output (a
pooled readout) takes one graph per request, and a batch of more than one is a `400` before
inference. `edge_attr` is given for every graph or none. The axis bounds apply to the whole
batch, so a batch can be refused although each graph would fit alone.

A safetensors body can carry a batch too: `x`, `edge_index` and `edge_attr` concatenated,
plus int64 vectors `num_nodes` and `num_edges` of length G (graph count). Each `edge_index`
column uses its own graph's local ids, as in JSON.

Status codes: the same table as `/predict`, plus the `400` "not graph-shaped" case above.

## Wire formats

Every tensor field (each value in `/predict`'s `inputs`, and `x`/`edge_index`/`edge_attr`
on `/predict/graph`) accepts either form; detection is by JSON type, so existing
nested-list clients keep working unmodified.

**Nested list** - a plain JSON array, arbitrarily nested to match the tensor's rank:

```json
{"inputs": {"x": [[1.0, 2.0], [3.0, 4.0]]}}
```

**Typed object** (`TypedArray` in `serve/schemas.py`) - `{"data", "dtype", "shape"}`, where
`data` is either a nested list (dtype/shape then optional, used only to cast/reshape) or a
base64 string (dtype/shape then mandatory, since a shape can't be inferred from raw bytes):

```json
{"inputs": {"x": {"data": "AACAPwAAAEA=", "dtype": "float32", "shape": [1, 2]}}}
```

Base64 rules, enforced by `to_numpy`/`_from_base64` in `serve/schemas.py`:

- `dtype` and `shape` are both required when `data` is a string.
- The byte order must be little-endian; a dtype string starting with `>` (e.g. `>f4`) is
  rejected.
- The decoded length must equal `prod(shape) * itemsize`; a mismatch is a `400` naming
  both the expected and actual byte counts.
- The decoded size is checked against `--max-input-bytes` (default 256 MiB; the body cap
  `--max-body-bytes` defaults to 32 MiB, so it is the tighter of the two unless raised) *before*
  decoding, from the declared shape and dtype alone.

The codec itself (`serve/codec.py`) is `pybase64` when the `[fast]` extra is installed
(about 12x faster than the standard library in both directions), otherwise Python's
`base64` module; both expose the same `b64encode`/`b64decode` signatures, standard
alphabet, with padding.

`output_encoding` (a field on both request bodies, not a URL parameter) controls only the
*response* encoding, independently of which format the request used: `"json"` writes
nested lists (the default, `DOWNSHIFT_OUTPUT_ENCODING` / `--output-encoding` sets the
server-wide default when the field is omitted), `"base64"` writes `{"data", "dtype",
"shape"}` dicts, matching `_base64_ready` in `serve/predict.py`:

```python
{
    "data": b64encode(arr).decode("ascii"),
    "dtype": arr.dtype.name,
    "shape": list(arr.shape),
}
```

`shapes` and `dtypes` in the response body are populated the same way regardless of
`output_encoding`.

**safetensors body** - instead of JSON, send the whole request as one safetensors file with
`Content-Type: application/vnd.safetensors` (`application/octet-stream` is accepted as an
alias), one tensor per input name. It is the cheapest body to parse: the arrays are
zero-copy views of the request bytes. The decoder executes nothing and is strict:

- dtypes `F64`, `F32`, `F16`, `I64`, `I32`, `I16`, `I8`, `U8`, `BOOL` (no `BF16` on the wire);
  a dtype other than the model's own input dtype is a `400`, never a silent cast;
- no repeated names; each tensor's byte range matches its shape and dtype; the tensors tile
  the data buffer exactly; each one is within `--max-input-bytes`;
- tensor names must be the route's inputs (on `/predict/graph`: `x`, `edge_index`,
  `edge_attr`, `num_nodes`, `num_edges`);
- `__metadata__` may carry `output_encoding`; `text` is JSON only.

Any violation is a `400` naming the problem. A safetensors response is chosen by
`Accept: application/vnd.safetensors` or `output_encoding: "safetensors"` (in JSON or in the
request's `__metadata__`): each output is a tensor under its own name, and `__metadata__`
carries `downshift.predictions` (a classifier's predictions, as a JSON string) and
`downshift.embedding` (the embedding recipe). A graph batch's outputs are named
`graphs.<i>.<output>`, with the graph count in `downshift.graphs`.

## Headers

| Header | Direction | Meaning |
|---|---|---|
| `X-Request-Id` | Request (optional) / Response (always) | Echoed back if the client sent it, otherwise a generated 16-hex-char id (`RequestIdMiddleware`). A `500` body's `request_id` field and every server log line written while the request was served (the `downshift.access` request line included) carry the same value. |
| `Authorization` | Request | `Bearer <key>`, required on every route but `/health` and `/ready` when `DOWNSHIFT_SERVER_API_KEY` is set. |
| `WWW-Authenticate` | Response (`401` only) | `Bearer`. |
| `Server-Timing` | Response (`/predict`, `/predict/graph`, `200` only) | `parse` (JSON or safetensors decode and validation), `prep_wait` (waiting for a prep thread), `prep` (conversion to arrays, tokenizing, input checks), `infer_wait` (waiting for an inference slot), `infer` (the backend call), `encode` (the response body), each `;dur=<ms>`. The same split is on the access log line as `timings_ms`. |
| `Retry-After` | Response (`503` only) | `2` for "not ready yet" (`/metadata` and the predict routes), `1` for "at capacity". Not set on the "queued past `--request-timeout`" `503`. |
| `Content-Length` | Request (optional) | Checked against `--max-body-bytes` before the body is read at all, when present; a chunked body without one is read and checked as it lands. |

## Error body shape

Every non-`200` response from a route handler is `{"detail": <string>}`, except:

- The `401` from the API key check, which is the same `{"detail": ...}` plus a
  `WWW-Authenticate` header.
- The unhandled-exception (`500`) handler, which adds `"request_id"`:
  `{"detail": "inference failed on the server; see the server log", "request_id": "..."}`.
- `422` validation errors, which use FastAPI/Pydantic's own body shape
  (`{"detail": [{"type": ..., "loc": [...], "msg": ..., ...}]}`).
