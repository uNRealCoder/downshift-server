# HTTP API reference

This page describes every route that `build_app` registers: the request and response schemas, the status codes, the headers, the authentication, and the wire formats for tensor data. The routes and schemas are the same for both backends (`onnxruntime` and `torch`).

## Routes

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness. |
| `GET` | `/ready` | Readiness. |
| `GET` | `/metadata` | Model, backend, verdict, limits, boot timings and warmup statistics. |
| `GET` | `/schema` | What to POST: input names, dtypes, shapes and an example body. |
| `POST` | `/predict` | Named tensor inputs or `text`, for any model. JSON or safetensors body. |
| `POST` | `/predict/graph` | One graph (`x`, `edge_index` and optional `edge_attr`) or a batch of graphs. JSON or safetensors body. |

Each served model comes from this machine. It is one of these:

- A downloaded `.onnx` file.
- A downloaded PyTorch checkpoint.
- A downloaded Hugging Face repo directory. Downshift identifies it by the `config.json` in it.
- A module that the server process can import.

The `source` block of `/schema` shows which one applies to each server. Downshift fetches nothing to answer a request.

## Authentication

Authentication is optional. It uses one shared key. The key is active when `DOWNSHIFT_SERVER_API_KEY` is set. It is also active when you give a value to `build_app(..., api_key=...)` or `app_for(..., api_key=...)`. Both default to the variable. When the key is active, every route except `/health` and `/ready` needs `Authorization: Bearer <key>`.

Downshift matches the scheme without regard to case. It compares the key in constant time (`hmac.compare_digest`). A missing or wrong credential gives a `401` with the header `WWW-Authenticate: Bearer` and this body:

```json
{"detail": "Authorization header is not set or incorrect"}
```

The check runs before the route. It therefore applies to unknown paths, `/docs` and `/openapi.json`.

If the key is unset or an empty string, downshift checks nothing. It logs one `WARNING` at startup that the endpoints are unauthenticated. To fix this, set the key or add your own authentication middleware.

The rest of this page does not repeat the `401` for each route.

```bash
curl -s localhost:8000/schema -H "Authorization: Bearer $DOWNSHIFT_SERVER_API_KEY"
```

### `GET /health`

This route returns `200` when the process is up and the event loop runs. It also returns `200` while a model is still loading. The route runs directly on the event loop (`async def`). A predict never blocks it. The response model is `HealthResponse`:

```json
{"status": "ok"}
```

```bash
curl -s localhost:8000/health
```

### `GET /ready`

This route returns `200` with `{"ready": true}` after the model is loaded, exported, verified and warmed up. Before that, it returns `503`. There are two different `503` shapes. Neither has a `Retry-After` header:

- No `ServingState` exists yet. The background loader started because `serve` got no `state=` argument, and it has not made one. The body is `{"ready": false, "phase": "export"}`. `phase` is the step that the loader is in now. It is one of `load`, `export`, `verify`, `session` and `warmup`. These are the values of the `Phase` enum in `core/phase.py`. They are also the keys of `boot` in `/metadata` and the names in the `Boot` row of the banner. `phase` is `null` for an app that you built with `state=`, because such an app has no loading window. The `200` body has no `phase`.
- A `ServingState` exists, but its `ready` flag is `False`. The body is `{"ready": false}`. In practice, `prepare_serving()` sets `ready` to `True` as its last step. This happens inside `warmup()`, also with `--warmup 0`. This second shape cannot occur in the current boot paths for one worker or for `--workers N`. A `ServingState` object always appears after the warmup.

The response model is `ReadyResponse` (`{"ready": bool, "phase": str | None}`). The route writes its bodies by hand. A ready server answers `{"ready": true}` with no `phase` key.

```bash
curl -s -i localhost:8000/ready
```

### `GET /metadata`

While no `ServingState` exists, this route returns `503` with `{"detail": "model is not ready"}` and the header `Retry-After: 2`. After that, it returns `200` with a `MetadataResponse`:

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
| `model` | The source that `serve` or `app_for` received. It is always a form that is already on this machine: a `.onnx` file, a checkpoint, a downloaded Hugging Face repo directory, or an import spec. A file or directory has only its last path component (`/srv/models/bert` gives `bert`). The server never sends its directory layout. An import spec is reported as given. |
| `family` | `verdict.model_family`. |
| `verdict` | `ExportVerdict.to_dict()` without `onnx_path`, which is a location on the disk of the server. Refer to [`python-api.md`](python-api.md#exportverdict). Paths in `reason`, `warnings` and `notes` are also cut to their file name. |
| `backend` | `BackendMeta.to_dict()`: `{"name", "device", "inputs": [...], "outputs": [...]}`. Each input and output is an `IOSpec` (`name`, `dtype`, `shape`). |
| `input_names` | The flat input names, in the order of the forward arguments. |
| `notes` | Notes for the banner, for example warnings about `--force-onnx`. |
| `version` | `downshift.__version__`. |
| `limits` | `max_body_bytes`, `max_input_bytes`, `max_concurrency`, `max_queue` and `request_timeout`, read from `ServeOptions`. The `limits` block of `/schema` is the same. |
| `execution` | The value of `--execution`: `"threadpool"` or `"inline"`. |
| `boot` | The wall-clock seconds of each phase (`load`, `export`, `verify`, `session` and `warmup`, for the phases that ran). The `Boot` row of the CLI banner prints the same dict. |
| `warmup` | `{"count", "mean_ms", "synthesized"}` from `WarmupStats`. It is `null` if the warmup did not run. |

```bash
curl -s localhost:8000/metadata | python -m json.tool
```

### `GET /schema`

This route answers the question "What do I POST?". `/metadata` is the view of the operator. It shows how downshift judged the model and how long each boot phase took. `/schema` is the view of the caller. It shows the name, dtype and shape of each input. It also gives an example body that you can post back to `/predict` unchanged.

While no `ServingState` exists, the route returns `503` with `{"detail": "model is not ready"}` and `Retry-After: 2`. This is the same as `/metadata`. After that, it returns `200` with a `SchemaResponse`. The server builds the answer for each request from the IO that the backend declares (`downshift/serve/describe.py`). The answer always describes the graph that is running. It never describes what the source model promised.

| Field | Meaning |
|---|---|
| `model` | The source that the server received. It has only the file name or directory name. An import spec is reported as given. This is the same rule as `model` in `/metadata`. |
| `source` | `{"spec", "kind", "description", "fetched_at_runtime": false}`. `kind` is one of `onnx-file`, `torch-checkpoint`, `hf-repo-dir`, `import-spec` and `unknown`. `description` says the same in a sentence. |
| `family` | `verdict.model_family`. |
| `backend` | `"onnxruntime"` or `"torch"`. |
| `device` | The ONNX Runtime execution provider, or the torch device. |
| `endpoint` | `"/predict"`. This is the route that receives these inputs. |
| `graph_endpoint` | `"/predict/graph"` if the inputs of the model include `x` and `edge_index`. Otherwise `null`. |
| `inputs` | One `TensorSchema` for each input, in the order of the forward arguments. All are required. For a model that downshift exported, each one has the axis names of the adapter and `bounds` for each axis (see below). |
| `outputs` | One `TensorSchema` for each output. It is empty for a torch backend that serves without example inputs. Nothing ran through the model yet, so nothing is known. A `notes` entry says this. |
| `example_request` | A complete and correct `/predict` body. It is `null` if the body would have more than 256 elements across all inputs. A `notes` entry then gives the shape to build. The values are filler: zeros, or ones for an attention mask. A zero attention mask is a correct request that returns NaN. Copy only the names, dtypes and shapes. |
| `example_curl` | `example_request` as a `curl` command for the URL of this server. It is `null` if the example is `null`. |
| `input_formats` | The three wire forms that a tensor value can have: nested list, typed object and base64. The rules are in "Wire formats" below. |
| `output_encodings` | `["json", "base64"]`. |
| `default_output_encoding` | The encoding that this server uses when a request has no `output_encoding`. |
| `limits` | The same five limits as `limits` in `/metadata`. |
| `notes` | Information that could surprise a caller. Refer to the fields above. |

Each entry in `inputs` and `outputs` is a `TensorSchema`:

```python
class TensorSchema(BaseModel):
    name: str
    dtype: str | None = None  # numpy name: "float32", "int64" - not ORT's "tensor(float)"
    shape: list[int | str] | None = None  # int: fixed. str: a dynamic axis, named or "dynamic"
    required: bool | None = None  # True on inputs, null on outputs
    example_shape: list[int] | None = None  # `shape` with each dynamic axis pinned to 1
    bounds: list[AxisBound | None] | None = None  # parallel to `shape`; inputs only
```

A dynamic axis that an adapter named keeps its name (`"batch"`, `"seq"`). `torch.export` makes names for other axes (`"s77"`, `"u3"`). These names mean nothing outside the tracer, so downshift reports them as `"dynamic"`.

`bounds` has the same length as `shape`. For each dynamic axis of a model that downshift exported, it gives `{"min": int, "max": int}`. This is the range that the export was traced for (the limits of the `torch.export.Dim`). `null` is the entry for all other axes.

If a request has an axis outside this range, the server refuses it with a `400`. The message names the input, the axis, the size and the range (`input_ids axis 1 is 65; this model accepts 1 to 64`). It does not use the wording of ONNX Runtime or torch.

`bounds` is `null` for a bare `.onnx` file that is served with no `--reference`. The axes show `"dynamic"`. There is no adapter to name or bound them. With `--workers N`, the parent sends its names and bounds to the workers. They answer in the same way.

The dtypes are the wire dtypes. A bfloat16 model shows as `float32`. Numpy has no bfloat16, and the torch backend casts inside itself. It takes and returns float32. For the same reason, the torch backend reports float16 as `float32`.

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

To check that a server is usable, send the example back to the server. This is the quickest way:

```bash
curl -s localhost:8000/predict -H 'content-type: application/json' \
  -d "$(curl -s localhost:8000/schema | python -c 'import json,sys; print(json.dumps(json.load(sys.stdin)["example_request"]))')"
```

### `POST /predict`

The request model is `PredictRequest`:

```python
class PredictRequest(BaseModel):
    inputs: dict[str, Any] = {}  # name -> nested list, or a {"data", "dtype", "shape"} object
    text: str | list[str] | None = None  # instead of `inputs`; see "Text input" below
    prompt_name: str | None = None  # a named prompt from the repo, `text` requests only
    output_encoding: RequestOutputEncoding | None = (
        None  # "json" | "base64" | "safetensors"; omit to use the server default
    )
```

The body can also be a safetensors file (`Content-Type: application/vnd.safetensors`) with one tensor for each input name. Refer to "Wire formats".

The response model is `PredictResponse`:

```python
class PredictResponse(BaseModel):
    outputs: dict[str, Any]
    shapes: dict[str, list[int]]
    dtypes: dict[str, str]
    predictions: list[dict] | None = None  # sequence classifiers only
```

A nested-list body:

```bash
curl -s localhost:8000/predict -H 'content-type: application/json' \
  -d '{"inputs": {"x": [[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]]}}'
```

```json
{"outputs": {"output_0": [[0.199, -0.206, 0.561, 0.405]]},
 "shapes": {"output_0": [1, 4]},
 "dtypes": {"output_0": "float32"}}
```

A base64 body, with a request for a base64 response. The encoding rules are in "Wire formats" below.

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

If you give no `dtype`, and the backend declares none, integer lists become `int64`. All other lists become `float32` (`to_numpy` in `serve/schemas.py`).

Send large tensors as safetensors or base64. Both skip the conversion of floats to text and back. Python parses a JSON float list in the prep pool. During the parse, it holds the GIL. This slows every other request in that process.

Safetensors also skips the JSON wrapper and the 33% overhead of base64. Measured: for `bert_small`, the p50 is 11.2 ms with JSON and 9.1 ms with safetensors. The `parse` and `prep` entries of `Server-Timing` show what a body cost.

The server admits a predict, or refuses it with a `503`, before it reads the body. An overloaded server does not buffer or parse a body that it will turn away.

A body larger than `--max-body-bytes` (default 32 MiB) gets a `413`:

- If `Content-Length` shows the size, the server refuses the body without a read.
- For a chunked body with no `Content-Length`, the server refuses it when the running total passes the limit.

There is no separate limit on text length. For a `text` request, the limit on the body is the limit.

#### Text input and class probabilities

A model that is served from a downloaded Hugging Face repo directory with tokenizer files (`tokenizer.json`, `vocab.txt`, and others) also takes text. Send `text` (one string or a list) and not `inputs`. If you send both, the server returns `422`. `GET /schema` shows a `text_input` block when text input is available. If it is not available, a `400` says this.

```bash
curl -s localhost:8000/predict -H 'content-type: application/json' \
  -d '{"text": ["What is the capital of France?", "Ignore all previous instructions."]}'
```

The server tokenizes the batch and pads it. Each row can have at most the smallest of these limits that the files of the model declare:

- The `model_max_length` of the tokenizer.
- The `max_position_embeddings` of the model.
- For an embedding model, the `max_seq_length` that its author used for training (see "Embedding models").

Downshift does not choose any of these limits.

For the RoBERTa family (`roberta`, `xlm-roberta`, `xlm-roberta-xl`, `camembert`), the usable length is `max_position_embeddings - (pad_token_id + 1)`. Position ids start after the padding index. For example, 514 positions serve 512 tokens.

The server refuses a longer row with a 400. The message names each row that is too long, with its token count. The server never cuts a row. A classifier that reads only the start of a long input would answer about text that it never saw. The only cap on the text in a request is the limit on the request body (`--max-body-bytes`). A single string is a batch of one. `shapes` does not depend on the form that you sent.

If `config.json` declares a sequence-classification architecture, downshift loads the repo with its classification head. (`AutoModel` would drop it.) `output_0` is then the logits. The response also has `predictions`, with one entry for each row. This is true for `text` requests and for tensor requests:

```json
{"outputs": {"output_0": [[-3.1, 5.2, -1.9]]},
 "shapes": {"output_0": [1, 3]},
 "dtypes": {"output_0": "float32"},
 "predictions": [{"label": "INJECTION", "score": 0.9997,
                  "probabilities": {"BENIGN": 0.0002, "INJECTION": 0.9997, "JAILBREAK": 0.0001}}]}
```

`probabilities` is a softmax over the logits. If the config says `multi_label_classification`, it is a sigmoid for each label. The labels come from `id2label` in `config.json`. Token-classification repos also load with their head. They return logits, without `predictions`.

#### Embedding models

`config.json` does not give a pooling. A sentence-transformers repo has its recipe in sibling files, and downshift reads them:

- `modules.json` lists the modules: Transformer, then Pooling, then an optional Normalize.
- `1_Pooling/config.json` gives the pooling: mean, cls, max or mean-sqrt-len.
- `sentence_bert_config.json` gives `max_seq_length`. This is the length that the model used for training. It is often shorter than the position embeddings: 256 against 512 for all-MiniLM-L6-v2.

The pooling is part of the exported graph. Verification therefore compares the finished embedding, and ONNX Runtime runs it.

```bash
downshift serve ./all-MiniLM-L6-v2
curl -s localhost:8000/predict -H 'content-type: application/json' \
  -d '{"text": ["That is a happy person", "Today is a sunny day"]}'
```

`output_0` is `[batch, dim]` (`[2, 384]` here). It has one L2-normalised vector for each text. It matches the output of sentence-transformers to within float32 noise. `GET /schema` reports what you got:

```json
"embedding": {"pooling": "mean", "normalized": true, "dimension": 384,
              "max_seq_length": 256, "from": "modules.json"}
```

Downshift never serves a model as something that it is not. These rules apply:

- **Downshift refuses a recipe that it does not apply at load.** It does not approximate. Examples: a Dense module, and several poolings that are active together. The error says which one. `--pooling none` serves the token vectors of the encoder instead.
- **Downshift does not pool a repo with no `modules.json`.** `output_0` stays `[batch, seq, hidden]`, and `/schema` has a note about this. To pool such a repo, use `--pooling mean|cls|max|mean_sqrt_len|lasttoken|weightedmean`. Add `--normalize` for L2 normalisation. The server does not guess a pooling.
- `--pooling` and `--normalize` also override a recipe that the repo declares. They have no meaning for a repo with a classification head.
- **The client adds prefixes.** Models such as E5 and BGE expect a query or passage prefix in the text. The prefix depends on what you embed, so the server sends the text that it receives.
- Downshift supports decoder-only embedders only if the repo has an embedding recipe (for example Qwen3-Embedding, with last-token pooling and left padding). It does not support models that need `trust_remote_code`.

Status codes:

| Status | Body | When |
|---|---|---|
| `200` | `PredictResponse` | The inference ran. |
| `400` | `{"detail": "missing inputs: [...]"}` | A declared input name is not in `inputs`. |
| `400` | `{"detail": "input 'name': ..."}` | One input has a bad shape, dtype or base64 (from `to_numpy`). |
| `400` | `{"detail": "this model takes tensors only. ..."}` | A `text` request to a model with no tokenizer. |
| `400` | `{"detail": "input_ids axis 1 is 65; this model accepts 1 to 64"}` | An axis is outside the `bounds` that `GET /schema` reports for that input. |
| `400` | `{"detail": "input_ids contains -1, outside the vocabulary [0, 30522)"}` | A model served from a Hugging Face repo directory has an `input_ids` value outside `[0, vocab_size)`. Downshift checks this before the inference, because ONNX Runtime wraps a negative index and does not refuse it. |
| `400` | `{"detail": "x axis 1 is 5; this model takes 16"}` | The rank or a fixed axis is different from the declared inputs of the model. Downshift checks this before the inference on both backends. |
| `400` | `{"detail": "..."}` | ONNX Runtime rejected the inputs (`InvalidArgument`). Any other failure inside the inference, on either backend, is a `500`. |
| `401` | `{"detail": "Authorization header is not set or incorrect"}`, header `WWW-Authenticate: Bearer` | `DOWNSHIFT_SERVER_API_KEY` is set, and the request has no matching bearer token. This applies to every route except `/health` and `/ready`. |
| `413` | `{"detail": "request body is N bytes; the server limit is M bytes (--max-body-bytes)"}` | The body is larger than `--max-body-bytes` (default 32 MiB). The server rejects it before it parses it. `N` is the declared `Content-Length`. For a chunked body, `N` is the running total when it passed the limit. |
| `415` | `{"detail": "unsupported Content-Type; ..."}` | The body is not JSON and not safetensors (`application/vnd.safetensors` or `application/octet-stream`). |
| `422` | The validation error body of FastAPI | The body is empty (`missing`), the JSON is malformed (`json_invalid`, with the byte offset in `loc`), or a field fails validation (`loc` starts with `"body"`). |
| `500` | `{"detail": "inference failed on the server; see the server log", "request_id": "..."}` | An unhandled exception on the server. The real error is only in the server log. |
| `503` | `{"detail": "model is not ready"}`, header `Retry-After: 2` | No `ServingState` exists yet. |
| `503` | `{"detail": "server is at capacity (N running, M queued)"}`, header `Retry-After: 1` | `--max-concurrency + --max-queue` predicts are already admitted. |
| `503` | `{"detail": "request waited X.Xs in queue, past the Y.Ys --request-timeout"}`, no `Retry-After` | The predict was admitted, but it waited longer than `--request-timeout` before the inference started. Downshift checks this after the body is prepared, and again when the inference starts. |

A `200` response has this header: `Server-Timing: parse;dur=<ms>, prep_wait;dur=<ms>, prep;dur=<ms>, infer_wait;dur=<ms>, infer;dur=<ms>, encode;dur=<ms>` (see "Headers"). Every response, on every route and with every status code, has `X-Request-Id`. If the client sent the header, downshift returns it. Otherwise, downshift makes an ID of 16 hexadecimal characters.

### `POST /predict/graph`

The request model is `GraphPredictRequest`. It has one graph at the top level, or a batch under `graphs`. It never has both.

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

If the `input_names` of the serving model do not include `x` and `edge_index`, the route returns `400` with `"model is not graph-shaped: inputs are [...], expected at least 'x' and 'edge_index'"`.

**One graph** gets a `PredictResponse`, with the same shape as `/predict`:

```bash
curl -s localhost:8000/predict/graph -H 'content-type: application/json' \
  -d '{"x": [[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
             [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
             [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]],
       "edge_index": [[0, 1, 2], [1, 2, 0]]}'
```

**A batch** (`graphs`) runs as one inference. Downshift joins the graphs into one disjoint graph. Each `edge_index` uses the local node ids of its own graph, and the server adds the offsets. The outputs are split again for each graph. The response is `{"graphs": [{"outputs", "shapes", "dtypes"}, ...]}`, in the order of the request.

The split is exact only for outputs whose axis 0 follows the nodes or the edges. The gate finds this during verification. A model with a fixed-size output (a pooled readout) takes one graph for each request. A batch of more than one graph gets a `400` before the inference.

Give `edge_attr` for every graph or for none. The axis bounds apply to the whole batch. A batch can therefore be refused although each graph fits alone.

A safetensors body can also carry a batch. It has `x`, `edge_index` and `edge_attr` in a concatenated form. It also has the int64 vectors `num_nodes` and `num_edges` of length G (the number of graphs). As in JSON, each `edge_index` column uses the local ids of its own graph.

The status codes are the same as for `/predict`. There is also the `400` case "not graph-shaped" above.

## Wire formats

Every tensor field accepts either form. This applies to each value in `inputs` of `/predict`, and to `x`, `edge_index` and `edge_attr` on `/predict/graph`. Downshift detects the form from the JSON type. Existing nested-list clients continue to work without a change.

**Nested list.** A plain JSON array. Its nesting matches the rank of the tensor:

```json
{"inputs": {"x": [[1.0, 2.0], [3.0, 4.0]]}}
```

**Typed object** (`TypedArray` in `serve/schemas.py`). It has `{"data", "dtype", "shape"}`. `data` is a nested list or a base64 string:

- For a nested list, `dtype` and `shape` are optional. Downshift uses them only to cast or reshape.
- For a base64 string, `dtype` and `shape` are mandatory. The server cannot find a shape from raw bytes.

```json
{"inputs": {"x": {"data": "AACAPwAAAEA=", "dtype": "float32", "shape": [1, 2]}}}
```

`to_numpy` and `_from_base64` in `serve/schemas.py` enforce these base64 rules:

- `dtype` and `shape` are both mandatory when `data` is a string.
- The byte order must be little-endian. The server rejects a dtype string that starts with `>` (for example `>f4`).
- The decoded length must equal `prod(shape) * itemsize`. A mismatch gives a `400` with the expected and the actual number of bytes.
- The server checks the decoded size against `--max-input-bytes` (default 256 MiB) before it decodes. It uses only the declared shape and dtype for this check. The limit on the body, `--max-body-bytes`, is 32 MiB by default. It is the smaller limit, unless you increase it.

The codec (`serve/codec.py`) is `pybase64` if the `[fast]` extra is installed. It is about 12x faster than the standard library in both directions. Otherwise, it is the `base64` module of Python. Both have the same `b64encode` and `b64decode` signatures, with the standard alphabet and padding.

`output_encoding` is a field in both request bodies. It is not a URL parameter. It controls only the encoding of the response. It does not depend on the format of the request.

- `"json"` writes nested lists. This is the default. `DOWNSHIFT_OUTPUT_ENCODING` or `--output-encoding` sets the default for the whole server, when the field is not in the request.
- `"base64"` writes `{"data", "dtype", "shape"}` dicts. This agrees with `_base64_ready` in `serve/predict.py`:

```python
{
    "data": b64encode(arr).decode("ascii"),
    "dtype": arr.dtype.name,
    "shape": list(arr.shape),
}
```

`shapes` and `dtypes` in the response body are filled in the same way for each `output_encoding`.

**Safetensors body.** Instead of JSON, send the whole request as one safetensors file. Use `Content-Type: application/vnd.safetensors` (`application/octet-stream` is an alias). Send one tensor for each input name. This is the cheapest body to parse. The arrays are zero-copy views of the request bytes. The decoder executes nothing, and it is strict:

- The accepted dtypes are `F64`, `F32`, `F16`, `I64`, `I32`, `I16`, `I8`, `U8` and `BOOL`. There is no `BF16` on the wire. A dtype that is not the input dtype of the model gives a `400`. Downshift never casts silently.
- Names must not repeat. The byte range of each tensor must match its shape and dtype. The tensors must fill the data buffer exactly. Each tensor must be within `--max-input-bytes`.
- The tensor names must be the inputs of the route. On `/predict/graph`, they are `x`, `edge_index`, `edge_attr`, `num_nodes` and `num_edges`.
- `__metadata__` can have `output_encoding`. `text` is JSON only.

Any violation gives a `400` that names the problem.

To get a safetensors response, send `Accept: application/vnd.safetensors`, or set `output_encoding: "safetensors"` in the JSON or in the `__metadata__` of the request. Each output is a tensor under its own name. `__metadata__` has `downshift.predictions` (the predictions of a classifier, as a JSON string) and `downshift.embedding` (the embedding recipe). The outputs of a graph batch have the names `graphs.<i>.<output>`. The number of graphs is in `downshift.graphs`.

## Headers

| Header | Direction | Meaning |
|---|---|---|
| `X-Request-Id` | Request (optional), response (always) | Downshift returns it if the client sent it. Otherwise it makes an ID of 16 hexadecimal characters (`RequestIdMiddleware`). The `request_id` field of a `500` body has the same value. Each server log line that is written while the request is served (including the request line of `downshift.access`) has it too. |
| `Authorization` | Request | `Bearer <key>`. It is required on every route except `/health` and `/ready` when `DOWNSHIFT_SERVER_API_KEY` is set. |
| `WWW-Authenticate` | Response (`401` only) | `Bearer`. |
| `Server-Timing` | Response (`/predict` and `/predict/graph`, `200` only) | The stages are listed below. Each one has `;dur=<ms>`. The access log line has the same split as `timings_ms`. |
| `Retry-After` | Response (`503` only) | `2` for "not ready yet" (`/metadata` and the predict routes). `1` for "at capacity". The `503` for "queued past `--request-timeout`" does not have it. |
| `Content-Length` | Request (optional) | If it is present, downshift checks it against `--max-body-bytes` before it reads the body. Downshift reads a chunked body without it and checks it while it arrives. |

The `Server-Timing` stages:

- `parse`: the decode and validation of the JSON or safetensors body.
- `prep_wait`: the wait for a prep thread.
- `prep`: the conversion to arrays, the tokenizing and the input checks.
- `infer_wait`: the wait for an inference slot.
- `infer`: the call to the backend.
- `encode`: the response body.

## Error body shape

Each response that is not `200` from a route handler is `{"detail": <string>}`. There are three exceptions:

- The `401` from the API key check. It is the same `{"detail": ...}` plus a `WWW-Authenticate` header.
- The handler for an unhandled exception (`500`). It adds `"request_id"`: `{"detail": "inference failed on the server; see the server log", "request_id": "..."}`.
- `422` validation errors. They use the body shape of FastAPI and Pydantic: `{"detail": [{"type": ..., "loc": [...], "msg": ..., ...}]}`.
