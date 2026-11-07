# OpenAI-compatible routes

A downshift server for a text embedding model has the embeddings routes of the OpenAI API. An OpenAI client can use them. Only `base_url` and `api_key` change.

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/v1/models` | A list with the one model that the server serves. |
| `GET` | `/v1/models/{id}` | The card of that model. Another `id` gives a `404`. |
| `POST` | `/v1/embeddings` | The vectors of one or more texts. |
| `GET` | `/version` | `{"version": "<downshift version>"}`. |

The routes use the same queue, pools, timeout, body limit, result cache and API key as `POST /predict`. The vectors are the same as the `text` output of `/predict`.

## Which models qualify

The model must be a text embedding model: a Hugging Face repo with a tokenizer and a pooling that gives one vector for each text. Pooling `none` (one vector for each token) does not qualify. For any other model, use `POST /predict`. `POST /v1/embeddings` then gives the error `not_embedding_model`.

## Quick start

1. Start the server:

   ```bash
   downshift serve ./bge-small --port 8000
   ```

2. Wait until `GET /ready` gives `200`.
3. Use the `openai` package. Set `base_url` and `api_key`. If the server has no API key, `api_key` can be any text.

   ```python
   from openai import OpenAI

   client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-used")
   response = client.embeddings.create(model="bge-small", input=["hello", "world"])
   vectors = [row.embedding for row in response.data]
   ```

   The `model` must be the name of the served model. To find the name, send `GET /v1/models`.

With curl:

```bash
curl -s localhost:8000/v1/models

curl -s localhost:8000/v1/embeddings -H 'content-type: application/json' \
  -d '{"model": "bge-small", "input": ["hello", "world"]}'
```

If the server has an API key, add `-H 'Authorization: Bearer <key>'`.

## The request

| Field | Meaning |
|---|---|
| `model` | The name of the served model. Another name gives a `404` (`model_not_found`). |
| `input` | A text, or a list of texts. An empty list is a `400`. A text with more tokens than the model reads is a `400` (`text_too_long`). Downshift does not cut a text. |
| `encoding_format` | `float` (default) is a JSON list of numbers. `base64` is the little-endian `float32` bytes of the vector, in base64. The `openai` package asks for `base64` and decodes it. |
| `dimensions` | The size of each vector. Refer to "Dimensions and --matryoshka". |
| `user` | Accepted and ignored. |

Downshift ignores all other fields.

The response is `{"object": "list", "data": [...], "model": "<name>", "usage": {...}}`. Each item of `data` is `{"object": "embedding", "index": <n>, "embedding": ...}`.

## Model name

By default, the name is the last part of the model path. For example, `./models/bge-small` has the name `bge-small`. For an import spec such as `pkg.mod:make_model`, the name is the whole spec. To give clients a short name, set `--served-model-name`. To set another name, start the server with `--served-model-name NAME` (or `DOWNSHIFT_SERVED_MODEL_NAME`). Use it if a client needs a fixed name that does not depend on the path, for example `text-embedding-3-small`.

## Dimensions and --matryoshka

A Matryoshka model has vectors that you can cut: the first `n` values of a vector are a good vector of size `n`. Only some models are trained in this way. Cutting the vector of another model gives bad results, so downshift does not allow it by default.

If the model was trained for Matryoshka cuts, start the server with `--matryoshka` (or `DOWNSHIFT_MATRYOSHKA=1`). Then `dimensions` is valid:

- The server keeps the first `dimensions` values of each vector. It then scales each vector to unit length.
- `dimensions` must be from 1 to the width of the vector. If it is not, the server gives a `400` (`invalid_input`, `param` is `dimensions`). The message gives the range.
- The result cache keeps the full vector. One cache entry serves all sizes.

If the server did not start with `--matryoshka`, a request with `dimensions` gives a `400` (`dimensions_not_supported`).

```python
response = client.embeddings.create(model="bge-small", input="hello", dimensions=256)
```

`dimensions` is only on `/v1/embeddings`. `/predict` gives the full vectors.

## Usage

`usage.prompt_tokens` and `usage.total_tokens` are the same number: the count of tokens of all texts, without the padding. A request that the result cache answers has the same count as a request that the model answers. The count includes the prompt prefix and the special tokens, because the tokenizer produced them.

## Errors

On `/v1/*`, each error body has the OpenAI shape:

```json
{"error": {"message": "model 'x' is not served. Set 'model' to 'bge-small' (GET /v1/models)",
           "type": "invalid_request_error", "param": null, "code": "model_not_found"}}
```

`type` is `invalid_request_error`, `authentication_error` (`401`) or `server_error` (status `500` or more). `param` is the field at fault, or `null`. `code` is a code from "Error codes" in [`http-api.md`](http-api.md#error-codes). The `openai` package puts the body in the `body` of the exception, and the `code` in its `code`.

The other routes keep the body `{"detail", "code"}`.

## Authentication

If the server has an API key, send it as `Authorization: Bearer <key>`. The `openai` package does this with `api_key`.

## Not supported

- Token-id input (`int[]` and `int[][]`). Send texts.
- The other routes of the OpenAI API (`/v1/chat/completions` and others). They give a `404`.
- Many models in one server. The server serves one model.
