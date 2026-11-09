"""The OpenAI-compatible routes: the model list, and the parse and encode steps of
POST /v1/embeddings. The embeddings request uses run_predict with these two steps, so it has the
same admission, pools, timeout, body limit, result cache and row checks as /predict."""

import time
from typing import Any

import numpy as np
from fastapi import Response

from downshift.serve.backends import FIRST_OUTPUT_NAME
from downshift.serve.codec import b64encode
from downshift.serve.engine import ServingState
from downshift.serve.errors import ApiError
from downshift.serve.predict import (
    GraphLayout,
    NumpyJSONResponse,
    _Request,
    _validate_json,
    add_server_timing,
    as_batch,
    resolve_prompt,
)
from downshift.serve.schemas import EmbeddingsRequest, OutputEncoding

# The "created" time of each model: the time that the server process started.
CREATED = int(time.time())


def model_card(state: ServingState) -> dict[str, Any]:
    return {
        "id": state.served_name,
        "object": "model",
        "created": CREATED,
        "owned_by": "downshift",
    }


def _check_dimensions(dimensions: int, width: int | None) -> None:
    if dimensions < 1 or (width is not None and dimensions > width):
        limit = "1 or more" if width is None else f"1 to {width}"
        raise ApiError(
            400,
            "invalid_input",
            f"'dimensions' is {dimensions}. Set 'dimensions' to a value from {limit}",
            param="dimensions",
        )


def require_model(state: ServingState, model_id: str) -> None:
    """A 404 if `model_id` is not the model that the server serves."""
    name = state.served_name
    if model_id != name:
        raise ApiError(
            404,
            "model_not_found",
            f"model {model_id!r:.64} is not served. Set 'model' to {name!r} (GET /v1/models)",
        )


def _matryoshka_cut(vectors: np.ndarray, dimensions: int) -> np.ndarray:
    """The first `dimensions` values of each vector, scaled to unit length. The request check
    has already tested `dimensions` against a fixed width of the model. If the width is not
    fixed, only the vector can tell."""
    if dimensions > vectors.shape[-1]:
        _check_dimensions(dimensions, vectors.shape[-1])
    cut = vectors[:, :dimensions].astype(np.float32, copy=False)  # float16 would overflow
    norms = np.linalg.norm(cut, axis=1, keepdims=True)
    scaled: np.ndarray = cut / np.where(norms == 0, 1.0, norms)  # numpy 1.x stubs give Any
    return scaled


def parse_embeddings(state: ServingState, body: bytes, timings_ms: dict[str, float]) -> _Request:
    """The body of /v1/embeddings as a text _Request, or a 422, 404 or 400."""
    start = time.perf_counter()
    req = _validate_json(EmbeddingsRequest, body)
    # pooling "none" gives the vector of each token, so it is not an embedding model.
    recipe = state.embedding
    if state.text is None or recipe is None or recipe.pooling == "none":
        raise ApiError(
            400,
            "not_embedding_model",
            "this model is not a text embedding model. Send the request to POST /predict",
        )
    require_model(state, req.model)
    if req.dimensions is not None:
        if not state.options.matryoshka:
            raise ApiError(
                400,
                "dimensions_not_supported",
                "Start the server with --matryoshka if the model was trained for Matryoshka cuts.",
            )
        _check_dimensions(req.dimensions, state.embedding_width)
    text = as_batch(req.input)
    if not text:
        raise ApiError(400, "invalid_input", "'input' is an empty list. Send at least one text")
    encoding = OutputEncoding.base64 if req.encoding_format == "base64" else OutputEncoding.json
    timings_ms["parse"] = round((time.perf_counter() - start) * 1000, 2)
    return _Request(
        text=text,
        prompt=resolve_prompt(state, None, True),
        encoding=encoding,
        dimensions=req.dimensions,
    )


def embeddings_response(
    state: ServingState,
    timings_ms: dict[str, float],
    outputs: dict[str, np.ndarray],
    request: _Request,
    tokens: list[int] | None,
    layout: GraphLayout | None,
) -> Response:
    """The /v1/embeddings body. A vector is a JSON list of floats, or the base64 of its
    little-endian float32 bytes."""
    start = time.perf_counter()
    vectors = outputs[FIRST_OUTPUT_NAME]
    if request.dimensions is not None:
        vectors = _matryoshka_cut(vectors, request.dimensions)
    vectors = np.require(vectors, dtype="<f4", requirements="C")
    base64 = request.encoding == OutputEncoding.base64
    count = sum(tokens or ())
    response = NumpyJSONResponse(
        {
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "index": i,
                    "embedding": b64encode(vector).decode("ascii") if base64 else vector,
                }
                for i, vector in enumerate(vectors)
            ],
            "model": state.served_name,
            "usage": {"prompt_tokens": count, "total_tokens": count},
        }
    )
    timings_ms["encode"] = round((time.perf_counter() - start) * 1000, 2)
    return add_server_timing(state, response, timings_ms)
