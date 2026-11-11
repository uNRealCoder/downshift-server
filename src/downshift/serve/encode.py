"""The response of a predict: JSON, base64 or safetensors, and the Server-Timing header."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import numpy as np
import orjson
from fastapi import Response
from fastapi.responses import JSONResponse

from downshift.entities.options import OutputEncoding, RequestOutputEncoding
from downshift.entities.serving import ServingState
from downshift.entities.signatures import GraphLayout
from downshift.serve.backends import FIRST_OUTPUT_NAME
from downshift.serve.codec import b64encode, encode_safetensors
from downshift.serve.errors import ApiError
from downshift.serve.graphs import split_outputs

SAFETENSORS_MEDIA_TYPE = "application/vnd.safetensors"


# The labels that a binary request can have. octet-stream is an alias for safetensors.
BINARY_REQUEST_TYPES = frozenset({SAFETENSORS_MEDIA_TYPE, "application/octet-stream"})


# The Server-Timing stages, in the order of the request.
STAGES = ("parse", "prep_wait", "prep", "infer_wait", "infer", "encode")


# The dtypes that OPT_SERIALIZE_NUMPY of orjson writes directly from the array buffer (orjson >= 3.9).
ORJSON_DTYPES = frozenset(
    "float16 float32 float64 int8 int16 int32 int64 uint8 uint16 uint32 uint64 bool".split()
)


class NumpyJSONResponse(JSONResponse):
    """orjson with OPT_SERIALIZE_NUMPY: arrays straight from their buffers, NaN and Inf as
    null. (FastAPI's ORJSONResponse lacks the flag in old versions and is deprecated in new.)"""

    def render(self, content: Any) -> bytes:
        return orjson.dumps(content, option=orjson.OPT_SERIALIZE_NUMPY)


def _json_ready(arr: np.ndarray) -> np.ndarray | list:
    """The contiguous array itself when orjson can write it in one pass, else a list."""
    return (
        arr if arr.dtype.name in ORJSON_DTYPES and arr.ndim else arr.tolist()
    )  # 0-d: orjson rejects it


def _base64_ready(arr: np.ndarray) -> dict[str, Any]:
    """{data, dtype, shape}: base64 straight off the contiguous buffer."""
    return {
        "data": b64encode(arr).decode("ascii"),
        "dtype": arr.dtype.name,
        "shape": list(arr.shape),
    }


def encode_response(
    state: ServingState,
    outputs: dict[str, np.ndarray],
    encoding: OutputEncoding | RequestOutputEncoding | None,
    timings_ms: dict[str, float],
    accept_safetensors: bool = False,
    layout: GraphLayout | None = None,
) -> Response:
    """The response for `outputs`, with the Server-Timing header of all stages if the
    server_timing option is on.

    A safetensors body has each output under its own name, with `predictions` and the
    embedding recipe as JSON in `__metadata__`. A graph batch (`layout`) is split first: JSON
    {"graphs": [...]}, or safetensors names `graphs.<i>.<output>`."""
    encoding = encoding or state.options.output_encoding
    encode = _base64_ready if encoding == OutputEncoding.base64 else _json_ready
    safetensors = accept_safetensors or encoding == RequestOutputEncoding.safetensors
    start = time.perf_counter()
    # np.require keeps 0-d arrays 0-d (ascontiguousarray does not).
    arrays = {name: np.require(arr, requirements="C") for name, arr in outputs.items()}
    response: Response
    if layout is not None:
        try:
            per_graph = split_outputs(arrays, state.verdict.output_axes, *layout)
        except ValueError as exc:
            raise ApiError(400, "invalid_input", str(exc)) from exc
        if safetensors:
            flat = {
                f"graphs.{g}.{name}": arr
                for g, part in enumerate(per_graph)
                for name, arr in part.items()
            }
            response = Response(
                encode_safetensors(flat, {"downshift.graphs": str(len(per_graph))}),
                media_type=SAFETENSORS_MEDIA_TYPE,
            )
        else:
            response = NumpyJSONResponse(
                {"graphs": [_tensor_body(part, encode) for part in per_graph]}
            )
    elif safetensors:
        response = _safetensors_response(state, arrays)
    else:
        # PredictResponse built by hand, so orjson writes the buffers directly.
        body = _tensor_body(arrays, encode)
        predictions = _predictions(state, arrays)
        if predictions is not None:
            body["predictions"] = predictions
        response = NumpyJSONResponse(body)
    timings_ms["encode"] = round((time.perf_counter() - start) * 1000, 2)
    return add_server_timing(state, response, timings_ms)


def add_server_timing(
    state: ServingState, response: Response, timings_ms: dict[str, float]
) -> Response:
    """Set the Server-Timing header of all stages if the server_timing option is on."""
    if not state.options.server_timing:
        return response
    stages = [f"{stage};dur={timings_ms.get(stage, 0.0):.2f}" for stage in STAGES]
    if "cache" in timings_ms:  # only a server with the cache on has this stage
        hits, total = int(timings_ms["cache_hits"]), int(timings_ms["cache_rows"])
        stages.append(f'cache;dur={timings_ms["cache"]:.2f};desc="{hits}/{total}"')
    response.headers["Server-Timing"] = ", ".join(stages)
    return response


def _tensor_body(
    arrays: dict[str, np.ndarray], encode: Callable[[np.ndarray], Any]
) -> dict[str, Any]:
    return {
        "outputs": {name: encode(arr) for name, arr in arrays.items()},
        "shapes": {name: list(arr.shape) for name, arr in arrays.items()},
        "dtypes": {name: arr.dtype.name for name, arr in arrays.items()},
    }


def _predictions(state: ServingState, arrays: dict[str, np.ndarray]) -> list | None:
    if state.text is None or FIRST_OUTPUT_NAME not in arrays:
        return None
    return state.text.predictions(arrays[FIRST_OUTPUT_NAME])


def _safetensors_response(state: ServingState, arrays: dict[str, np.ndarray]) -> Response:
    metadata: dict[str, str] = {}
    predictions = _predictions(state, arrays)
    if predictions is not None:
        metadata["downshift.predictions"] = _json_str(predictions)
    if state.embedding is not None:
        first = next(iter(arrays.values()), None)
        dimension = first.shape[-1] if first is not None and first.ndim else None
        metadata["downshift.embedding"] = _json_str(state.embedding.info(dimension))
    return Response(encode_safetensors(arrays, metadata), media_type=SAFETENSORS_MEDIA_TYPE)


def _json_str(value: Any) -> str:
    return orjson.dumps(value, option=orjson.OPT_SERIALIZE_NUMPY).decode()
