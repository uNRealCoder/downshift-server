"""The /predict and /predict/graph internals: request conversion, admission bookkeeping and
the inference call itself. Kept apart from serve/app.py's routing so a body-handling change
(P1 to P3) touches one file (M8).
"""

import asyncio
import contextvars
import logging
import time
from functools import partial
from typing import Any

import numpy as np
import orjson
from fastapi import HTTPException
from fastapi.responses import JSONResponse

from downshift.serve.backends import FIRST_OUTPUT_NAME, InferenceInputError
from downshift.serve.codec import b64encode
from downshift.serve.engine import DimBound, ServingState
from downshift.serve.schemas import OutputEncoding, to_numpy

logger = logging.getLogger("downshift.serve")

# The predict routes: the only ones OrjsonRoute admits against before reading the body (P1).
PREDICT_PATHS = frozenset({"/predict", "/predict/graph"})

# dtypes orjson's OPT_SERIALIZE_NUMPY writes straight from the array buffer (orjson >= 3.9).
_ORJSON_DTYPES = frozenset(
    "float16 float32 float64 int8 int16 int32 int64 uint8 uint16 uint32 uint64 bool".split()
)


class NumpyJSONResponse(JSONResponse):
    """orjson with OPT_SERIALIZE_NUMPY: arrays are written from their buffers; NaN/Inf become null.

    Not built on fastapi's ORJSONResponse: older versions lack the numpy flag, newer ones
    deprecate the class and warn at import.
    """

    def render(self, content: Any) -> bytes:
        return orjson.dumps(content, option=orjson.OPT_SERIALIZE_NUMPY)


def _json_ready(arr: np.ndarray) -> np.ndarray | list:
    """The contiguous array itself when orjson can write it in one pass, else a list."""
    return (
        arr if arr.dtype.name in _ORJSON_DTYPES and arr.ndim else arr.tolist()
    )  # 0-d: orjson rejects


def _base64_ready(arr: np.ndarray) -> dict[str, Any]:
    """{data, dtype, shape}: base64 straight off the contiguous buffer."""
    return {
        "data": b64encode(arr).decode("ascii"),
        "dtype": arr.dtype.name,
        "shape": list(arr.shape),
    }


def as_batch(value: str | list[str]) -> list[str]:
    """A single string is a batch of one, so the response shape never depends on which
    form the client used."""
    return [value] if isinstance(value, str) else list(value)


def _text_feeds(
    state: ServingState, text: list[str], declared: dict[str, str | None]
) -> dict[str, np.ndarray]:
    """Tokenize a `text` request into the graph's own inputs. ValueError means the client's
    text was unusable."""
    assert state.text is not None  # run_predict refuses a text request without one
    encoded = state.text.encode(text)
    missing = [n for n in state.input_names if n not in encoded]
    if missing:
        raise ValueError(f"the tokenizer does not produce the model's inputs {missing}")
    return {n: to_numpy(n, encoded[n], declared.get(n)) for n in state.input_names}


def _vocab_violation(state: ServingState, feeds: dict[str, np.ndarray]) -> str | None:
    """B3: ORT wraps a negative input_ids index instead of refusing it. One vectorised
    min/max comparison against the hf model's own vocab_size, so garbage in doesn't become a
    confident 200 out."""
    if state.vocab_size is None:
        return None
    ids = feeds.get("input_ids")
    if ids is None or ids.size == 0:
        return None
    lo, hi = int(ids.min()), int(ids.max())
    if lo >= 0 and hi < state.vocab_size:
        return None
    bad = lo if lo < 0 else hi
    return f"input_ids contains {bad}, outside the vocabulary [0, {state.vocab_size})"


def _edge_index_violation(feeds: dict[str, np.ndarray]) -> str | None:
    """The /predict/graph analogue of _vocab_violation: edge_index must index into x's node
    dimension, or a backend that trusts it (gather/scatter with no bounds check of its own)
    reads or writes out of bounds instead of refusing the request."""
    edge_index = feeds.get("edge_index")
    x = feeds.get("x")
    if edge_index is None or x is None or edge_index.size == 0:
        return None
    num_nodes = x.shape[0]
    lo, hi = int(edge_index.min()), int(edge_index.max())
    if lo >= 0 and hi < num_nodes:
        return None
    bad = lo if lo < 0 else hi
    return f"edge_index contains {bad}, outside the node range [0, {num_nodes})"


def _bound_message(name: str, axis: int, size: int, bound: DimBound) -> str:
    return f"{name} axis {axis} is {size}; this model accepts {bound.min} to {bound.max}"


def _bound_violation(state: ServingState, feeds: dict[str, np.ndarray]) -> str | None:
    """U2: the first input whose shape falls outside an axis downshift's own export traced,
    or None when every known bound is satisfied (including when none are known at all)."""
    for name, arr in feeds.items():
        bounds = state.axis_bounds.get(name)
        if not bounds:
            continue
        for axis, bound in bounds.items():
            if axis >= arr.ndim:
                continue
            size = arr.shape[axis]
            if not (bound.min <= size <= bound.max):
                return _bound_message(name, axis, size, bound)
    return None


def _predict_body(
    state: ServingState,
    inputs: dict[str, Any],
    encoding: OutputEncoding | None,
    admitted_at: float,
    text: list[str] | None,
    parse_ms: float,
    timings_ms: dict[str, float],
) -> NumpyJSONResponse:
    """Runs in state.executor, never on the event loop: conversion of a large JSON body,
    the inference itself, and response encoding. Admission and the cheap missing-input check
    already happened on the loop by the time this is submitted.

    On success the parse/codec/infer split (ms) is also written into `timings_ms`, for the
    request log line (U1).

    Raises HTTPException(400) for anything the client got wrong (bad shape/dtype/JSON).
    Anything else (a backend bug, OOM, ...) propagates so the app-level handler turns it into
    a 500 without leaking the exception text to the client.
    """
    request_timeout = state.options.request_timeout
    if request_timeout > 0:
        waited = time.monotonic() - admitted_at
        if waited > request_timeout:
            # Never raised once infer() has begun: only a request that sat in the queue.
            raise HTTPException(
                503,
                f"request waited {waited:.1f}s in queue, past the {request_timeout:.1f}s "
                "--request-timeout",
            )

    declared = state.declared_dtypes
    max_bytes = state.options.max_input_bytes
    codec_start = time.perf_counter()
    try:
        if text is not None:
            feeds = _text_feeds(state, text, declared)
        else:
            feeds = {
                n: to_numpy(n, inputs[n], declared.get(n), max_bytes=max_bytes)
                for n in state.input_names
            }
    except (ValueError, TypeError, KeyError) as exc:
        raise HTTPException(400, str(exc)) from exc
    codec_ms = (time.perf_counter() - codec_start) * 1000

    vocab_violation = _vocab_violation(state, feeds)
    if vocab_violation is not None:
        raise HTTPException(400, vocab_violation)
    edge_index_violation = _edge_index_violation(feeds)
    if edge_index_violation is not None:
        raise HTTPException(400, edge_index_violation)
    bound_violation = _bound_violation(state, feeds)
    if bound_violation is not None:
        raise HTTPException(400, bound_violation)

    infer_start = time.perf_counter()
    try:
        outputs = state.backend.infer(feeds)
    except HTTPException:
        raise
    except InferenceInputError as exc:
        # The bound pre-check above already answers the common case; this is the backend
        # rejecting something we had no bound for.
        raise HTTPException(400, str(exc)) from exc
    infer_ms = (time.perf_counter() - infer_start) * 1000

    encoding = encoding or state.options.output_encoding
    encode = _base64_ready if encoding == OutputEncoding.base64 else _json_ready
    encode_start = time.perf_counter()
    # C-contiguous once, up front (np.require keeps 0-d arrays 0-d; ascontiguousarray does not).
    arrays = {name: np.require(arr, requirements="C") for name, arr in outputs.items()}
    # Same keys as PredictResponse; built by hand so orjson serializes the buffers directly.
    body = {
        "outputs": {name: encode(arr) for name, arr in arrays.items()},
        "shapes": {name: list(arr.shape) for name, arr in arrays.items()},
        "dtypes": {name: arr.dtype.name for name, arr in arrays.items()},
    }
    if state.text is not None and FIRST_OUTPUT_NAME in arrays:
        predictions = state.text.predictions(arrays[FIRST_OUTPUT_NAME])
        if predictions is not None:
            body["predictions"] = predictions
    codec_ms += (time.perf_counter() - encode_start) * 1000

    timings_ms.update(parse=round(parse_ms, 2), codec=round(codec_ms, 2), infer=round(infer_ms, 2))
    response = NumpyJSONResponse(body)
    response.headers["Server-Timing"] = (
        f"parse;dur={parse_ms:.2f}, codec;dur={codec_ms:.2f}, infer;dur={infer_ms:.2f}"
    )
    return response


async def run_predict(
    state: ServingState,
    inputs: dict[str, Any],
    encoding: OutputEncoding | None,
    text: list[str] | None = None,
    *,
    admitted_at: float,
    parse_ms: float,
    timings_ms: dict[str, float],
) -> NumpyJSONResponse:
    """Missing-input check happens here, on the loop; everything else runs in state.executor
    (see _predict_body) so a slow body or a slow backend never blocks /health or /ready, which
    share this process's single event loop. Tokenizing a `text` request is part of that
    executor work. Admission itself already happened in OrjsonRoute, before the body was even
    read (P1); `admitted_at` is when that happened.
    """
    if text is not None:
        if state.text is None:
            raise HTTPException(
                400,
                "this model takes tensors only: it was not loaded from a Hugging Face repo "
                "directory with tokenizer files. Send 'inputs' (see GET /schema).",
            )
    else:
        missing = [n for n in state.input_names if n not in inputs]
        if missing:
            raise HTTPException(400, f"missing inputs: {missing}")

    loop = asyncio.get_running_loop()
    body = partial(
        _predict_body,
        state,
        inputs,
        encoding,
        admitted_at,
        text,
        parse_ms,
        timings_ms,
    )
    # run_in_executor does not carry contextvars over; without this, log lines from the
    # executor thread would lose the request id.
    return await loop.run_in_executor(state.executor, contextvars.copy_context().run, body)


__all__ = [
    "PREDICT_PATHS",
    "NumpyJSONResponse",
    "as_batch",
    "run_predict",
]
