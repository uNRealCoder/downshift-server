"""The /predict and /predict/graph internals: request conversion, admission bookkeeping and
the inference call itself. Kept apart from serve/app.py's routing so a body-handling change
(P1 to P3) touches one file (M8).
"""

import asyncio
import contextvars
import logging
import time
from collections.abc import Callable
from concurrent.futures import Executor
from typing import Any

import numpy as np
import orjson
from fastapi import HTTPException, Response
from fastapi.responses import JSONResponse

from downshift.serve.backends import FIRST_OUTPUT_NAME, InferenceInputError
from downshift.serve.codec import b64encode, encode_safetensors
from downshift.serve.engine import DimBound, ServingState
from downshift.serve.metrics import STAGES
from downshift.serve.schemas import OutputEncoding, RequestOutputEncoding, normalize_dtype, to_numpy

logger = logging.getLogger("downshift.serve")

# The predict routes: the only ones OrjsonRoute admits against before reading the body (P1).
PREDICT_PATHS = frozenset({"/predict", "/predict/graph"})

SAFETENSORS_MEDIA_TYPE = "application/vnd.safetensors"
# What a binary request may be labelled; octet-stream is an alias for safetensors.
BINARY_REQUEST_TYPES = frozenset({SAFETENSORS_MEDIA_TYPE, "application/octet-stream"})

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


def resolve_prompt(state: ServingState, prompt_name: str | None, has_text: bool) -> str:
    """The prompt text to put before every row: the named one, else the repo's default, else
    none. A name the repo does not define (or one sent without `text`) is a 400 that lists the
    names there are; the client's own value is echoed only as a truncated repr."""
    recipe = state.embedding
    prompts = recipe.prompts if recipe is not None else {}
    if prompt_name is None:
        default = recipe.default_prompt if recipe is not None else None
        return prompts[default] if has_text and default is not None else ""
    if not has_text:
        raise HTTPException(400, "'prompt_name' applies to a 'text' request; none was sent")
    if prompt_name not in prompts:
        available = sorted(prompts)
        raise HTTPException(
            400,
            f"unknown prompt_name {prompt_name!r:.64}; "
            + (f"this model has: {available}" if available else "this model has no named prompts"),
        )
    return prompts[prompt_name]


def _text_feeds(
    state: ServingState, text: list[str], declared: dict[str, str | None], prompt: str = ""
) -> dict[str, np.ndarray]:
    """Tokenize a `text` request into the graph's own inputs, each row after `prompt`.
    ValueError means the client's text was unusable."""
    assert state.text is not None  # run_predict refuses a text request without one
    encoded = state.text.encode([prompt + row for row in text] if prompt else text)
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


def _check_timeout(state: ServingState, admitted_at: float) -> None:
    request_timeout = state.options.request_timeout
    if request_timeout > 0:
        waited = time.monotonic() - admitted_at
        if waited > request_timeout:
            # Only ever raised before infer() has begun: a request that sat in a queue.
            raise HTTPException(
                503,
                f"request waited {waited:.1f}s in queue, past the {request_timeout:.1f}s "
                "--request-timeout",
            )


def _binary_feeds(
    state: ServingState, tensors: dict[str, np.ndarray], declared: dict[str, str | None]
) -> dict[str, np.ndarray]:
    """A safetensors request's arrays are already NumPy and never cast: a dtype other than the
    graph's own is a 400 rather than a silent copy."""
    feeds = {n: tensors[n] for n in state.input_names}
    for name, arr in feeds.items():
        expected = normalize_dtype(declared.get(name))
        if expected is not None and arr.dtype.name != expected:
            raise ValueError(f"input {name!r} is {arr.dtype.name}; this model takes {expected}")
    return feeds


def _prepare_feeds(
    state: ServingState,
    inputs: dict[str, Any],
    text: list[str] | None,
    prompt: str = "",
    tensors: dict[str, np.ndarray] | None = None,
) -> tuple[dict[str, np.ndarray], float]:
    """Request body to the graph's own input arrays (base64 decode, to_numpy, tokenize) and
    the checks that need them: vocab, edge index, axis bounds. Runs in state.prep_executor.
    Returns the feeds and the milliseconds spent. `tensors` (a safetensors body, already
    decoded) takes the place of `inputs`.

    Raises HTTPException(400) for anything the client got wrong (bad shape/dtype/JSON).
    """
    declared = state.declared_dtypes
    max_bytes = state.options.max_input_bytes
    start = time.perf_counter()
    try:
        if tensors is not None:
            feeds = _binary_feeds(state, tensors, declared)
        elif text is not None:
            feeds = _text_feeds(state, text, declared, prompt)
        else:
            feeds = {
                n: to_numpy(n, inputs[n], declared.get(n), max_bytes=max_bytes)
                for n in state.input_names
            }
    except (ValueError, TypeError, KeyError) as exc:
        raise HTTPException(400, str(exc)) from exc

    vocab_violation = _vocab_violation(state, feeds)
    if vocab_violation is not None:
        raise HTTPException(400, vocab_violation)
    edge_index_violation = _edge_index_violation(feeds)
    if edge_index_violation is not None:
        raise HTTPException(400, edge_index_violation)
    bound_violation = _bound_violation(state, feeds)
    if bound_violation is not None:
        raise HTTPException(400, bound_violation)
    return feeds, (time.perf_counter() - start) * 1000


def _infer(
    state: ServingState, feeds: dict[str, np.ndarray]
) -> tuple[dict[str, np.ndarray], float]:
    """Only the backend call, so an inference slot is never held for conversion or encoding.
    Runs in state.executor. Returns the outputs and the milliseconds spent.

    Anything but a client error (a backend bug, OOM, ...) propagates so the app-level handler
    turns it into a 500 without leaking the exception text to the client.
    """
    start = time.perf_counter()
    try:
        outputs = state.backend.infer(feeds)
    except InferenceInputError as exc:
        # The bound pre-check already answers the common case; this is the backend
        # rejecting something we had no bound for.
        raise HTTPException(400, str(exc)) from exc
    return outputs, (time.perf_counter() - start) * 1000


def _encode_response(
    state: ServingState,
    outputs: dict[str, np.ndarray],
    encoding: OutputEncoding | RequestOutputEncoding | None,
    timings_ms: dict[str, float],
    accept_safetensors: bool = False,
) -> Response:
    """Outputs to the response body, in state.prep_executor. `timings_ms` already holds the
    earlier stages; the encode stage and the Server-Timing header are added here.

    A safetensors body (an `Accept` naming it, or output_encoding "safetensors") carries each
    output under its own name, with `predictions` and the embedding recipe as JSON strings in
    `__metadata__`."""
    encoding = encoding or state.options.output_encoding
    encode = _base64_ready if encoding == OutputEncoding.base64 else _json_ready
    start = time.perf_counter()
    # C-contiguous once, up front (np.require keeps 0-d arrays 0-d; ascontiguousarray does not).
    arrays = {name: np.require(arr, requirements="C") for name, arr in outputs.items()}
    if accept_safetensors or encoding == RequestOutputEncoding.safetensors:
        response: Response = _safetensors_response(state, arrays)
        timings_ms["encode"] = round((time.perf_counter() - start) * 1000, 2)
        _add_server_timing(response, timings_ms)
        return response
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
    timings_ms["encode"] = round((time.perf_counter() - start) * 1000, 2)

    response = NumpyJSONResponse(body)
    _add_server_timing(response, timings_ms)
    return response


def _add_server_timing(response: Response, timings_ms: dict[str, float]) -> None:
    response.headers["Server-Timing"] = ", ".join(
        f"{stage};dur={timings_ms.get(stage, 0.0):.2f}" for stage in STAGES
    )


def _safetensors_response(state: ServingState, arrays: dict[str, np.ndarray]) -> Response:
    metadata: dict[str, str] = {}
    if state.text is not None and FIRST_OUTPUT_NAME in arrays:
        predictions = state.text.predictions(arrays[FIRST_OUTPUT_NAME])
        if predictions is not None:
            metadata["downshift.predictions"] = _json_str(predictions)
    recipe = state.embedding
    if recipe is not None:
        first = next(iter(arrays.values()), None)
        dimension = first.shape[-1] if first is not None and first.ndim else None
        metadata["downshift.embedding"] = _json_str(
            {
                "pooling": recipe.pooling,
                "normalized": recipe.normalize,
                "dimension": dimension,
                "max_seq_length": recipe.max_seq_length,
                "from": recipe.origin,
            }
        )
    return Response(encode_safetensors(arrays, metadata), media_type=SAFETENSORS_MEDIA_TYPE)


def _json_str(value: Any) -> str:
    return orjson.dumps(value, option=orjson.OPT_SERIALIZE_NUMPY).decode()


async def _run_in(
    executor: Executor | None,
    wait_key: str,
    timings_ms: dict[str, float],
    fn: Callable[..., Any],
    *args: Any,
) -> Any:
    """fn(*args) in `executor`, carrying contextvars (run_in_executor does not; without this,
    log lines from the executor thread would lose the request id). The time from submit to
    start is added to timings_ms[wait_key]. `executor` None (--execution inline) calls fn
    right here on the event loop, with no wait."""
    if executor is None:
        timings_ms.setdefault(wait_key, 0.0)
        return fn(*args)
    submitted = time.perf_counter()

    def call() -> Any:
        waited = (time.perf_counter() - submitted) * 1000
        timings_ms[wait_key] = round(timings_ms.get(wait_key, 0.0) + waited, 2)
        return fn(*args)

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(executor, contextvars.copy_context().run, call)


async def run_predict(
    state: ServingState,
    inputs: dict[str, Any],
    encoding: OutputEncoding | RequestOutputEncoding | None,
    text: list[str] | None = None,
    prompt_name: str | None = None,
    *,
    admitted_at: float,
    parse_ms: float,
    timings_ms: dict[str, float],
    inline: bool = False,
    tensors: dict[str, np.ndarray] | None = None,
    accept_safetensors: bool = False,
) -> Response:
    """Missing-input check happens here, on the loop; the rest is three stages so only the
    middle one holds an inference thread: _prepare_feeds (prep pool), _infer (inference
    executor), _encode_response (prep pool). A slow body or encode therefore never blocks
    /health or /ready (they share this process's single event loop) nor another request's
    inference. Admission itself already happened in OrjsonRoute, before the body was even
    read (P1); `admitted_at` is when that happened, and the request keeps its slot until
    the response is built.

    `timings_ms` collects the per-stage milliseconds (parse, prep_wait, prep, infer_wait,
    infer, encode) for Server-Timing and the request log line (U1).

    `inline` (--execution inline, a small JSON body; see OrjsonRoute) runs all three stages
    on the event loop instead, with zero waits.

    `tensors` is a decoded safetensors request body, used instead of `inputs`; with
    `accept_safetensors` the response is safetensors too.
    """
    prep_pool = None if inline else state.prep_executor
    infer_pool = None if inline else state.executor
    if text is not None:
        if state.text is None:
            raise HTTPException(
                400,
                "this model takes tensors only: it was not loaded from a Hugging Face repo "
                "directory with tokenizer files. Send 'inputs' (see GET /schema).",
            )
    else:
        missing = [n for n in state.input_names if n not in (tensors or inputs)]
        if missing:
            raise HTTPException(400, f"missing inputs: {missing}")

    prompt = resolve_prompt(state, prompt_name, text is not None)

    timings_ms["parse"] = round(parse_ms, 2)
    feeds, prep_ms = await _run_in(
        prep_pool, "prep_wait", timings_ms, _prepare_feeds, state, inputs, text, prompt, tensors
    )
    timings_ms["prep"] = round(prep_ms, 2)

    _check_timeout(state, admitted_at)

    def infer() -> tuple[dict[str, np.ndarray], float]:
        # Re-checked at start: the request may have queued behind a running inference.
        _check_timeout(state, admitted_at)
        return _infer(state, feeds)

    outputs, infer_ms = await _run_in(infer_pool, "infer_wait", timings_ms, infer)
    timings_ms["infer"] = round(infer_ms, 2)

    response: Response = await _run_in(
        prep_pool,
        "prep_wait",
        timings_ms,
        _encode_response,
        state,
        outputs,
        encoding,
        timings_ms,
        accept_safetensors,
    )
    return response


__all__ = [
    "PREDICT_PATHS",
    "NumpyJSONResponse",
    "as_batch",
    "run_predict",
]
