"""The request path of /predict and /predict/graph, from the raw body to the response: parse,
convert and check (prep pool), infer and encode (inference pool), and the result cache.
"""

import asyncio
import contextvars
import logging
import time
from collections.abc import Callable
from concurrent.futures import Executor, Future
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import orjson
from fastapi import Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError

from downshift import settings
from downshift.adapters.text import TextTooLongError
from downshift.core.axes import DimBound
from downshift.serve.backends import FIRST_OUTPUT_NAME, InferenceInputError
from downshift.serve.codec import b64encode, decode_safetensors, encode_safetensors
from downshift.serve.engine import ServingState
from downshift.serve.errors import ApiError
from downshift.serve.graphs import (
    GRAPH_COUNTS,
    GRAPH_INPUTS,
    GRAPH_TENSORS,
    batch_graphs,
    binary_batch,
    edge_index_violation,
    index_range_violation,
    split_outputs,
    split_refusal,
)
from downshift.serve.result_cache import Lookup, ResultCache, Row, tensor_keys, text_keys
from downshift.serve.schemas import (
    GraphPredictRequest,
    OutputEncoding,
    PredictRequest,
    RequestOutputEncoding,
    normalize_dtype,
    to_numpy,
)

logger = logging.getLogger("downshift.serve")

# The (node_counts, edge_counts) of each graph in a graph batch. The response is split by them.
GraphLayout = tuple[list[int], list[int]]

# The names that a safetensors body on /predict/graph can use.
_GRAPH_TENSOR_NAMES = frozenset((*GRAPH_TENSORS, *GRAPH_COUNTS))

SAFETENSORS_MEDIA_TYPE = "application/vnd.safetensors"
# The labels that a binary request can have. octet-stream is an alias for safetensors.
BINARY_REQUEST_TYPES = frozenset({SAFETENSORS_MEDIA_TYPE, "application/octet-stream"})
# The Server-Timing stages, in the order of the request.
STAGES = ("parse", "prep_wait", "prep", "infer_wait", "infer", "encode")

# The dtypes that OPT_SERIALIZE_NUMPY of orjson writes directly from the array buffer (orjson >= 3.9).
_ORJSON_DTYPES = frozenset(
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
        arr if arr.dtype.name in _ORJSON_DTYPES and arr.ndim else arr.tolist()
    )  # 0-d: orjson rejects it


def _base64_ready(arr: np.ndarray) -> dict[str, Any]:
    """{data, dtype, shape}: base64 straight off the contiguous buffer."""
    return {
        "data": b64encode(arr).decode("ascii"),
        "dtype": arr.dtype.name,
        "shape": list(arr.shape),
    }


def as_batch(value: str | list[str]) -> list[str]:
    """A single string is a batch of one, so the response shape is the same for both forms."""
    return [value] if isinstance(value, str) else list(value)


def resolve_prompt(state: ServingState, prompt_name: str | None, has_text: bool) -> str:
    """The prompt to put before each text row: the named one, else the repo default, else ""."""
    recipe = state.embedding
    prompts = recipe.prompts if recipe is not None else {}
    if prompt_name is None:
        default = recipe.default_prompt if recipe is not None else None
        return prompts[default] if has_text and default is not None else ""
    if not has_text:
        raise ApiError(
            400, "invalid_input", "'prompt_name' applies to a 'text' request. None was sent"
        )
    if prompt_name not in prompts:
        available = sorted(prompts)
        raise ApiError(
            400,
            "unknown_prompt",
            f"unknown prompt_name {prompt_name!r:.64}. "
            + (f"this model has: {available}" if available else "this model has no named prompts"),
        )
    return prompts[prompt_name]


def _text_feeds(
    state: ServingState,
    text: list[str],
    declared: dict[str, str | None],
    prompt: str = "",
    count_tokens: bool = True,
) -> tuple[dict[str, np.ndarray], list[int] | None]:
    """Tokenize `text` (each row after `prompt`) into the graph inputs. With `count_tokens`,
    also returns the token count of each row, from the tokenizer output before it is cut to the
    model inputs."""
    assert state.text is not None  # run_predict refuses a text request without it
    encoded = state.text.encode([prompt + row for row in text] if prompt else text)
    missing = [n for n in state.input_names if n not in encoded]
    if missing:
        raise ValueError(f"the tokenizer does not produce the model's inputs {missing}")
    feeds = {n: to_numpy(n, encoded[n], declared.get(n)) for n in state.input_names}
    return feeds, _row_tokens(encoded) if count_tokens else None


def _vocab_violation(state: ServingState, feeds: dict[str, np.ndarray]) -> str | None:
    """ORT wraps a negative input_ids index: check the range before the inference."""
    if state.vocab_size is None:
        return None
    return index_range_violation(
        "input_ids", feeds.get("input_ids"), state.vocab_size, "vocabulary"
    )


def _bound_message(name: str, axis: int, size: int, bound: DimBound) -> str:
    return f"{name} axis {axis} is {size}; this model accepts {bound.min} to {bound.max}"


def _shape_violation(state: ServingState, feeds: dict[str, np.ndarray]) -> str | None:
    """The first input whose rank or fixed axis differs from the backend's declaration, so a
    shape error inside the model is never blamed on the client."""
    for name, arr in feeds.items():
        spec = state.input_specs.get(name)
        if spec is None or spec.shape is None:
            continue
        if arr.ndim != len(spec.shape):
            return f"{name} has {arr.ndim} dimensions; this model takes {len(spec.shape)}"
        for axis, dim in enumerate(spec.shape):
            if isinstance(dim, int) and arr.shape[axis] != dim:
                return f"{name} axis {axis} is {arr.shape[axis]}; this model takes {dim}"
    return None


def _bound_violation(
    state: ServingState, feeds: dict[str, np.ndarray], skip_batch_axis: bool = False
) -> str | None:
    """The first input outside the axis bounds that the export traced. `skip_batch_axis` is
    for cache-miss rows: the caller already checked the whole batch."""
    for name, arr in feeds.items():
        bounds = state.axis_bounds.get(name)
        if not bounds:
            continue
        for axis, bound in bounds.items():
            if axis >= arr.ndim or (skip_batch_axis and axis == 0):
                continue
            size = arr.shape[axis]
            if not (bound.min <= size <= bound.max):
                return _bound_message(name, axis, size, bound)
    return None


def _violation(
    state: ServingState,
    feeds: dict[str, np.ndarray],
    *,
    edges: bool = True,
    skip_batch_axis: bool = False,
) -> str | None:
    """The first client error in `feeds`: a token ID outside the vocabulary, an edge index
    outside the nodes, a wrong rank or fixed axis, or an axis outside its bounds."""
    return (
        _vocab_violation(state, feeds)
        or (edge_index_violation(feeds) if edges else None)
        or _shape_violation(state, feeds)
        or _bound_violation(state, feeds, skip_batch_axis)
    )


def _check_timeout(state: ServingState, deadline: float | None) -> None:
    """A 503 if `deadline` (a time.monotonic value, None for no limit) has passed."""
    if deadline is not None and (now := time.monotonic()) > deadline:
        request_timeout = state.options.request_timeout
        waited = now - deadline + request_timeout
        raise ApiError(
            503,
            "request_timeout",
            f"request waited {waited:.1f}s in queue, past the {request_timeout:.1f}s "
            "--request-timeout",
        )


def _binary_feeds(
    state: ServingState, tensors: dict[str, np.ndarray], declared: dict[str, str | None]
) -> dict[str, np.ndarray]:
    """The arrays of a safetensors body, never cast: a wrong dtype is a 400, not a copy."""
    feeds = {n: tensors[n] for n in state.input_names}
    for name, arr in feeds.items():
        expected = normalize_dtype(declared.get(name))
        if expected is not None and arr.dtype.name != expected:
            raise ValueError(f"input {name!r} is {arr.dtype.name}. This model takes {expected}")
    return feeds


def _tensor_feeds(state: ServingState, request: "_Request") -> dict[str, np.ndarray]:
    """The arrays of an `inputs` or safetensors request."""
    declared = state.declared_dtypes
    if request.tensors is not None:
        return _binary_feeds(state, request.tensors, declared)
    max_bytes = state.options.max_input_bytes
    return {
        n: to_numpy(n, request.inputs[n], declared.get(n), max_bytes=max_bytes)
        for n in state.input_names
    }


def _graph_item(
    state: ServingState, index: int, graph: dict[str, Any], max_bytes: int
) -> dict[str, np.ndarray]:
    """One `graphs` entry (JSON values) to arrays; an error names the graph."""
    declared = state.declared_dtypes
    try:
        return {
            n: to_numpy(n, graph[n], declared.get(n), max_bytes=max_bytes)
            for n in GRAPH_TENSORS
            if n in state.input_names and graph.get(n) is not None
        }
    except (ValueError, TypeError) as exc:
        raise ValueError(f"graphs[{index}]: {exc}") from exc


def _batch_feeds(
    state: ServingState,
    graphs: list[dict[str, Any]] | None,
    tensors: dict[str, np.ndarray] | None,
    max_bytes: int,
) -> tuple[dict[str, np.ndarray], GraphLayout]:
    """One batched feed set, and the counts for each graph that split the response, from a
    `graphs` list or a safetensors body with num_nodes and num_edges."""
    names = [spec.name for spec in state.backend.metadata().outputs]
    if tensors is not None:
        _binary_feeds(state, tensors, state.declared_dtypes)
        feeds, node_counts, edge_counts = binary_batch(tensors)
    else:
        assert graphs is not None
        items = [_graph_item(state, i, graph, max_bytes) for i, graph in enumerate(graphs)]
        feeds, node_counts, edge_counts = batch_graphs(items)
    refusal = split_refusal(state.verdict.output_axes, names, len(node_counts))
    if refusal is not None:
        raise ValueError(refusal)
    missing = [n for n in state.input_names if n not in feeds]
    if missing:
        raise ValueError(f"missing inputs: {missing}")
    return {n: feeds[n] for n in state.input_names}, (node_counts, edge_counts)


@dataclass
class _Request:
    """A predict body, decoded and validated: what _prepare_feeds takes, plus the encoding."""

    inputs: dict[str, Any] = field(default_factory=dict)
    text: list[str] | None = None
    prompt: str = ""
    tensors: dict[str, np.ndarray] | None = None
    graphs: list[dict[str, Any]] | None = None
    encoding: OutputEncoding | RequestOutputEncoding | None = None
    dimensions: int | None = None  # /v1/embeddings: keep the first values of each vector
    tokens: list[int] | None = None  # set by _prepare_feeds for a text request


Parse = Callable[[], _Request]
# (outputs, request, tokens, layout) -> the response. `tokens` is the count of tokens for each
# row, only for a text request and only if run_predict was asked for it. `layout` is for a graph
# batch.
Encode = Callable[[dict[str, np.ndarray], _Request, list[int] | None, GraphLayout | None], Response]

# The key of the token count in a cached text row. The encoders never see it as an output.
TOKENS_KEY = "__tokens__"


def _row_tokens(feeds: dict[str, np.ndarray]) -> list[int]:
    """The tokens that are not padding, for each row of text feeds."""
    mask = feeds.get("attention_mask")
    if mask is not None:
        return mask.reshape(mask.shape[0], -1).sum(axis=1).tolist()  # type: ignore[no-any-return]
    ids = feeds["input_ids"] if "input_ids" in feeds else next(iter(feeds.values()))
    return [ids.shape[1]] * ids.shape[0]


def _input_error(exc: Exception) -> ApiError:
    code = "text_too_long" if isinstance(exc, TextTooLongError) else "invalid_input"
    return ApiError(400, code, str(exc))


def _prepare_feeds(
    state: ServingState,
    request: _Request,
    skip_batch_axis: bool = False,
    count_tokens: bool = False,
) -> tuple[dict[str, np.ndarray], float, GraphLayout | None]:
    """The graph inputs of a validated request (decode, convert, tokenize, join graphs), after
    the checks that need them. Returns (feeds, milliseconds, graph layout or None). A text
    request gets `request.tokens` only with `count_tokens`. Raises a 400 for each client
    error."""
    tensors = request.tensors
    start = time.perf_counter()
    layout: GraphLayout | None = None
    try:
        if request.graphs is not None or (
            tensors is not None and ("num_nodes" in tensors or "num_edges" in tensors)
        ):
            feeds, layout = _batch_feeds(
                state, request.graphs, tensors, state.options.max_input_bytes
            )
        elif request.text is not None:
            feeds, request.tokens = _text_feeds(
                state, request.text, state.declared_dtypes, request.prompt, count_tokens
            )
        else:
            feeds = _tensor_feeds(state, request)
    except (ValueError, TypeError, KeyError) as exc:
        raise _input_error(exc) from exc

    # A graph batch was already checked graph by graph.
    violation = _violation(state, feeds, edges=layout is None, skip_batch_axis=skip_batch_axis)
    if violation is not None:
        raise ApiError(400, "invalid_input", violation)
    return feeds, (time.perf_counter() - start) * 1000, layout


def _infer(
    state: ServingState, feeds: dict[str, np.ndarray]
) -> tuple[dict[str, np.ndarray], float]:
    """The backend call: (outputs, milliseconds). Only InferenceInputError is a 400. Any
    other error propagates to the 500 handler, which hides its text from the client."""
    start = time.perf_counter()
    try:
        outputs = state.backend.infer(feeds)
    except InferenceInputError as exc:
        raise ApiError(400, "invalid_input", str(exc)) from exc
    return outputs, (time.perf_counter() - start) * 1000


def _encode_response(
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


def _validate_json[ModelT: BaseModel](model: type[ModelT], body: bytes) -> ModelT:
    """orjson, then the pydantic model, with the same 422 errors that FastAPI's own request
    handler builds (it has no public helper for them)."""
    if not body:
        raise RequestValidationError(
            [{"type": "missing", "loc": ("body",), "msg": "Field required", "input": None}]
        )
    try:
        parsed = orjson.loads(body)
    except orjson.JSONDecodeError as exc:
        raise RequestValidationError(
            [
                {
                    "type": "json_invalid",
                    "loc": ("body", exc.pos),
                    "msg": "JSON decode error",
                    "input": {},
                    "ctx": {"error": exc.msg},
                }
            ]
        ) from None
    try:
        return model.model_validate(parsed)
    except ValidationError as exc:
        errors = exc.errors(include_url=False)
        raise RequestValidationError(
            [{**error, "loc": ("body", *error["loc"])} for error in errors]
        ) from None


def _json_request(state: ServingState, graph: bool, body: bytes) -> _Request:
    if graph:
        graph_req = _validate_json(GraphPredictRequest, body)
        if graph_req.graphs is not None:
            graphs = [
                {"x": g.x, "edge_index": g.edge_index, "edge_attr": g.edge_attr}
                for g in graph_req.graphs
            ]
            return _Request(graphs=graphs, encoding=graph_req.output_encoding)
        inputs = {"x": graph_req.x, "edge_index": graph_req.edge_index}
        if graph_req.edge_attr is not None:
            inputs["edge_attr"] = graph_req.edge_attr
        return _Request(inputs=inputs, encoding=graph_req.output_encoding)

    req = _validate_json(PredictRequest, body)
    if req.text is not None and state.text is None:
        raise ApiError(
            400,
            "text_not_supported",
            "this model takes tensors only. It was not loaded from a Hugging Face repo "
            "directory with tokenizer files. Send 'inputs' (see GET /schema).",
        )
    text = as_batch(req.text) if req.text is not None else None
    if text == []:
        raise ApiError(400, "invalid_input", "'text' is an empty list. Send at least one text")
    return _Request(
        inputs=req.inputs,
        text=text,
        prompt=resolve_prompt(state, req.prompt_name, req.text is not None),
        encoding=req.output_encoding,
    )


def _binary_request(state: ServingState, graph: bool, body: bytes) -> _Request:
    """A safetensors body: its arrays and the output_encoding of its __metadata__."""
    try:
        arrays, metadata = decode_safetensors(body, max_input_bytes=state.options.max_input_bytes)
    except ValueError as exc:
        raise ApiError(400, "invalid_input", str(exc)) from exc
    if "text" in metadata:
        raise ApiError(
            400, "invalid_input", "text is JSON only. A safetensors body carries tensors"
        )
    allowed = _GRAPH_TENSOR_NAMES if graph else frozenset(state.input_names)
    unknown = [name for name in arrays if name not in allowed]
    if unknown:
        raise ApiError(
            400,
            "invalid_input",
            f"unknown tensor name {unknown[0]!r:.64}. This route takes {sorted(allowed)}",
        )
    encoding = metadata.get("output_encoding")
    try:
        return _Request(
            tensors=arrays, encoding=RequestOutputEncoding(encoding) if encoding else None
        )
    except ValueError:
        raise ApiError(
            400,
            "invalid_input",
            f"unknown output_encoding {encoding!r:.64} in __metadata__. This route takes "
            f"{[e.value for e in RequestOutputEncoding]}",
        ) from None


def _parse(
    state: ServingState, body: bytes, graph: bool, binary: bool, timings_ms: dict[str, float]
) -> _Request:
    """The body (JSON or safetensors) as a validated _Request, or a 422 or 400."""
    start = time.perf_counter()
    if graph and not GRAPH_INPUTS.issubset(state.input_names):
        raise ApiError(
            400,
            "not_graph_model",
            f"model is not graph-shaped: inputs are {list(state.input_names)}, "
            "expected at least 'x' and 'edge_index'",
        )
    request = (_binary_request if binary else _json_request)(state, graph, body)
    if request.text is None and request.graphs is None:
        missing = [n for n in state.input_names if n not in (request.tensors or request.inputs)]
        if missing:
            raise ApiError(400, "missing_input", f"missing inputs: {missing}")
    timings_ms["parse"] = round((time.perf_counter() - start) * 1000, 2)
    return request


async def _run_in(
    executor: Executor | None,
    wait_key: str,
    timings_ms: dict[str, float],
    fn: Callable[..., Any],
    *args: Any,
) -> Any:
    """fn(*args) in `executor` with the contextvars (so log lines keep the request ID), or on
    the event loop if `executor` is None. timings_ms[wait_key] is the queue time."""
    if executor is None:
        timings_ms[wait_key] = 0.0
        return fn(*args)
    submitted = time.perf_counter()

    def call() -> Any:
        timings_ms[wait_key] = round((time.perf_counter() - submitted) * 1000, 2)
        return fn(*args)

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(executor, contextvars.copy_context().run, call)


@dataclass
class _CachePlan:
    """The prep result of a cached request: its row keys, the lookup, and the feeds of the rows
    that this request must compute."""

    request: _Request
    arrays: dict[str, np.ndarray] | None  # the full input arrays of a tensor request
    keys: list[bytes]
    found: Lookup
    feeds: dict[str, np.ndarray] | None  # for the missing rows only. None if there are none
    owned: list[bytes] = field(default_factory=list)  # the keys that this request claimed
    passthrough: bool = False  # an empty batch: run the model once and skip the cache
    tokens: list[int] | None = None  # the token count of each row of `feeds`, for a text request


def _row_count_violation(state: ServingState, rows: int) -> str | None:
    """The batch-axis bound for a text request, whose hit rows are never tokenized."""
    for name, bounds in state.axis_bounds.items():
        bound = bounds.get(0)
        if bound is not None and not (bound.min <= rows <= bound.max):
            return _bound_message(name, 0, rows, bound)
    return None


def _subset(request: _Request, arrays: dict[str, np.ndarray] | None, rows: list[int]) -> _Request:
    """A request with only `rows` of the batch."""
    if request.text is not None:
        return _Request(text=[request.text[i] for i in rows], prompt=request.prompt)
    assert arrays is not None
    return _Request(tensors={n: a[rows] for n, a in arrays.items()})


def _cache_prepare(
    state: ServingState,
    cache: ResultCache,
    request: _Request,
    timings_ms: dict[str, float],
    use_lookup: bool = True,
    deadline: float | None = None,
) -> _CachePlan:
    """The prep hop of a cached request. A tensor request is checked whole before the lookup,
    so a hit never hides a client error. Only text rows that missed are tokenized. The lookup
    claims the missed rows; a later failure in this hop releases them. A request that is past
    its `deadline` claims nothing."""
    start = time.perf_counter()
    arrays: dict[str, np.ndarray] | None = None
    try:
        if request.text is not None:
            keys = text_keys(request.prompt, request.text)
            violation = _row_count_violation(state, len(keys))
        else:
            arrays = _tensor_feeds(state, request)
            counts = {a.shape[0] if a.ndim else -1 for a in arrays.values()}
            if len(counts) != 1 or -1 in counts:
                raise ValueError("each input needs the same number of rows on axis 0")
            keys = tensor_keys(arrays, counts.pop())
            violation = _violation(state, arrays)
    except (ValueError, TypeError, KeyError) as exc:
        raise _input_error(exc) from exc
    if violation is not None:
        raise ApiError(400, "invalid_input", violation)

    if not keys:
        whole, _, _ = _prepare_feeds(state, request)
        return _CachePlan(request, arrays, keys, Lookup(), whole, passthrough=True)
    _check_timeout(state, deadline)
    cache_start = time.perf_counter()
    # Cache-Control: no-cache reads and claims nothing. The rows are still stored.
    found = cache.lookup(keys) if use_lookup else Lookup(missing=list(range(len(keys))))
    claimed = [keys[i] for i in found.missing] if use_lookup else []
    timings_ms["cache"] = round((time.perf_counter() - cache_start) * 1000, 2)
    timings_ms["cache_hits"] = len(found.hits) + len(found.pending)
    timings_ms["cache_rows"] = len(keys)
    tokens: list[int] | None = None
    try:
        feeds: dict[str, np.ndarray] | None = None
        if found.missing:
            if arrays is not None and len(found.missing) == len(keys):
                feeds = arrays
            else:
                subset = _subset(request, arrays, found.missing)
                feeds, _, _ = _prepare_feeds(state, subset, skip_batch_axis=True, count_tokens=True)
                tokens = subset.tokens
    except BaseException:
        cache.fail(claimed)
        raise
    timings_ms["prep"] = round((time.perf_counter() - start) * 1000, 2)
    return _CachePlan(request, arrays, keys, found, feeds, claimed, tokens=tokens)


def _release_claims(cache: ResultCache, prep: asyncio.Future[_CachePlan]) -> None:
    """The request left before its plan arrived: fail its claimed rows, so the waiters compute
    them themselves."""
    if not prep.cancelled() and prep.exception() is None:
        cache.fail(prep.result().owned)


async def _await_row(future: Future[Row]) -> Row:
    """The row that another request computes. asyncio.shield (unlike awaiting the wrapped
    future) does not cancel it if this request leaves, so the other waiters still get the row."""
    wrapped = asyncio.wrap_future(future)
    wrapped.add_done_callback(lambda f: f.cancelled() or f.exception())  # no unread errors
    return await asyncio.shield(wrapped)


async def _run_cached(
    state: ServingState,
    cache: ResultCache,
    parsed: _Request | None,
    *,
    body: bytes,
    graph: bool,
    binary: bool,
    accept_safetensors: bool,
    parse: Parse | None,
    encode: Encode | None,
    with_tokens: bool,
    run_here: bool,
    deadline: float | None,
    timings_ms: dict[str, float],
    use_lookup: bool = True,
) -> Response:
    """run_predict with the result cache. The model sees only the missed rows. Hits, rows from
    other requests and new rows are joined in request order and encoded on the prep pool. A
    text row keeps its token count under TOKENS_KEY."""
    pool = None if run_here else state.prep_executor

    def prepare() -> _CachePlan:
        request = parsed or (
            parse() if parse is not None else _parse(state, body, graph, binary, timings_ms)
        )
        return _cache_prepare(state, cache, request, timings_ms, use_lookup, deadline)

    # If the client leaves during prep, the thread still finishes and claims rows: release them.
    prep = asyncio.ensure_future(_run_in(pool, "prep_wait", timings_ms, prepare))
    try:
        plan: _CachePlan = await asyncio.shield(prep)
    except asyncio.CancelledError:
        prep.add_done_callback(lambda t: _release_claims(cache, t))
        raise
    owned = plan.owned

    rows: dict[int, Row] = dict(plan.found.hits)
    served = len(plan.found.hits)  # the rows that this request did not compute

    async def compute(
        feeds: dict[str, np.ndarray], indexes: list[int], tokens: list[int] | None
    ) -> Response | None:
        """Run the model on `feeds` (the rows `indexes`) and give each row to the cache."""

        def infer_rows() -> Response | None:
            _check_timeout(state, deadline)
            outputs, infer_ms = _infer(state, feeds)
            timings_ms["infer"] = round(infer_ms, 2)
            by_row = all(o.ndim and o.shape[0] == len(indexes) for o in outputs.values())
            if plan.passthrough or (not by_row and len(indexes) == len(plan.keys)):
                # An empty batch, or outputs without a row axis: nothing to cache or join. The
                # waiters of the claimed rows compute them themselves.
                cache.fail(owned)
                owned.clear()
                if encode is not None:
                    return encode(outputs, plan.request, None, None)
                return _encode_response(
                    state, outputs, plan.request.encoding, timings_ms, accept_safetensors, None
                )
            if not by_row:
                raise RuntimeError("an output of the model does not follow the batch size")
            for n, i in enumerate(indexes):
                rows[i] = {name: out[n] for name, out in outputs.items()}
                if tokens is not None:
                    rows[i][TOKENS_KEY] = np.asarray(tokens[n])
                cache.complete(plan.keys[i], rows[i])
                if plan.keys[i] in owned:  # a later fail() must not touch a new owner's claim
                    owned.remove(plan.keys[i])
            return None

        done: Response | None = await _run_in(
            None if run_here else state.executor, "infer_wait", timings_ms, infer_rows
        )
        return done

    try:
        _check_timeout(state, deadline)
        if plan.feeds is not None:
            done = await compute(plan.feeds, plan.found.missing, plan.tokens)
            if done is not None:
                return done
        # A waiter whose owner failed computes the row itself. It never sees the owner's error.
        retry: list[int] = []
        for i, future in plan.found.pending.items():
            try:
                rows[i] = await _await_row(future)
                served += 1
            except Exception:  # noqa: BLE001 - any failure of the owner means "compute it here"
                retry.append(i)
        if retry:
            subset = _subset(plan.request, plan.arrays, retry)
            feeds = await _run_in(
                pool,
                "prep_wait",
                timings_ms,
                lambda: _prepare_feeds(state, subset, skip_batch_axis=True, count_tokens=True)[0],
            )
            done = await compute(feeds, retry, subset.tokens)
            if done is not None:
                return done
    except BaseException:
        cache.fail(owned)
        raise

    def finish() -> Response:
        names = [n for n in rows[0] if n != TOKENS_KEY]
        arrays = {n: np.stack([rows[i][n] for i in range(len(plan.keys))]) for n in names}
        tokens = None
        if with_tokens and plan.request.text is not None:
            tokens = [int(rows[i][TOKENS_KEY]) for i in range(len(plan.keys))]
        if encode is not None:
            return encode(arrays, plan.request, tokens, None)
        return _encode_response(
            state, arrays, plan.request.encoding, timings_ms, accept_safetensors, None
        )

    response: Response = await _run_in(pool, "encode_wait", timings_ms, finish)
    computed = len(plan.keys) - served
    response.headers[settings.CACHE_HEADER] = (
        "hit" if computed == 0 else "miss" if served == 0 else "partial"
    )
    return response


async def run_predict(
    state: ServingState,
    body: bytes,
    *,
    graph: bool,
    binary: bool,
    inline: bool,
    deadline: float | None,
    timings_ms: dict[str, float],
    accept_safetensors: bool,
    parse: Parse | None,
    encode: Encode | None,
    with_tokens: bool,
    use_cache: bool = True,
) -> Response:
    """One admitted predict, from its raw body, in two hops: parse and prepare on the prep
    pool, then infer and encode on an inference thread. The event loop does neither, so
    /health and /ready never wait.

    `binary` is a safetensors body. `inline` (--execution inline, a small JSON body) does all
    of it on the event loop, except tokenization. `timings_ms` collects the milliseconds of
    each stage for the access log and the optional Server-Timing header.

    `parse` (None: decode `body` as a /predict body) builds the _Request, and `encode` (None: the
    /predict response) builds the response. /v1/embeddings sets both and has the same admission,
    pools, timeout, cache and row checks. `with_tokens` gives `encode` the token count of each
    text row. `deadline` is a time.monotonic value (None for no limit).
    """
    parsed: _Request | None = None
    if inline:
        parsed = parse() if parse is not None else _parse(state, body, graph, binary, timings_ms)
    run_here = parsed is not None and parsed.text is None
    if state.result_cache is not None and not graph:
        return await _run_cached(
            state,
            state.result_cache,
            parsed,
            body=body,
            graph=graph,
            binary=binary,
            accept_safetensors=accept_safetensors,
            parse=parse,
            encode=encode,
            with_tokens=with_tokens,
            run_here=run_here,
            deadline=deadline,
            timings_ms=timings_ms,
            use_lookup=use_cache,
        )

    def prepare() -> tuple[_Request, dict[str, np.ndarray], GraphLayout | None]:
        request = parsed or (
            parse() if parse is not None else _parse(state, body, graph, binary, timings_ms)
        )
        feeds, prep_ms, layout = _prepare_feeds(state, request, count_tokens=with_tokens)
        timings_ms["prep"] = round(prep_ms, 2)
        return request, feeds, layout

    request, feeds, layout = await _run_in(
        None if run_here else state.prep_executor, "prep_wait", timings_ms, prepare
    )

    _check_timeout(state, deadline)

    def infer_and_encode() -> Response:
        # Again: the request may have waited behind a running inference.
        _check_timeout(state, deadline)
        outputs, infer_ms = _infer(state, feeds)
        timings_ms["infer"] = round(infer_ms, 2)
        tokens = request.tokens if with_tokens else None
        if encode is not None:
            return encode(outputs, request, tokens, layout)
        return _encode_response(
            state, outputs, request.encoding, timings_ms, accept_safetensors, layout
        )

    response: Response = await _run_in(
        None if run_here else state.executor, "infer_wait", timings_ms, infer_and_encode
    )
    return response


__all__ = [
    "NumpyJSONResponse",
    "add_server_timing",
    "as_batch",
    "run_predict",
]
