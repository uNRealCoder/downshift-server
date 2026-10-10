"""The internals of /predict and /predict/graph: the request conversion, the admission
bookkeeping and the inference call itself. This code is separate from the routing in
serve/app.py. A change in the handling of the body (P1 to P3) then touches one file (M8).
"""

import asyncio
import contextvars
import logging
import time
from collections.abc import Callable
from concurrent.futures import Executor
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import orjson
from fastapi import HTTPException, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError

from downshift.core.axes import DimBound
from downshift.serve.backends import FIRST_OUTPUT_NAME, InferenceInputError
from downshift.serve.codec import b64encode, decode_safetensors, encode_safetensors
from downshift.serve.engine import ServingState
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
    """orjson with OPT_SERIALIZE_NUMPY. It writes arrays from their buffers. NaN and Inf become
    null.

    It is not built on ORJSONResponse of fastapi. Older versions do not have the numpy flag.
    Newer versions deprecate the class and give a warning at import.
    """

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
    """A single string is a batch of one. The response shape therefore never depends on the
    form that the client used."""
    return [value] if isinstance(value, str) else list(value)


def resolve_prompt(state: ServingState, prompt_name: str | None, has_text: bool) -> str:
    """The prompt text to put at the start of each row: the named one, otherwise the default of
    the repo, otherwise none. A name that the repo does not define (or a name that is sent
    without `text`) gives a 400 that lists the names that exist. Downshift returns the value of
    the client only as a truncated repr."""
    recipe = state.embedding
    prompts = recipe.prompts if recipe is not None else {}
    if prompt_name is None:
        default = recipe.default_prompt if recipe is not None else None
        return prompts[default] if has_text and default is not None else ""
    if not has_text:
        raise HTTPException(400, "'prompt_name' applies to a 'text' request. None was sent")
    if prompt_name not in prompts:
        available = sorted(prompts)
        raise HTTPException(
            400,
            f"unknown prompt_name {prompt_name!r:.64}. "
            + (f"this model has: {available}" if available else "this model has no named prompts"),
        )
    return prompts[prompt_name]


def _text_feeds(
    state: ServingState, text: list[str], declared: dict[str, str | None], prompt: str = ""
) -> dict[str, np.ndarray]:
    """Tokenize a `text` request into the inputs of the graph, with `prompt` at the start of
    each row. ValueError means that the text of the client was not usable."""
    assert state.text is not None  # run_predict refuses a text request without it
    encoded = state.text.encode([prompt + row for row in text] if prompt else text)
    missing = [n for n in state.input_names if n not in encoded]
    if missing:
        raise ValueError(f"the tokenizer does not produce the model's inputs {missing}")
    return {n: to_numpy(n, encoded[n], declared.get(n)) for n in state.input_names}


def _vocab_violation(state: ServingState, feeds: dict[str, np.ndarray]) -> str | None:
    """B3: ORT wraps a negative input_ids index and does not refuse it. Wrong input would then
    become a confident 200 response."""
    if state.vocab_size is None:
        return None
    return index_range_violation(
        "input_ids", feeds.get("input_ids"), state.vocab_size, "vocabulary"
    )


def _bound_message(name: str, axis: int, size: int, bound: DimBound) -> str:
    return f"{name} axis {axis} is {size}; this model accepts {bound.min} to {bound.max}"


def _shape_violation(state: ServingState, feeds: dict[str, np.ndarray]) -> str | None:
    """The first input whose rank, or one of whose fixed axes, is different from what the
    backend declares. Downshift checks this before infer. A shape error of the model itself is
    then never a 400 for the client."""
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


def _bound_violation(state: ServingState, feeds: dict[str, np.ndarray]) -> str | None:
    """U2: the first input whose shape is outside an axis that the export of downshift
    traced. None if each known bound is satisfied (also if no bound is known)."""
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
            # Raised only before infer() has started: a request that waited in a queue.
            raise HTTPException(
                503,
                f"request waited {waited:.1f}s in queue, past the {request_timeout:.1f}s "
                "--request-timeout",
            )


def _binary_feeds(
    state: ServingState, tensors: dict[str, np.ndarray], declared: dict[str, str | None]
) -> dict[str, np.ndarray]:
    """The arrays of a safetensors request are already NumPy, and downshift never casts them. A
    dtype that is not the dtype of the graph gives a 400 and not a silent copy."""
    feeds = {n: tensors[n] for n in state.input_names}
    for name, arr in feeds.items():
        expected = normalize_dtype(declared.get(name))
        if expected is not None and arr.dtype.name != expected:
            raise ValueError(f"input {name!r} is {arr.dtype.name}. This model takes {expected}")
    return feeds


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
    """Turn a `graphs` list, or a binary body with num_nodes and num_edges, into one batched feed
    set and the counts for each graph that split the response. ValueError for each client
    error."""
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


def _prepare_feeds(
    state: ServingState, request: _Request
) -> tuple[dict[str, np.ndarray], float, GraphLayout | None]:
    """Turn a validated request into the input arrays of the graph (base64 decode,
    to_numpy, tokenize), and do the checks that need them: vocabulary, edge index and axis
    bounds. It runs in state.prep_executor. It returns the feeds, the milliseconds that it used
    and, for a graph batch, the counts for each graph. `request.tensors` (a safetensors body
    that is already decoded) takes the place of `inputs`. `request.graphs` (or a `tensors` body
    with num_nodes) is a batch of graphs. Downshift joins them here into one feed set.

    It raises HTTPException(400) for each client error (a bad shape, dtype or JSON).
    """
    declared = state.declared_dtypes
    max_bytes = state.options.max_input_bytes
    tensors = request.tensors
    start = time.perf_counter()
    layout: GraphLayout | None = None
    try:
        if request.graphs is not None or (
            tensors is not None and ("num_nodes" in tensors or "num_edges" in tensors)
        ):
            feeds, layout = _batch_feeds(state, request.graphs, tensors, max_bytes)
        elif tensors is not None:
            feeds = _binary_feeds(state, tensors, declared)
        elif request.text is not None:
            feeds = _text_feeds(state, request.text, declared, request.prompt)
        else:
            feeds = {
                n: to_numpy(n, request.inputs[n], declared.get(n), max_bytes=max_bytes)
                for n in state.input_names
            }
    except (ValueError, TypeError, KeyError) as exc:
        raise HTTPException(400, str(exc)) from exc

    vocab_violation = _vocab_violation(state, feeds)
    if vocab_violation is not None:
        raise HTTPException(400, vocab_violation)
    # Downshift checked a batch graph by graph, against the node count of each graph.
    edge_violation = None if layout is not None else edge_index_violation(feeds)
    if edge_violation is not None:
        raise HTTPException(400, edge_violation)
    violation = _shape_violation(state, feeds) or _bound_violation(state, feeds)
    if violation is not None:
        raise HTTPException(400, violation)
    return feeds, (time.perf_counter() - start) * 1000, layout


def _infer(
    state: ServingState, feeds: dict[str, np.ndarray]
) -> tuple[dict[str, np.ndarray], float]:
    """Only the backend call. _encode_response follows it on the same inference thread. It runs
    in state.executor. It returns the outputs and the milliseconds that it used.

    Each error that is not a client error (a bug of the backend, OOM, ...) propagates. The
    handler of the app then turns it into a 500. The text of the exception does not go to the
    client.
    """
    start = time.perf_counter()
    try:
        outputs = state.backend.infer(feeds)
    except InferenceInputError as exc:
        # The check of the bounds before this already answers the common case. This is the
        # backend that rejects something for which we had no bound.
        raise HTTPException(400, str(exc)) from exc
    return outputs, (time.perf_counter() - start) * 1000


def _encode_response(
    state: ServingState,
    outputs: dict[str, np.ndarray],
    encoding: OutputEncoding | RequestOutputEncoding | None,
    timings_ms: dict[str, float],
    accept_safetensors: bool = False,
    layout: GraphLayout | None = None,
) -> Response:
    """Turn the outputs into the response body, directly after _infer on the same thread.
    `timings_ms` already has the earlier stages. This function adds the encode stage and the
    Server-Timing header.

    A safetensors body (an `Accept` that names it, or output_encoding "safetensors") has each
    output under its own name. `predictions` and the embedding recipe are JSON strings in
    `__metadata__`.

    `layout` (a graph batch) first splits the outputs for each graph. The body is {"graphs":
    [{outputs, shapes, dtypes}, ...]} in the order of the request. In safetensors, each output
    has the name `graphs.<i>.<output>`, and the number of graphs is in `downshift.graphs`."""
    encoding = encoding or state.options.output_encoding
    encode = _base64_ready if encoding == OutputEncoding.base64 else _json_ready
    safetensors = accept_safetensors or encoding == RequestOutputEncoding.safetensors
    start = time.perf_counter()
    # C-contiguous one time, at the start (np.require keeps 0-d arrays 0-d. ascontiguousarray does not).
    arrays = {name: np.require(arr, requirements="C") for name, arr in outputs.items()}
    response: Response
    if layout is not None:
        try:
            per_graph = split_outputs(arrays, state.verdict.output_axes, *layout)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
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
        # The same keys as PredictResponse. Built by hand, so orjson serializes the buffers directly.
        body = _tensor_body(arrays, encode)
        predictions = _predictions(state, arrays)
        if predictions is not None:
            body["predictions"] = predictions
        response = NumpyJSONResponse(body)
    timings_ms["encode"] = round((time.perf_counter() - start) * 1000, 2)
    response.headers["Server-Timing"] = ", ".join(
        f"{stage};dur={timings_ms.get(stage, 0.0):.2f}" for stage in STAGES
    )
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
    """orjson, then the pydantic model of the route. Both give the 422 errors of FastAPI. An
    empty body is the missing body. Malformed JSON is json_invalid. The errors of the model are
    under "body". The error dicts copy what the request handler of FastAPI builds
    (fastapi/routing.py, get_request_handler). It has no public helper for them. Tests pin all
    three."""
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
            # Downshift needs no dtype hints. to_numpy takes the dtype that the backend declares
            # (int64 for edge_index on both backends). Integer lists default to int64 in any case.
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
        raise HTTPException(
            400,
            "this model takes tensors only. It was not loaded from a Hugging Face repo "
            "directory with tokenizer files. Send 'inputs' (see GET /schema).",
        )
    return _Request(
        inputs=req.inputs,
        text=as_batch(req.text) if req.text is not None else None,
        prompt=resolve_prompt(state, req.prompt_name, req.text is not None),
        encoding=req.output_encoding,
    )


def _binary_request(state: ServingState, graph: bool, body: bytes) -> _Request:
    """A safetensors body: its arrays (only names that this route takes), and the
    output_encoding of __metadata__."""
    try:
        arrays, metadata = decode_safetensors(body, max_input_bytes=state.options.max_input_bytes)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if "text" in metadata:
        raise HTTPException(400, "text is JSON only. A safetensors body carries tensors")
    allowed = _GRAPH_TENSOR_NAMES if graph else frozenset(state.input_names)
    unknown = [name for name in arrays if name not in allowed]
    if unknown:
        raise HTTPException(
            400, f"unknown tensor name {unknown[0]!r:.64}. This route takes {sorted(allowed)}"
        )
    encoding = metadata.get("output_encoding")
    try:
        return _Request(
            tensors=arrays, encoding=RequestOutputEncoding(encoding) if encoding else None
        )
    except ValueError:
        raise HTTPException(
            400,
            f"unknown output_encoding {encoding!r:.64} in __metadata__. This route takes "
            f"{[e.value for e in RequestOutputEncoding]}",
        ) from None


def _parse(
    state: ServingState, body: bytes, graph: bool, binary: bool, timings_ms: dict[str, float]
) -> _Request:
    """The request body (JSON or safetensors) as a validated _Request. Downshift records its time
    as timings_ms["parse"]. It raises the 422 or 400 for the client."""
    start = time.perf_counter()
    if graph and not GRAPH_INPUTS.issubset(state.input_names):
        raise HTTPException(
            400,
            f"model is not graph-shaped: inputs are {list(state.input_names)}, "
            "expected at least 'x' and 'edge_index'",
        )
    request = (_binary_request if binary else _json_request)(state, graph, body)
    if request.text is None and request.graphs is None:
        missing = [n for n in state.input_names if n not in (request.tensors or request.inputs)]
        if missing:
            raise HTTPException(400, f"missing inputs: {missing}")
    timings_ms["parse"] = round((time.perf_counter() - start) * 1000, 2)
    return request


async def _run_in(
    executor: Executor | None,
    wait_key: str,
    timings_ms: dict[str, float],
    fn: Callable[..., Any],
    *args: Any,
) -> Any:
    """Run fn(*args) in `executor`, and carry the contextvars. (run_in_executor does not. Without
    this, the log lines from the executor thread would lose the request ID.) The time from the
    submit to the start is timings_ms[wait_key]. If `executor` is None (--execution inline), it
    calls fn here on the event loop, with no wait."""
    if executor is None:
        timings_ms[wait_key] = 0.0
        return fn(*args)
    submitted = time.perf_counter()

    def call() -> Any:
        timings_ms[wait_key] = round((time.perf_counter() - submitted) * 1000, 2)
        return fn(*args)

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(executor, contextvars.copy_context().run, call)


async def run_predict(
    state: ServingState,
    body: bytes,
    *,
    graph: bool,
    binary: bool,
    inline: bool,
    admitted_at: float,
    timings_ms: dict[str, float],
    accept_safetensors: bool,
) -> Response:
    """One /predict request (or /predict/graph, with `graph`), from its raw body, in two hops.
    First, _parse and _prepare_feeds run in the prep pool. Then _infer and _encode_response run
    on an inference thread. The encoding therefore holds the inference slot. The event loop
    does neither of them. A large body therefore never blocks /health or /ready (they share the
    one event loop of this process). PredictRoute already did the admission before it read the
    body (P1). `admitted_at` is the time of that event. The request keeps its slot until the
    response is built.

    `binary` means a safetensors body. `inline` (--execution inline, a small JSON body) runs
    everything on the event loop, with no waits. A `text` request still tokenizes in the prep
    pool.

    `timings_ms` collects the milliseconds of each stage (parse, prep_wait, prep, infer_wait,
    infer, encode) for Server-Timing and the request log line (U1). With `accept_safetensors`,
    the response is safetensors.

    A graph batch (a `graphs` list, or a safetensors body with num_nodes and num_edges) is
    batched in prep, runs as one inference, and is split in encode.
    """
    parsed = _parse(state, body, graph, binary, timings_ms) if inline else None
    run_here = parsed is not None and parsed.text is None

    def prepare() -> tuple[Any, dict[str, np.ndarray], GraphLayout | None]:
        # Only the encoding stays after this. Downshift frees the parsed body before the inference.
        request = parsed or _parse(state, body, graph, binary, timings_ms)
        feeds, prep_ms, layout = _prepare_feeds(state, request)
        timings_ms["prep"] = round(prep_ms, 2)
        return request.encoding, feeds, layout

    encoding, feeds, layout = await _run_in(
        None if run_here else state.prep_executor, "prep_wait", timings_ms, prepare
    )

    _check_timeout(state, admitted_at)

    def infer_and_encode() -> Response:
        # Checked again at the start. The request can have waited behind an inference that was running.
        _check_timeout(state, admitted_at)
        outputs, infer_ms = _infer(state, feeds)
        timings_ms["infer"] = round(infer_ms, 2)
        return _encode_response(state, outputs, encoding, timings_ms, accept_safetensors, layout)

    response: Response = await _run_in(
        None if run_here else state.executor, "infer_wait", timings_ms, infer_and_encode
    )
    return response


__all__ = [
    "NumpyJSONResponse",
    "as_batch",
    "run_predict",
]
