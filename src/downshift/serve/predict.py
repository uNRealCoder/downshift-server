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

# Per-graph (node_counts, edge_counts) of a graph batch, what the response is split by.
GraphLayout = tuple[list[int], list[int]]

# What a safetensors body on /predict/graph may name.
_GRAPH_TENSOR_NAMES = frozenset((*GRAPH_TENSORS, *GRAPH_COUNTS))

SAFETENSORS_MEDIA_TYPE = "application/vnd.safetensors"
# What a binary request may be labelled; octet-stream is an alias for safetensors.
BINARY_REQUEST_TYPES = frozenset({SAFETENSORS_MEDIA_TYPE, "application/octet-stream"})
# The Server-Timing stages, in request order.
STAGES = ("parse", "prep_wait", "prep", "infer_wait", "infer", "encode")

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
    """B3: ORT wraps a negative input_ids index instead of refusing it, so garbage in would
    become a confident 200 out."""
    if state.vocab_size is None:
        return None
    return index_range_violation(
        "input_ids", feeds.get("input_ids"), state.vocab_size, "vocabulary"
    )


def _bound_message(name: str, axis: int, size: int, bound: DimBound) -> str:
    return f"{name} axis {axis} is {size}; this model accepts {bound.min} to {bound.max}"


def _shape_violation(state: ServingState, feeds: dict[str, np.ndarray]) -> str | None:
    """The first input whose rank, or one of whose fixed axes, differs from what the backend
    declares. Checked before infer so a model's own shape error is never the client's 400."""
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
    """A `graphs` list, or a binary body with num_nodes/num_edges, to one batched feed set and
    the per-graph counts the response is split by. ValueError for anything the client got wrong."""
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
    """A validated request to the graph's own input arrays (base64 decode, to_numpy,
    tokenize) and the checks that need them: vocab, edge index, axis bounds. Runs in
    state.prep_executor. Returns the feeds, the milliseconds spent and, for a graph batch,
    its per-graph counts. `request.tensors` (a safetensors body, already decoded) takes the
    place of `inputs`; `request.graphs` (or a `tensors` body carrying num_nodes) is a batch of
    graphs, concatenated here into one feed set.

    Raises HTTPException(400) for anything the client got wrong (bad shape/dtype/JSON).
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
    # A batch was checked graph by graph against each graph's own node count.
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
    """Only the backend call; _encode_response follows it on the same inference thread.
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
    layout: GraphLayout | None = None,
) -> Response:
    """Outputs to the response body, right after _infer on the same thread. `timings_ms` already holds the
    earlier stages; the encode stage and the Server-Timing header are added here.

    A safetensors body (an `Accept` naming it, or output_encoding "safetensors") carries each
    output under its own name, with `predictions` and the embedding recipe as JSON strings in
    `__metadata__`.

    `layout` (a graph batch) splits the outputs per graph first: the body is
    {"graphs": [{outputs, shapes, dtypes}, ...]} in request order, or in safetensors each
    output named `graphs.<i>.<output>` with the graph count in `downshift.graphs`."""
    encoding = encoding or state.options.output_encoding
    encode = _base64_ready if encoding == OutputEncoding.base64 else _json_ready
    safetensors = accept_safetensors or encoding == RequestOutputEncoding.safetensors
    start = time.perf_counter()
    # C-contiguous once, up front (np.require keeps 0-d arrays 0-d; ascontiguousarray does not).
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
        # Same keys as PredictResponse; built by hand so orjson serializes the buffers directly.
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
    """orjson, then the route's pydantic model, with FastAPI's own 422s for both: an empty
    body is the missing body, malformed JSON is json_invalid, the model's errors sit under
    "body". The error dicts copy what FastAPI's request handler builds (fastapi/routing.py,
    get_request_handler), which has no public helper for them; tests pin all three."""
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
            # No dtype hints needed: to_numpy takes the backend's declared dtype (int64 for
            # edge_index on both backends), and integer lists default to int64 anyway.
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
            "this model takes tensors only: it was not loaded from a Hugging Face repo "
            "directory with tokenizer files. Send 'inputs' (see GET /schema).",
        )
    return _Request(
        inputs=req.inputs,
        text=as_batch(req.text) if req.text is not None else None,
        prompt=resolve_prompt(state, req.prompt_name, req.text is not None),
        encoding=req.output_encoding,
    )


def _binary_request(state: ServingState, graph: bool, body: bytes) -> _Request:
    """A safetensors body: its arrays (only names this route takes), and the __metadata__
    output_encoding."""
    try:
        arrays, metadata = decode_safetensors(body, max_input_bytes=state.options.max_input_bytes)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if "text" in metadata:
        raise HTTPException(400, "text is JSON only; a safetensors body carries tensors")
    allowed = _GRAPH_TENSOR_NAMES if graph else frozenset(state.input_names)
    unknown = [name for name in arrays if name not in allowed]
    if unknown:
        raise HTTPException(
            400, f"unknown tensor name {unknown[0]!r:.64}; this route takes {sorted(allowed)}"
        )
    encoding = metadata.get("output_encoding")
    try:
        return _Request(
            tensors=arrays, encoding=RequestOutputEncoding(encoding) if encoding else None
        )
    except ValueError:
        raise HTTPException(
            400,
            f"unknown output_encoding {encoding!r:.64} in __metadata__; this route takes "
            f"{[e.value for e in RequestOutputEncoding]}",
        ) from None


def _parse(
    state: ServingState, body: bytes, graph: bool, binary: bool, timings_ms: dict[str, float]
) -> _Request:
    """The request body (JSON or safetensors) to a validated _Request, its time recorded as
    timings_ms["parse"]. Raises the client's 422 or 400."""
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
    """fn(*args) in `executor`, carrying contextvars (run_in_executor does not; without this,
    log lines from the executor thread would lose the request id). The time from submit to
    start is timings_ms[wait_key]. `executor` None (--execution inline) calls fn right here on
    the event loop, with no wait."""
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
    """One /predict (or, with `graph`, /predict/graph) request, from its raw body, in two
    hops: _parse + _prepare_feeds in the prep pool, then _infer + _encode_response on an
    inference thread, so encoding holds the inference slot. The event loop does neither, so a
    large body never blocks /health or /ready (they share this process's single event loop).
    Admission already happened in PredictRoute, before the body was even read (P1);
    `admitted_at` is when that happened, and the request keeps its slot until the response
    is built.

    `binary` means a safetensors body. `inline` (--execution inline, a small JSON body) runs
    everything on the event loop instead, with zero waits; a `text` request still tokenizes in
    the prep pool.

    `timings_ms` collects the per-stage milliseconds (parse, prep_wait, prep, infer_wait,
    infer, encode) for Server-Timing and the request log line (U1). With
    `accept_safetensors` the response is safetensors.

    A graph batch (a `graphs` list, or a safetensors body with num_nodes/num_edges) is
    batched in prep, run as one inference and split in encode.
    """
    parsed = _parse(state, body, graph, binary, timings_ms) if inline else None
    run_here = parsed is not None and parsed.text is None

    def prepare() -> tuple[Any, dict[str, np.ndarray], GraphLayout | None]:
        # Only the encoding outlives this: the parsed body is freed before inference.
        request = parsed or _parse(state, body, graph, binary, timings_ms)
        feeds, prep_ms, layout = _prepare_feeds(state, request)
        timings_ms["prep"] = round(prep_ms, 2)
        return request.encoding, feeds, layout

    encoding, feeds, layout = await _run_in(
        None if run_here else state.prep_executor, "prep_wait", timings_ms, prepare
    )

    _check_timeout(state, admitted_at)

    def infer_and_encode() -> Response:
        # Re-checked at start: the request may have queued behind a running inference.
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
