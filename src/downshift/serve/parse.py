"""The request body of a predict: JSON or safetensors, to a ParsedRequest."""

from __future__ import annotations

import time

import orjson
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, ValidationError

from downshift.entities.options import RequestOutputEncoding
from downshift.entities.requests import ParsedRequest
from downshift.entities.serving import ServingState
from downshift.serve.codec import decode_safetensors
from downshift.serve.errors import ApiError
from downshift.serve.feeds import as_batch, resolve_prompt
from downshift.serve.graphs import GRAPH_COUNTS, GRAPH_INPUTS, GRAPH_TENSORS
from downshift.serve.schemas import GraphPredictRequest, PredictRequest

# The names that a safetensors body on /predict/graph can use.
_GRAPH_TENSOR_NAMES = frozenset((*GRAPH_TENSORS, *GRAPH_COUNTS))


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


def _json_request(state: ServingState, graph: bool, body: bytes) -> ParsedRequest:
    if graph:
        graph_req = _validate_json(GraphPredictRequest, body)
        if graph_req.graphs is not None:
            graphs = [
                {"x": g.x, "edge_index": g.edge_index, "edge_attr": g.edge_attr}
                for g in graph_req.graphs
            ]
            return ParsedRequest(graphs=graphs, encoding=graph_req.output_encoding)
        inputs = {"x": graph_req.x, "edge_index": graph_req.edge_index}
        if graph_req.edge_attr is not None:
            inputs["edge_attr"] = graph_req.edge_attr
        return ParsedRequest(inputs=inputs, encoding=graph_req.output_encoding)

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
    return ParsedRequest(
        inputs=req.inputs,
        text=text,
        prompt=resolve_prompt(state, req.prompt_name, req.text is not None),
        encoding=req.output_encoding,
    )


def _binary_request(state: ServingState, graph: bool, body: bytes) -> ParsedRequest:
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
        return ParsedRequest(
            tensors=arrays, encoding=RequestOutputEncoding(encoding) if encoding else None
        )
    except ValueError:
        raise ApiError(
            400,
            "invalid_input",
            f"unknown output_encoding {encoding!r:.64} in __metadata__. This route takes "
            f"{[e.value for e in RequestOutputEncoding]}",
        ) from None


def parse_request(
    state: ServingState, body: bytes, graph: bool, binary: bool, timings_ms: dict[str, float]
) -> ParsedRequest:
    """The body (JSON or safetensors) as a validated ParsedRequest, or a 422 or 400."""
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
