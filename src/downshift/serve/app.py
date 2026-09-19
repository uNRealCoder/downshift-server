"""FastAPI app over a ServingState. The same routes regardless of which backend is behind it."""

import asyncio
import logging
import time
import uuid
from collections.abc import Callable, Coroutine, Sequence
from functools import partial
from typing import Any

import numpy as np
import orjson
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

import downshift
from downshift.serve.backends import InferenceInputError
from downshift.serve.codec import b64encode
from downshift.serve.engine import ServingState
from downshift.serve.middleware import load_middleware
from downshift.serve.schemas import (
    GraphPredictRequest,
    HealthResponse,
    MetadataResponse,
    OutputEncoding,
    PredictRequest,
    PredictResponse,
    ReadyResponse,
    to_numpy,
)

logger = logging.getLogger("downshift.serve")

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


class _OrjsonRequest(Request):
    async def json(self) -> Any:
        if not hasattr(self, "_json"):
            self._json = orjson.loads(await self.body())
        return self._json


def _body_too_large(observed: int, limit: int) -> JSONResponse:
    return JSONResponse(
        {
            "detail": (
                f"request body is {observed} bytes; the server limit is {limit} bytes "
                "(--max-body-bytes)"
            )
        },
        status_code=413,
    )


class OrjsonRoute(APIRoute):
    """Parse request bodies with orjson: several times faster than stdlib on MiB-scale bodies.

    orjson's JSONDecodeError subclasses the stdlib one, so malformed JSON still maps to 422.
    Also enforces --max-body-bytes: a Content-Length over the limit is rejected without
    reading the body; without one (chunked), the body is read and checked as it lands.
    """

    def get_route_handler(self) -> Callable[[Request], Coroutine[Any, Any, Response]]:
        handler = super().get_route_handler()

        async def route_handler(request: Request) -> Response:
            state: ServingState = request.app.state.serving
            limit = state.options.max_body_bytes
            content_length = request.headers.get("content-length")
            wrapped = _OrjsonRequest(request.scope, request.receive)
            if content_length is not None and content_length.isdigit():
                if int(content_length) > limit:
                    return _body_too_large(int(content_length), limit)
            else:
                body = await request.body()
                if len(body) > limit:
                    return _body_too_large(len(body), limit)
                wrapped._body = body
            return await handler(wrapped)

        return route_handler


# The predict routes: orjson response, and PredictResponse kept in the OpenAPI schema without
# the response_model re-validation walk over every output.
_PREDICT_ROUTE: dict[str, Any] = {
    "response_class": NumpyJSONResponse,
    "responses": {200: {"model": PredictResponse}},
}


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


class QueuedTooLong(HTTPException):
    """Raised inside the executor when a request waited past --request-timeout before its
    conversion/inference even started. Never raised once infer() has begun."""

    def __init__(self, waited: float, limit: float) -> None:
        super().__init__(
            503, f"request waited {waited:.1f}s in queue, past the {limit:.1f}s --request-timeout"
        )


def _predict_body(
    state: ServingState,
    inputs: dict[str, Any],
    encoding: OutputEncoding | None,
    admitted_at: float,
) -> NumpyJSONResponse:
    """Runs in state.executor, never on the event loop: conversion of a large JSON body,
    the inference itself, and response encoding. Admission and the cheap missing-input check
    already happened on the loop by the time this is submitted.

    Raises HTTPException(400) for anything the client got wrong (bad shape/dtype/JSON).
    Anything else (a backend bug, OOM, ...) propagates so the app-level handler turns it into
    a 500 without leaking the exception text to the client.
    """
    request_timeout = state.options.request_timeout
    if request_timeout > 0:
        waited = time.monotonic() - admitted_at
        if waited > request_timeout:
            raise QueuedTooLong(waited, request_timeout)

    declared = state.declared_dtypes
    max_bytes = state.options.max_input_bytes
    codec_start = time.perf_counter()
    try:
        feeds = {
            n: to_numpy(n, inputs[n], declared.get(n), max_bytes=max_bytes)
            for n in state.input_names
        }
    except (ValueError, TypeError, KeyError) as exc:
        raise HTTPException(400, str(exc)) from exc
    codec_ms = (time.perf_counter() - codec_start) * 1000

    infer_start = time.perf_counter()
    try:
        outputs = state.backend.infer(feeds)
    except HTTPException:
        raise
    except (InferenceInputError, KeyError) as exc:
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
    codec_ms += (time.perf_counter() - encode_start) * 1000

    response = NumpyJSONResponse(body)
    response.headers["Server-Timing"] = f"codec;dur={codec_ms:.2f}, infer;dur={infer_ms:.2f}"
    return response


def _capacity_error(state: ServingState) -> HTTPException:
    running = min(state.in_flight, state.options.max_concurrency)
    queued = max(0, state.in_flight - state.options.max_concurrency)
    return HTTPException(
        503,
        f"server is at capacity ({running} running, {queued} queued)",
        headers={"Retry-After": "1"},
    )


async def run_predict(
    state: ServingState, inputs: dict[str, Any], encoding: OutputEncoding | None
) -> NumpyJSONResponse:
    """Missing-input check and admission happen here, on the loop; everything else runs in
    state.executor (see _predict_body) so a slow body or a slow backend never blocks /health
    or /ready, which share this process's single event loop.
    """
    missing = [n for n in state.input_names if n not in inputs]
    if missing:
        raise HTTPException(400, f"missing inputs: {missing}")

    if not state.try_admit():
        raise _capacity_error(state)
    try:
        loop = asyncio.get_running_loop()
        body = partial(_predict_body, state, inputs, encoding, time.monotonic())
        return await loop.run_in_executor(state.executor, body)
    finally:
        state.release()


class RequestIdMiddleware:
    """Pure ASGI, not BaseHTTPMiddleware (which buffers the whole response body to let a
    handler rewrite headers, extra copies this doesn't need): reads X-Request-Id or makes
    one, stores it on request.state, and echoes it on the response. "See the server log"
    then has something to search for.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = Headers(scope=scope).get("x-request-id") or uuid.uuid4().hex[:16]
        scope.setdefault("state", {})["request_id"] = request_id

        async def send_with_request_id(message: Message) -> None:
            if message["type"] == "http.response.start":
                MutableHeaders(scope=message).append("x-request-id", request_id)
            await send(message)

        await self.app(scope, receive, send_with_request_id)


def build_app(state: ServingState, middleware: Sequence[str] = ()) -> FastAPI:
    app = FastAPI(title="downshift", version=downshift.__version__)
    app.router.route_class = OrjsonRoute
    app.state.serving = state
    app.add_middleware(RequestIdMiddleware)
    load_middleware(app, middleware)

    @app.exception_handler(Exception)
    async def _unhandled_exception(request: Request, exc: Exception) -> JSONResponse:
        # ServerErrorMiddleware, which dispatches this handler, sits outside
        # RequestIdMiddleware (Starlette always makes it the outermost layer), so its
        # response bypasses our send wrapper; set the header here too, directly on the
        # response, rather than relying on that wrapper for this one path.
        request_id = getattr(request.state, "request_id", None)
        logger.exception(
            "unhandled exception serving %s %s (request_id=%s)",
            request.method,
            request.url.path,
            request_id,
        )
        return JSONResponse(
            {
                "detail": "inference failed on the server; see the server log",
                "request_id": request_id,
            },
            status_code=500,
            headers={"X-Request-Id": request_id} if request_id else None,
        )

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse()

    @app.get("/ready", response_model=ReadyResponse)
    async def ready() -> JSONResponse:
        status = 200 if state.ready else 503
        return JSONResponse({"ready": state.ready}, status_code=status)

    @app.get("/metadata", response_model=MetadataResponse)
    async def metadata() -> MetadataResponse:
        return MetadataResponse(
            model=state.source,
            family=state.verdict.model_family,
            verdict=state.verdict.to_dict(),
            backend=state.backend.metadata().to_dict(),
            input_names=list(state.input_names),
            notes=list(state.notes),
            version=downshift.__version__,
            limits={
                "max_body_bytes": state.options.max_body_bytes,
                "max_input_bytes": state.options.max_input_bytes,
                "max_concurrency": state.options.max_concurrency,
                "max_queue": state.options.max_queue,
                "request_timeout": state.options.request_timeout,
            },
        )

    @app.post("/predict", **_PREDICT_ROUTE)
    async def predict(req: PredictRequest) -> NumpyJSONResponse:
        return await run_predict(state, req.inputs, req.output_encoding)

    @app.post("/predict/graph", **_PREDICT_ROUTE)
    async def predict_graph(req: GraphPredictRequest) -> NumpyJSONResponse:
        if not {"x", "edge_index"} <= set(state.input_names):
            raise HTTPException(
                400,
                f"model is not graph-shaped: inputs are {list(state.input_names)}, "
                "expected at least 'x' and 'edge_index'",
            )
        # No dtype hints needed: to_numpy takes the backend's declared dtype (int64 for
        # edge_index on both backends), and integer lists default to int64 anyway.
        inputs: dict[str, Any] = {"x": req.x, "edge_index": req.edge_index}
        if req.edge_attr is not None:
            inputs["edge_attr"] = req.edge_attr
        return await run_predict(state, inputs, req.output_encoding)

    return app
