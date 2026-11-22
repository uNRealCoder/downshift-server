"""The FastAPI app over a ServingState. The routes are the same for both backends.

To read this file, go in this order:

1. `build_app`: makes the app. With `loader=`, the app binds first and the lifespan
   (`_loader_lifespan`) loads the model on a background thread. Until then, `/ready` is 503.
2. `PredictRoute`: admits a slot (or 503), applies --request-timeout, and releases the slot.
3. `_predict`: checks the content type and the body size, then calls `run_predict`
   (serve/predict.py). The request path continues there.
4. `RequestIdMiddleware`, `ApiKeyMiddleware`, `load_middleware`: the middleware stack, from the
   outside in.
"""

import asyncio
import contextvars
import logging
import os
import signal
import threading
import time
from collections.abc import AsyncIterator, Callable, Coroutine, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import asdict
from functools import partial
from typing import Any

from fastapi import Depends, FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from starlette.exceptions import HTTPException as StarletteHTTPException

import downshift
from downshift import settings
from downshift.core.phase import CURRENT_PROGRESS
from downshift.entities.metrics import Metrics
from downshift.entities.options import ExecutionChoice
from downshift.entities.phase import LoadProgress
from downshift.entities.serving import ServingState
from downshift.logs import request_id_var
from downshift.serve.encode import BINARY_REQUEST_TYPES, SAFETENSORS_MEDIA_TYPE, NumpyJSONResponse
from downshift.serve.errors import (
    ApiError,
    error_response,
    http_error_handler,
    validation_error_handler,
)
from downshift.serve.metrics import CONTENT_TYPE, render
from downshift.serve.middleware import ApiKeyMiddleware, RequestIdMiddleware, load_middleware
from downshift.serve.openai import (
    embeddings_response,
    model_card,
    parse_embeddings,
    require_model,
)
from downshift.serve.predict import run_predict
from downshift.serve.schemas import (
    BackendInfo,
    CacheInfo,
    EmbeddingsRequest,
    GraphPredictRequest,
    HealthResponse,
    IndexResponse,
    MetadataResponse,
    PredictRequest,
    PredictResponse,
    ReadyResponse,
    SchemaResponse,
    VerdictInfo,
)
from downshift.settings import API_KEY
from downshift.sources import LABEL_KINDS, display_source, hide_paths

logger = logging.getLogger("downshift.serve")

# The Retry-After seconds of a 503: while the model loads, and when all slots are in use.
NOT_READY_RETRY_AFTER = 2
CAPACITY_RETRY_AFTER = 1


# The largest body that --execution inline parses on the event loop. A chunked body (no
# Content-Length) always goes to the pools.
INLINE_MAX_BODY_BYTES = 64 * 1024

_UNSUPPORTED_MEDIA_TYPE = (
    "unsupported Content-Type; send JSON (application/json) or safetensors "
    f"({SAFETENSORS_MEDIA_TYPE}, or application/octet-stream)"
)


def _body_too_large(request: Request, observed: int, limit: int) -> JSONResponse:
    return error_response(
        request.scope,
        413,
        "body_too_large",
        f"request body is {observed} bytes; the server limit is {limit} bytes (--max-body-bytes)",
    )


def _not_ready_error() -> ApiError:
    return ApiError(
        503,
        "not_ready",
        "model is not ready",
        headers={"Retry-After": str(NOT_READY_RETRY_AFTER)},
    )


def _capacity_error(state: ServingState) -> ApiError:
    running = min(state.in_flight, state.options.max_concurrency)
    queued = max(0, state.in_flight - state.options.max_concurrency)
    return ApiError(
        503,
        "at_capacity",
        f"server is at capacity ({running} running, {queued} queued)",
        headers={"Retry-After": str(CAPACITY_RETRY_AFTER)},
    )


async def _read_body_limited(request: Request, limit: int) -> bytes | JSONResponse:
    """The streamed body, or a 413 as soon as the running total passes `limit`."""
    chunks: list[bytes] = []
    total = 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > limit:
            return _body_too_large(request, total, limit)
        chunks.append(chunk)
    return b"".join(chunks)


def _content_type(request: Request) -> str:
    return request.headers.get("content-type", "").split(";")[0].strip().lower()


async def _predict(
    request: Request,
    state: ServingState,
    graph: bool,
    is_embeddings: bool,
    deadline: float | None,
) -> Response:
    """An admitted predict: check the content type and --max-body-bytes (before reading, if
    Content-Length is given), then run_predict on the raw body. The stage timings go to the
    scope state, for the access log."""
    content_type = _content_type(request)
    binary = content_type in BINARY_REQUEST_TYPES
    if content_type and not binary and "json" not in content_type:
        return error_response(request.scope, 415, "unsupported_media_type", _UNSUPPORTED_MEDIA_TYPE)
    limit = state.options.max_body_bytes
    content_length = request.headers.get("content-length")
    if content_length is not None and content_length.isdigit():
        if int(content_length) > limit:
            return _body_too_large(request, int(content_length), limit)
        body = await request.body()
    else:
        read = await _read_body_limited(request, limit)
        if isinstance(read, JSONResponse):
            return read
        body = read

    timings: dict[str, float] = {}
    request.state.timings_ms = timings
    parse = partial(parse_embeddings, state, body, timings) if is_embeddings else None
    encode = partial(embeddings_response, state, timings) if is_embeddings else None
    response = await run_predict(
        state,
        body,
        graph=graph,
        binary=binary,
        inline=state.options.execution == ExecutionChoice.inline
        and not binary
        and content_length is not None
        and len(body) <= INLINE_MAX_BODY_BYTES,
        deadline=deadline,
        timings_ms=timings,
        accept_safetensors=SAFETENSORS_MEDIA_TYPE in request.headers.get("accept", "").lower(),
        use_cache="no-cache" not in request.headers.get("cache-control", "").lower(),
        parse=parse,
        encode=encode,
        with_tokens=is_embeddings,
    )
    if request.app.state.metrics is not None:  # only the request counters read it
        request.state.cache = response.headers.get(settings.CACHE_HEADER)
    return response


class PredictRoute(APIRoute):
    """/predict, /predict/graph and /v1/embeddings, served from the raw body: FastAPI does not
    parse it, and run_predict does so off the event loop. A slot is admitted first, so an overloaded server
    answers 503 before it reads a byte. One --request-timeout limit covers the whole request,
    the body read included. A thread that already runs is not stopped. `finally` releases the
    slot on every exit."""

    def get_route_handler(self) -> Callable[[Request], Coroutine[Any, Any, Response]]:
        graph = self.path == "/predict/graph"
        is_embeddings = self.path == "/v1/embeddings"

        async def route_handler(request: Request) -> Response:
            state: ServingState | None = request.app.state.serving
            if state is None:
                raise _not_ready_error()
            if not state.try_admit():
                raise _capacity_error(state)
            timeout = state.options.request_timeout
            deadline = time.monotonic() + timeout if timeout > 0 else None
            limit: asyncio.Timeout | None = None
            try:
                call = _predict(request, state, graph, is_embeddings, deadline)
                if deadline is None:
                    return await call
                async with asyncio.timeout(timeout) as limit:
                    return await call
            except TimeoutError:
                if limit is None or not limit.expired():
                    raise  # a timeout inside a backend or user code is not ours
                raise ApiError(
                    503,
                    "request_timeout",
                    f"request took more than the {timeout:.1f}s --request-timeout",
                ) from None
            finally:
                state.release()

        return route_handler


async def predict(req: PredictRequest | None = None) -> Response:
    """Never called (PredictRoute serves the route): documents the JSON body for OpenAPI."""
    raise NotImplementedError


async def predict_graph(req: GraphPredictRequest | None = None) -> Response:
    """Never called (PredictRoute serves the route): documents the JSON body for OpenAPI."""
    raise NotImplementedError


async def embeddings(req: EmbeddingsRequest | None = None) -> Response:
    """Never called (PredictRoute serves the route): documents the JSON body for OpenAPI."""
    raise NotImplementedError


# PredictResponse documents the JSON body without a response_model validation of each output.
_BINARY_BODY = {"schema": {"type": "string", "format": "binary"}}
_PREDICT_ROUTE: dict[str, Any] = {
    "response_class": NumpyJSONResponse,
    "responses": {
        200: {
            "model": PredictResponse,
            "content": {SAFETENSORS_MEDIA_TYPE: _BINARY_BODY},
        }
    },
    "openapi_extra": {
        "requestBody": {
            "content": {
                SAFETENSORS_MEDIA_TYPE: _BINARY_BODY,
                "application/octet-stream": _BINARY_BODY,
            }
        }
    },
}


def _routes(request: Request) -> dict[str, str]:
    """{name: "METHOD /path"} for each route of the app (with the mount prefix), plus the
    OpenAPI pages."""
    app = request.app
    prefix = request.scope.get("root_path", "")
    routes = {
        route.name: f"{min(route.methods)} {prefix}{route.path}"
        for route in app.routes
        if isinstance(route, APIRoute) and route.methods
    }
    if app.openapi_url:
        routes["openapi"] = f"GET {prefix}{app.openapi_url}"
    if app.docs_url:
        routes["docs"] = f"GET {prefix}{app.docs_url}"
    return routes


def require_serving(request: Request) -> ServingState:
    """The loaded model, or a 503. A dependency of each route that needs the model."""
    current: ServingState | None = request.app.state.serving
    if current is None:
        raise _not_ready_error()
    return current


def _flag_stopping_on_signal(app: FastAPI) -> None:
    """Set app.state.stopping when a stop signal comes, then run the handler of uvicorn.

    uvicorn runs the lifespan shutdown only after the connections drain. /ready must turn 503
    before that, so the load balancer stops new requests. This works in each `--workers`
    process too. uvicorn installs its handlers before the lifespan starts. Only the main
    thread can set a handler: in any other thread (a test client) this does nothing.
    """
    if threading.current_thread() is not threading.main_thread():
        return
    from uvicorn.server import HANDLED_SIGNALS

    def wrap(previous: Callable[[int, Any], Any]) -> Callable[[int, Any], None]:
        def handler(sig: int, frame: Any) -> None:
            app.state.stopping = True
            previous(sig, frame)

        return handler

    for sig in HANDLED_SIGNALS:
        previous = signal.getsignal(sig)
        if callable(previous):
            signal.signal(sig, wrap(previous))


def _loader_lifespan(
    loader: Callable[[], ServingState],
) -> Callable[[FastAPI], AbstractAsyncContextManager[None]]:
    """Run `loader` on a background thread when the app starts to serve, and put its result on
    app.state.serving. If it raises, the state stays None: the caller must end the process."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        progress = LoadProgress()
        app.state.load_progress = progress
        _flag_stopping_on_signal(app)

        def run() -> None:
            # In the thread's own Context, report() calls deep in core reach `progress`.
            CURRENT_PROGRESS.set(progress)
            try:
                app.state.serving = loader()
            except Exception:
                # The caller reports the error. Log it only at DEBUG, to print it once.
                logger.debug("model failed to load; /ready will not turn 200", exc_info=True)

        thread = threading.Thread(
            target=lambda: contextvars.Context().run(run), name="downshift-loader", daemon=True
        )
        app.state.loader_thread = thread
        thread.start()
        yield

    return lifespan


def build_app(
    state: ServingState | None = None,
    *,
    loader: Callable[[], ServingState] | None = None,
    middleware: Sequence[str] = (),
    api_key: str | None = API_KEY,
    access_log: bool = True,
    metrics: bool = False,
) -> FastAPI:
    """The app over `state`, or over the result of `loader`. With a loader, the app binds at
    once: /health is 200, and /ready and the model routes are 503 until the loader finishes.

    `api_key` defaults to DOWNSHIFT_API_KEY. None or "" turns it off (with a warning).
    `access_log=False` drops the request lines of `downshift.access`.
    `metrics=True` counts the requests and adds GET /metrics.
    """
    if (state is None) == (loader is None):
        raise ValueError("build_app needs exactly one of state or loader")

    if "DOWNSHIFT_SERVER_API_KEY" in os.environ:
        logger.warning(
            "DOWNSHIFT_SERVER_API_KEY is the old name and stops working in 1.1 at the earliest. "
            + (
                "Both names are set, so downshift uses DOWNSHIFT_API_KEY. Remove the old name."
                if "DOWNSHIFT_API_KEY" in os.environ
                else "Set DOWNSHIFT_API_KEY instead."
            )
        )
    api_key = api_key or None
    if api_key is None:
        logger.warning(
            "DOWNSHIFT_API_KEY is not set, so the endpoints (including /metadata) are "
            "unauthenticated. Set it to require a fixed API key, or add your own "
            "authentication middleware."
        )

    app = FastAPI(
        title="downshift",
        version=downshift.__version__,
        lifespan=_loader_lifespan(loader) if loader is not None else None,
    )
    app.state.serving = state
    app.state.stopping = False
    # The last middleware added is the outermost. User middleware sits next to the routes and
    # sees only authenticated traffic. The request ID wraps everything, the 401s included.
    load_middleware(app, middleware)
    app.add_middleware(ApiKeyMiddleware, api_key=api_key)
    request_id_header = settings.REQUEST_ID_HEADER
    counters = Metrics() if metrics else None
    app.state.metrics = counters
    app.add_middleware(
        RequestIdMiddleware, access_log=access_log, header=request_id_header, metrics=counters
    )

    app.add_exception_handler(StarletteHTTPException, http_error_handler)  # type: ignore[arg-type]
    app.add_exception_handler(RequestValidationError, validation_error_handler)  # type: ignore[arg-type]

    @app.exception_handler(Exception)
    async def _unhandled_exception(request: Request, exc: Exception) -> JSONResponse:
        # This runs outside RequestIdMiddleware, which has already reset request_id_var and
        # cannot add its header here.
        request_id = getattr(request.state, "request_id", None)
        if isinstance(exc, StarletteHTTPException):
            # A user middleware sits outside the exception middleware of the routes.
            response = await http_error_handler(request, exc)
            if request_id:
                response.headers[request_id_header] = request_id
            return response
        token = request_id_var.set(request_id)
        try:
            logger.exception(
                "unhandled exception serving %s %s (request_id=%s)",
                request.method,
                request.url.path,
                request_id,
            )
        finally:
            request_id_var.reset(token)
        return error_response(
            request.scope,
            500,
            "internal_error",
            "inference failed on the server; see the server log",
            {request_id_header: request_id} if request_id else None,
            request_id=request_id,
        )

    @app.get("/", response_model=IndexResponse)
    async def index(request: Request) -> IndexResponse:
        current: ServingState | None = request.app.state.serving
        model = None
        if current is not None and current.ready:
            model = display_source(current.source, current.source_kind)
        return IndexResponse(
            version=downshift.__version__,
            ready=model is not None,
            model=model,
            routes=_routes(request),
        )

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse()

    @app.get("/ready", response_model=ReadyResponse)
    async def ready(request: Request) -> JSONResponse:
        if request.app.state.stopping:
            return JSONResponse({"ready": False, "phase": "stopping"}, status_code=503)
        current: ServingState | None = request.app.state.serving
        if current is None:
            progress: LoadProgress | None = getattr(request.app.state, "load_progress", None)
            phase = progress.phase.value if progress is not None else None
            return JSONResponse({"ready": False, "phase": phase}, status_code=503)
        status = 200 if current.ready else 503
        return JSONResponse({"ready": current.ready}, status_code=status)

    @app.get("/metadata", response_model=MetadataResponse)
    async def metadata(current: ServingState = Depends(require_serving)) -> MetadataResponse:
        # Free text can quote local paths (ORT and transformers errors): show names only.
        located = () if current.source_kind in LABEL_KINDS else (current.source,)
        paths = (*located, current.verdict.onnx_path, current.hf_source)
        from downshift.serve.describe import axes_info, limits_info

        return MetadataResponse(
            model=display_source(current.source, current.source_kind),
            family=current.verdict.model_family,
            verdict=VerdictInfo.model_validate(current.verdict.redacted_dict(paths)),
            axes=axes_info(current),
            backend=BackendInfo.model_validate(current.backend.metadata().to_dict()),
            input_names=list(current.input_names),
            notes=[hide_paths(note, paths) for note in current.notes],
            version=downshift.__version__,
            limits=limits_info(current),
            cache=(
                CacheInfo(**current.result_cache.stats())
                if current.result_cache is not None
                else None
            ),
            execution=current.options.execution.value,
            boot=dict(current.timings),
            warmup=asdict(current.warmup_stats) if current.warmup_stats is not None else None,
            serving_numerics=(
                current.serving_numerics.to_dict() if current.serving_numerics else None
            ),
        )

    @app.get("/schema", response_model=SchemaResponse)
    async def schema(
        request: Request, current: ServingState = Depends(require_serving)
    ) -> SchemaResponse:
        """What to POST: the name, dtype and shape of each input, and an example body."""
        from downshift.serve.describe import describe

        return describe(current, f"{str(request.base_url).rstrip('/')}/predict")

    @app.get("/version")
    async def version() -> dict[str, str]:
        return {"version": downshift.__version__}

    @app.get("/v1/models")
    async def list_models(current: ServingState = Depends(require_serving)) -> dict[str, Any]:
        return {"object": "list", "data": [model_card(current)]}

    @app.get("/v1/models/{model_id:path}")
    async def get_model(
        model_id: str, current: ServingState = Depends(require_serving)
    ) -> dict[str, Any]:
        require_model(current, model_id)
        return model_card(current)

    if counters is not None:

        @app.get("/metrics", include_in_schema=False)
        async def metrics_text(request: Request) -> Response:
            text = render(counters, request.app.state.serving)
            return Response(text, media_type=CONTENT_TYPE)

    for path, endpoint, options in (
        ("/predict", predict, _PREDICT_ROUTE),
        ("/predict/graph", predict_graph, _PREDICT_ROUTE),
        ("/v1/embeddings", embeddings, {"response_class": NumpyJSONResponse}),
    ):
        app.router.add_api_route(
            path, endpoint, methods=["POST"], route_class_override=PredictRoute, **options
        )

    return app
