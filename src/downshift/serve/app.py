"""The FastAPI app over a ServingState. The routes are the same for both backends."""

import contextvars
import hmac
import logging
import threading
import time
import uuid
from collections.abc import AsyncIterator, Callable, Coroutine, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import asdict
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

import downshift
from downshift.core.phase import CURRENT_PROGRESS, LoadProgress
from downshift.logs import request_id_var
from downshift.serve.engine import ServingState
from downshift.serve.middleware import load_middleware
from downshift.serve.options import ExecutionChoice
from downshift.serve.predict import (
    BINARY_REQUEST_TYPES,
    SAFETENSORS_MEDIA_TYPE,
    NumpyJSONResponse,
    run_predict,
)
from downshift.serve.schemas import (
    BackendInfo,
    GraphPredictRequest,
    HealthResponse,
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
access_logger = logging.getLogger("downshift.access")

NOT_READY_RETRY_AFTER = 2  # the number of seconds that a client waits before it asks again

# Probes: the API key does not apply to them, and downshift logs them at DEBUG. They then do not flood the default output.
_PROBE_PATHS = frozenset({"/health", "/ready"})


def _route_path(scope: Scope) -> str:
    """The path relative to the place where this app is mounted. From Starlette 0.33,
    scope["path"] keeps the mount prefix (app_for(...) is made to be mounted). "/model/health"
    must therefore still count as the /health probe."""
    path: str = scope["path"]
    return path.removeprefix(scope.get("root_path", ""))


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


def _not_ready_error() -> HTTPException:
    """/metadata, /schema and the predict routes raise this while an app that a loader built
    has no ServingState yet (see the loader parameter of build_app)."""
    return HTTPException(
        503, "model is not ready", headers={"Retry-After": str(NOT_READY_RETRY_AFTER)}
    )


def _capacity_error(state: ServingState) -> HTTPException:
    running = min(state.in_flight, state.options.max_concurrency)
    queued = max(0, state.in_flight - state.options.max_concurrency)
    return HTTPException(
        503,
        f"server is at capacity ({running} running, {queued} queued)",
        headers={"Retry-After": "1"},
    )


async def _read_body_limited(request: Request, limit: int) -> bytes | JSONResponse:
    """request.stream(), with a running total. It gives a 413 when the total passes `limit`. It
    does not first collect the whole body (P2 - request.body() does this)."""
    chunks: list[bytes] = []
    total = 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > limit:
            return _body_too_large(total, limit)
        chunks.append(chunk)
    return b"".join(chunks)


# The largest body that --execution inline parses on the event loop. A chunked body (no
# Content-Length) always goes to the pools.
INLINE_MAX_BODY_BYTES = 64 * 1024


_UNSUPPORTED_MEDIA_TYPE = (
    "unsupported Content-Type; send JSON (application/json) or safetensors "
    f"({SAFETENSORS_MEDIA_TYPE}, or application/octet-stream)"
)


def _content_type(request: Request) -> str:
    return request.headers.get("content-type", "").split(";")[0].strip().lower()


async def _predict(
    request: Request,
    state: ServingState,
    graph: bool,
    admitted_at: float,
) -> Response:
    """An admitted predict: the content type, then --max-body-bytes, then run_predict on the raw
    body.

    If Content-Length is more than the limit, the server refuses the request without a read of
    the body. It reads a chunked body while it arrives. It refuses it when the running total
    passes the limit (P2), and not after the whole body arrived. The timings of each stage are in
    the state of the scope. RequestIdMiddleware reads them again from there for the request line
    (U1)."""
    content_type = _content_type(request)
    binary = content_type in BINARY_REQUEST_TYPES
    if content_type and not binary and "json" not in content_type:
        return JSONResponse({"detail": _UNSUPPORTED_MEDIA_TYPE}, status_code=415)
    limit = state.options.max_body_bytes
    content_length = request.headers.get("content-length")
    if content_length is not None and content_length.isdigit():
        if int(content_length) > limit:
            return _body_too_large(int(content_length), limit)
        body = await request.body()
    else:
        read = await _read_body_limited(request, limit)
        if isinstance(read, JSONResponse):
            return read
        body = read

    timings: dict[str, float] = {}
    request.state.timings_ms = timings
    return await run_predict(
        state,
        body,
        graph=graph,
        binary=binary,
        inline=state.execution == ExecutionChoice.inline
        and not binary
        and content_length is not None
        and len(body) <= INLINE_MAX_BODY_BYTES,
        admitted_at=admitted_at,
        timings_ms=timings,
        accept_safetensors=SAFETENSORS_MEDIA_TYPE in request.headers.get("accept", "").lower(),
    )


class PredictRoute(APIRoute):
    """/predict and /predict/graph. _predict serves them directly from the raw body. FastAPI does
    not resolve the body. run_predict parses it (orjson), validates it (the pydantic model of the
    route), and prepares it in one prep-pool hop. It never does this on the event loop (P3). The
    endpoint functions only document the JSON body in the OpenAPI schema.

    Downshift first admits a slot (P1). An overloaded server answers 503 without a buffer or a
    parse of one byte. Downshift releases the slot in `finally`. Each exit (413, 400, a
    validation error, a client that disconnects) therefore releases it exactly one time.
    """

    def get_route_handler(self) -> Callable[[Request], Coroutine[Any, Any, Response]]:
        graph = self.path == "/predict/graph"

        async def route_handler(request: Request) -> Response:
            state: ServingState | None = request.app.state.serving
            if state is None:
                raise _not_ready_error()
            if not state.try_admit():
                raise _capacity_error(state)
            try:
                return await _predict(request, state, graph, time.monotonic())
            finally:
                state.release()

        return route_handler


async def predict(req: PredictRequest | None = None) -> Response:
    """Never called (PredictRoute serves the route): documents the JSON body for OpenAPI."""
    raise NotImplementedError


async def predict_graph(req: GraphPredictRequest | None = None) -> Response:
    """Never called (PredictRoute serves the route): documents the JSON body for OpenAPI."""
    raise NotImplementedError


# The predict routes: an orjson response. PredictResponse stays in the OpenAPI schema, without
# the validation of response_model again over each output.
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


def require_serving(request: Request) -> ServingState:
    """The loaded model, or a 503.

    Each route that needs a ServingState depends on this. A route that you add later therefore
    cannot forget the guard by omission of a copy of it. /ready is not a caller, by design. Its
    answer is that the model is not loaded. This is not an error.
    """
    current: ServingState | None = request.app.state.serving
    if current is None:
        raise _not_ready_error()
    return current


class RequestIdMiddleware:
    """This is pure ASGI and not BaseHTTPMiddleware. BaseHTTPMiddleware buffers the whole
    response body to let a handler rewrite headers. This code does not need these extra copies.
    The middleware reads X-Request-Id or makes one, stores it on request.state, and returns it
    in the response. "See the server log" then has a search key.

    It also binds the ID to `request_id_var` for the request. Each log line that downshift
    emits while it serves the request then has the ID. With `access_log`, it writes the one
    request line (U1): method, path, status, duration and, for the predict routes, the split
    of parse, codec and infer. The routes leave the split in the state of the scope as
    "timings_ms". The level is WARNING from status 400, DEBUG for the probes /health and /ready
    (a 503 from /ready during the load is its normal answer), and INFO for all other requests.
    """

    def __init__(self, app: ASGIApp, access_log: bool = True) -> None:
        self.app = app
        self.access_log = access_log

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        start = time.perf_counter()
        request_id = Headers(scope=scope).get("x-request-id") or uuid.uuid4().hex[:16]
        state = scope.setdefault("state", {})
        state["request_id"] = request_id
        status: int | None = None

        async def send_with_request_id(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                MutableHeaders(scope=message).append("x-request-id", request_id)
            await send(message)

        token = request_id_var.set(request_id)
        try:
            await self.app(scope, receive, send_with_request_id)
        except Exception:
            # ServerErrorMiddleware, outside this middleware, turns it into the 500 response.
            if status is None:
                status = 500
            raise
        finally:
            try:
                if self.access_log and status is not None:
                    self._log(scope, status, (time.perf_counter() - start) * 1000, state)
            finally:
                request_id_var.reset(token)

    @staticmethod
    def _log(scope: Scope, status: int, duration_ms: float, state: dict[str, Any]) -> None:
        path = scope["path"]
        if _route_path(scope) in _PROBE_PATHS:
            level = logging.DEBUG
        elif status >= 400:
            level = logging.WARNING
        else:
            level = logging.INFO
        if not access_logger.isEnabledFor(level):
            return
        extra: dict[str, Any] = {
            "method": scope["method"],
            "path": path,
            "status": status,
            "duration_ms": round(duration_ms, 2),
        }
        timings = state.get("timings_ms")
        if timings:
            extra["timings_ms"] = timings
        access_logger.log(
            level, "%s %s %d %.1f ms", scope["method"], path, status, duration_ms, extra=extra
        )


class ApiKeyMiddleware:
    """DOWNSHIFT_SERVER_API_KEY (U7). Downshift reads it through settings.py, so it also applies
    to app_for() (ruling 5). None (unset) means no authentication. The caller already gave a
    warning about this at build_app time. This is pure ASGI, like RequestIdMiddleware.
    /health and /ready are exempt, so a probe never needs the key.
    """

    def __init__(self, app: ASGIApp, api_key: str | None) -> None:
        self.app = app
        self.api_key = api_key
        self._key_bytes = api_key.encode("utf-8") if api_key is not None else b""

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or self.api_key is None or _route_path(scope) in _PROBE_PATHS:
            await self.app(scope, receive, send)
            return

        header = Headers(scope=scope).get("authorization", "")
        scheme, _, token = header.partition(" ")
        # Starlette decodes header bytes as latin-1. Latin-1 therefore gives the raw bytes of the
        # client back. compare_digest on a str that is not ASCII would raise an error (a 500) and
        # would not fail the check.
        if scheme.lower() != "bearer" or not hmac.compare_digest(
            token.encode("latin-1"), self._key_bytes
        ):
            response = JSONResponse(
                {"detail": "Authorization header is not set or incorrect"},
                status_code=401,
                headers={"WWW-Authenticate": "Bearer"},
            )
            await response(scope, receive, send)
            return
        await self.app(scope, receive, send)


def _loader_lifespan(
    loader: Callable[[], ServingState],
) -> Callable[[FastAPI], AbstractAsyncContextManager[None]]:
    """Run `loader` on a background thread. It starts when the app begins to serve (and not when
    build_app is called). The result goes to app.state.serving. If the loader raises an error,
    app.state.serving stays None forever, as far as this function can see. The caller
    (serve_cmd) makes sure that this ends the process. It holds the uvicorn.Server and sets
    should_exit itself before it raises the error again.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        progress = LoadProgress()
        app.state.load_progress = progress

        def run() -> None:
            # This runs in a new Context (see below). This set() is therefore visible only to the
            # call stack of this thread. The calls that report the phase inside the builders of
            # engine.py (U4) read it again with CURRENT_PROGRESS.get(). Downshift does not pass
            # a parameter through each builder and through the own loader closure of the CLI.
            CURRENT_PROGRESS.set(progress)
            try:
                app.state.serving = loader()
            except Exception:
                # Not logged at the error or exception level. A caller with its own reporting
                # path (serve_cmd raises this error again on the main thread) would otherwise
                # print the traceback two times. Once here without a condition, and once with
                # the gate --log-level debug. --log-level debug still sees it here.
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
) -> FastAPI:
    """With `state`, the app serves it immediately (library use, tests). With `loader` instead,
    the app binds with no ServingState. /health is 200 immediately. /ready is 503. /metadata,
    /schema and the predict routes are 503 until the loader gives one. (The lifespan starts the
    loader on a background thread when the app is served.)

    If you set `api_key`, every route except /health and /ready needs "Authorization: Bearer
    <api_key>" (U7). It defaults to DOWNSHIFT_SERVER_API_KEY (settings.API_KEY). The
    environment variable then reaches app_for() and this function in the same way (ruling 5).
    To override that default for a library caller, pass a value (or None to turn it off). An
    empty string counts as unset. DOWNSHIFT_SERVER_API_KEY="" must not make "Bearer " a valid
    credential.

    `access_log` False removes the one request line that RequestIdMiddleware writes on the
    `downshift.access` logger (U1). The request ID is still set and returned.
    """
    if (state is None) == (loader is None):
        raise ValueError("build_app needs exactly one of state or loader")

    api_key = api_key or None
    if api_key is None:
        logger.warning(
            "DOWNSHIFT_SERVER_API_KEY is not set, so the endpoints (including /metadata) are "
            "unauthenticated. Set it to require a fixed API key, or add your own "
            "authentication middleware."
        )

    app = FastAPI(
        title="downshift",
        version=downshift.__version__,
        lifespan=_loader_lifespan(loader) if loader is not None else None,
    )
    app.state.serving = state
    # Starlette makes the *last* registered middleware the outermost layer. The registration
    # order here is therefore from back to front. The user middleware of load_middleware goes
    # first (innermost, directly next to the routes, so it sees only authenticated traffic).
    # ApiKeyMiddleware is next (the gate). RequestIdMiddleware is last (outermost). The access
    # log and the X-Request-Id header then cover 401 responses and all other responses in the
    # same way.
    load_middleware(app, middleware)
    app.add_middleware(ApiKeyMiddleware, api_key=api_key)
    app.add_middleware(RequestIdMiddleware, access_log=access_log)

    @app.exception_handler(Exception)
    async def _unhandled_exception(request: Request, exc: Exception) -> JSONResponse:
        # ServerErrorMiddleware dispatches this handler. It is outside RequestIdMiddleware
        # (Starlette always makes it the outermost layer). Its response therefore bypasses our
        # send wrapper. Set the header here too, directly on the response. Do not rely on that
        # wrapper for this one path.
        request_id = getattr(request.state, "request_id", None)
        # RequestIdMiddleware has already reset request_id_var when this code runs.
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
    async def ready(request: Request) -> JSONResponse:
        current: ServingState | None = request.app.state.serving
        if current is None:
            progress: LoadProgress | None = getattr(request.app.state, "load_progress", None)
            phase = progress.phase.value if progress is not None else None
            return JSONResponse({"ready": False, "phase": phase}, status_code=503)
        status = 200 if current.ready else 503
        return JSONResponse({"ready": current.ready}, status_code=status)

    @app.get("/metadata", response_model=MetadataResponse)
    async def metadata(current: ServingState = Depends(require_serving)) -> MetadataResponse:
        # Free-text fields quote the path that an exception received (the "Load model from ..."
        # of ORT, the "Can't load tokenizer for ..." of transformers). A client gets the name
        # only.
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
            execution=current.execution.value,
            boot=dict(current.timings),
            warmup=asdict(current.warmup_stats) if current.warmup_stats is not None else None,
        )

    @app.get("/schema", response_model=SchemaResponse)
    async def schema(
        request: Request, current: ServingState = Depends(require_serving)
    ) -> SchemaResponse:
        """What to POST: the name, dtype and shape of each input, and an example body.

        It is cheap to build for each request (it reads the IO that the backend declares and
        fills a small example). Nothing is cached, so it always matches the running graph.
        """
        from downshift.serve.describe import describe

        return describe(current, f"{str(request.base_url).rstrip('/')}/predict")

    for path, endpoint in (("/predict", predict), ("/predict/graph", predict_graph)):
        app.router.add_api_route(
            path, endpoint, methods=["POST"], route_class_override=PredictRoute, **_PREDICT_ROUTE
        )

    return app
