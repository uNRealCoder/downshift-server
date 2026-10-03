"""FastAPI app over a ServingState. The same routes regardless of which backend is behind it."""

import asyncio
import contextvars
import hmac
import logging
import threading
import time
import uuid
from collections.abc import AsyncIterator, Callable, Coroutine, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import asdict
from functools import partial
from typing import Any

import orjson
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

import downshift
from downshift.core.phase import CURRENT_PROGRESS, LoadProgress
from downshift.logs import request_id_var
from downshift.serve.codec import decode_safetensors
from downshift.serve.engine import ServingState
from downshift.serve.middleware import load_middleware
from downshift.serve.options import ExecutionChoice
from downshift.serve.predict import (
    BINARY_REQUEST_TYPES,
    PREDICT_PATHS,
    SAFETENSORS_MEDIA_TYPE,
    NumpyJSONResponse,
    as_batch,
    run_predict,
)
from downshift.serve.schemas import (
    GRAPH_INPUTS,
    AxisInfo,
    BackendInfo,
    ExecutionInfo,
    GraphPredictRequest,
    HealthResponse,
    MetadataResponse,
    PredictRequest,
    PredictResponse,
    ReadyResponse,
    SchemaResponse,
    VerdictInfo,
)
from downshift.settings import API_KEY, DEFAULT_MAX_BODY_BYTES
from downshift.sources import LABEL_KINDS, display_source, hide_paths

logger = logging.getLogger("downshift.serve")
access_logger = logging.getLogger("downshift.access")

NOT_READY_RETRY_AFTER = 2  # seconds a client should wait before asking again

# Probes: exempt from the API key, and logged at DEBUG so they don't flood the default output.
_PROBE_PATHS = frozenset({"/health", "/ready"})


def _route_path(scope: Scope) -> str:
    """The path relative to where this app is mounted: from Starlette 0.33 on, scope["path"]
    keeps the mount prefix (app_for(...) is meant to be mounted), so "/model/health" must
    still count as the /health probe."""
    path: str = scope["path"]
    return path.removeprefix(scope.get("root_path", ""))


class _OrjsonRequest(Request):
    async def json(self) -> Any:
        if not hasattr(self, "_json"):
            parse_start = time.perf_counter()
            self._json = orjson.loads(await self.body())
            self.state.parse_ms = (time.perf_counter() - parse_start) * 1000
        return self._json

    async def parse_in(self, executor: Any) -> None:
        """Pre-parse the body in `executor`, off the event loop; json() (called later, as
        part of FastAPI's own request-body-to-Pydantic resolution) then just returns the
        cached result instead of doing the parse itself. Used for the predict routes (P3):
        without this, a body near --max-body-bytes runs orjson.loads() straight on the loop,
        stalling /health, /ready and every other in-flight request for however long that
        takes.

        Malformed JSON is left for json() to raise the usual way: self._json stays unset, so
        FastAPI's own request-body-to-Pydantic resolution calls json() itself and gets the
        same orjson.JSONDecodeError (a json.JSONDecodeError subclass) it already turns into a
        422 - paying for a second, synchronous parse only on that error path.
        """
        parse_start = time.perf_counter()
        body = await self.body()
        loop = asyncio.get_running_loop()
        try:
            parsed = await loop.run_in_executor(executor, orjson.loads, body)
        except ValueError:
            return
        self._json = parsed
        self.state.parse_ms = (time.perf_counter() - parse_start) * 1000


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
    """Raised by /metadata, /schema and the predict routes while a loader-built app has no
    ServingState yet (see build_app's loader parameter)."""
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
    """request.stream(), with a running total: a 413 the moment it passes `limit`, instead of
    accumulating the whole body first (P2 - request.body() does exactly that)."""
    chunks: list[bytes] = []
    total = 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > limit:
            return _body_too_large(total, limit)
        chunks.append(chunk)
    return b"".join(chunks)


# Largest declared body --execution inline will parse on the event loop.
INLINE_MAX_BODY_BYTES = 64 * 1024


async def _try_inline(
    wrapped: _OrjsonRequest, state: ServingState, content_length: str | None
) -> bool:
    """--execution inline: parse a small JSON tensor body right here on the loop and mark the
    request so run_predict prepares, infers and encodes it on the loop too. A chunked body
    (no Content-Length), a larger one, a non-JSON content type or a `text` request returns
    False and takes the pools; the body is already buffered by then, so nothing is re-read."""
    if (
        state.execution != ExecutionChoice.inline
        or content_length is None
        or not content_length.isdigit()
        or int(content_length) > INLINE_MAX_BODY_BYTES
        or "json" not in wrapped.headers.get("content-type", "").lower()
    ):
        return False
    parse_start = time.perf_counter()
    try:
        parsed = orjson.loads(await wrapped.body())
    except ValueError:
        return False
    if not isinstance(parsed, dict) or parsed.get("text") is not None:
        return False
    wrapped._json = parsed
    wrapped.state.parse_ms = (time.perf_counter() - parse_start) * 1000
    wrapped.state.inline = True
    return True


_UNSUPPORTED_MEDIA_TYPE = (
    "unsupported Content-Type; send JSON (application/json) or safetensors "
    f"({SAFETENSORS_MEDIA_TYPE}, or application/octet-stream)"
)
_GRAPH_TENSOR_NAMES = frozenset({"x", "edge_index", "edge_attr"})


def _content_type(request: Request) -> str:
    return request.headers.get("content-type", "").split(";")[0].strip().lower()


def _as_json_request(request: Request) -> _OrjsonRequest:
    """A request over the same body whose Content-Type reads application/json: FastAPI only
    asks for request.json() on a JSON type, and a binary route stashes its placeholder there."""
    scope = dict(request.scope)
    scope["headers"] = [
        (name, b"application/json" if name == b"content-type" else value)
        for name, value in request.scope["headers"]
    ]
    return _OrjsonRequest(scope, request.receive)


async def _load_binary(
    wrapped: _OrjsonRequest, state: ServingState, path: str, body: bytes
) -> None:
    """Decode a safetensors body in the prep pool (its time is the request's `parse`) and hand
    the arrays to the endpoint as request.state.tensors. The JSON the pydantic model validates
    is a placeholder carrying only what the body's __metadata__ can say (output_encoding)."""
    parse_start = time.perf_counter()
    loop = asyncio.get_running_loop()
    try:
        arrays, metadata = await loop.run_in_executor(
            state.prep_executor,
            partial(decode_safetensors, body, max_input_bytes=state.options.max_input_bytes),
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if "text" in metadata:
        raise HTTPException(400, "text is JSON only; a safetensors body carries tensors")
    graph = path == "/predict/graph"
    allowed = _GRAPH_TENSOR_NAMES if graph else frozenset(state.input_names)
    unknown = [name for name in arrays if name not in allowed]
    if unknown:
        raise HTTPException(
            400, f"unknown tensor name {unknown[0]!r:.64}; this route takes {sorted(allowed)}"
        )
    placeholder: dict[str, Any] = {"output_encoding": metadata.get("output_encoding")}
    placeholder.update({"x": [], "edge_index": []} if graph else {"inputs": {}})
    wrapped._json = placeholder
    wrapped.state.tensors = arrays
    wrapped.state.parse_ms = (time.perf_counter() - parse_start) * 1000


class OrjsonRoute(APIRoute):
    """Parse request bodies with orjson: several times faster than stdlib on MiB-scale bodies.

    orjson's JSONDecodeError subclasses the stdlib one, so malformed JSON still maps to 422.
    Also enforces --max-body-bytes: a Content-Length over the limit is rejected without
    reading the body; without one (chunked), the body is read as it lands and the 413 fires
    the moment the running total passes the limit (P2), not after the whole body arrived.

    For /predict and /predict/graph, a slot is admitted *before* any of that (P1): an
    overloaded server answers 503 without buffering or parsing a byte. The slot is released
    in `finally`, so every exit (413, 400, a validation error, the client disconnecting)
    releases it exactly once. Once the body is in hand, orjson.loads() itself also runs in
    state.executor rather than on the loop (P3): a body near --max-body-bytes would otherwise
    block /health, /ready and every other in-flight request for the parse.
    """

    def get_route_handler(self) -> Callable[[Request], Coroutine[Any, Any, Response]]:
        handler = super().get_route_handler()
        is_predict = self.path in PREDICT_PATHS

        async def route_handler(request: Request) -> Response:
            current: ServingState | None = request.app.state.serving
            # Before the loader lands there is no ServingState to read a limit from yet;
            # fall back to the default so a large body is still rejected before parsing.
            limit = (
                current.options.max_body_bytes if current is not None else DEFAULT_MAX_BODY_BYTES
            )
            content_type = _content_type(request) if is_predict else ""
            binary = content_type in BINARY_REQUEST_TYPES
            wrapped = (
                _as_json_request(request)
                if binary
                else _OrjsonRequest(request.scope, request.receive)
            )

            admitted = False
            if is_predict:
                if current is None:
                    raise _not_ready_error()
                if not current.try_admit():
                    raise _capacity_error(current)
                admitted = True
                wrapped.state.admitted_at = time.monotonic()

            try:
                if content_type and not binary and "json" not in content_type:
                    return JSONResponse({"detail": _UNSUPPORTED_MEDIA_TYPE}, status_code=415)
                content_length = request.headers.get("content-length")
                if content_length is not None and content_length.isdigit():
                    if int(content_length) > limit:
                        return _body_too_large(int(content_length), limit)
                else:
                    result = await _read_body_limited(request, limit)
                    if isinstance(result, JSONResponse):
                        return result
                    wrapped._body = result
                if is_predict:
                    assert current is not None
                    if binary:
                        await _load_binary(wrapped, current, self.path, await wrapped.body())
                    elif not await _try_inline(wrapped, current, content_length):
                        await wrapped.parse_in(current.prep_executor)
                return await handler(wrapped)
            finally:
                if admitted:
                    assert current is not None
                    current.release()

        return route_handler


# The predict routes: orjson response, and PredictResponse kept in the OpenAPI schema without
# the response_model re-validation walk over every output.
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

    Every route that needs a ServingState depends on this, so a route added later cannot
    forget the guard by omitting a copy of it. /ready is deliberately not a caller: being
    unloaded is its answer, not an error.
    """
    current: ServingState | None = request.app.state.serving
    if current is None:
        raise _not_ready_error()
    return current


def _predict_context(request: Request) -> dict[str, Any]:
    """run_predict's per-request keywords. The parse/codec/infer split dict it fills lives in
    the scope's state, which is where RequestIdMiddleware reads it back for the request
    line (U1)."""
    timings: dict[str, float] = {}
    request.state.timings_ms = timings
    return {
        "admitted_at": request.state.admitted_at,
        "parse_ms": getattr(request.state, "parse_ms", 0.0),
        "inline": getattr(request.state, "inline", False),
        "timings_ms": timings,
        "tensors": getattr(request.state, "tensors", None),
        "accept_safetensors": SAFETENSORS_MEDIA_TYPE in request.headers.get("accept", "").lower(),
    }


class RequestIdMiddleware:
    """Pure ASGI, not BaseHTTPMiddleware (which buffers the whole response body to let a
    handler rewrite headers, extra copies this doesn't need): reads X-Request-Id or makes
    one, stores it on request.state, and echoes it on the response. "See the server log"
    then has something to search for.

    It also binds the id to `request_id_var` for the request, so every log line emitted
    while serving it carries it, and (with `access_log`) writes the one request line (U1):
    method, path, status, duration and, for the predict routes, the parse/codec/infer split
    they leave in the scope's state as "timings_ms". WARNING from status 400 up, DEBUG for
    the /health and /ready probes (a 503 from /ready while loading is its normal answer),
    INFO otherwise.
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
            # ServerErrorMiddleware, outside this one, turns it into the 500 response.
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
    """DOWNSHIFT_SERVER_API_KEY (U7), read via settings.py so it applies to app_for() too
    (ruling 5). None (unset) means unauthenticated - the caller already warned about that at
    build_app time. Pure ASGI, like RequestIdMiddleware; /health and /ready are exempt so a
    probe never needs the key.
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
        # Starlette decodes header bytes as latin-1, so latin-1 gets the client's raw bytes
        # back; compare_digest on a non-ASCII str would raise (a 500) instead of failing.
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
    """Runs `loader` on a background thread, started when the app actually begins serving
    (not when build_app is called), and lands the result on app.state.serving. A raising
    loader leaves app.state.serving None forever from this function's point of view alone;
    the caller (serve_cmd) arranges for that to end the process instead, by closing over
    the uvicorn.Server and setting should_exit itself before re-raising.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        progress = LoadProgress()
        app.state.load_progress = progress

        def run() -> None:
            # Runs in a fresh Context (see below), so this set() is only ever visible to this
            # thread's call stack - the phase-reporting calls inside engine.py's builders
            # (U4) read it back via CURRENT_PROGRESS.get() without a parameter threaded
            # through every builder and the CLI's own loader closure.
            CURRENT_PROGRESS.set(progress)
            try:
                app.state.serving = loader()
            except Exception:
                # Not logged at error/exception level: a caller with a reporting path of its
                # own (serve_cmd re-raises this on the main thread) would otherwise print the
                # traceback twice, once here unconditionally and once gated by --log-level
                # debug. --log-level debug still sees it here.
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
    """With `state`, the app serves it immediately (library use, tests). With `loader`
    instead, the app binds with no ServingState at all: /health is 200 right away, /ready
    is 503, and /metadata, /schema and the predict routes are 503, until the loader (run on
    a background thread started by the lifespan, once the app is actually served) lands one.

    `api_key`, when set, requires "Authorization: Bearer <api_key>" on every route but
    /health and /ready (U7); it defaults to DOWNSHIFT_SERVER_API_KEY (settings.API_KEY) so
    the env var reaches app_for() and this function equally (ruling 5). Pass a value (or
    None to force it off) to override that default for a library caller. An empty string
    counts as unset: DOWNSHIFT_SERVER_API_KEY="" must not make "Bearer " a valid credential.

    `access_log` False drops the one request line RequestIdMiddleware writes on the
    `downshift.access` logger (U1); the request id is still set and echoed.
    """
    if (state is None) == (loader is None):
        raise ValueError("build_app needs exactly one of state or loader")

    api_key = api_key or None
    if api_key is None:
        logger.warning(
            "DOWNSHIFT_SERVER_API_KEY is not set, so the endpoints are unauthenticated. Set "
            "it to require a fixed API key, or add your own authentication middleware."
        )

    app = FastAPI(
        title="downshift",
        version=downshift.__version__,
        lifespan=_loader_lifespan(loader) if loader is not None else None,
    )
    app.router.route_class = OrjsonRoute
    app.state.serving = state
    # Starlette makes the *last*-registered middleware the outermost layer, so registration
    # order here is back to front: load_middleware's user middleware goes on first (innermost,
    # right next to the routes, so it only ever sees authenticated traffic), ApiKeyMiddleware
    # next (the gate), and RequestIdMiddleware last (outermost, so access logging and the
    # X-Request-Id header still cover 401s and everything else uniformly).
    load_middleware(app, middleware)
    app.add_middleware(ApiKeyMiddleware, api_key=api_key)
    app.add_middleware(RequestIdMiddleware, access_log=access_log)

    @app.exception_handler(Exception)
    async def _unhandled_exception(request: Request, exc: Exception) -> JSONResponse:
        # ServerErrorMiddleware, which dispatches this handler, sits outside
        # RequestIdMiddleware (Starlette always makes it the outermost layer), so its
        # response bypasses our send wrapper; set the header here too, directly on the
        # response, rather than relying on that wrapper for this one path.
        request_id = getattr(request.state, "request_id", None)
        # RequestIdMiddleware has already reset request_id_var by the time this runs.
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
        # Free-text fields quote the path an exception was handed (ORT's "Load model from
        # ...", transformers' "Can't load tokenizer for ..."); a client gets the name only.
        located = () if current.source_kind in LABEL_KINDS else (current.source,)
        paths = (*located, current.verdict.onnx_path, current.hf_source)
        verdict = current.verdict.to_dict()
        verdict["reason"] = hide_paths(verdict["reason"], paths)
        verdict["warnings"] = [hide_paths(w, paths) for w in verdict["warnings"]]
        return MetadataResponse(
            model=display_source(current.source, current.source_kind),
            family=current.verdict.model_family,
            verdict=VerdictInfo.model_validate(verdict),
            axes=[AxisInfo.model_validate(fact.to_dict()) for fact in current.verdict.axes],
            backend=BackendInfo.model_validate(current.backend.metadata().to_dict()),
            input_names=list(current.input_names),
            notes=[hide_paths(note, paths) for note in current.notes],
            version=downshift.__version__,
            limits={
                "max_body_bytes": current.options.max_body_bytes,
                "max_input_bytes": current.options.max_input_bytes,
                "max_concurrency": current.options.max_concurrency,
                "max_queue": current.options.max_queue,
                "request_timeout": current.options.request_timeout,
            },
            execution=ExecutionInfo(mode=current.execution.value),
            boot=dict(current.timings),
            warmup=asdict(current.warmup_stats) if current.warmup_stats is not None else None,
        )

    @app.get("/schema", response_model=SchemaResponse)
    async def schema(
        request: Request, current: ServingState = Depends(require_serving)
    ) -> SchemaResponse:
        """What to POST: every input's name, dtype and shape, and an example body.

        Cheap enough to build per request (it reads the backend's declared IO and fills a
        small example), so nothing is cached and it always matches the running graph.
        """
        from downshift.serve.describe import describe

        return describe(current, f"{str(request.base_url).rstrip('/')}/predict")

    @app.post("/predict", **_PREDICT_ROUTE)
    async def predict(
        req: PredictRequest, request: Request, current: ServingState = Depends(require_serving)
    ) -> Response:
        text = as_batch(req.text) if req.text is not None else None
        return await run_predict(
            current,
            req.inputs,
            req.output_encoding,
            text,
            req.prompt_name,
            **_predict_context(request),
        )

    @app.post("/predict/graph", **_PREDICT_ROUTE)
    async def predict_graph(
        req: GraphPredictRequest,
        request: Request,
        current: ServingState = Depends(require_serving),
    ) -> Response:
        if not GRAPH_INPUTS <= set(current.input_names):
            raise HTTPException(
                400,
                f"model is not graph-shaped: inputs are {list(current.input_names)}, "
                "expected at least 'x' and 'edge_index'",
            )
        # No dtype hints needed: to_numpy takes the backend's declared dtype (int64 for
        # edge_index on both backends), and integer lists default to int64 anyway.
        inputs: dict[str, Any] = {"x": req.x, "edge_index": req.edge_index}
        if getattr(request.state, "tensors", None) is not None:
            inputs = {}
        elif req.edge_attr is not None:
            inputs["edge_attr"] = req.edge_attr
        return await run_predict(current, inputs, req.output_encoding, **_predict_context(request))

    return app
