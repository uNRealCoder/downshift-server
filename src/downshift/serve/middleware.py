"""The ASGI middleware of the server: the request ID with the access log, the API key, and the loading of --middleware."""

from __future__ import annotations

import hmac
import inspect
import logging
import re
import time
import uuid
from collections.abc import Sequence
from typing import Any, cast

from fastapi import FastAPI
from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from downshift._imports import import_object
from downshift.entities.metrics import Metrics
from downshift.logs import request_id_var
from downshift.serve.errors import error_response, route_path

access_logger = logging.getLogger("downshift.access")


# The probes: no API key, and logged at DEBUG.
PROBE_PATHS = frozenset({"/health", "/ready"})


# A request ID from a client goes to the logs, a header and an error body: only safe characters.
_REQUEST_ID = re.compile(r"[A-Za-z0-9._:-]{1,128}")


class RequestIdMiddleware:
    """Pure ASGI (BaseHTTPMiddleware would buffer each response). It takes the request ID from
    the `header` of the request (DOWNSHIFT_REQUEST_ID_HEADER, default X-Request-Id) or makes one, binds it to `request_id_var` for the log lines, and returns it. With
    `access_log`, it writes one line for each request: WARNING from status 400, DEBUG for the
    probes, INFO otherwise, with the stage timings of a predict."""

    def __init__(
        self,
        app: ASGIApp,
        access_log: bool = True,
        header: str = "X-Request-Id",
        metrics: Metrics | None = None,
    ) -> None:
        self.app = app
        self.access_log = access_log
        self.header = header.lower()
        self.metrics = metrics

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        start = time.perf_counter() if self.access_log or self.metrics is not None else 0.0
        supplied = Headers(scope=scope).get(self.header)
        request_id = (
            supplied if supplied and _REQUEST_ID.fullmatch(supplied) else uuid.uuid4().hex[:16]
        )
        state = scope.setdefault("state", {})
        state["request_id"] = request_id
        status: int | None = None

        async def send_with_request_id(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                headers = MutableHeaders(scope=message)
                headers.append(self.header, request_id)
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
                if status is not None:
                    elapsed = time.perf_counter() - start
                    if self.metrics is not None:
                        self._count(scope, status, elapsed, state)
                    if self.access_log:
                        self._log(scope, status, elapsed * 1000, state)
            finally:
                request_id_var.reset(token)

    def _count(self, scope: Scope, status: int, seconds: float, state: dict[str, Any]) -> None:
        assert self.metrics is not None
        # FastAPI puts the matched route in the scope. No route is "other": a request path
        # is never a label.
        route = scope.get("route")
        self.metrics.record_request(route.path if route is not None else "other", status, seconds)
        self.metrics.record_rejected(state.get("error_code"))
        self.metrics.record_cache_result(state.get("cache"))
        timings = state.get("timings_ms")
        if timings:
            self.metrics.record_stages(timings)

    @staticmethod
    def _log(scope: Scope, status: int, duration_ms: float, state: dict[str, Any]) -> None:
        path = scope["path"]
        if route_path(scope) in PROBE_PATHS:
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
    """Pure ASGI: every route except the probes needs "Authorization: Bearer <api_key>".
    None turns it off."""

    def __init__(self, app: ASGIApp, api_key: str | None) -> None:
        self.app = app
        self.api_key = api_key
        self._key_bytes = api_key.encode("utf-8") if api_key is not None else b""

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or self.api_key is None or route_path(scope) in PROBE_PATHS:
            await self.app(scope, receive, send)
            return

        header = Headers(scope=scope).get("authorization", "")
        scheme, _, token = header.partition(" ")
        # Starlette decodes headers as latin-1, so this gives back the client's bytes.
        # compare_digest on a non-ASCII str would raise (a 500) instead of failing the check.
        if scheme.lower() != "bearer" or not hmac.compare_digest(
            token.encode("latin-1"), self._key_bytes
        ):
            response = error_response(
                scope,
                401,
                "unauthorized",
                "Authorization header is not set or incorrect",
                headers={"WWW-Authenticate": "Bearer"},
            )
            await response(scope, receive, send)
            return
        await self.app(scope, receive, send)


def load_middleware(app: FastAPI, specs: Sequence[str]) -> None:
    """Attach each "pkg.module:attr" spec: a middleware class built as cls(app) (pure ASGI or
    BaseHTTPMiddleware), or an async (request, call_next) function."""
    for spec in specs:
        obj = import_object(spec)
        if inspect.isclass(obj):
            app.add_middleware(cast(Any, obj))
        elif inspect.iscoroutinefunction(obj):
            app.middleware("http")(obj)
        else:
            raise ValueError(
                f"{spec!r} is not middleware. Expected a middleware class (pure ASGI or "
                "BaseHTTPMiddleware) or an async function that takes (request, call_next). "
                f"Got {type(obj).__name__}"
            )
