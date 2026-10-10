"""Attach user-supplied middleware given as "pkg.module:attr" import specs."""

import inspect
from collections.abc import Sequence
from typing import Any, cast

from fastapi import FastAPI

from downshift._imports import import_object


def load_middleware(app: FastAPI, specs: Sequence[str]) -> None:
    """Each spec names a middleware class built as cls(app), either pure ASGI (like Starlette's
    GZipMiddleware) or a BaseHTTPMiddleware subclass, or an async (request, call_next)
    function. Pure ASGI is the cheaper one: BaseHTTPMiddleware wraps every request and
    response in extra tasks and streams."""
    for spec in specs:
        obj = import_object(spec)
        if inspect.isclass(obj):
            app.add_middleware(cast(Any, obj))  # any cls(app): pure ASGI or BaseHTTPMiddleware
        elif inspect.iscoroutinefunction(obj):
            app.middleware("http")(obj)
        else:
            raise ValueError(
                f"{spec!r} is not middleware: expected a middleware class (pure ASGI or "
                "BaseHTTPMiddleware) or an async function taking (request, call_next), got "
                f"{type(obj).__name__}"
            )
