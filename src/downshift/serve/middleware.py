"""Attach the middleware that the user supplies as "pkg.module:attr" import specs."""

import inspect
from collections.abc import Sequence
from typing import Any, cast

from fastapi import FastAPI

from downshift._imports import import_object


def load_middleware(app: FastAPI, specs: Sequence[str]) -> None:
    """Each spec names a middleware class that is built as cls(app). It is pure ASGI (like
    GZipMiddleware of Starlette) or a subclass of BaseHTTPMiddleware. A spec can also name an
    async (request, call_next) function. Pure ASGI is cheaper. BaseHTTPMiddleware wraps each
    request and each response in extra tasks and streams."""
    for spec in specs:
        obj = import_object(spec)
        if inspect.isclass(obj):
            app.add_middleware(cast(Any, obj))  # any cls(app): pure ASGI or BaseHTTPMiddleware
        elif inspect.iscoroutinefunction(obj):
            app.middleware("http")(obj)
        else:
            raise ValueError(
                f"{spec!r} is not middleware. Expected a middleware class (pure ASGI or "
                "BaseHTTPMiddleware) or an async function that takes (request, call_next). "
                f"Got {type(obj).__name__}"
            )
