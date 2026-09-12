"""Attach user-supplied middleware given as "pkg.module:attr" import specs."""

import inspect
from collections.abc import Sequence

from fastapi import FastAPI
from starlette.middleware.base import BaseHTTPMiddleware

from downshift.loading import import_object


def load_middleware(app: FastAPI, specs: Sequence[str]) -> None:
    """Each spec must name a BaseHTTPMiddleware subclass or an async (request, call_next) function."""
    for spec in specs:
        obj = import_object(spec)
        if inspect.isclass(obj) and issubclass(obj, BaseHTTPMiddleware):
            app.add_middleware(obj)
        elif inspect.iscoroutinefunction(obj):
            app.middleware("http")(obj)
        else:
            raise ValueError(
                f"{spec!r} is not middleware: expected a BaseHTTPMiddleware subclass or an "
                f"async function taking (request, call_next), got {type(obj).__name__}"
            )
