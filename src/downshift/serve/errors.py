"""The error codes of the HTTP API. An error body is {"detail": <text>, "code": <code>}. The
list is stable: a client can branch on a code. Add a code only with a new row in
docs/http-api.md.

On a /v1 path the body has the shape of the OpenAI API instead:
{"error": {"message", "type", "param", "code"}}. error_response is the only place that
chooses the shape."""

from collections.abc import Mapping
from typing import Any

from fastapi import HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.types import Scope

# code -> when the server returns it
ERROR_CODES: dict[str, str] = {
    "invalid_request": "The body is empty, is not valid JSON, or has a field that is not valid.",
    "invalid_input": "An input is not valid: its shape, dtype, values or encoding.",
    "missing_input": "A declared input is not in the request.",
    "text_too_long": "A text row has more tokens than the model reads.",
    "unknown_prompt": "The prompt_name is not one of the prompts of the model.",
    "text_not_supported": "The request has text, and the model takes tensors only.",
    "not_graph_model": "The request is for /predict/graph, and the model is not a graph model.",
    "unauthorized": "The API key is not set or is not correct.",
    "body_too_large": "The body is larger than --max-body-bytes.",
    "unsupported_media_type": "The Content-Type is not JSON and not safetensors.",
    "not_found": "The path does not exist.",
    "method_not_allowed": "The path does not take this HTTP method.",
    "http_error": "Another HTTP error, for example from a middleware that you added.",
    "not_ready": "The model is not loaded yet. Try again after the Retry-After seconds.",
    "at_capacity": "All slots are in use. Try again after the Retry-After seconds.",
    "request_timeout": "The request took more than --request-timeout.",
    "model_not_found": "The model of the request is not the model that the server serves.",
    "not_embedding_model": "The route is for a text embedding model, and the model is not one.",
    "dimensions_not_supported": "The request has dimensions, and the server did not start with --matryoshka.",
    "internal_error": "The server failed. The server log has the cause.",
}

_STATUS_CODES = {404: "not_found", 405: "method_not_allowed"}


class ApiError(HTTPException):
    code: str
    param: str | None = None


def api_error(
    status: int,
    code: str,
    message: str,
    headers: Mapping[str, str] | None = None,
    param: str | None = None,
) -> ApiError:
    """The exception for a route: the error handler turns it into the error body. `param` is
    the field at fault, for the OpenAI shape."""
    exc = ApiError(status, message, headers=headers)
    exc.code = code
    exc.param = param
    return exc


def error_body(code: str, message: Any, **extra: Any) -> dict[str, Any]:
    return {"detail": message, "code": code, **extra}


def route_path(scope: Scope) -> str:
    """The path without the mount prefix, so "/model/health" is still the /health probe."""
    path: str = scope["path"]
    return path.removeprefix(scope.get("root_path", ""))


def is_openai_path(scope: Scope) -> bool:
    return route_path(scope).startswith("/v1/")


def error_response(
    scope: Scope,
    status: int,
    code: str,
    message: Any,
    headers: Mapping[str, str] | None = None,
    *,
    param: str | None = None,
    **extra: Any,
) -> JSONResponse:
    """The error response of a place that cannot raise (a middleware, or a handler). On a /v1
    path it has the OpenAI shape; `param` (the field at fault) and `extra` are for the two
    shapes, respectively."""
    if is_openai_path(scope):
        kind = (
            "authentication_error"
            if status == 401
            else "invalid_request_error"
            if status < 500
            else "server_error"
        )
        body: dict[str, Any] = {
            "error": {"message": message, "type": kind, "param": param, "code": code}
        }
    else:
        body = error_body(code, message, **extra)
    return JSONResponse(body, status_code=status, headers=headers)


async def http_error_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    code = getattr(exc, "code", None) or _STATUS_CODES.get(exc.status_code, "http_error")
    param = getattr(exc, "param", None)
    return error_response(
        request.scope, exc.status_code, code, exc.detail, exc.headers, param=param
    )


async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    errors = jsonable_encoder(exc.errors())
    first = errors[0]
    names = [part for part in first["loc"] if isinstance(part, str) and part != "body"]
    # The OpenAI shape has the first error only. The detail list is for the other shape.
    message = first["msg"] if is_openai_path(request.scope) else errors
    return error_response(
        request.scope, 422, "invalid_request", message, param=names[-1] if names else None
    )
