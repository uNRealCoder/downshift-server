"""The error codes of the HTTP API. An error body is {"detail": <text>, "code": <code>}. The
list is stable: a client can branch on a code. Add a code only with a new row in
docs/http-api.md."""

from collections.abc import Mapping
from typing import Any

from fastapi import HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

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
    "internal_error": "The server failed. The server log has the cause.",
}

_STATUS_CODES = {404: "not_found", 405: "method_not_allowed"}


class ApiError(HTTPException):
    code: str


def api_error(
    status: int, code: str, message: str, headers: Mapping[str, str] | None = None
) -> ApiError:
    """The exception for a route: the error handler turns it into the error body."""
    exc = ApiError(status, message, headers=headers)
    exc.code = code
    return exc


def error_body(code: str, message: Any, **extra: Any) -> dict[str, Any]:
    return {"detail": message, "code": code, **extra}


def error_response(
    status: int, code: str, message: str, headers: Mapping[str, str] | None = None
) -> JSONResponse:
    """The error response of a place that cannot raise (a middleware, or a handler)."""
    return JSONResponse(error_body(code, message), status_code=status, headers=headers)


async def http_error_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    code = getattr(exc, "code", None) or _STATUS_CODES.get(exc.status_code, "http_error")
    return error_response(exc.status_code, code, exc.detail, exc.headers)


async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(
        error_body("invalid_request", jsonable_encoder(exc.errors())), status_code=422
    )
