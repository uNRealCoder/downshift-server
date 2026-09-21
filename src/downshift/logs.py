"""The one logging sink: plain text, on stdout, no colours.

Everything the CLI and server say goes through `logging` into the single handler
`setup_logging` installs: our own loggers, uvicorn's (re-routed here), `warnings` (captured),
the boot banner and the `check`/`export` reports. Stdlib only, so importing it costs nothing
and `--help` stays fast.
"""

from __future__ import annotations

import logging
import sys
from contextvars import ContextVar
from typing import TextIO

# Set by the ASGI request-id middleware for the life of one request; RequestIdFilter copies it
# onto every record logged meanwhile, so an operator can find all lines of the request whose id
# a client quotes from the X-Request-ID response header.
request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)

_UVICORN_LOGGERS = ("uvicorn", "uvicorn.error", "uvicorn.access")

# The boot banner and the check/export reports: the command's own output, not diagnostics,
# so they print at any --log-level (warning, the default, would otherwise hide them).
REPORT_LOGGER = "downshift.report"


class RequestIdFilter(logging.Filter):
    """Stamps `record.request_id` from `request_id_var`, unless the caller already set one
    (the request log line is emitted after the context var may have been reset)."""

    def filter(self, record: logging.LogRecord) -> bool:
        if getattr(record, "request_id", None) is None:
            record.request_id = request_id_var.get()
        return True


class TextFormatter(logging.Formatter):
    """`time LEVEL logger: message`, with ` request_id=<id>` after the message when set."""

    def __init__(self) -> None:
        super().__init__(
            "%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )

    def formatMessage(self, record: logging.LogRecord) -> str:
        line = super().formatMessage(record)
        request_id = getattr(record, "request_id", None)
        return f"{line} request_id={request_id}" if request_id else line


def setup_logging(level: str, *, stream: TextIO | None = None) -> None:
    """Route root, warnings and uvicorn's loggers into one StreamHandler on `stream`
    (default: sys.stdout, looked up now so a redirected stdout is honoured). Safe to call
    again: it replaces the previous handler."""
    handler = logging.StreamHandler(stream if stream is not None else sys.stdout)
    handler.addFilter(RequestIdFilter())
    handler.setFormatter(TextFormatter())
    level_name = level.upper()
    logging.basicConfig(level=level_name, handlers=[handler], force=True)
    logging.captureWarnings(True)
    logging.getLogger(REPORT_LOGGER).setLevel(logging.INFO)
    # The ONNX optimizer passes log every rewrite at INFO. Only show them when debugging.
    if level_name != "DEBUG":
        for name in ("onnxscript", "onnx_ir"):
            logging.getLogger(name).setLevel(logging.WARNING)
    # uvicorn gives these their own handlers and stops them propagating; undo both so its
    # startup, error and (if enabled) access lines share this handler and format.
    for name in _UVICORN_LOGGERS:
        uvicorn_logger = logging.getLogger(name)
        uvicorn_logger.handlers.clear()
        uvicorn_logger.propagate = True
        uvicorn_logger.setLevel(level_name)
