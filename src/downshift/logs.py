"""The one logging sink: plain text on stdout, no colours. Downshift, uvicorn, `warnings`,
the banner and the reports all go through it.
"""

from __future__ import annotations

import logging
import sys
from contextvars import ContextVar
from typing import TextIO

# Set for each request by RequestIdMiddleware; RequestIdFilter adds it to each record.
request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)

_UVICORN_LOGGERS = ("uvicorn", "uvicorn.error", "uvicorn.access")

# The banner and the reports: command output, printed at every --log-level.
REPORT_LOGGER = "downshift.report"


class RequestIdFilter(logging.Filter):
    """Sets `record.request_id` from `request_id_var`, unless the caller already set one. (The
    request log line is emitted after the context variable can be reset.)"""

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
    """Send the root logger, the warnings and the loggers of uvicorn to one StreamHandler on
    `stream`. The default is sys.stdout, which downshift looks up now, so a redirected stdout
    works. You can call it again. It replaces the previous handler."""
    handler = logging.StreamHandler(stream if stream is not None else sys.stdout)
    handler.addFilter(RequestIdFilter())
    handler.setFormatter(TextFormatter())
    level_name = level.upper()
    logging.basicConfig(level=level_name, handlers=[handler], force=True)
    logging.captureWarnings(True)
    logging.getLogger(REPORT_LOGGER).setLevel(logging.INFO)
    # The ONNX optimizer passes log each rewrite at INFO. Show them only when you debug.
    if level_name != "DEBUG":
        for name in ("onnxscript", "onnx_ir"):
            logging.getLogger(name).setLevel(logging.WARNING)
    # Uvicorn gives these loggers their own handlers and stops them from propagating. Undo
    # both. The startup lines, the error lines and (if enabled) the access lines of uvicorn then
    # share this handler and format.
    for name in _UVICORN_LOGGERS:
        uvicorn_logger = logging.getLogger(name)
        uvicorn_logger.handlers.clear()
        uvicorn_logger.propagate = True
        uvicorn_logger.setLevel(level_name)
