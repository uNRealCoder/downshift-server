"""The one logging sink: plain text, on stdout, no colours.

All output of the CLI and the server goes through `logging` to the one handler that
`setup_logging` installs. This includes the loggers of downshift, the loggers of uvicorn
(which downshift routes here), `warnings` (captured), the boot banner, and the `check` and
`export` reports. This module uses only the standard library. An import costs nothing, and
`--help` stays fast.
"""

from __future__ import annotations

import logging
import sys
from contextvars import ContextVar
from typing import TextIO

# The ASGI request-id middleware sets it for the life of one request. RequestIdFilter copies it
# to each record that is logged in that time. An operator can then find all lines of the request
# whose ID a client quotes from the X-Request-ID response header.
request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)

_UVICORN_LOGGERS = ("uvicorn", "uvicorn.error", "uvicorn.access")

# The boot banner and the check and export reports. They are the own output of the command and
# not diagnostics. They therefore print at all values of --log-level. (The default, warning,
# would hide them otherwise.)
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
