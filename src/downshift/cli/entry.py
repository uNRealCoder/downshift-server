"""The console entry point. It reports a bad DOWNSHIFT_* variable as a usage error (exit 4)."""

from __future__ import annotations

import sys

EXIT_USAGE = 4


def main() -> None:
    try:
        from downshift.cli.main import app
    except ValueError as exc:  # downshift.settings reads the environment at import
        import logging

        from downshift.logs import setup_logging

        setup_logging("warning")
        logging.getLogger("downshift.cli").error("%s", exc)
        sys.exit(EXIT_USAGE)
    app()
