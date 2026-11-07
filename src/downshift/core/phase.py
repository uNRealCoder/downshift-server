"""The boot phases, and how the loader thread reports them to /ready. No heavy imports."""

from contextvars import ContextVar
from dataclasses import dataclass
from enum import StrEnum

__all__ = ["CURRENT_PROGRESS", "LoadProgress", "Phase", "report"]


class Phase(StrEnum):
    """A boot phase: a key of ServingState.timings, a part of the Boot row (in this order), and
    the `phase` of /ready."""

    load = "load"
    export = "export"
    verify = "verify"
    session = "session"
    warmup = "warmup"


@dataclass
class LoadProgress:
    """The phase that /ready reports while the loader thread runs."""

    phase: Phase = Phase.load


# Set in the loader thread's own Context (serve/app.py), so report() calls deep in core reach
# the LoadProgress of /ready without a parameter through each builder.
CURRENT_PROGRESS: ContextVar[LoadProgress | None] = ContextVar(
    "downshift_load_progress", default=None
)


def report(phase: Phase) -> None:
    """Tell /ready which phase the loader is in. A no-op outside a serve loader."""
    progress = CURRENT_PROGRESS.get()
    if progress is not None:
        progress.phase = phase
