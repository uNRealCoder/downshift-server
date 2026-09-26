"""Boot phases and the progress value the serve loader reads them back from. Imports nothing
heavy, so core (which the loader calls into) and the CLI (which must stay torch-free for
`--help`) can both use it.
"""

from contextvars import ContextVar
from dataclasses import dataclass
from enum import StrEnum

__all__ = ["CURRENT_PROGRESS", "LoadProgress", "Phase", "report"]


class Phase(StrEnum):
    """A boot phase, shared by ServingState.timings' keys (serve/engine.py, core/verdict.py),
    the CLI's Boot banner row (rendered in declaration order) and /ready's `phase` field
    (LoadProgress below)."""

    load = "load"
    export = "export"
    verify = "verify"
    session = "session"
    warmup = "warmup"


@dataclass
class LoadProgress:
    """What GET /ready reports while there is no ServingState yet (U4): which phase the
    background loader thread is in. See CURRENT_PROGRESS."""

    phase: Phase = Phase.load


# Set by build_app's loader lifespan, in a fresh contextvars.Context handed to the loader
# thread, so report() (called from whichever thread is actually loading, including core's
# export and verify) reaches the LoadProgress /ready reads - without threading a parameter
# through every builder, core and the CLI's own loader closure (see serve/app.py's
# _loader_lifespan).
CURRENT_PROGRESS: ContextVar[LoadProgress | None] = ContextVar(
    "downshift_load_progress", default=None
)


def report(phase: Phase) -> None:
    """Tell /ready which phase the loader is in; a no-op when nothing is listening (library
    use, the CLI's `check`/`export`)."""
    progress = CURRENT_PROGRESS.get()
    if progress is not None:
        progress.phase = phase
