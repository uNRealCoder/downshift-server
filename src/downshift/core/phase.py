"""The boot phases, and the progress value from which the serve loader reads them again. This
module imports nothing heavy. core (which the loader calls) and the CLI (which must stay free
of torch for `--help`) can both use it.
"""

from contextvars import ContextVar
from dataclasses import dataclass
from enum import StrEnum

__all__ = ["CURRENT_PROGRESS", "LoadProgress", "Phase", "report"]


class Phase(StrEnum):
    """A boot phase. These are the keys of ServingState.timings (serve/engine.py and
    core/verdict.py), the rows of the Boot banner of the CLI (in the order of declaration), and
    the `phase` field of /ready (LoadProgress below)."""

    load = "load"
    export = "export"
    verify = "verify"
    session = "session"
    warmup = "warmup"


@dataclass
class LoadProgress:
    """What GET /ready reports while there is no ServingState yet (U4): the phase that the
    background loader thread is in. See CURRENT_PROGRESS."""

    phase: Phase = Phase.load


# The loader lifespan of build_app sets it, in a new contextvars.Context that it gives to the
# loader thread. report() is called from the thread that loads (this includes the export and the
# verification in core). It then reaches the LoadProgress that /ready reads. Downshift does not
# need to pass a parameter through each builder, through core, and through the loader closure
# of the CLI (see _loader_lifespan in serve/app.py).
CURRENT_PROGRESS: ContextVar[LoadProgress | None] = ContextVar(
    "downshift_load_progress", default=None
)


def report(phase: Phase) -> None:
    """Tell /ready which phase the loader is in. It does nothing if nothing listens (library
    use, and `check` and `export` of the CLI)."""
    progress = CURRENT_PROGRESS.get()
    if progress is not None:
        progress.phase = phase
