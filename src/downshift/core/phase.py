"""The boot phases, and how the loader thread reports them to /ready. No heavy imports."""

from contextvars import ContextVar

from downshift.entities.phase import LoadProgress, Phase

__all__ = ["CURRENT_PROGRESS", "LoadProgress", "Phase", "report"]


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
