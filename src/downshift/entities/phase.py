"""The boot phases of a server, and the phase that /ready reports while the model loads.
Used by core/phase.py (`report`), core/verdict.py, serve/engine.py, serve/app.py and cli/render.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


# A boot phase. It is the key of `ServingState.timings`, a part of the Boot row of the banner,
# and the `phase` that /ready reports. core/phase.py `report()` sets it.
class Phase(StrEnum):
    """A boot phase: a key of ServingState.timings, a part of the Boot row (in this order), and
    the `phase` of /ready."""

    load = "load"
    export = "export"
    verify = "verify"
    session = "session"
    warmup = "warmup"


# The phase that the loader thread has reached. /ready reads it (serve/app.py).
@dataclass
class LoadProgress:
    """The phase that /ready reports while the loader thread runs."""

    phase: Phase = Phase.load
