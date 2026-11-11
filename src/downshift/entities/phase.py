"""Entities: phase."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


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
