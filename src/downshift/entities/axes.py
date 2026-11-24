"""The dynamic axes of a model. `torch.export` gives each one a name and a range (DimBound);
the verdict reports what verification covered and what the server accepts (AxisFact).
Used by core/axes.py, core/shapes.py, core/verdict.py, serve/feeds.py and serve/describe.py.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import NamedTuple


# One dynamic axis as torch.export traced it. serve/feeds.py checks a request against it.
class DimBound(NamedTuple):
    """The Dim name and (min, max) of one dynamic axis, as torch.export traced it."""

    name: str
    min: int
    max: int


# The bounds of all dynamic axes of a model: input name -> axis -> DimBound.
AxisBounds = dict[str, dict[int, DimBound]]


# What the verdict reports about one dynamic axis: the range that the server accepts
# (served_*) and the range that verification ran (sampled_*). core/axes.py makes it.
@dataclass(frozen=True)
class AxisFact:
    input: str
    axis: int
    name: str
    served_min: int
    served_max: int
    sampled_min: int | None
    sampled_max: int | None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> AxisFact:
        return cls(**data)
