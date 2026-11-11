"""Entities: axes."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import NamedTuple


class DimBound(NamedTuple):
    """The Dim name and (min, max) of one dynamic axis, as torch.export traced it."""

    name: str
    min: int
    max: int


# input -> axis -> its DimBound
AxisBounds = dict[str, dict[int, DimBound]]


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
