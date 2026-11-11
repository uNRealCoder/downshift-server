"""Entities: backend."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from downshift.entities.verdict import BackendName


@dataclass
class IOSpec:
    name: str
    dtype: str | None
    shape: list[int | str | None] | None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BackendMeta:
    name: BackendName
    device: str
    inputs: list[IOSpec]
    outputs: list[IOSpec]

    def to_dict(self) -> dict:
        return asdict(self)
