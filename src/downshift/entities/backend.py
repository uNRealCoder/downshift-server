"""The description of a backend: its device, and the name, dtype and shape of each input and
output. Used by serve/backends.py (it makes them), serve/describe.py (GET /schema) and
serve/schemas.py (GET /metadata).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from downshift.entities.verdict import BackendName


# The name, dtype and shape of one input or output of a backend.
@dataclass
class IOSpec:
    name: str
    dtype: str | None
    shape: list[int | str | None] | None

    def to_dict(self) -> dict:
        return asdict(self)


# What a backend reports about itself: the device, and the IOSpec of its inputs and outputs.
@dataclass
class BackendMeta:
    name: BackendName
    device: str
    inputs: list[IOSpec]
    outputs: list[IOSpec]

    def to_dict(self) -> dict:
        return asdict(self)
