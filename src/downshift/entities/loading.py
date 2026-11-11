"""Entities: loading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from downshift.sources import UNKNOWN_SOURCE

if TYPE_CHECKING:
    from torch import nn


@dataclass(frozen=True)
class LoadSpec:
    """One MODEL argument and its flags. pooling and normalize apply to HF repos only."""

    model: str
    inputs: str | None = None
    model_class: str | None = None
    unsafe_load: bool = False
    pooling: str | None = None
    normalize: bool | None = None


@dataclass
class LoadedModel:
    source: str
    model: nn.Module | None = None
    onnx_path: Path | None = None
    example_inputs: tuple | None = None
    adapter_hint: str | None = None
    # The accepted form of `source` (downshift.sources).
    kind: str = UNKNOWN_SOURCE

    @property
    def source_path(self) -> Path | None:
        path = Path(self.source)
        return path if path.exists() else None
