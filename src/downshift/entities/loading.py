"""How a model is loaded: the request (LoadSpec) and the result (LoadedModel).
Used by loading.py (`load_model`), cli/main.py, cli/runtime.py and serve/engine.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from downshift.sources import UNKNOWN_SOURCE

if TYPE_CHECKING:
    from torch import nn


# One MODEL argument and its flags. cli/main.py makes it from the command line, and it
# goes to each `--workers` worker as JSON.
@dataclass(frozen=True)
class LoadSpec:
    """One MODEL argument and its flags. pooling and normalize apply to HF repos only."""

    model: str
    inputs: str | None = None
    model_class: str | None = None
    unsafe_load: bool = False
    pooling: str | None = None
    normalize: bool | None = None


# What `load_model` found: the module, or the .onnx path, with the example inputs.
# serve/engine.py takes it to build a ServingState.
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
