"""The data that `serve --workers N` sends from the parent to each worker, as JSON.
Used by cli/main.py and cli/runtime.py.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from downshift.entities.options import BackendChoice, ExecutionChoice, OutputEncoding, ServeOptions

if TYPE_CHECKING:
    from downshift.entities.cache import ExportEntry
    from downshift.entities.loading import LoadSpec


# The parent's export for `--workers N`. A worker boots from it like from a cache hit.
# cli/runtime.py makes it and reads it.
@dataclass
class ArtifactHandoff:
    """The parent's export for `--workers N`. A worker boots from it like from a cache hit. A
    torch worker loads the model itself: weights are not sent between processes."""

    verdict: dict
    input_names: list[str]
    kind: str  # the parent's source_kind: an ONNX worker loads nothing to find it
    axis_bounds: dict[str, list[list]] = field(default_factory=dict)
    onnx_path: str | None = None  # the graph, if the parent serves it on onnxruntime
    feeds_path: str | None = None  # example inputs for the warmup, if any

    def entry(self) -> ExportEntry:
        import numpy as np

        from downshift.core.export_cache import unpack_feeds
        from downshift.entities.cache import ExportEntry

        feeds = None
        samples: list[dict[str, np.ndarray]] = []
        if self.feeds_path:
            with np.load(self.feeds_path, allow_pickle=False) as archive:
                feeds, samples = unpack_feeds(archive)
        return ExportEntry(
            verdict=self.verdict,
            input_names=self.input_names,
            axis_bounds=self.axis_bounds,
            feeds=feeds,
            samples=samples,
            onnx_path=Path(self.onnx_path) if self.onnx_path else None,
        )


# Everything that is needed to build the ServingState and the app again. It goes to each
# worker as JSON (to_json and from_json). cli/main.py makes it.
@dataclass
class ServeArgs:
    """Everything needed to build the ServingState and app again, as JSON for the workers."""

    load: LoadSpec
    options: ServeOptions
    reference: str | None
    middleware: list[str] | None
    log_level: str
    artifact: ArtifactHandoff | None = None
    access_log: bool = True
    # The validated --tokenizer-from directory.
    tokenizer_from: str | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, raw: str) -> ServeArgs:
        from downshift.entities.loading import LoadSpec

        data = json.loads(raw)
        data["load"] = LoadSpec(**data["load"])
        options = data["options"]
        options["backend"] = BackendChoice(options["backend"])
        options["output_encoding"] = OutputEncoding(options["output_encoding"])
        options["execution"] = ExecutionChoice(options["execution"])
        data["options"] = ServeOptions(**options)
        if data.get("artifact") is not None:
            data["artifact"] = ArtifactHandoff(**data["artifact"])
        return cls(**data)
