"""The data of the two caches.

ExportEntry is a verified export that a boot can reuse (core/memo.py and core/export_cache.py).
Row and Lookup belong to the result cache of serve/result_cache.py.
"""

from __future__ import annotations

from concurrent.futures import Future
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


# A verified export that a boot can reuse. core/memo.py keeps it in memory,
# core/export_cache.py keeps it on disk, and a `--workers` worker gets it from the parent.
@dataclass
class ExportEntry:
    """What a boot needs to skip the export and the verification. A graph with external data
    has no `onnx_bytes`; `onnx_path` names its .onnx file next to the data file."""

    verdict: dict
    input_names: list[str]
    axis_bounds: dict[str, list[list]]
    feeds: dict[str, np.ndarray] | None = None
    # The input feeds of the verification samples, for the serving check. Empty in an old entry.
    samples: list[dict[str, np.ndarray]] = field(default_factory=list, repr=False)
    onnx_bytes: bytes = field(default=b"", repr=False)
    onnx_path: Path | None = None

    @property
    def status(self) -> str:
        return str(self.verdict["status"])

    @property
    def external(self) -> bool:
        return not self.onnx_bytes and self.onnx_path is not None

    def example_inputs(self) -> tuple | None:
        if self.feeds is None:
            return None
        return tuple(self.feeds[name] for name in self.input_names)


# The output of one row: output name -> array, with no batch axis. The result cache stores it.
Row = dict[str, np.ndarray]  # output name -> the array of this row, with no batch axis


# The answer of the result cache for the rows of one request: hits, rows that another
# request computes, and rows that this request must compute (serve/result_cache.py).
@dataclass
class Lookup:
    """The answer for the rows of one request. `hits` and `pending` map a row index to its
    result (or to a Future of it). `missing` lists the rows that this request must compute. It
    has claimed them: it must call complete() or fail() for each of them."""

    hits: dict[int, Row] = field(default_factory=dict)
    pending: dict[int, Future[Row]] = field(default_factory=dict)
    missing: list[int] = field(default_factory=list)
