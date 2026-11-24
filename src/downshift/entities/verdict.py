"""The verdict and the numbers behind it.

ExportVerdict is the one object that all other code reads. It holds the status, the reason, the
numerics, the axis facts and the recommended backend. core/capture.py makes a CaptureResult,
core/verify.py makes a NumericsReport, and core/verdict.py combines them.
Used by almost every module of core/, serve/ and cli/.
"""

from __future__ import annotations

import copy
import tempfile
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field, fields
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from downshift.entities.adapters import Prepared
from downshift.entities.axes import AxisFact
from downshift.sources import hide_paths

if TYPE_CHECKING:
    import torch


# The worst element of a failed comparison. NumericsReport keeps it.
@dataclass
class WorstMismatch:
    """The element with the largest absolute error, across all samples."""

    sample: int
    output: int
    index: tuple[int, ...]
    expected: float
    got: float
    input_shapes: list[tuple[int, ...]]


# The result of the numeric check: the error numbers, the samples that ran and the worst
# mismatch. core/verify.py makes it, and ExportVerdict holds it.
@dataclass
class NumericsReport:
    samples_tested: int
    max_abs_err: float
    max_rel_err: float
    failures: int
    # None if the baseline sample failed, False if only a varied shape failed.
    shape_generalization: bool | None
    tolerance_abs: float
    tolerance_rel: float
    tolerance_dtype: str = "float32"  # the dtype whose default tolerances applied
    tolerance_overridden: bool = False  # --atol or --rtol set the values
    baseline_failed: bool = False
    # For an embedding model: the smallest cosine between a torch row and an ONNX Runtime row.
    min_cosine: float | None = None
    tolerance_cosine: float | None = None
    worst: WorstMismatch | None = None
    sample_shapes: list[list[tuple[int, ...]]] = field(default_factory=list)
    output_shapes: list[list[tuple[int, ...]]] = field(default_factory=list)  # [sample][output]
    seed: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.failures == 0

    def to_dict(self) -> dict:
        return asdict(self) | {"passed": self.passed}

    @classmethod
    def from_dict(cls, data: dict) -> NumericsReport:
        """The inverse of to_dict()."""
        values = {f.name: data[f.name] for f in fields(cls) if f.name in data}
        if values.get("worst") is not None:
            worst = dict(values["worst"])
            worst["index"] = tuple(worst["index"])
            worst["input_shapes"] = [tuple(s) for s in worst["input_shapes"]]
            values["worst"] = WorstMismatch(**worst)
        for name in ("sample_shapes", "output_shapes"):
            values[name] = [[tuple(s) for s in sample] for sample in values.get(name, [])]
        return cls(**values)


# What `torch.export` and the ONNX translation gave: the graph, or the error. core/capture.py
# makes it and core/verdict.py reads it.
@dataclass
class CaptureResult:
    success: bool
    capture_strategy: str | None  # a _STRATEGIES name. None if nothing traced
    onnx_program: torch.onnx.ONNXProgram | None = None
    onnx_bytes: bytes = field(default=b"", repr=False)
    # In place of onnx_bytes if the weights went to external data. `tmpdir` owns the files.
    onnx_path: Path | None = None
    tmpdir: tempfile.TemporaryDirectory | None = field(default=None, repr=False)
    opset: int | None = None
    op_types: dict[str, int] = field(default_factory=dict)  # histogram, highest count first
    exception: Exception | None = None  # quoted as the reason
    exceptions: list[tuple[str, Exception]] = field(default_factory=list)  # each strategy tried
    stderr: str = ""  # what torch printed, for the debug log


# The four verdicts: CLEAN, DEGRADED, FAILED, UNVERIFIED.
class Status(StrEnum):
    CLEAN = "CLEAN"
    DEGRADED = "DEGRADED"
    FAILED = "FAILED"
    UNVERIFIED = "UNVERIFIED"


# A running backend: ONNX Runtime or torch.
class BackendName(StrEnum):
    """A concrete backend. "auto" is only a CLI choice (serve.options.BackendChoice)."""

    onnxruntime = "onnxruntime"
    torch = "torch"


# The exit code of `check` and `export` for each status.
EXIT_CODES = {Status.CLEAN: 0, Status.FAILED: 1, Status.DEGRADED: 2, Status.UNVERIFIED: 3}


# The one object that all other code reads: status, reason, numerics, axis facts and the
# recommended backend. core/verdict.py (`build_verdict`, `intake`) makes it. The server,
# the CLI reports, the manifest and the export cache read it.
@dataclass
class ExportVerdict:
    status: Status
    model_family: str
    recommended_backend: BackendName
    reason: str = ""
    capture_strategy: str | None = None
    opset: int | None = None
    op_types: dict[str, int] = field(default_factory=dict)  # highest count first
    numerics: NumericsReport | None = None
    input_names: tuple[str, ...] = ()
    dynamic_dims: dict[str, list[int]] = field(default_factory=dict)
    axes: list[AxisFact] = field(default_factory=list)
    output_axes: list[str] = field(default_factory=list)  # PyG only: node/edge/fixed/unknown
    unsupported_ops: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    # Does a row give the same output alone and in a batch (core/rows.py)? None: not checked.
    row_independent: bool | None = None
    row_reason: str | None = None
    onnx_path: Path | None = None
    onnx_program: object | None = field(default=None, repr=False)  # torch.onnx.ONNXProgram
    onnx_bytes: bytes = field(default=b"", repr=False)
    # Not serialized: the temporary directory that holds onnx_path (external data), and the
    # CPU session of verify, which take_session() gives to the server.
    _tmpdir: object | None = field(default=None, repr=False)
    _session: object | None = field(default=None, repr=False, compare=False)
    # The input feeds of the verification samples, for the serving check. Not serialized.
    sample_feeds: list = field(default_factory=list, repr=False, compare=False)
    prepared: Prepared | None = field(default=None, repr=False)
    # For the debug log of a FAILED verdict. Not serialized.
    capture_stderr: str = field(default="", repr=False)
    capture_exceptions: list[tuple[str, Exception]] = field(default_factory=list, repr=False)

    @property
    def shape_generalization(self) -> bool | None:
        return self.numerics.shape_generalization if self.numerics else None

    @property
    def shape_generalization_reason(self) -> str | None:
        if self.numerics is not None and self.numerics.baseline_failed:
            return "baseline sample failed; shape generalization was never evaluated"
        return None

    @property
    def exit_code(self) -> int:
        return EXIT_CODES[self.status]

    @property
    def rows_note(self) -> str | None:
        """Why the rows are not known to be independent. None if they are."""
        if self.row_independent:
            return None
        return self.row_reason or f"not checked: the verdict is {self.status}"

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "model_family": self.model_family,
            "capture_strategy": self.capture_strategy,
            "opset": self.opset,
            "op_types": self.op_types,
            "numerics": self.numerics.to_dict() if self.numerics else None,
            "shape_generalization": self.shape_generalization,
            "shape_generalization_reason": self.shape_generalization_reason,
            "recommended_backend": self.recommended_backend,
            "reason": self.reason,
            "input_names": list(self.input_names),
            "dynamic_dims": self.dynamic_dims,
            "axes": [fact.to_dict() for fact in self.axes],
            "output_axes": list(self.output_axes),
            "unsupported_ops": self.unsupported_ops,
            "warnings": self.warnings,
            "row_independent": self.row_independent,
            "row_reason": self.rows_note,
            "onnx_path": str(self.onnx_path) if self.onnx_path else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ExportVerdict:
        """The inverse of to_dict(). `prepared` and `onnx_program` come back as None."""
        data = copy.deepcopy(data)
        values = {f.name: data[f.name] for f in fields(cls) if f.name in data}
        values["status"] = Status(data["status"])
        values["recommended_backend"] = BackendName(data["recommended_backend"])
        if data.get("numerics") is not None:
            values["numerics"] = NumericsReport.from_dict(data["numerics"])
        values["input_names"] = tuple(data.get("input_names", ()))
        values["axes"] = [AxisFact.from_dict(fact) for fact in data.get("axes", [])]
        values["onnx_path"] = Path(data["onnx_path"]) if data.get("onnx_path") else None
        return cls(**values)

    def take_session(self) -> Any:
        """The ORT session of verify, one time. The verdict lets go of it, so an unused
        session does not keep a second copy of the weights alive."""
        session, self._session = self._session, None
        return session

    def redacted_dict(self, paths: Iterable[str | Path | None]) -> dict:
        """to_dict() with `paths` reduced to their names in the free-text fields."""
        data = self.to_dict()
        data["reason"] = hide_paths(data["reason"], paths)
        data["warnings"] = [hide_paths(w, paths) for w in data["warnings"]]
        if data["row_reason"] is not None:
            data["row_reason"] = hide_paths(data["row_reason"], paths)
        return data
