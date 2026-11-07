"""ExportVerdict: the one object that all other code reads.

CLEAN      exports, the numbers match, works on unseen shapes  -> serve via ORT
DEGRADED   exports, but the numbers differ beyond tolerance    -> serve via torch
FAILED     does not export (or ORT cannot run it)              -> serve via torch
UNVERIFIED a .onnx file with no reference model                -> serve via ORT, say so
"""

import copy
import re
import time
import traceback
from collections.abc import Iterable
from dataclasses import dataclass, field, fields
from enum import StrEnum
from pathlib import Path
from typing import Any

import onnx
import torch

from downshift._imports import import_object
from downshift.adapters import registry
from downshift.adapters.base import Adapter, Family, Prepared, VaryFn
from downshift.core import memo
from downshift.core.axes import AxisFact, axis_facts, classify_outputs
from downshift.core.capture import capture, op_type_histogram
from downshift.core.phase import Phase, report
from downshift.core.rows import check_rows
from downshift.core.shapes import (
    apply_dynamic_override,
    dynamic_bounds,
    lower_axis_max,
    pin_vary_fn,
    safe_capture_inputs,
)
from downshift.core.verify import (
    NumericsReport,
    OnnxRuntimeError,
    first_line,
    load_session,
    verify,
)
from downshift.settings import DEFAULT_SAMPLES
from downshift.sources import hide_paths


class Status(StrEnum):
    CLEAN = "CLEAN"
    DEGRADED = "DEGRADED"
    FAILED = "FAILED"
    UNVERIFIED = "UNVERIFIED"


class BackendName(StrEnum):
    """A concrete backend. "auto" is only a CLI choice (serve.options.BackendChoice)."""

    onnxruntime = "onnxruntime"
    torch = "torch"


EXIT_CODES = {Status.CLEAN: 0, Status.FAILED: 1, Status.DEGRADED: 2, Status.UNVERIFIED: 3}

_ATEN_OP = re.compile(r"(?:torch\.ops\.)?aten\.(\w+)(?:\.\w+)?")


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
    def from_dict(cls, data: dict) -> "ExportVerdict":
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


def axes_for(prepared: Prepared, numerics: NumericsReport | None) -> list[AxisFact]:
    return axis_facts(
        dynamic_bounds(prepared.input_names, prepared.dynamic_shapes),
        prepared.input_names,
        numerics.sample_shapes if numerics is not None else None,
    )


def _verdict(
    prepared: Prepared, status: Status, backend: BackendName, **values: Any
) -> ExportVerdict:
    """An ExportVerdict with the facts of `prepared` filled in."""
    return ExportVerdict(
        status=status,
        model_family=prepared.family,
        recommended_backend=backend,
        input_names=prepared.input_names,
        dynamic_dims=prepared.dynamic_dims,
        axes=axes_for(prepared, None),
        prepared=prepared,
        **values,
    )


def unverified_verdict(prepared: Prepared, reason: str, output_axes: list[str]) -> ExportVerdict:
    """For a model whose export was skipped: served eagerly, with no numerics."""
    return _verdict(
        prepared, Status.UNVERIFIED, BackendName.torch, reason=reason, output_axes=output_axes
    )


def _tied_weight_warnings(model: torch.nn.Module) -> list[str]:
    seen: dict[int, str] = {}
    tied: list[str] = []
    for name, param in model.named_parameters(remove_duplicate=False):
        first = seen.setdefault(id(param), name)
        if first != name:
            tied.append(f"tied weights: {name} shares storage with {first}")
    return tied


def _drop_frame_locals(exc: BaseException | None) -> None:
    """A kept exception would hold the torch.export frames (FX graphs, the model) alive."""
    seen: set[int] = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if exc.__traceback__ is not None:
            traceback.clear_frames(exc.__traceback__)
        exc = exc.__cause__ or exc.__context__


def prepare_model(
    model: torch.nn.Module,
    example_inputs: tuple | None = None,
    adapter: Adapter | str | None = None,
    dynamic: dict[str, list[int]] | None = None,
    vary: VaryFn | str | None = None,
    axis_max: dict[str, int] | None = None,
) -> Prepared:
    """Select an adapter, find example inputs, and flatten the model for export.

    The example inputs are the ones given, else the adapter's guess, else an error. `vary`
    (a function or an import spec) replaces the sampler of the adapter. `axis_max` lowers the
    named Dims and pins verification sample 1 at those sizes. The adapter does this for its
    own axes. With `dynamic`, downshift does it here for the `<input>_<axis>` axes.
    """
    if isinstance(adapter, str):
        adapter = registry.get(adapter)
    if adapter is None:
        adapter = registry.detect(model, example_inputs)
    if example_inputs is None:
        example_inputs = adapter.example_inputs(model)
    if example_inputs is None:
        raise ValueError(
            f"Cannot find example inputs for {type(model).__name__} with the "
            f"{adapter.name!r} adapter. Pass them explicitly. From Python, use "
            "check(model, example_inputs=(tensor, ...)). From the CLI, use --inputs "
            "module:function. The function returns a tuple of forward() arguments."
        )
    prepared = adapter.prepare(model, example_inputs, axis_max=None if dynamic else axis_max)
    if dynamic:
        prepared.dynamic_shapes = lower_axis_max(
            apply_dynamic_override(prepared.input_names, prepared.inputs, dynamic), axis_max
        )
        if axis_max:
            prepared.vary_fn = pin_vary_fn(
                prepared.inputs, prepared.dynamic_shapes, axis_max, prepared.vary_fn
            )
    if vary is not None:
        prepared.vary_fn = import_object(vary) if isinstance(vary, str) else vary
    return prepared


def _verify_into(
    verdict: ExportVerdict,
    graph: bytes | Path,
    passed: str,
    failed: str,
    unrunnable: str,
    k: int,
    atol: float | None,
    rtol: float | None,
    seed: int,
    timings: dict[str, float] | None,
) -> ExportVerdict:
    """Run verify() and the row check on one CPU session, and set the outcome on `verdict`.
    The prefixes start the reason of a pass, a numeric failure, or an ORT failure."""
    prepared = verdict.prepared
    assert prepared is not None
    report(Phase.verify)
    start = time.perf_counter()
    try:
        session = load_session(graph)
        numerics = verify(
            prepared.model,
            session,
            prepared.inputs,
            prepared.dynamic_shapes,
            vary_fn=prepared.vary_fn,
            k=k,
            atol=atol,
            rtol=rtol,
            seed=seed,
        )
        verdict.row_independent, verdict.row_reason = check_rows(prepared, numerics, session)
    except OnnxRuntimeError as exc:
        message = first_line(exc)
        verdict.status, verdict.recommended_backend = Status.FAILED, BackendName.torch
        verdict.reason = f"{unrunnable}: {message}"
        verdict.warnings.append(message)
        return verdict
    finally:
        if timings is not None:
            timings[Phase.verify] = time.perf_counter() - start

    verdict.numerics = numerics
    verdict._session = session
    verdict.axes = axes_for(prepared, numerics)
    err = f"(max abs err {numerics.max_abs_err:.2e})"
    if numerics.passed:
        verdict.status, verdict.recommended_backend = Status.CLEAN, BackendName.onnxruntime
        verdict.reason = f"{passed} across {numerics.samples_tested} samples {err}"
    else:
        verdict.status, verdict.recommended_backend = Status.DEGRADED, BackendName.torch
        verdict.reason = f"{failed} on {numerics.failures}/{numerics.samples_tested} samples {err}"
    return verdict


def build_verdict(
    prepared: Prepared,
    k: int = DEFAULT_SAMPLES,
    verify_numerics: bool = True,
    atol: float | None = None,
    rtol: float | None = None,
    seed: int = 0,
    timings: dict[str, float] | None = None,
    _external_data_threshold: int | None = None,
) -> ExportVerdict:
    """Capture, then verify. With verify_numerics=False (--no-verify), the graph is made but
    the verdict is UNVERIFIED. `timings` gets the seconds of Phase.export and Phase.verify."""
    warnings = _tied_weight_warnings(prepared.model)
    if prepared.model.training:
        warnings.append("model was in training mode; switched to eval() for export")
        prepared.model.eval()

    report(Phase.export)
    start = time.perf_counter()
    result = capture(
        prepared.model,
        safe_capture_inputs(prepared.inputs, prepared.dynamic_shapes),
        prepared.dynamic_shapes,
        external_data_threshold=_external_data_threshold,
    )
    if timings is not None:
        timings[Phase.export] = time.perf_counter() - start
    # A translation failure has no per-strategy exceptions, only the one it raised.
    capture_exceptions = result.exceptions or (
        [(result.capture_strategy or "translation", result.exception)]
        if result.exception is not None
        else []
    )
    for _, kept in capture_exceptions:
        _drop_frame_locals(kept)
    verdict = _verdict(
        prepared,
        Status.FAILED,
        BackendName.torch,
        capture_strategy=result.capture_strategy,
        opset=result.opset,
        op_types=result.op_types,
        warnings=warnings,
        onnx_program=result.onnx_program,
        onnx_bytes=result.onnx_bytes,
        onnx_path=result.onnx_path,
        _tmpdir=result.tmpdir,
        capture_stderr=result.stderr,
        capture_exceptions=capture_exceptions,
    )

    if not result.success:
        exc = result.exception
        message = f"{type(exc).__name__}: {exc}" if exc is not None else "export failed"
        verdict.reason = message.splitlines()[0]
        # Scan each strategy's message: strict=True often hides the real operation.
        messages = [str(e) for _, e in result.exceptions] if result.exceptions else [message]
        verdict.unsupported_ops = sorted({op for m in messages for op in _ATEN_OP.findall(m)})
        return verdict

    via = f"exported via {result.capture_strategy}"
    if not verify_numerics:
        verdict.status, verdict.recommended_backend = Status.UNVERIFIED, BackendName.onnxruntime
        verdict.reason = f"{via}; numerics never checked"
        return verdict

    graph = result.onnx_bytes or result.onnx_path
    assert graph is not None
    _verify_into(
        verdict,
        graph,
        f"{via}; numerics ok",
        f"{via} but numerics diverge",
        f"{via} but ONNX Runtime cannot run the graph",
        k,
        atol,
        rtol,
        seed,
        timings,
    )
    if verdict.numerics is not None and prepared.family == Family.pyg:
        verdict.output_axes = classify_outputs(
            verdict.numerics.sample_shapes,
            verdict.numerics.output_shapes,
            prepared.input_names.index("x"),
            prepared.input_names.index("edge_index"),
        )
    return verdict


def check(
    model: torch.nn.Module,
    example_inputs: tuple | None = None,
    k: int = DEFAULT_SAMPLES,
    adapter: Adapter | str | None = None,
    dynamic: dict[str, list[int]] | None = None,
    fp16: bool = False,
    verify_numerics: bool = True,
    atol: float | None = None,
    rtol: float | None = None,
    seed: int = 0,
    vary: VaryFn | str | None = None,
    axis_max: dict[str, int] | None = None,
    cache: bool = True,
    _memo_key: str | None = None,
) -> ExportVerdict:
    """Export in memory, verify, and return the verdict. Nothing is written to disk.

    check() never reads the export memo, because it is the audit gate. It does store a CLEAN
    or DEGRADED result there for export() and app_for(), unless `cache=False`.

    fp16=True exports a float16 copy; the model and inputs of the caller do not change.
    atol and rtol default to the tolerances of the model's dtype. `seed` makes the samples
    reproducible. See prepare_model for `vary` and `axis_max`.
    """
    key = None
    if cache and verify_numerics:
        key = _memo_key or memo.guarded(
            memo.model_key,
            model,
            example_inputs,
            adapter=adapter,
            dynamic=dynamic,
            k=k,
            seed=seed,
            atol=atol,
            rtol=rtol,
            vary=vary,
            axis_max=axis_max,
            fp16=fp16,
        )
    if fp16:
        model = copy.deepcopy(model).half()
        if example_inputs is not None:
            example_inputs = tuple(
                t.half() if isinstance(t, torch.Tensor) and t.is_floating_point() else t
                for t in example_inputs
            )
    prepared = prepare_model(model, example_inputs, adapter, dynamic, vary=vary, axis_max=axis_max)
    verdict = build_verdict(
        prepared, k=k, verify_numerics=verify_numerics, atol=atol, rtol=rtol, seed=seed
    )
    if key is not None:
        memo.store_verdict(key, verdict)
    return verdict


def intake(
    onnx_path: str | Path,
    reference: torch.nn.Module | None = None,
    example_inputs: tuple | None = None,
    adapter: Adapter | str | None = None,
    k: int = DEFAULT_SAMPLES,
    dynamic: dict[str, list[int]] | None = None,
    atol: float | None = None,
    rtol: float | None = None,
    seed: int = 0,
    vary: VaryFn | str | None = None,
    axis_max: dict[str, int] | None = None,
    timings: dict[str, float] | None = None,
) -> ExportVerdict:
    """The verdict for a .onnx file that another tool made. Without `reference`, it is
    UNVERIFIED. With it, the graph is verified against it like a new export."""
    onnx_path = Path(onnx_path)
    proto = onnx.load(str(onnx_path), load_external_data=False)
    opset = next((imp.version for imp in proto.opset_import if imp.domain in ("", "ai.onnx")), None)
    op_types = op_type_histogram(proto.graph.node)

    if reference is None:
        initializers = {init.name for init in proto.graph.initializer}
        return ExportVerdict(
            status=Status.UNVERIFIED,
            model_family="onnx",
            recommended_backend=BackendName.onnxruntime,
            reason="no reference model supplied; served as-is, numerics never checked",
            opset=opset,
            op_types=op_types,
            input_names=tuple(i.name for i in proto.graph.input if i.name not in initializers),
            onnx_path=onnx_path,
        )

    prepared = prepare_model(
        reference, example_inputs, adapter, dynamic, vary=vary, axis_max=axis_max
    )
    verdict = _verdict(prepared, Status.FAILED, BackendName.torch, opset=opset, op_types=op_types)
    verdict.onnx_path = onnx_path
    return _verify_into(
        verdict,
        onnx_path,
        "pre-built ONNX matches reference",
        "pre-built ONNX diverges from reference",
        "pre-built ONNX cannot run in ONNX Runtime",
        k,
        atol,
        rtol,
        seed,
        timings,
    )
