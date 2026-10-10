"""ExportVerdict: the one object that all other code reads.

CLEAN      exports, the numbers match, works on shapes that it was not traced on -> serve via ORT
DEGRADED   exports, but the numbers differ by more than the tolerance           -> serve via torch
FAILED     does not export                                                      -> serve via torch
UNVERIFIED a .onnx file that we received with no reference model                -> serve via ORT, say so
"""

import copy
import re
import time
import traceback
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import torch

from downshift._imports import import_object
from downshift.adapters import registry
from downshift.adapters.base import Adapter, Family, Prepared, VaryFn
from downshift.core import memo
from downshift.core.axes import AxisFact, axis_facts, classify_outputs
from downshift.core.capture import capture
from downshift.core.inputs import synthesize
from downshift.core.phase import Phase, report
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
    """The outcome of a verdict. See the module docstring. It is equal to its plain string. JSON
    and the manifest therefore carry "CLEAN" and the other values as before."""

    CLEAN = "CLEAN"
    DEGRADED = "DEGRADED"
    FAILED = "FAILED"
    UNVERIFIED = "UNVERIFIED"


class BackendName(StrEnum):
    """The concrete backends that a verdict can recommend and serve. It is never "auto". "auto"
    is a selection value for the CLI only and not a real backend. See engine.BackendChoice."""

    onnxruntime = "onnxruntime"
    torch = "torch"


EXIT_CODES = {Status.CLEAN: 0, Status.FAILED: 1, Status.DEGRADED: 2, Status.UNVERIFIED: 3}

_ATEN_OP = re.compile(r"(?:torch\.ops\.)?aten\.(\w+)(?:\.\w+)?")


@dataclass
class ExportVerdict:
    status: Status
    model_family: str
    capture_strategy: str | None
    opset: int | None
    op_types: dict[str, int]  # histogram, in descending order of count
    numerics: NumericsReport | None
    recommended_backend: BackendName
    reason: str
    input_names: tuple[str, ...] = ()
    dynamic_dims: dict[str, list[int]] = field(default_factory=dict)
    axes: list[AxisFact] = field(default_factory=list)
    output_axes: list[str] = field(default_factory=list)  # PyG only: node/edge/fixed/unknown
    unsupported_ops: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    onnx_path: Path | None = None
    onnx_program: object | None = field(default=None, repr=False)  # torch.onnx.ONNXProgram
    onnx_bytes: bytes = field(default=b"", repr=False)  # serialized once by capture()
    # Keeps the temporary directory for external data alive (onnx_path is in it). Never serialized.
    _tmpdir: object | None = field(default=None, repr=False)
    # The CPU session that verify ran on. Never serialized. take_session() gives it to the server.
    _session: object | None = field(default=None, repr=False, compare=False)
    prepared: Prepared | None = field(default=None, repr=False)
    # For debugging only. Not JSON-able, and to_dict() excludes them. The CLI logs them at
    # --log-level debug when the status is FAILED.
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
            "onnx_path": str(self.onnx_path) if self.onnx_path else None,
        }

    def take_session(self) -> Any:
        """The ONNX Runtime session that verify built, one time. The verdict lets go of it. A
        server that builds its own session then does not keep a second copy of the weights
        alive."""
        session, self._session = self._session, None
        return session

    def redacted_dict(self, paths: Iterable[str | Path | None]) -> dict:
        """to_dict() for someone outside this machine. Free-text fields quote the path that an
        exception received. Downshift therefore reduces `paths` to their names in reason and
        warnings."""
        data = self.to_dict()
        data["reason"] = hide_paths(data["reason"], paths)
        data["warnings"] = [hide_paths(w, paths) for w in data["warnings"]]
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "ExportVerdict":
        """Build the verdict again from the output of to_dict(). Example: a `serve --workers N`
        worker takes its verdict from the export of the parent and does not run one itself.
        Downshift did not serialize `prepared` and `onnx_program`, so they come back as None.
        If the ONNX graph is now at a temporary path in the worker, the caller sets `onnx_path`
        afterward. `shape_generalization` and `_reason` are derived properties, so this
        function ignores them.
        """
        numerics_data = data.get("numerics")
        numerics = NumericsReport.from_dict(numerics_data) if numerics_data is not None else None
        onnx_path = data.get("onnx_path")
        return cls(
            status=Status(data["status"]),
            model_family=data["model_family"],
            capture_strategy=data.get("capture_strategy"),
            opset=data.get("opset"),
            op_types=dict(data.get("op_types", {})),
            numerics=numerics,
            recommended_backend=BackendName(data["recommended_backend"]),
            reason=data.get("reason", ""),
            input_names=tuple(data.get("input_names", ())),
            dynamic_dims=dict(data.get("dynamic_dims", {})),
            axes=[AxisFact.from_dict(fact) for fact in data.get("axes", [])],
            output_axes=list(data.get("output_axes", [])),
            unsupported_ops=list(data.get("unsupported_ops", [])),
            warnings=list(data.get("warnings", [])),
            onnx_path=Path(onnx_path) if onnx_path else None,
        )


def axes_for(prepared: Prepared, numerics: NumericsReport | None) -> list[AxisFact]:
    """The served bounds from the dynamic_shapes of the export, next to what verify sampled
    (None if `numerics` is None)."""
    return axis_facts(
        dynamic_bounds(prepared.input_names, prepared.dynamic_shapes),
        prepared.input_names,
        numerics.sample_shapes if numerics is not None else None,
    )


def numerics_outcome(
    numerics: NumericsReport, passed_prefix: str, failed_prefix: str
) -> tuple[Status, BackendName, str]:
    """The numerics decide the verdict. If they pass, the verdict is CLEAN via ORT. If they fail,
    it is DEGRADED via torch.

    The prefixes start the reason string. Downshift adds the sample counts and the error after
    them.
    """
    err = f"(max abs err {numerics.max_abs_err:.2e})"
    if numerics.passed:
        reason = f"{passed_prefix} across {numerics.samples_tested} samples {err}"
        return Status.CLEAN, BackendName.onnxruntime, reason
    reason = f"{failed_prefix} on {numerics.failures}/{numerics.samples_tested} samples {err}"
    return Status.DEGRADED, BackendName.torch, reason


def _tied_weight_warnings(model: torch.nn.Module) -> list[str]:
    seen: dict[int, str] = {}
    tied: list[str] = []
    for name, param in model.named_parameters(remove_duplicate=False):
        first = seen.setdefault(id(param), name)
        if first != name:
            tied.append(f"tied weights: {name} shares storage with {first}")
    return tied


def _drop_frame_locals(exc: BaseException | None) -> None:
    """The traceback of a kept exception holds the locals of each torch.export frame (FX graphs,
    fake tensors, the model) as long as the verdict lives. Downshift clears them. The formatted
    traceback stays. The debug log reads only that."""
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
    """Select an adapter, synthesize inputs if necessary, and flatten the model into the form
    for export.

    vary replaces the vary_fn of the adapter. Downshift imports a spec string in the same
    way as the custom-file form of --adapter (fn(i) -> inputs, and fn(0) must return the
    example).

    axis_max ({axis name: largest size to serve}, --axis-max) lowers the maximum of the named
    Dims. It also pins verification sample 1 at those sizes. The adapter does this for its own
    axes. With `dynamic`, downshift replaces the axes of the adapter with `<input>_<axis>`
    axes. In that case, downshift applies axis_max to those here.
    """
    if isinstance(adapter, str):
        adapter = registry.get(adapter)
    if adapter is None:
        adapter = registry.detect(model, example_inputs)
    example_inputs = synthesize(model, adapter, example_inputs)
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
    """Capture, then verify. verify_numerics=False is the --no-verify escape. Downshift still
    makes the graph, but the verdict is UNVERIFIED and never CLEAN.

    If you give `timings`, downshift adds the wall-clock seconds of Phase.export (the capture()
    call) and Phase.verify (the verify() call) to it. It also tells /ready of the serve loader
    which of the two runs (see core/phase.py). The Boot row of the CLI banner and the `boot`
    field of /metadata read it again from ServingState.timings (see serve/engine.py).
    """
    warnings = _tied_weight_warnings(prepared.model)
    if prepared.model.training:
        warnings.append("model was in training mode; switched to eval() for export")
        prepared.model.eval()

    report(Phase.export)
    capture_start = time.perf_counter()
    result = capture(
        prepared.model,
        safe_capture_inputs(prepared.inputs, prepared.dynamic_shapes),
        prepared.dynamic_shapes,
        external_data_threshold=_external_data_threshold,
    )
    if timings is not None:
        timings[Phase.export] = time.perf_counter() - capture_start
    # exceptions normally has one entry for each strategy that was tried. A translation failure
    # (torch.export itself succeeded) has none of these. It then uses the one exception that it
    # raised.
    capture_exceptions = result.exceptions or (
        [(result.capture_strategy or "translation", result.exception)]
        if result.exception is not None
        else []
    )
    for _, kept in capture_exceptions:
        _drop_frame_locals(kept)
    verdict = ExportVerdict(
        status=Status.FAILED,
        model_family=prepared.family,
        capture_strategy=result.capture_strategy,
        opset=result.opset,
        op_types=result.op_types,
        numerics=None,
        recommended_backend=BackendName.torch,
        reason="",
        input_names=prepared.input_names,
        dynamic_dims=prepared.dynamic_dims,
        axes=axes_for(prepared, None),
        warnings=warnings,
        onnx_program=result.onnx_program,
        onnx_bytes=result.onnx_bytes,
        onnx_path=result.onnx_path,
        _tmpdir=result.tmpdir,
        prepared=prepared,
        capture_stderr=result.stderr,
        capture_exceptions=capture_exceptions,
    )

    if not result.success:
        exc = result.exception
        message = f"{type(exc).__name__}: {exc}" if exc is not None else "export failed"
        verdict.reason = message.splitlines()[0]
        # Taken from the message of each strategy, if there were several. The message of
        # strict=True is often generic. It would lose what strict=False said about the real
        # operation.
        messages = [str(e) for _, e in result.exceptions] if result.exceptions else [message]
        verdict.unsupported_ops = sorted({op for m in messages for op in _ATEN_OP.findall(m)})
        return verdict

    if not verify_numerics:
        verdict.status, verdict.recommended_backend = Status.UNVERIFIED, BackendName.onnxruntime
        verdict.reason = f"exported via {result.capture_strategy}; numerics never checked"
        return verdict

    report(Phase.verify)
    verify_start = time.perf_counter()
    try:
        session = load_session(result.onnx_bytes or result.onnx_path)
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
    except OnnxRuntimeError as exc:
        message = first_line(exc)
        verdict.reason = (
            f"exported via {result.capture_strategy} but ONNX Runtime cannot run the "
            f"graph: {message}"
        )
        verdict.warnings.append(message)
        return verdict
    finally:
        if timings is not None:
            timings[Phase.verify] = time.perf_counter() - verify_start

    verdict.numerics = numerics
    verdict._session = session
    verdict.axes = axes_for(prepared, numerics)
    if prepared.family == Family.pyg:
        verdict.output_axes = classify_outputs(
            numerics.sample_shapes,
            numerics.output_shapes,
            prepared.input_names.index("x"),
            prepared.input_names.index("edge_index"),
        )
    verdict.status, verdict.recommended_backend, verdict.reason = numerics_outcome(
        numerics,
        f"exported via {result.capture_strategy}; numerics ok",
        f"exported via {result.capture_strategy} but numerics diverge",
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
    """Export in memory, verify, and return the verdict. It writes nothing to disk.

    Downshift also keeps a CLEAN or DEGRADED result in the in-process export memo
    (core/memo.py). export() and app_for() can reuse it. check() itself never reads the memo,
    because it is the audit gate. `cache=False` keeps nothing.

    fp16=True casts a deep copy of `model` to float16. The model of the caller and its
    parameters do not change. If you give `example_inputs`, downshift casts the copies that it
    uses for the export. It does not cast the tensors that the caller passed in. Downshift
    still switches the model to eval() in place if it was in training mode (see the "training
    mode" warning on the returned verdict). With fp16=True, this happens to the copy. The model
    of the caller then keeps the mode that it already had.

    atol and rtol are None by default. This means "select by the floating dtype of the model"
    (see verify.default_tolerances). seed makes the verification samples reproducible. vary
    replaces the vary_fn of the adapter. See prepare_model, which also says what axis_max
    does.
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
    if key is not None and verdict.status in memo.STORED_STATUSES:
        entry = memo.entry_from_verdict(verdict)
        if entry is not None:
            memo.MEMO.put(key, entry)
    return verdict


def unverified_verdict(prepared: Prepared, reason: str, output_axes: list[str]) -> ExportVerdict:
    """The verdict for a model whose export was skipped (`--backend torch`). Downshift serves it
    eagerly, with the served axis bounds and without numerics."""
    return ExportVerdict(
        status=Status.UNVERIFIED,
        model_family=prepared.family,
        capture_strategy=None,
        opset=None,
        op_types={},
        numerics=None,
        recommended_backend=BackendName.torch,
        reason=reason,
        input_names=prepared.input_names,
        dynamic_dims=prepared.dynamic_dims,
        axes=axes_for(prepared, None),
        output_axes=output_axes,
        prepared=prepared,
    )
