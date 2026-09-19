"""ExportVerdict: the one object everything else reads.

CLEAN      exports, numerics match, survives shapes it wasn't traced on -> serve via ORT
DEGRADED   exports but numerics drift past tolerance             -> serve via torch
FAILED     won't export                                          -> serve via torch
UNVERIFIED a .onnx handed to us with no reference model         -> serve via ORT, say so
"""

import copy
import re
import time
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Literal

import torch

from downshift.adapters import registry
from downshift.adapters.base import Adapter, Prepared, VaryFn
from downshift.core.capture import capture
from downshift.core.inputs import synthesize
from downshift.core.shapes import apply_dynamic_override, safe_capture_inputs
from downshift.core.verify import NumericsReport, OnnxRuntimeError, verify
from downshift.loading import import_object

Status = Literal["CLEAN", "DEGRADED", "FAILED", "UNVERIFIED"]


class BackendName(StrEnum):
    """The concrete backends a verdict can recommend/serve; never "auto" (that's a CLI-only
    selection sentinel, not a real backend) - see engine.BackendChoice."""

    onnxruntime = "onnxruntime"
    torch = "torch"


EXIT_CODES: dict[str, int] = {"CLEAN": 0, "FAILED": 1, "DEGRADED": 2, "UNVERIFIED": 3}

_ATEN_OP = re.compile(r"(?:torch\.ops\.)?aten\.(\w+)(?:\.\w+)?")


@dataclass
class ExportVerdict:
    status: Status
    model_family: str
    capture_strategy: str | None
    opset: int | None
    op_types: dict[str, int]  # count-descending histogram
    numerics: NumericsReport | None
    recommended_backend: BackendName
    reason: str
    input_names: tuple[str, ...] = ()
    dynamic_dims: dict[str, list[int]] = field(default_factory=dict)
    unsupported_ops: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    onnx_path: Path | None = None
    onnx_program: object | None = field(default=None, repr=False)  # torch.onnx.ONNXProgram
    onnx_bytes: bytes = field(default=b"", repr=False)  # serialized once by capture()
    prepared: Prepared | None = field(default=None, repr=False)
    # Debug-only: not JSON-able, excluded from to_dict(); the CLI logs these at --log-level
    # debug when the status is FAILED.
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
            "unsupported_ops": self.unsupported_ops,
            "warnings": self.warnings,
            "onnx_path": str(self.onnx_path) if self.onnx_path else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ExportVerdict":
        """Rebuild from to_dict()'s output, e.g. in a `serve --workers N` worker that takes
        its verdict from the parent's export instead of running one itself. `prepared` and
        `onnx_program` weren't serialized, so they come back None; the caller sets
        `onnx_path` afterwards if the ONNX graph now lives at a worker-local temp path.
        `shape_generalization`/`_reason` are derived properties, so they're ignored here.
        """
        numerics_data = data.get("numerics")
        numerics = NumericsReport.from_dict(numerics_data) if numerics_data is not None else None
        onnx_path = data.get("onnx_path")
        return cls(
            status=data["status"],
            model_family=data["model_family"],
            capture_strategy=data.get("capture_strategy"),
            opset=data.get("opset"),
            op_types=dict(data.get("op_types", {})),
            numerics=numerics,
            recommended_backend=BackendName(data["recommended_backend"]),
            reason=data.get("reason", ""),
            input_names=tuple(data.get("input_names", ())),
            dynamic_dims=dict(data.get("dynamic_dims", {})),
            unsupported_ops=list(data.get("unsupported_ops", [])),
            warnings=list(data.get("warnings", [])),
            onnx_path=Path(onnx_path) if onnx_path else None,
        )


def numerics_outcome(
    numerics: NumericsReport, passed_prefix: str, failed_prefix: str
) -> tuple[Status, BackendName, str]:
    """Numerics decide the verdict: pass -> CLEAN via ORT, fail -> DEGRADED via torch.

    The prefixes open the reason string; the sample counts and error are appended.
    """
    err = f"(max abs err {numerics.max_abs_err:.2e})"
    if numerics.passed:
        reason = f"{passed_prefix} across {numerics.samples_tested} samples {err}"
        return "CLEAN", BackendName.onnxruntime, reason
    reason = f"{failed_prefix} on {numerics.failures}/{numerics.samples_tested} samples {err}"
    return "DEGRADED", BackendName.torch, reason


def _tied_weight_warnings(model: torch.nn.Module) -> list[str]:
    seen: dict[int, str] = {}
    tied: list[str] = []
    for name, param in model.named_parameters(remove_duplicate=False):
        first = seen.setdefault(id(param), name)
        if first != name:
            tied.append(f"tied weights: {name} shares storage with {first}")
    return tied


def prepare_model(
    model: torch.nn.Module,
    example_inputs: tuple | None = None,
    adapter: Adapter | str | None = None,
    dynamic: dict[str, list[int]] | None = None,
    vary: VaryFn | str | None = None,
) -> Prepared:
    """Pick an adapter, synthesise inputs if needed, and flatten into export form.

    vary overrides the adapter's own vary_fn: a spec string is imported like --adapter's
    custom-file form (fn(i) -> inputs; fn(0) should return the example).
    """
    if isinstance(adapter, str):
        adapter = registry.get(adapter)
    if adapter is None:
        adapter = registry.detect(model, example_inputs)
    example_inputs = synthesize(model, adapter, example_inputs)
    prepared = adapter.prepare(model, example_inputs)
    if dynamic:
        prepared.dynamic_shapes = apply_dynamic_override(
            prepared.input_names, prepared.inputs, dynamic
        )
    if vary is not None:
        prepared.vary_fn = import_object(vary) if isinstance(vary, str) else vary
    return prepared


def build_verdict(
    prepared: Prepared,
    k: int = 8,
    verify_numerics: bool = True,
    atol: float | None = None,
    rtol: float | None = None,
    seed: int = 0,
    timings: dict[str, float] | None = None,
) -> ExportVerdict:
    """Capture, then verify. verify_numerics=False is the --no-verify escape hatch: the
    graph is still produced but the verdict is UNVERIFIED, never CLEAN.

    `timings`, when given, gets "export" (the capture() call) and "verify" (the verify()
    call) wall-clock seconds added to it - the CLI's Boot banner row and /metadata's
    `boot` field read it back from ServingState.timings (see serve/engine.py).
    """
    warnings = _tied_weight_warnings(prepared.model)
    if prepared.model.training:
        warnings.append("model was in training mode; switched to eval() for export")
        prepared.model.eval()

    capture_start = time.perf_counter()
    result = capture(
        prepared.model,
        safe_capture_inputs(prepared.inputs, prepared.dynamic_shapes),
        prepared.dynamic_shapes,
    )
    if timings is not None:
        timings["export"] = time.perf_counter() - capture_start
    # exceptions is normally one entry per strategy tried; a translation failure (torch.export
    # itself succeeded) has none of those, so it falls back to the single exception it raised.
    capture_exceptions = result.exceptions or (
        [(result.capture_strategy or "translation", result.exception)]
        if result.exception is not None
        else []
    )
    verdict = ExportVerdict(
        status="FAILED",
        model_family=prepared.family,
        capture_strategy=result.capture_strategy,
        opset=result.opset,
        op_types=result.op_types,
        numerics=None,
        recommended_backend=BackendName.torch,
        reason="",
        input_names=prepared.input_names,
        dynamic_dims=prepared.dynamic_dims,
        warnings=warnings,
        onnx_program=result.onnx_program,
        onnx_bytes=result.onnx_bytes,
        prepared=prepared,
        capture_stderr=result.stderr,
        capture_exceptions=capture_exceptions,
    )

    if not result.success:
        exc = result.exception
        message = f"{type(exc).__name__}: {exc}" if exc is not None else "export failed"
        verdict.reason = message.splitlines()[0]
        # Mined from every strategy's message when there were several (strict=True's message
        # is often generic and would lose whatever strict=False said about the real op).
        messages = [str(e) for _, e in result.exceptions] if result.exceptions else [message]
        verdict.unsupported_ops = sorted({op for m in messages for op in _ATEN_OP.findall(m)})
        return verdict

    if not verify_numerics:
        verdict.status, verdict.recommended_backend = "UNVERIFIED", BackendName.onnxruntime
        verdict.reason = f"exported via {result.capture_strategy}; numerics never checked"
        return verdict

    verify_start = time.perf_counter()
    try:
        numerics = verify(
            prepared.model,
            result.onnx_bytes,
            prepared.inputs,
            prepared.dynamic_shapes,
            vary_fn=prepared.vary_fn,
            k=k,
            atol=atol,
            rtol=rtol,
            seed=seed,
        )
    except OnnxRuntimeError as exc:
        if timings is not None:
            timings["verify"] = time.perf_counter() - verify_start
        message = str(exc).splitlines()[0]
        verdict.reason = (
            f"exported via {result.capture_strategy} but ONNX Runtime cannot run the "
            f"graph: {message}"
        )
        verdict.warnings.append(message)
        return verdict
    if timings is not None:
        timings["verify"] = time.perf_counter() - verify_start

    verdict.numerics = numerics
    verdict.status, verdict.recommended_backend, verdict.reason = numerics_outcome(
        numerics,
        f"exported via {result.capture_strategy}; numerics ok",
        f"exported via {result.capture_strategy} but numerics diverge",
    )
    return verdict


def check(
    model: torch.nn.Module,
    example_inputs: tuple | None = None,
    k: int = 8,
    adapter: Adapter | str | None = None,
    dynamic: dict[str, list[int]] | None = None,
    fp16: bool = False,
    verify_numerics: bool = True,
    atol: float | None = None,
    rtol: float | None = None,
    seed: int = 0,
    vary: VaryFn | str | None = None,
) -> ExportVerdict:
    """Export in memory, verify, and return the verdict. Writes nothing to disk.

    fp16=True casts a deep copy of `model` to float16 and leaves the caller's model and
    its parameters untouched; `example_inputs`, if given, are cast on the copies used for
    export, not the tensors the caller passed in. The model IS still switched to eval()
    in place if it was in training mode (see the "training mode" warning on the returned
    verdict) - with fp16=True that happens to the copy, so the caller's model keeps
    whichever mode it was already in.

    atol/rtol default to None, meaning "pick by the model's floating dtype" (see
    verify.default_tolerances). seed makes the verification samples reproducible. vary
    overrides the adapter's own vary_fn; see prepare_model.
    """
    if fp16:
        model = copy.deepcopy(model).half()
        if example_inputs is not None:
            example_inputs = tuple(
                t.half() if isinstance(t, torch.Tensor) and t.is_floating_point() else t
                for t in example_inputs
            )
    prepared = prepare_model(model, example_inputs, adapter, dynamic, vary=vary)
    return build_verdict(
        prepared, k=k, verify_numerics=verify_numerics, atol=atol, rtol=rtol, seed=seed
    )
