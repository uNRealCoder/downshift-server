"""ExportVerdict: the one object that all other code reads.

CLEAN      exports, the numbers match, works on unseen shapes  -> serve via ORT
DEGRADED   exports, but the numbers differ beyond tolerance    -> serve via torch
FAILED     does not export (or ORT cannot run it)              -> serve via torch
UNVERIFIED a .onnx file with no reference model                -> serve via ORT, say so

To read this file, go in this order:

1. `ExportVerdict`: the fields. Everything else in the file makes one.
2. `check`: the public gate for an `nn.Module`. It calls `prepare_model`, then `build_verdict`.
3. `prepare_model`: adapter, example inputs, dynamic shapes. It does not export.
4. `build_verdict`: capture (core/capture.py), then `_verify_into`.
5. `_verify_into`: verify (core/verify.py) and the row check, then it sets the status.
6. `intake`: the gate for a .onnx file that another tool made.
"""

import re
import time
import traceback
from pathlib import Path
from typing import Any

import onnx
import torch

from downshift._imports import import_object
from downshift.adapters import registry
from downshift.adapters.base import Adapter
from downshift.core import memo
from downshift.core.axes import axis_facts, classify_outputs
from downshift.core.capture import capture, has_half_weights, op_type_histogram, to_half_precision
from downshift.core.feeds import bound_samples
from downshift.core.phase import report
from downshift.core.rows import check_rows
from downshift.core.shapes import (
    apply_dynamic_override,
    dynamic_bounds,
    lower_axis_max,
    pin_vary_fn,
    safe_capture_inputs,
)
from downshift.core.verify import OnnxRuntimeError, first_line, load_session, verify
from downshift.entities.adapters import Family, Prepared, VaryFn
from downshift.entities.axes import AxisFact
from downshift.entities.phase import Phase
from downshift.entities.verdict import BackendName, ExportVerdict, NumericsReport, Status
from downshift.settings import DEFAULT_SAMPLES

_FP16_NO_KERNEL_HINT = (
    ". If an fp16 operation has no CPU kernel, export without --fp16, or use a newer ONNX Runtime"
)
_ATEN_OP = re.compile(r"(?:torch\.ops\.)?aten\.(\w+)(?:\.\w+)?")


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
    samples: int,
    atol: float | None,
    rtol: float | None,
    seed: int,
    timings: dict[str, float] | None,
    fp16: bool = False,
) -> ExportVerdict:
    """Run verify() and the row check on one CPU session, and set the outcome on `verdict`.
    The prefixes start the reason of a pass, a numeric failure, or an ORT failure."""
    prepared = verdict.prepared
    assert prepared is not None
    report(Phase.verify)
    start = time.perf_counter()
    kept_feeds: list = []
    try:
        session = load_session(graph)
        numerics = verify(
            prepared.model,
            session,
            prepared.inputs,
            prepared.dynamic_shapes,
            vary_fn=prepared.vary_fn,
            samples=samples,
            atol=atol,
            rtol=rtol,
            seed=seed,
            embedding=prepared.embedding,
            keep_feeds=kept_feeds,
            fp16=fp16,
        )
        verdict.row_independent, verdict.row_reason = check_rows(prepared, numerics, session)
    except OnnxRuntimeError as exc:
        message = first_line(exc)
        verdict.status, verdict.recommended_backend = Status.FAILED, BackendName.torch
        verdict.reason = f"{unrunnable}: {message}"
        if fp16:
            verdict.reason += _FP16_NO_KERNEL_HINT
        verdict.warnings.append(message)
        return verdict
    finally:
        if timings is not None:
            timings[Phase.verify] = time.perf_counter() - start

    # The checks ran. Keep the session and the samples: the server reuses the session (it
    # saves a second load), and the GPU serving check replays the samples.
    verdict.numerics = numerics
    verdict._session = session
    verdict.sample_feeds = bound_samples(kept_feeds)
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
    samples: int = DEFAULT_SAMPLES,
    verify_numerics: bool = True,
    atol: float | None = None,
    rtol: float | None = None,
    seed: int = 0,
    timings: dict[str, float] | None = None,
    fp16: bool = False,
    _external_data_threshold: int | None = None,
) -> ExportVerdict:
    """Capture, then verify. With verify_numerics=False (--no-verify), the graph is made but
    the verdict is UNVERIFIED. `timings` gets the seconds of Phase.export and Phase.verify.
    With fp16=True, the fp32 graph changes to fp16 (fp32 inputs and outputs) before it is
    verified against the fp32 model, with the float16 tolerances."""
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
    if fp16 and result.success:
        to_half_precision(result)
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
    # The verdict starts as FAILED on torch. Each step that works changes it: a failed capture
    # keeps it, --no-verify makes it UNVERIFIED, and `_verify_into` makes it CLEAN or DEGRADED.
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
        samples,
        atol,
        rtol,
        seed,
        timings,
        fp16,
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
    samples: int = DEFAULT_SAMPLES,
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

    fp16=True changes the fp32 graph to fp16 after the export. The graph keeps fp32 inputs and
    outputs, and it is verified against the fp32 model with the float16 tolerances.
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
            samples=samples,
            seed=seed,
            atol=atol,
            rtol=rtol,
            vary=vary,
            axis_max=axis_max,
            fp16=fp16,
        )
    prepared = prepare_model(model, example_inputs, adapter, dynamic, vary=vary, axis_max=axis_max)
    verdict = build_verdict(
        prepared,
        samples=samples,
        verify_numerics=verify_numerics,
        atol=atol,
        rtol=rtol,
        seed=seed,
        fp16=fp16,
    )
    if key is not None:
        memo.store_verdict(key, verdict)
    return verdict


def intake(
    onnx_path: str | Path,
    reference: torch.nn.Module | None = None,
    example_inputs: tuple | None = None,
    adapter: Adapter | str | None = None,
    samples: int = DEFAULT_SAMPLES,
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
        samples,
        atol,
        rtol,
        seed,
        timings,
        has_half_weights(proto),
    )
