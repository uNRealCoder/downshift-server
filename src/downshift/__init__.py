"""downshift: serve a PyTorch model over HTTP, with its ONNX export verified against PyTorch first.

Lazy by PEP 562: importing `downshift` (and so `downshift.cli.main`, which imports this module
first) doesn't pull in torch, onnx or onnxruntime. Every name in `__all__` resolves on first
access via `__getattr__`, from the `downshift.core`/`downshift.adapters`/`downshift.serve`
module that actually defines it, and is cached on the module so later access skips
`__getattr__` entirely.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from downshift._version import __version__
from downshift.settings import DEFAULT_SAMPLES

if TYPE_CHECKING:
    from downshift.adapters.base import Adapter, Prepared, VaryFn
    from downshift.core.prevalidated import intake
    from downshift.core.verdict import ExportVerdict, build_verdict, check, prepare_model
    from downshift.core.verify import NumericsReport, OnnxRuntimeError

# Spelled out rather than derived from _LAZY: a literal list is what lets ruff see the
# TYPE_CHECKING imports above as exported (F401) and what `from downshift import *` and
# doc tooling read. A name added to _LAZY must be added here too.
__all__ = [
    "Adapter",
    "ExportVerdict",
    "NumericsReport",
    "OnnxRuntimeError",
    "Prepared",
    "app_for",
    "build_verdict",
    "check",
    "export",
    "intake",
    "prepare_model",
    "__version__",
]

# name -> (module that defines it, attribute name); resolved on first access.
_LAZY = {
    "Adapter": ("downshift.adapters.base", "Adapter"),
    "Prepared": ("downshift.adapters.base", "Prepared"),
    "ExportVerdict": ("downshift.core.verdict", "ExportVerdict"),
    "build_verdict": ("downshift.core.verdict", "build_verdict"),
    "check": ("downshift.core.verdict", "check"),
    "prepare_model": ("downshift.core.verdict", "prepare_model"),
    "NumericsReport": ("downshift.core.verify", "NumericsReport"),
    "OnnxRuntimeError": ("downshift.core.verify", "OnnxRuntimeError"),
    "intake": ("downshift.core.prevalidated", "intake"),
    "app_for": ("downshift.serve", "app_for"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attr = _LAZY[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    import importlib

    value = getattr(importlib.import_module(module_name), attr)
    globals()[name] = value  # cache: subsequent access is a plain attribute lookup
    return value


def export(
    model,
    output: str | Path,
    example_inputs: tuple | None = None,
    k: int = DEFAULT_SAMPLES,
    adapter: Adapter | str | None = None,
    dynamic: dict[str, list[int]] | None = None,
    fp16: bool = False,
    source_path: Path | None = None,
    verify_numerics: bool = True,
    atol: float | None = None,
    rtol: float | None = None,
    seed: int = 0,
    vary: VaryFn | str | None = None,
    axis_max: dict[str, int] | None = None,
) -> ExportVerdict:
    """check() plus writing the .onnx and its manifest. `output` is the .onnx path.

    A FAILED verdict writes nothing; a DEGRADED one still writes the artifact because
    the manifest records exactly how far off it is.
    """
    from downshift.core.manifest import write_manifest
    from downshift.core.verdict import check

    verdict = check(
        model,
        example_inputs,
        k=k,
        adapter=adapter,
        dynamic=dynamic,
        fp16=fp16,
        verify_numerics=verify_numerics,
        atol=atol,
        rtol=rtol,
        seed=seed,
        vary=vary,
        axis_max=axis_max,
    )
    if verdict.onnx_program is None:
        return verdict
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # An external-data export (onnx_path is its temp copy) is saved through the program again,
    # which names the data file after `output` (<name>.onnx.data) and records that location in
    # the .onnx; the temp pair is never renamed or loaded back into memory.
    external = verdict.onnx_path is not None
    verdict.onnx_program.save(str(output), external_data=external or None)  # type: ignore[attr-defined]
    verdict.onnx_path = output
    write_manifest(output, verdict, source_path, __version__)
    return verdict
