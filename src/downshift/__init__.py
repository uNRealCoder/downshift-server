"""downshift: serve a PyTorch model over HTTP, with its ONNX export verified against PyTorch first.

Lazy by PEP 562: importing `downshift` (and so `downshift.cli.main`, which imports this module
first) doesn't pull in torch, onnx or onnxruntime. Every name in `__all__` resolves on first
access via `__getattr__`, from the `downshift.core`/`downshift.adapters` module that actually
defines it, and is cached on the module so later access skips `__getattr__` entirely.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from downshift._version import __version__

if TYPE_CHECKING:
    from downshift.adapters.base import Adapter, Prepared, VaryFn
    from downshift.core.prevalidated import intake
    from downshift.core.verdict import ExportVerdict, build_verdict, check, prepare_model
    from downshift.core.verify import NumericsReport, OnnxRuntimeError

__all__ = [
    "Adapter",
    "ExportVerdict",
    "NumericsReport",
    "OnnxRuntimeError",
    "Prepared",
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
    k: int = 8,
    adapter: Adapter | str | None = None,
    dynamic: dict[str, list[int]] | None = None,
    fp16: bool = False,
    source_path: Path | None = None,
    verify_numerics: bool = True,
    atol: float | None = None,
    rtol: float | None = None,
    seed: int = 0,
    vary: VaryFn | str | None = None,
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
    )
    if verdict.onnx_program is None:
        return verdict
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    verdict.onnx_program.save(str(output))  # type: ignore[attr-defined]
    verdict.onnx_path = output
    write_manifest(output, verdict, source_path, __version__)
    return verdict
