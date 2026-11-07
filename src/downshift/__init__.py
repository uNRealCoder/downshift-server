"""downshift: serve a PyTorch model over HTTP, after it verifies the ONNX export against
PyTorch.

The names in `__all__` import lazily (PEP 562), so `import downshift` (and the CLI) does not
import torch, onnx or onnxruntime.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from downshift._defaults import DEFAULT_SAMPLES
from downshift._version import __version__

if TYPE_CHECKING:
    from downshift.adapters.base import Adapter, Prepared, VaryFn
    from downshift.core.memo import ExportEntry
    from downshift.core.verdict import (
        ExportVerdict,
        build_verdict,
        check,
        intake,
        prepare_model,
    )
    from downshift.core.verify import NumericsReport, OnnxRuntimeError
    from downshift.serve.app import build_app

# A literal list, so ruff sees the TYPE_CHECKING imports as exported. Keep it in step with _LAZY.
__all__ = [
    "Adapter",
    "ExportVerdict",
    "NumericsReport",
    "OnnxRuntimeError",
    "Prepared",
    "app_for",
    "build_app",
    "build_verdict",
    "check",
    "export",
    "intake",
    "prepare_model",
    "__version__",
]

# name -> (the module that defines it, the attribute name). Resolved on first access.
_LAZY = {
    "Adapter": ("downshift.adapters.base", "Adapter"),
    "Prepared": ("downshift.adapters.base", "Prepared"),
    "ExportVerdict": ("downshift.core.verdict", "ExportVerdict"),
    "build_verdict": ("downshift.core.verdict", "build_verdict"),
    "check": ("downshift.core.verdict", "check"),
    "prepare_model": ("downshift.core.verdict", "prepare_model"),
    "NumericsReport": ("downshift.core.verify", "NumericsReport"),
    "OnnxRuntimeError": ("downshift.core.verify", "OnnxRuntimeError"),
    "intake": ("downshift.core.verdict", "intake"),
    "app_for": ("downshift.serve", "app_for"),
    "build_app": ("downshift.serve.app", "build_app"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attr = _LAZY[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    import importlib

    value = getattr(importlib.import_module(module_name), attr)
    globals()[name] = value  # cache: later access is a plain attribute lookup
    return value


def export(
    model,
    output: str | Path,
    example_inputs: tuple | None = None,
    samples: int = DEFAULT_SAMPLES,
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
    cache: bool = True,
    export_cache_dir: str | Path | None = None,
) -> ExportVerdict:
    """check(), then write the .onnx file (`output`) and its manifest. A FAILED verdict writes
    nothing. A DEGRADED one writes both: the manifest records how far the numbers differ.

    Unlike check(), export() reuses an earlier CLEAN or DEGRADED export of the same model and
    options, from the in-process memo and from `export_cache_dir` (None: the value of
    DOWNSHIFT_EXPORT_CACHE_DIR; unset means no disk). `cache=False` skips both.
    """
    from downshift.core import memo
    from downshift.core.export_cache import ExportCache, lookup, store
    from downshift.core.manifest import write_manifest
    from downshift.core.verdict import check
    from downshift.settings import EXPORT_CACHE_DIR

    if export_cache_dir is None:
        export_cache_dir = EXPORT_CACHE_DIR
    run = dict(
        samples=samples,
        adapter=adapter,
        dynamic=dynamic,
        fp16=fp16,
        atol=atol,
        rtol=rtol,
        seed=seed,
        vary=vary,
        axis_max=axis_max,
    )
    disk = ExportCache(export_cache_dir) if cache and export_cache_dir else None
    key = None
    if cache and verify_numerics:
        key = memo.guarded(memo.model_key, model, example_inputs, **run)
    output = Path(output)
    entry, _ = lookup(key, disk, key)
    if entry is not None:
        return _export_from_entry(entry, output, source_path)

    verdict = check(
        model,
        example_inputs,
        verify_numerics=verify_numerics,
        cache=cache,
        _memo_key=key,
        **run,  # type: ignore[arg-type]
    )
    if verdict.status == "FAILED":
        return verdict
    if disk is not None and key is not None:
        entry = memo.entry_from_verdict(verdict)
        if entry is not None:
            store(entry, None, disk, key)  # check() already put it in the memo
    output.parent.mkdir(parents=True, exist_ok=True)
    if verdict.onnx_bytes:
        output.write_bytes(verdict.onnx_bytes)
    elif verdict.onnx_program is not None:
        # External data: save through the program, so the data file is named after `output`.
        verdict.onnx_program.save(str(output), external_data=True)  # type: ignore[attr-defined]
    else:
        assert verdict.onnx_path is not None
        _copy_external_graph(verdict.onnx_path, output)
    verdict.onnx_path = output
    write_manifest(output, verdict, source_path, __version__)
    return verdict


def _copy_external_graph(source: Path, output: Path) -> None:
    """Write the graph at `source` (weights in external data) to `output`. Name the data file
    after `output`, as a new export does. Only the data files are copied."""
    import shutil

    import onnx
    from onnx.external_data_helper import ExternalDataInfo, uses_external_data

    from downshift.core.manifest import external_data_files

    proto = onnx.load(str(source), load_external_data=False)
    renamed = {
        name: f"{output.name}.data" + (str(i) if i else "")
        for i, name in enumerate(external_data_files(source))
    }
    for init in proto.graph.initializer:
        if uses_external_data(init):
            for item in init.external_data:
                if item.key == "location":
                    item.value = renamed[ExternalDataInfo(init).location]
    for name, new_name in renamed.items():
        shutil.copyfile(source.parent / name, output.parent / new_name)
    onnx.save_model(proto, str(output))


def _export_from_entry(entry: ExportEntry, output: Path, source_path: Path | None) -> ExportVerdict:
    """export() from a reused export: write the cached graph and a manifest."""
    from downshift.core.manifest import write_manifest
    from downshift.core.verdict import ExportVerdict

    verdict = ExportVerdict.from_dict(entry.verdict)
    if verdict.status == "FAILED":
        return verdict
    output.parent.mkdir(parents=True, exist_ok=True)
    if entry.onnx_bytes:
        output.write_bytes(entry.onnx_bytes)
    else:
        assert entry.onnx_path is not None
        _copy_external_graph(entry.onnx_path, output)
    verdict.onnx_path = output
    write_manifest(output, verdict, source_path, __version__)
    return verdict
