"""downshift: serve a PyTorch model over HTTP, with its ONNX export verified against PyTorch first.

Lazy by PEP 562: importing `downshift` (and so `downshift.cli.main`, which imports this module
first) doesn't pull in torch, onnx or onnxruntime. Every name in `__all__` resolves on first
access via `__getattr__`, from the `downshift.core`/`downshift.adapters`/`downshift.serve`
module that actually defines it, and is cached on the module so later access skips
`__getattr__` entirely.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from downshift._version import __version__
from downshift.settings import DEFAULT_SAMPLES, EXPORT_CACHE_DIR

if TYPE_CHECKING:
    from downshift.adapters.base import Adapter, Prepared, VaryFn
    from downshift.core.memo import ExportEntry
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
    cache: bool = True,
    export_cache_dir: str | Path | None = EXPORT_CACHE_DIR,
) -> ExportVerdict:
    """check() plus writing the .onnx and its manifest. `output` is the .onnx path.

    A FAILED verdict writes nothing; a DEGRADED one still writes the artifact because
    the manifest records exactly how far off it is.

    Unlike check(), export() reuses an earlier CLEAN or DEGRADED export of the same model and
    options from this process's memo (core/memo.py), and from `export_cache_dir` when given
    (default: DOWNSHIFT_EXPORT_CACHE_DIR; unset writes nothing to disk). `cache=False` skips
    both, in both directions.
    """
    from downshift.core import memo
    from downshift.core.export_cache import ExportCache
    from downshift.core.manifest import write_manifest
    from downshift.core.verdict import check

    disk = ExportCache(export_cache_dir) if cache and export_cache_dir else None
    key = None
    if cache and verify_numerics:
        key = memo.guarded(
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
    output = Path(output)
    if key is not None:
        entry = memo.MEMO.get(key) or (disk.get(key) if disk is not None else None)
        if entry is not None:
            return _export_from_entry(entry, output, source_path)

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
        cache=cache,
        _memo_key=key,
    )
    if verdict.onnx_program is None:
        return verdict
    if disk is not None and key is not None and verdict.status in memo.STORED_STATUSES:
        entry = memo.entry_from_verdict(verdict)
        if entry is not None:
            try:
                disk.put(key, entry)
            except OSError as exc:
                logging.getLogger("downshift.export_cache").warning(
                    "export cache write failed in %s: %s", disk.root, exc
                )
    output.parent.mkdir(parents=True, exist_ok=True)
    # An external-data export (onnx_path is its temp copy) is saved through the program again,
    # which names the data file after `output` (<name>.onnx.data) and records that location in
    # the .onnx; the temp pair is never renamed or loaded back into memory.
    external = verdict.onnx_path is not None
    verdict.onnx_program.save(str(output), external_data=external or None)  # type: ignore[attr-defined]
    verdict.onnx_path = output
    write_manifest(output, verdict, source_path, __version__)
    return verdict


def _export_from_entry(entry: ExportEntry, output: Path, source_path: Path | None) -> ExportVerdict:
    """export() for a reused export: write the cached graph where `output` says, with a
    manifest, instead of exporting again."""
    import shutil

    from downshift.core.manifest import external_data_files, write_manifest
    from downshift.core.verdict import ExportVerdict

    verdict = ExportVerdict.from_dict(entry.verdict)
    output.parent.mkdir(parents=True, exist_ok=True)
    if entry.onnx_bytes:
        output.write_bytes(entry.onnx_bytes)
    else:
        # The data file is renamed after `output`, as a fresh external export names it, and the
        # locations recorded in the .onnx follow; the (large) data itself is only copied.
        import onnx
        from onnx.external_data_helper import ExternalDataInfo, uses_external_data

        assert entry.onnx_path is not None
        proto = onnx.load(str(entry.onnx_path), load_external_data=False)
        renamed = {
            name: f"{output.name}.data" + (str(i) if i else "")
            for i, name in enumerate(external_data_files(entry.onnx_path))
        }
        for init in proto.graph.initializer:
            if uses_external_data(init):
                for item in init.external_data:
                    if item.key == "location":
                        item.value = renamed[ExternalDataInfo(init).location]
        for name, new_name in renamed.items():
            shutil.copyfile(entry.onnx_path.parent / name, output.parent / new_name)
        onnx.save_model(proto, str(output))
    verdict.onnx_path = output
    write_manifest(output, verdict, source_path, __version__)
    return verdict
