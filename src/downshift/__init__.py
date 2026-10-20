"""downshift: serve a PyTorch model over HTTP. Downshift verifies the ONNX export against
PyTorch first.

The import is lazy (PEP 562). An import of `downshift` (and so of `downshift.cli.main`, which
imports this module first) does not import torch, onnx or onnxruntime. Each name in `__all__`
resolves when you first use it. `__getattr__` takes it from the `downshift.core`,
`downshift.adapters` or `downshift.serve` module that defines it. Downshift caches it on this
module, so later access does not call `__getattr__`.
"""

from __future__ import annotations

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

# This list is written out and not derived from _LAZY. A literal list lets ruff see the
# TYPE_CHECKING imports above as exported (F401). `from downshift import *` and the doc tools
# also read it. If you add a name to _LAZY, add it here too.
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
    globals()[name] = value  # cache: later access is a plain attribute lookup
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
    """check() plus the writing of the .onnx file and its manifest. `output` is the .onnx path.

    A FAILED verdict writes nothing. A DEGRADED verdict writes the artifact, because the
    manifest records how far the numbers differ.

    Unlike check(), export() reuses an earlier CLEAN or DEGRADED export of the same model and
    options. It takes the export from the memo of this process (core/memo.py), and from
    `export_cache_dir` if you give it (default: DOWNSHIFT_EXPORT_CACHE_DIR. If it is unset,
    nothing is written to disk). `cache=False` skips both, for reads and for writes.
    """
    from downshift.core import memo
    from downshift.core.export_cache import ExportCache, lookup, store
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
    entry, _ = lookup(key, disk, key)
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
            store(entry, None, disk, key)  # check() already put it in the memo
    output.parent.mkdir(parents=True, exist_ok=True)
    if verdict.onnx_bytes:
        # The graph that capture() already serialized. A reused export writes the same bytes.
        output.write_bytes(verdict.onnx_bytes)
    else:
        # An export with external data (onnx_path is its temporary copy) is saved through the
        # program again. This names the data file after `output` (<name>.onnx.data) and records
        # that location in the .onnx file. Downshift never renames the temporary pair, and it
        # never loads it back into memory.
        verdict.onnx_program.save(str(output), external_data=True)  # type: ignore[attr-defined]
    verdict.onnx_path = output
    write_manifest(output, verdict, source_path, __version__)
    return verdict


def _export_from_entry(entry: ExportEntry, output: Path, source_path: Path | None) -> ExportVerdict:
    """export() for a reused export. Downshift writes the cached graph to `output`, with a
    manifest. It does not export again."""
    import shutil

    from downshift.core.manifest import external_data_files, write_manifest
    from downshift.core.verdict import ExportVerdict

    verdict = ExportVerdict.from_dict(entry.verdict)
    output.parent.mkdir(parents=True, exist_ok=True)
    if entry.onnx_bytes:
        output.write_bytes(entry.onnx_bytes)
    else:
        # Downshift names the data file after `output`, as a new export with external data does.
        # The locations in the .onnx file follow. Downshift only copies the (large) data.
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
