"""Deprecated alias for downshift.core; removed in 0.5.0.

Importing a submodule here (e.g. `downshift.export.verify`) makes Python's import system
rebind the `export` attribute on the `downshift` package to this shim module once the import
finishes, which would otherwise break `downshift.export(model, path)` for callers who do
both. `_export_fn` captures the real function before that rebind happens (this module always
finishes executing while `downshift.export` still names the function), and this module's
class is swapped for one whose `__call__` forwards to it, so the call keeps working. This
trick lives for one release; use `downshift.core` directly.
"""

from __future__ import annotations

import sys
import warnings
from types import ModuleType
from typing import Any

import downshift

warnings.warn(
    "downshift.export is downshift.core since 0.4.0; removed in 0.5.0",
    DeprecationWarning,
    stacklevel=2,
)

_export_fn = downshift.export


class _ExportShimModule(ModuleType):
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return _export_fn(*args, **kwargs)


sys.modules[__name__].__class__ = _ExportShimModule
