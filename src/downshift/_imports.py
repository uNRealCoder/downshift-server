"""Import-spec resolution: stdlib only, so core/adapters/serve/middleware don't have to
import downshift.loading (the CLI-argument resolver, which imports torch) just for this.
Same pattern as downshift.sources for source kinds; loading.py re-exports these names.
"""

import re
from importlib import import_module
from typing import Any

_IMPORT_SPEC_RE = re.compile(r"^[A-Za-z_][\w.]*:[A-Za-z_]\w*$")


class LoadError(ValueError):
    pass


def is_import_spec(spec: str) -> bool:
    return bool(_IMPORT_SPEC_RE.match(spec))


def import_object(spec: str) -> Any:
    if not is_import_spec(spec):
        raise LoadError(f"{spec!r} is not an import spec of the form package.module:attr")
    module_name, _, attr = spec.partition(":")
    try:
        module = import_module(module_name)
    except ImportError as exc:
        raise LoadError(f"can't import {module_name!r}: {exc}") from exc
    try:
        return getattr(module, attr)
    except AttributeError as exc:
        raise LoadError(f"{module_name!r} has no attribute {attr!r}") from exc
