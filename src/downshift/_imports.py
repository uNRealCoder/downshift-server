"""Import-spec resolution. This module uses only the standard library. core, adapters, serve and
middleware therefore do not need to import downshift.loading (the resolver for CLI arguments,
which imports torch) for this. It is the same pattern as downshift.sources for the source kinds.
loading.py re-exports these names.
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
        hint = ""
        if module_name.endswith(".py"):
            hint = (
                f". An import spec takes a module name, not a file name: "
                f"{module_name.removesuffix('.py')}:{attr}"
            )
        raise LoadError(f"cannot import {module_name!r}: {exc}{hint}") from exc
    try:
        return getattr(module, attr)
    except AttributeError as exc:
        raise LoadError(f"{module_name!r} has no attribute {attr!r}") from exc
