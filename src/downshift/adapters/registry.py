"""Adapter lookup: built-ins, anything registered under the `downshift.adapters` entry-point
group, and one-off adapters loaded straight from a user's .py file. Adapters whose optional
dependency is missing, or whose family hasn't been imported by anything yet, are skipped
silently.
"""

import importlib.util
import sys
from importlib import import_module
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any

from torch import nn

from downshift._imports import LoadError
from downshift.adapters.base import Adapter, Family

ENTRY_POINT_GROUP = "downshift.adapters"


# Most specific first; generic last so it only wins when nothing else matches. Each spec names
# an adapter class, which the registry instantiates. The third element is the module whose
# presence in sys.modules means the family is actually in play (so discovery never imports
# transformers/torch_geometric on their behalf); None for generic, which has no optional
# dependency to gate on.
_BUILTIN_SPECS = (
    (Family.hf, "downshift.adapters.hf:HFAdapter", "transformers"),
    (Family.pyg, "downshift.adapters.pyg:PyGAdapter", "torch_geometric"),
    (Family.generic, "downshift.adapters.generic:GenericAdapter", None),
)

# available() keyed by which built-ins are in play; a plugin's own imports (or a custom
# adapter's) can only add entries, never remove one already cached for this process.
_cache: dict[frozenset[str], dict[str, Adapter]] = {}


def _instantiate(obj: Any) -> Adapter:
    """An adapter class is instantiated with no args; an instance is used as-is."""
    adapter: Adapter = obj() if isinstance(obj, type) else obj
    return adapter


def _load_spec(spec: str) -> Adapter | None:
    module_name, _, attr = spec.partition(":")
    try:
        return _instantiate(getattr(import_module(module_name), attr))
    except ImportError:
        return None


def load_from_file(path_str: str, attr: str = "ADAPTER") -> Adapter:
    """Load a user's adapter from a standalone .py file, outside any installed package.

    `attr` names the adapter class, instantiated with no args (`ADAPTER = MyAdapter`, or
    `--adapter file.py:MyAdapter`), or an already-built `Adapter`-shaped instance.
    """
    path = Path(path_str)
    if not path.is_file():
        raise LoadError(f"{path} does not exist")
    spec = importlib.util.spec_from_file_location(f"downshift._custom_adapter_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise LoadError(f"can't import {path} as a Python module")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise LoadError(f"error while loading {path}: {exc}") from exc

    try:
        obj = getattr(module, attr)
    except AttributeError as exc:
        raise LoadError(f"{path} has no attribute {attr!r}") from exc

    obj = _instantiate(obj)
    if not isinstance(obj, Adapter):
        raise LoadError(
            f"{path}:{attr} is a {type(obj).__name__}, not an Adapter — it needs `name`, "
            "matches(), example_inputs(), and prepare(); see GenericAdapter for "
            "the shape to implement."
        )
    return obj


def available() -> dict[str, Adapter]:
    in_play = [spec for _, spec, mod in _BUILTIN_SPECS if mod is None or mod in sys.modules]
    key = frozenset(in_play)
    cached = _cache.get(key)
    if cached is not None:
        return cached

    adapters: dict[str, Adapter] = {}
    for ep in entry_points(group=ENTRY_POINT_GROUP):
        try:
            adapter = _instantiate(ep.load())
        except ImportError:
            continue
        adapters[adapter.name] = adapter
    for spec in in_play:
        builtin = _load_spec(spec)
        if builtin is not None:
            adapters.setdefault(builtin.name, builtin)
    # Generic must be tried last regardless of registration order.
    generic = adapters.pop(Family.generic, None)
    if generic is not None:
        adapters[Family.generic] = generic
    _cache[key] = adapters
    return adapters


def get(name: str) -> Adapter:
    """Load exactly one adapter by name, without pulling in every other family's import."""
    # path/to/adapter.py[:attr]. Split on ".py:", not the last ":", so a Windows drive
    # letter's colon (`C:\...`) is never taken for the path:attr separator.
    path, sep, attr = name.partition(".py:")
    if sep:
        return load_from_file(path + ".py", attr)
    if name.endswith(".py"):
        return load_from_file(name)

    builtin = next((spec for family, spec, _ in _BUILTIN_SPECS if family == name), None)
    adapter = _load_spec(builtin) if builtin is not None else None
    if adapter is not None:
        return adapter

    for ep in entry_points(group=ENTRY_POINT_GROUP):
        if ep.name == name:
            try:
                return _instantiate(ep.load())
            except ImportError:
                break

    raise LoadError(f"unknown adapter {name!r}; available: {', '.join(available())}")


def detect(model: nn.Module, example_inputs: tuple | None) -> Adapter:
    for adapter in available().values():
        if adapter.matches(model, example_inputs):
            return adapter
    raise RuntimeError("no adapter matched and the generic adapter is unavailable")
