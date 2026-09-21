"""Adapter lookup: built-ins, anything registered under the `downshift.adapters` entry-point
group, and one-off adapters loaded straight from a user's .py file. Adapters whose optional
dependency is missing, or whose family hasn't been imported by anything yet, are skipped
silently.
"""

import importlib.util
import sys
from enum import StrEnum
from importlib import import_module
from importlib.metadata import entry_points
from pathlib import Path

from torch import nn

from downshift._imports import LoadError
from downshift.adapters.base import Adapter

ENTRY_POINT_GROUP = "downshift.adapters"


class Family(StrEnum):
    """The built-in adapter names, so "hf"/"pyg"/"generic" aren't repeated as string literals
    across this module, loading.py and the CLI. A custom adapter's own `name` is still a plain
    string - this only names the three downshift ships."""

    hf = "hf"
    pyg = "pyg"
    generic = "generic"


# Most specific first; generic last so it only wins when nothing else matches. The third
# element is the module whose presence in sys.modules means the family is actually in play
# (so discovery never imports transformers/torch_geometric on their behalf); None for generic,
# which has no optional dependency to gate on.
_BUILTIN_SPECS = (
    (Family.hf, "downshift.adapters.hf:ADAPTER", "transformers"),
    (Family.pyg, "downshift.adapters.pyg:ADAPTER", "torch_geometric"),
    (Family.generic, "downshift.adapters.generic:ADAPTER", None),
)
_BUILTIN_VALUES = frozenset(spec for _, spec, _ in _BUILTIN_SPECS)

# available() keyed by which optional families are in play; a plugin's own imports (or a
# custom adapter's) can only add entries, never remove one already cached for this process.
_cache: dict[frozenset[str], dict[str, Adapter]] = {}


def _in_play(requires: str | None) -> bool:
    return requires is None or requires in sys.modules


def _load_spec(spec: str) -> Adapter | None:
    module_name, _, attr = spec.partition(":")
    try:
        adapter: Adapter = getattr(import_module(module_name), attr)
    except ImportError:
        return None
    return adapter


def _split_file_spec(name: str) -> tuple[str, str] | None:
    """Split path/to/adapter.py[:attr] into (path, attr); None if `name` isn't a .py spec.

    Splits on the literal ".py:" rather than the last ":", so a Windows drive letter's
    colon (`C:\\...`) is never mistaken for the path:attr separator.
    """
    path, sep, attr = name.partition(".py:")
    if sep:
        return path + ".py", attr
    if name.endswith(".py"):
        return name, "ADAPTER"
    return None


def load_from_file(path_str: str, attr: str = "ADAPTER") -> Adapter:
    """Load a user's adapter from a standalone .py file, outside any installed package.

    `attr` names either an `Adapter`-shaped instance — the `generic`/`pyg`/`hf` convention of
    a module-level `ADAPTER = MyAdapter()` — or the class itself, instantiated with no args.
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

    if isinstance(obj, type):
        obj = obj()
    if not isinstance(obj, Adapter):
        raise LoadError(
            f"{path}:{attr} is a {type(obj).__name__}, not an Adapter — it needs `name`, "
            "`family`, matches(), example_inputs(), and prepare(); see GenericAdapter for "
            "the shape to implement."
        )
    return obj


def available() -> dict[str, Adapter]:
    key = frozenset(mod for _, _, mod in _BUILTIN_SPECS if mod is not None and mod in sys.modules)
    cached = _cache.get(key)
    if cached is not None:
        return cached

    adapters: dict[str, Adapter] = {}
    for ep in entry_points(group=ENTRY_POINT_GROUP):
        if ep.value in _BUILTIN_VALUES:
            continue  # a stale install's metadata may still list these; pyproject no longer does
        try:
            adapter = ep.load()
        except ImportError:
            continue
        adapters[adapter.name] = adapter
    for _, spec, requires in _BUILTIN_SPECS:
        if not _in_play(requires):
            continue
        adapter = _load_spec(spec)
        if adapter is not None:
            adapters.setdefault(adapter.name, adapter)
    # Generic must be tried last regardless of registration order.
    generic = adapters.pop(Family.generic, None)
    if generic is not None:
        adapters[Family.generic] = generic
    _cache[key] = adapters
    return adapters


def get(name: str) -> Adapter:
    """Load exactly one adapter by name, without pulling in every other family's import."""
    file_spec = _split_file_spec(name)
    if file_spec is not None:
        return load_from_file(*file_spec)

    for builtin_name, spec, _ in _BUILTIN_SPECS:
        if builtin_name == name:
            adapter = _load_spec(spec)
            if adapter is not None:
                return adapter
            break  # the built-in is registered but its optional dependency isn't installed

    for ep in entry_points(group=ENTRY_POINT_GROUP):
        if ep.name == name and ep.value not in _BUILTIN_VALUES:
            try:
                loaded: Adapter = ep.load()
            except ImportError:
                break
            return loaded

    raise LoadError(f"unknown adapter {name!r}; available: {', '.join(available())}")


def detect(model: nn.Module, example_inputs: tuple | None) -> Adapter:
    for adapter in available().values():
        if adapter.matches(model, example_inputs):
            return adapter
    raise RuntimeError("no adapter matched and the generic adapter is unavailable")
