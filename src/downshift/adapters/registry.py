"""Adapter lookup: the built-in adapters, all adapters that are registered under the
`downshift.adapters` entry-point group, and one-off adapters that downshift loads directly from
a .py file of the user. Downshift skips an adapter without a message if its optional dependency
is missing, or if nothing has imported its family yet.
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


# The most specific first. generic is last, so it wins only if nothing else matches. Each spec
# names an adapter class, and the registry creates an instance of it. The third element is the
# module whose presence in sys.modules means that the family is in use. (Discovery therefore
# never imports transformers or torch_geometric for them.) It is None for generic, which has no
# optional dependency to gate on.
_BUILTIN_SPECS = (
    (Family.hf, "downshift.adapters.hf:HFAdapter", "transformers"),
    (Family.pyg, "downshift.adapters.pyg:PyGAdapter", "torch_geometric"),
    (Family.generic, "downshift.adapters.generic:GenericAdapter", None),
)

# The cache of available(). The key is the set of built-in adapters that are in use. The imports
# of a plugin (or of a custom adapter) can only add entries. They never remove an entry that is
# already in the cache of this process.
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
    """Load the adapter of a user from a standalone .py file, outside all installed packages.

    `attr` names the adapter class, and downshift creates an instance with no arguments
    (`ADAPTER = MyAdapter`, or `--adapter file.py:MyAdapter`). It can also name an instance of
    the right shape for an `Adapter` that is already built.
    """
    path = Path(path_str)
    if not path.is_file():
        raise LoadError(f"{path} does not exist")
    spec = importlib.util.spec_from_file_location(f"downshift._custom_adapter_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise LoadError(f"cannot import {path} as a Python module")
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
            f"{path}:{attr} is a {type(obj).__name__} and not an Adapter. It needs `name`, "
            "matches(), example_inputs() and prepare(). See GenericAdapter for "
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
    # generic must be tried last, for all orders of registration.
    generic = adapters.pop(Family.generic, None)
    if generic is not None:
        adapters[Family.generic] = generic
    _cache[key] = adapters
    return adapters


def get(name: str) -> Adapter:
    """Load exactly one adapter by name, without pulling in every other family's import."""
    # path/to/adapter.py[:attr]. Split on ".py:" and not on the last ":". The colon of a Windows
    # drive letter (`C:\...`) is then never the separator between the path and attr.
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
