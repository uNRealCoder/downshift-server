"""Adapter lookup: built-ins, anything registered under the `downshift.adapters` entry-point
group, and one-off adapters loaded straight from a user's .py file. Adapters whose optional
dependency is missing are skipped silently.
"""

import importlib.util
from importlib import import_module
from importlib.metadata import entry_points
from pathlib import Path

from torch import nn

from downshift.adapters.base import Adapter
from downshift.loading import LoadError

ENTRY_POINT_GROUP = "downshift.adapters"

# Most specific first; generic last so it only wins when nothing else matches.
_BUILTIN_SPECS = (
    "downshift.adapters.hf:ADAPTER",
    "downshift.adapters.pyg:ADAPTER",
    "downshift.adapters.generic:ADAPTER",
)


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
    adapters: dict[str, Adapter] = {}
    for ep in entry_points(group=ENTRY_POINT_GROUP):
        try:
            adapter = ep.load()
        except ImportError:
            continue
        adapters[adapter.name] = adapter
    for spec in _BUILTIN_SPECS:
        adapter = _load_spec(spec)
        if adapter is not None:
            adapters.setdefault(adapter.name, adapter)
    # Generic must be tried last regardless of registration order.
    generic = adapters.pop("generic", None)
    if generic is not None:
        adapters["generic"] = generic
    return adapters


def get(name: str) -> Adapter:
    file_spec = _split_file_spec(name)
    if file_spec is not None:
        return load_from_file(*file_spec)
    adapters = available()
    if name not in adapters:
        raise KeyError(f"unknown adapter {name!r}; available: {', '.join(adapters)}")
    return adapters[name]


def detect(model: nn.Module, example_inputs: tuple | None) -> Adapter:
    for adapter in available().values():
        if adapter.matches(model, example_inputs):
            return adapter
    raise RuntimeError("no adapter matched and the generic adapter is unavailable")
