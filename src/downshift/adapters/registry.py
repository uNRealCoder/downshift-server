"""Adapter lookup: built-ins plus anything registered under the `downshift.adapters`
entry-point group. Adapters whose optional dependency is missing are skipped silently.
"""

from importlib import import_module
from importlib.metadata import entry_points

from torch import nn

from downshift.adapters.base import Adapter

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
    adapters = available()
    if name not in adapters:
        raise KeyError(f"unknown adapter {name!r}; available: {', '.join(adapters)}")
    return adapters[name]


def detect(model: nn.Module, example_inputs: tuple | None) -> Adapter:
    for adapter in available().values():
        if adapter.matches(model, example_inputs):
            return adapter
    raise RuntimeError("no adapter matched and the generic adapter is unavailable")
