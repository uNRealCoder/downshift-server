"""Import-spec entry points for the real CLI to load a bench fixture directly.

`downshift serve <spec>` builds the model in whatever process it is told to import: the parent
when `--workers` is 1, each worker when it is more. Every one of those must build the model
under `bench.cases.SEED`, or the correctness numbers are noise the moment the workers-sweep
compares a response against `torch_reference` (`bench.cases.load_fixture` already does this for
the in-process variants). This module gives the CLI's `pkg.module:attr` loader something to
point at: one zero-arg model factory and one zero-arg inputs factory per case in
`bench.cases.CASE_NAMES`.

Deliberately does not import `bench._path`: this module runs inside the server, and the server's
PYTHONPATH (`bench._path.cli_env`) decides which downshift it is.
"""

from __future__ import annotations

from typing import Any

from bench.cases import CASE_NAMES, load_fixture


def _factories(name: str) -> tuple[Any, Any]:
    def model() -> Any:
        return load_fixture(name)[0]

    def inputs() -> Any:
        return load_fixture(name)[1]

    model.__name__ = model.__qualname__ = name
    inputs.__name__ = inputs.__qualname__ = f"{name}_inputs"
    return model, inputs


for _name in CASE_NAMES:
    _model, _inputs = _factories(_name)
    globals()[_name] = _model
    globals()[f"{_name}_inputs"] = _inputs

del _name, _model, _inputs
