"""The contract of the lazy `downshift/__init__.py`. Each `__all__` name still imports. The
version in `pyproject.toml` (the source of truth) is the same as `downshift.__version__`.
"""

import tomllib
from pathlib import Path

import downshift

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


def test_version_matches_pyproject() -> None:
    data = tomllib.loads(PYPROJECT.read_text())
    assert downshift.__version__ == data["project"]["version"]


def test_every_dunder_all_name_is_importable() -> None:
    """`__all__` is the lazy `__init__`'s contract: `from downshift import X` for each name."""
    names = list(downshift.__all__)
    namespace: dict = {}
    exec(f"from downshift import {', '.join(names)}", namespace)
    for name in names:
        assert namespace[name] is not None
