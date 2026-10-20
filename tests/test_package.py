"""The package contract: the lazy `__all__` names import, the version matches pyproject,
and the entry points (`downshift`, `python -m downshift`) start fast, without torch."""

import subprocess
import sys
import time
import tomllib
from pathlib import Path

import downshift
from tests.conftest import subprocess_env

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


def test_python_dash_m_downshift_help() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "downshift", "--help"],
        capture_output=True,
        text=True,
        env=subprocess_env(),
    )
    assert result.returncode == 0, result.stderr
    assert "check" in result.stdout
    assert "serve" in result.stdout


HELP_BUDGET_SECONDS = 1.5


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", "from downshift.cli.main import app; app()", *args],
        capture_output=True,
        text=True,
        env=subprocess_env(),
    )


def test_cli_main_import_leaves_torch_out_of_sys_modules() -> None:
    result = subprocess.run(
        [sys.executable, "-c", "import downshift.cli.main, sys; assert 'torch' not in sys.modules"],
        capture_output=True,
        text=True,
        env=subprocess_env(),
    )
    assert result.returncode == 0, result.stderr


def test_help_is_fast() -> None:
    start = time.perf_counter()
    result = _run("--help")
    elapsed = time.perf_counter() - start
    assert result.returncode == 0, result.stderr
    assert elapsed < HELP_BUDGET_SECONDS, elapsed


def test_serve_help_is_fast() -> None:
    start = time.perf_counter()
    result = _run("serve", "--help")
    elapsed = time.perf_counter() - start
    assert result.returncode == 0, result.stderr
    assert elapsed < HELP_BUDGET_SECONDS, elapsed
