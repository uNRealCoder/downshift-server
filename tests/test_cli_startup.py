"""U6: `downshift --help` must not pay torch's import cost. Runs in a subprocess so it
measures a cold interpreter, the way a user's shell actually invokes the CLI.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

HELP_BUDGET_SECONDS = 1.5
SRC = str(Path(__file__).resolve().parent.parent / "src")


def _env() -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = SRC + os.pathsep + env.get("PYTHONPATH", "")
    return env


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", "from downshift.cli.main import app; app()", *args],
        capture_output=True,
        text=True,
        env=_env(),
    )


def test_cli_main_import_leaves_torch_out_of_sys_modules() -> None:
    result = subprocess.run(
        [sys.executable, "-c", "import downshift.cli.main, sys; assert 'torch' not in sys.modules"],
        capture_output=True,
        text=True,
        env=_env(),
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
