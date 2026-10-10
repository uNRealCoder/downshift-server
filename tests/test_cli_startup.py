"""U6: `downshift --help` must not pay the import cost of torch. It runs in a subprocess. It
therefore measures a cold interpreter, as the shell of a user starts the CLI.
"""

import subprocess
import sys
import time

from tests.conftest import subprocess_env

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
