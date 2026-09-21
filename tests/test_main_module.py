"""U7: `python -m downshift` is a working entry point, equivalent to the `downshift` script."""

import subprocess
import sys

from tests.conftest import subprocess_env


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
