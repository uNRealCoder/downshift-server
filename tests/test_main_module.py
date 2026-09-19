"""U7: `python -m downshift` is a working entry point, equivalent to the `downshift` script."""

import os
import subprocess
import sys
from pathlib import Path

SRC = str(Path(__file__).resolve().parent.parent / "src")


def _env() -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = SRC + os.pathsep + env.get("PYTHONPATH", "")
    return env


def test_python_dash_m_downshift_help() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "downshift", "--help"],
        capture_output=True,
        text=True,
        env=_env(),
    )
    assert result.returncode == 0, result.stderr
    assert "check" in result.stdout
    assert "serve" in result.stdout
