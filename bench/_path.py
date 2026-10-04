"""Which downshift each bench process imports.

The orchestrator and the naive baselines always import the checkout: `src/` goes ahead of
site-packages, because the venv also has a released downshift pip-installed and `python -m
bench.run` from the repo root would otherwise silently pick that up. Import this module first
from every entry point.

The downshift servers are separate `downshift serve` processes, and `cli_env(target)` picks
which downshift they run:

  checkout   src/ on PYTHONPATH: the version under development
  installed  the copy in site-packages: the previous release, for a version-to-version run
"""

from __future__ import annotations

import os
import subprocess
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
TARGETS = ("checkout", "installed")

for entry in (str(ROOT), str(SRC)):
    while entry in sys.path:
        sys.path.remove(entry)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SRC))

warnings.filterwarnings("ignore")


def verify() -> str:
    """Confirm the checkout won in this process, and return its version."""
    import downshift

    resolved = Path(downshift.__file__).resolve()
    if SRC not in resolved.parents:
        raise RuntimeError(
            f"benchmarking the wrong downshift: {resolved}\nexpected something under {SRC}"
        )
    return downshift.__version__


def cli_env(target: str) -> dict[str, str]:
    """Environment for a subprocess that must import the `target` downshift."""
    env = os.environ.copy()
    paths = [str(ROOT)] if target == "installed" else [str(SRC), str(ROOT)]
    env["PYTHONPATH"] = os.pathsep.join(paths)
    return env


def target_version(target: str) -> str:
    """The version `target` resolves to, checked against where it was meant to come from."""
    out = subprocess.run(
        [
            sys.executable,
            "-c",
            "import downshift; print(downshift.__version__, downshift.__file__)",
        ],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        env=cli_env(target),
        check=True,
    ).stdout.split()
    version, location = out[0], Path(out[1]).resolve()
    if (SRC in location.parents) != (target == "checkout"):
        raise RuntimeError(f"target {target!r} resolved to {location}")
    return version
