"""Make every bench process import the working tree, not whatever pip put in site-packages.

This venv has downshift pip-installed. Running `python -m bench.run` from the repo root puts
the root on sys.path but not `src/`, so `import downshift` silently resolves to the installed
copy and the benchmark measures a released version instead of the checkout. Import this first
from every entry point.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"

for entry in (str(ROOT), str(SRC)):
    while entry in sys.path:
        sys.path.remove(entry)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SRC))

warnings.filterwarnings("ignore")


def verify() -> str:
    """Confirm the checkout won, and return the version being benchmarked."""
    import downshift

    resolved = Path(downshift.__file__).resolve()
    if SRC not in resolved.parents:
        raise RuntimeError(
            f"benchmarking the wrong downshift: {resolved}\nexpected something under {SRC}"
        )
    return downshift.__version__
