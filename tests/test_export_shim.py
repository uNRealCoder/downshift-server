"""V8: `downshift.export` is a one-release deprecation shim for `downshift.core`.

Run in a subprocess: the DeprecationWarning fires once per process (module import is cached),
and the shim's class-swap trick only matters the first time `downshift.export.<submodule>`
is imported, so a clean interpreter is the only way to observe both reliably.
"""

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

_SCRIPT = """
import warnings

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    import downshift.export.verify as shim_verify

import downshift.core.verify as core_verify

dep_warnings = [w for w in caught if issubclass(w.category, DeprecationWarning)]
assert len(dep_warnings) == 1, dep_warnings
assert "downshift.core since 0.4.0" in str(dep_warnings[0].message)
assert "removed in 0.5.0" in str(dep_warnings[0].message)
assert shim_verify.verify is core_verify.verify
assert shim_verify.NumericsReport is core_verify.NumericsReport

import downshift
from tests.models import clean_mlp

verdict = downshift.export(clean_mlp.make_model(), r"{out}", clean_mlp.make_inputs())
assert verdict.status == "CLEAN", verdict.reason
print("OK")
"""


def test_export_submodule_warns_once_and_matches_core(tmp_path) -> None:
    script = _SCRIPT.format(out=str(tmp_path / "shim.onnx").replace("\\", "\\\\"))
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT / "src") + os.pathsep + str(REPO_ROOT)
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env=env,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout
