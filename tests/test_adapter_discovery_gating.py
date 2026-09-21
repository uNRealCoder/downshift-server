"""P1: adapter discovery must not import optional model families that aren't in play.

Both checks need a fresh interpreter: `tests/test_adapters.py` imports transformers and
torch_geometric at module level, which would make them look "in play" no matter what the
registry does.
"""

import subprocess
import sys

from tests.conftest import REPO_ROOT, subprocess_env


def _run(script: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env=subprocess_env(REPO_ROOT),
    )


def test_prepare_serving_on_clean_mlp_does_not_import_optional_families() -> None:
    script = """
import sys

from downshift.demo import clean_mlp
from downshift.loading import LoadedModel
from downshift.serve.engine import prepare_serving

loaded = LoadedModel(source="downshift.demo.clean_mlp", model=clean_mlp.make_model(),
                      example_inputs=clean_mlp.make_inputs())
state = prepare_serving(loaded)
assert state.verdict.status == "CLEAN", state.verdict.reason
assert "transformers" not in sys.modules
assert "torch_geometric" not in sys.modules
print("OK")
"""
    result = _run(script)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout


def test_detect_on_tiny_bert_still_returns_hf() -> None:
    script = """
from downshift.adapters import registry
from tests.models import tiny_bert

adapter = registry.detect(tiny_bert.make_model(), tiny_bert.make_inputs())
assert adapter.name == "hf", adapter.name
print("OK")
"""
    result = _run(script)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout
