"""Fixtures that the test modules share.

The export and the verification are the slow part of the suite. Downshift builds a fixture
that more than one module needs one time for each session.
"""

import base64
import os
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

import downshift
from downshift.core.memo import MEMO
from downshift.loading import LoadSpec, load_model
from downshift.serve.app import build_app
from downshift.serve.engine import ServeOptions, ServingState, prepare_serving
from tests.models import clean_mlp

REPO_ROOT = Path(__file__).resolve().parent.parent

_TORCH_VERSION = tuple(int(p) for p in torch.__version__.split("+")[0].split(".")[:2])


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip `needs_torch_26` tests on the CI floor job, which uses torch 2.5.1.

    A marker next to the tests replaces an inline `--deselect` list in the CI YAML. The marker
    cannot become old when you add, rename or remove tests.
    """
    if _TORCH_VERSION >= (2, 6):
        return
    skip = pytest.mark.skip(reason=f"needs torch>=2.6, running {torch.__version__}")
    for item in items:
        if "needs_torch_26" in item.keywords:
            item.add_marker(skip)


class WorkerExit(Exception):
    """os._exit, called from a test. A `--workers` worker does this when its model fails to load
    (cli.runtime._serve_app_factory)."""


@pytest.fixture(autouse=True)
def _no_real_os_exit() -> Iterator[None]:
    """A worker-path test whose load fails would call the real os._exit. This would end the
    whole pytest run with no report. An exception raised here fails only that test.

    The patch is done by hand and not through `monkeypatch`. An autouse fixture that requests
    `monkeypatch` would set it up first. Its undo would then run after the teardowns of other
    fixtures (for example, test_settings reloads the module when its environment variables are
    gone)."""

    def fake_exit(code: int) -> None:
        raise WorkerExit(f"os._exit({code})")

    real_exit = os._exit
    os._exit = fake_exit  # type: ignore[assignment]
    try:
        yield
    finally:
        os._exit = real_exit


@pytest.fixture(autouse=True)
def _fresh_export_memo() -> Iterator[None]:
    """The export memo is global for the process. A test that counts export phases, or patches a
    loader, must never see a hit that an earlier test left."""
    MEMO.clear()
    yield
    MEMO.clear()


def subprocess_env(*extra_paths: Path) -> dict[str, str]:
    """The environment for a child interpreter that must import the `downshift` of this checkout.

    `src` always goes first. The venv also has a pip-installed copy. Without this, a child
    process uses that copy and not the working tree, and nobody sees it. `extra_paths` follow
    `src`.
    """
    env = dict(os.environ)
    paths = [str(REPO_ROOT / "src"), *map(str, extra_paths), env.get("PYTHONPATH", "")]
    env["PYTHONPATH"] = os.pathsep.join(paths)
    return env


def b64_input(arr: np.ndarray, **overrides) -> dict:
    """The base64 TypedArray form of `arr`; overrides drop (None) or replace fields."""
    payload = {
        "data": base64.b64encode(np.ascontiguousarray(arr)).decode("ascii"),
        "dtype": arr.dtype.name,
        "shape": list(arr.shape),
    }
    payload.update(overrides)
    return {k: v for k, v in payload.items() if v is not None}


def b64_output(entry: dict) -> np.ndarray:
    """Decode one base64-encoded response tensor back to an ndarray."""
    raw = base64.b64decode(entry["data"], validate=True)
    return np.frombuffer(raw, dtype=entry["dtype"]).reshape(entry["shape"])


@pytest.fixture(scope="session")
def serve_fixture() -> Callable[..., ServingState]:
    """Serve a tests/models fixture by name: serve_fixture("clean_mlp", force_onnx=True)."""

    def build(name: str, **opts) -> ServingState:
        loaded = load_model(LoadSpec(f"tests.models.{name}:make_model"))
        return prepare_serving(loaded, ServeOptions(warmup=1, **opts))

    return build


@pytest.fixture(scope="session")
def mlp_state(serve_fixture) -> ServingState:
    return serve_fixture("clean_mlp")


@pytest.fixture(scope="session")
def branch_state(serve_fixture) -> ServingState:
    return serve_fixture("data_dependent_branch")


@pytest.fixture(scope="module")
def mlp_client(mlp_state) -> TestClient:
    return TestClient(build_app(mlp_state))


@pytest.fixture(scope="module")
def gcn_client(serve_fixture) -> TestClient:
    return TestClient(build_app(serve_fixture("gnn_gcn")))


@pytest.fixture(scope="session")
def exported_mlp(tmp_path_factory) -> tuple[Path, torch.nn.Module, downshift.ExportVerdict]:
    """A CLEAN clean_mlp on disk: (onnx path, the model it came from, its verdict)."""
    model = clean_mlp.make_model()
    out = tmp_path_factory.mktemp("exported") / "m.onnx"
    verdict = downshift.export(model, out, clean_mlp.make_inputs())
    assert verdict.status == "CLEAN", verdict.reason
    return out, model, verdict
