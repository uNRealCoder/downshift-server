"""Fixtures shared across test modules.

Export and verification are the slow part of the suite, so anything more than one module
needs is built once per session.
"""

from collections.abc import Callable
from pathlib import Path

import pytest
import torch

import downshift
from downshift.loading import load_model
from downshift.serve.engine import ServeOptions, ServingState, prepare_serving
from tests.models import clean_mlp


@pytest.fixture(scope="session")
def serve_fixture() -> Callable[..., ServingState]:
    """Serve a tests/models fixture by name: serve_fixture("clean_mlp", force_onnx=True)."""

    def build(name: str, **opts) -> ServingState:
        loaded = load_model(f"tests.models.{name}:make_model")
        return prepare_serving(loaded, ServeOptions(warmup=1, **opts))

    return build


@pytest.fixture(scope="session")
def mlp_state(serve_fixture) -> ServingState:
    return serve_fixture("clean_mlp")


@pytest.fixture(scope="session")
def branch_state(serve_fixture) -> ServingState:
    return serve_fixture("data_dependent_branch")


@pytest.fixture(scope="session")
def exported_mlp(tmp_path_factory) -> tuple[Path, torch.nn.Module, downshift.ExportVerdict]:
    """A CLEAN clean_mlp on disk: (onnx path, the model it came from, its verdict)."""
    model = clean_mlp.make_model()
    out = tmp_path_factory.mktemp("exported") / "m.onnx"
    verdict = downshift.export(model, out, clean_mlp.make_inputs())
    assert verdict.status == "CLEAN", verdict.reason
    return out, model, verdict
