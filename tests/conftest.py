"""Fixtures shared across test modules.

Export and verification are the slow part of the suite, so anything more than one module
needs is built once per session.
"""

import base64
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import torch

import downshift
from downshift.loading import load_model
from downshift.serve.engine import ServeOptions, ServingState, prepare_serving
from tests.models import clean_mlp


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
