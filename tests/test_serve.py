"""HTTP contract tests. Each state is exported once per module; export is the slow part."""

import dataclasses

import numpy as np
import pytest
from fastapi import Request
from fastapi.testclient import TestClient
from starlette.middleware.base import BaseHTTPMiddleware

from downshift.loading import LoadedModel, load_model
from downshift.serve.app import build_app
from downshift.serve.engine import ServeOptions, ServingState, prepare_serving


def _state(fixture: str, **opts) -> ServingState:
    loaded = load_model(f"tests.models.{fixture}:make_model")
    return prepare_serving(loaded, ServeOptions(warmup=1, **opts))


@pytest.fixture(scope="module")
def mlp_state() -> ServingState:
    return _state("clean_mlp")


@pytest.fixture(scope="module")
def mlp_client(mlp_state) -> TestClient:
    return TestClient(build_app(mlp_state))


@pytest.fixture(scope="module")
def gcn_client() -> TestClient:
    return TestClient(build_app(_state("gnn_gcn")))


@pytest.fixture(scope="module")
def branch_client() -> TestClient:
    return TestClient(build_app(_state("data_dependent_branch")))


def test_health_ready_metadata(mlp_client, mlp_state):
    assert mlp_client.get("/health").json() == {"status": "ok"}

    ready = mlp_client.get("/ready")
    assert ready.status_code == 200
    assert ready.json() == {"ready": True}

    meta = mlp_client.get("/metadata").json()
    assert meta["model"] == "tests.models.clean_mlp:make_model"
    assert meta["family"] == mlp_state.verdict.model_family
    assert meta["verdict"]["status"] == "CLEAN"
    assert meta["backend"]["name"] == "onnxruntime"
    assert meta["input_names"] == ["x"]
    assert isinstance(meta["notes"], list)
    assert meta["version"]


def test_predict_batch(mlp_client):
    x = np.random.randn(3, 16).tolist()
    resp = mlp_client.post("/predict", json={"inputs": {"x": x}})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["shapes"]["output_0"] == [3, 4]
    assert np.asarray(body["outputs"]["output_0"]).shape == (3, 4)
    assert body["dtypes"]["output_0"] == "float32"


def test_predict_missing_input(mlp_client):
    resp = mlp_client.post("/predict", json={"inputs": {"y": [[0.0] * 16]}})
    assert resp.status_code == 400
    assert "x" in resp.json()["detail"]


def test_predict_wrong_feature_size(mlp_client):
    resp = mlp_client.post("/predict", json={"inputs": {"x": [[0.0] * 5]}})
    assert resp.status_code == 400
    assert resp.json()["detail"]


def test_predict_typed_input(mlp_client):
    payload = {"data": [0.0] * 32, "dtype": "float32", "shape": [2, 16]}
    resp = mlp_client.post("/predict", json={"inputs": {"x": payload}})
    assert resp.status_code == 200, resp.text
    assert resp.json()["shapes"]["output_0"] == [2, 4]


def test_predict_graph(gcn_client):
    x = np.random.randn(5, 8).tolist()
    edge_index = [[0, 1, 2, 3, 4, 0, 2], [1, 2, 3, 4, 0, 3, 4]]
    resp = gcn_client.post("/predict/graph", json={"x": x, "edge_index": edge_index})
    assert resp.status_code == 200, resp.text
    assert resp.json()["shapes"]["output_0"] == [5, 4]


def test_predict_graph_on_non_graph_model(mlp_client):
    resp = mlp_client.post("/predict/graph", json={"x": [[0.0] * 16], "edge_index": [[0], [0]]})
    assert resp.status_code == 400
    assert "not graph-shaped" in resp.json()["detail"]


def test_backends_agree_on_same_contract():
    loaded = load_model("tests.models.clean_mlp:make_model")
    same_weights = LoadedModel(
        source=loaded.source, model=loaded.model, example_inputs=loaded.example_inputs
    )
    ort_client = TestClient(
        build_app(prepare_serving(loaded, ServeOptions(warmup=1, backend="onnxruntime")))
    )
    torch_client = TestClient(
        build_app(prepare_serving(same_weights, ServeOptions(warmup=1, backend="torch")))
    )

    payload = {"inputs": {"x": np.random.randn(4, 16).tolist()}}
    ort_resp = ort_client.post("/predict", json=payload)
    torch_resp = torch_client.post("/predict", json=payload)
    assert ort_resp.status_code == 200, ort_resp.text
    assert torch_resp.status_code == 200, torch_resp.text
    assert ort_client.get("/metadata").json()["backend"]["name"] == "onnxruntime"
    assert torch_client.get("/metadata").json()["backend"]["name"] == "torch"
    np.testing.assert_allclose(
        ort_resp.json()["outputs"]["output_0"],
        torch_resp.json()["outputs"]["output_0"],
        atol=1e-5,
    )


def test_failed_export_falls_back_to_torch(branch_client):
    meta = branch_client.get("/metadata").json()
    assert meta["verdict"]["status"] == "FAILED"
    assert meta["backend"]["name"] == "torch"

    resp = branch_client.post("/predict", json={"inputs": {"x": np.random.randn(2, 8).tolist()}})
    assert resp.status_code == 200, resp.text
    assert resp.json()["shapes"]["output_0"] == [2, 8]


async def add_test_header(request: Request, call_next):
    response = await call_next(request)
    response.headers["x-test"] = "1"
    return response


class AddClassHeader(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        response.headers["x-class"] = "1"
        return response


def not_middleware():
    return None


def test_middleware_specs(mlp_state):
    app = build_app(
        mlp_state,
        middleware=["tests.test_serve:add_test_header", "tests.test_serve:AddClassHeader"],
    )
    resp = TestClient(app).get("/health")
    assert resp.status_code == 200
    assert resp.headers["x-test"] == "1"
    assert resp.headers["x-class"] == "1"

    with pytest.raises(ValueError, match="not middleware"):
        build_app(mlp_state, middleware=["tests.test_serve:not_middleware"])


def test_ready_503_when_not_ready(mlp_state):
    not_ready = dataclasses.replace(mlp_state, ready=False)
    resp = TestClient(build_app(not_ready)).get("/ready")
    assert resp.status_code == 503
    assert resp.json() == {"ready": False}
