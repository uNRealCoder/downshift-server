"""HTTP contract tests. Each state is exported once per module; export is the slow part."""

import dataclasses

import numpy as np
import pytest
from fastapi import HTTPException, Request
from fastapi.testclient import TestClient
from starlette.middleware.base import BaseHTTPMiddleware

from downshift.loading import LoadedModel, load_model
from downshift.serve import app as serve_app
from downshift.serve.app import build_app
from downshift.serve.engine import ServeOptions, prepare_serving

MLP_INPUT = {"inputs": {"x": [[0.0] * 16]}}


def _client_emitting(state, monkeypatch, output: np.ndarray) -> TestClient:
    """A client whose backend returns `output` (as output_0) whatever the input."""
    monkeypatch.setattr(state.backend, "infer", lambda inputs: {"output_0": output})
    return TestClient(build_app(state))


@pytest.fixture(scope="module")
def mlp_client(mlp_state) -> TestClient:
    return TestClient(build_app(mlp_state))


@pytest.fixture(scope="module")
def gcn_client(serve_fixture) -> TestClient:
    return TestClient(build_app(serve_fixture("gnn_gcn")))


@pytest.fixture(scope="module")
def branch_client(branch_state) -> TestClient:
    return TestClient(build_app(branch_state))


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


def test_predict_graph_with_edge_attr_is_accepted(gcn_client):
    x = np.random.randn(5, 8).tolist()
    edge_index = [[0, 1, 2, 3, 4, 0, 2], [1, 2, 3, 4, 0, 3, 4]]
    edge_attr = [[0.1]] * 7
    resp = gcn_client.post(
        "/predict/graph", json={"x": x, "edge_index": edge_index, "edge_attr": edge_attr}
    )
    assert resp.status_code == 200, resp.text


def test_predict_reraises_http_exception_raised_by_the_backend(mlp_state, monkeypatch):
    def boom(inputs):
        raise HTTPException(422, "custom backend error")

    monkeypatch.setattr(mlp_state.backend, "infer", boom)
    client = TestClient(build_app(mlp_state))

    resp = client.post("/predict", json={"inputs": {"x": [[0.0] * 16]}})
    assert resp.status_code == 422
    assert resp.json()["detail"] == "custom backend error"


def test_predict_nan_and_inf_become_null(mlp_state, monkeypatch):
    # Deliberate wire behaviour: valid JSON (null), not stdlib's NaN/Infinity extension.
    output = np.array([[np.nan, np.inf, 1.0]], dtype=np.float32)
    resp = _client_emitting(mlp_state, monkeypatch, output).post("/predict", json=MLP_INPUT)
    assert resp.status_code == 200, resp.text
    assert "null" in resp.text
    assert "NaN" not in resp.text and "Infinity" not in resp.text
    assert resp.json()["outputs"]["output_0"] == [[None, None, 1.0]]


def test_predict_float16_output_serializes(mlp_state, monkeypatch):
    output = np.array([[0.1, 1.5, -2.25], [65504.0, 3.14159, 0.0]], dtype=np.float16)
    resp = _client_emitting(mlp_state, monkeypatch, output).post("/predict", json=MLP_INPUT)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["dtypes"]["output_0"] == "float16"
    assert body["shapes"]["output_0"] == [2, 3]
    np.testing.assert_allclose(body["outputs"]["output_0"], output.astype(np.float32), rtol=1e-3)


def test_predict_non_contiguous_output_serializes(mlp_state, monkeypatch):
    output = np.arange(16, dtype=np.float32).reshape(4, 4)[:, ::2]  # strided view
    assert not output.flags.c_contiguous
    resp = _client_emitting(mlp_state, monkeypatch, output).post("/predict", json=MLP_INPUT)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["shapes"]["output_0"] == [4, 2]
    assert body["outputs"]["output_0"] == output.tolist()


def test_predict_scalar_output_serializes(mlp_state, monkeypatch):
    # 0-d arrays take the tolist() fallback: orjson rejects them and ascontiguousarray
    # would silently promote them to shape (1,).
    output = np.asarray(2.5, dtype=np.float32)
    resp = _client_emitting(mlp_state, monkeypatch, output).post("/predict", json=MLP_INPUT)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["outputs"]["output_0"] == 2.5
    assert body["shapes"]["output_0"] == []
    assert body["dtypes"]["output_0"] == "float32"


def test_predict_unsupported_dtype_falls_back_per_array(mlp_state, monkeypatch):
    # Simulate a dtype this orjson cannot write natively (e.g. float128 on Linux) without
    # depending on the platform: drop float32 from the supported set for this test only.
    monkeypatch.setattr(serve_app, "_ORJSON_DTYPES", serve_app._ORJSON_DTYPES - {"float32"})
    fallback = np.array([[1.0, 2.0]], dtype=np.float32)
    native = np.array([[3, 4]], dtype=np.int64)
    monkeypatch.setattr(
        mlp_state.backend, "infer", lambda inputs: {"a": fallback, "b": native}
    )
    resp = TestClient(build_app(mlp_state)).post("/predict", json=MLP_INPUT)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["outputs"] == {"a": [[1.0, 2.0]], "b": [[3, 4]]}
    assert body["dtypes"] == {"a": "float32", "b": "int64"}


def test_openapi_still_documents_predict_response(mlp_client):
    schema = mlp_client.get("/openapi.json").json()
    assert "PredictResponse" in schema["components"]["schemas"]
    for path in ("/predict", "/predict/graph"):
        ok = schema["paths"][path]["post"]["responses"]["200"]
        assert ok["content"]["application/json"]["schema"]["$ref"].endswith("/PredictResponse")


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
