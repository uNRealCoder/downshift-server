"""HTTP contract tests. Each state is exported once per module; export is the slow part."""

import dataclasses
import logging
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from fastapi import HTTPException, Request
from fastapi.testclient import TestClient
from starlette.middleware.base import BaseHTTPMiddleware

from downshift.loading import LoadedModel, LoadSpec, load_model
from downshift.serve import app_for
from downshift.serve import predict as serve_predict
from downshift.serve.app import build_app
from downshift.serve.engine import ServeOptions, prepare_serving
from downshift.serve.schemas import OutputEncoding
from tests.conftest import b64_input, b64_output
from tests.models import clean_mlp

MLP_INPUT = {"inputs": {"x": [[0.0] * 16]}}


def _client_emitting(state, monkeypatch, outputs: dict) -> TestClient:
    """A client whose backend returns `outputs` whatever the input."""
    monkeypatch.setattr(state.backend, "infer", lambda inputs: outputs)
    return TestClient(build_app(state))


def _assert_same_outputs(client: TestClient, path: str, reference_body: dict, body: dict) -> dict:
    """POST both bodies; `reference_body` must answer with JSON lists, `body` may answer base64.

    Asserts both succeed with the same output_0 and returns `body`'s response JSON.
    """
    reference = client.post(path, json=reference_body)
    resp = client.post(path, json=body)
    assert reference.status_code == 200, reference.text
    assert resp.status_code == 200, resp.text
    expected = reference.json()["outputs"]["output_0"]
    assert isinstance(expected, list)
    actual = resp.json()["outputs"]["output_0"]
    if isinstance(actual, dict):
        actual = b64_output(actual)
    np.testing.assert_allclose(actual, expected)
    return resp.json()


def _with_options(state, **overrides):
    return dataclasses.replace(state, options=dataclasses.replace(state.options, **overrides))


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
    assert meta["limits"] == {
        "max_body_bytes": mlp_state.options.max_body_bytes,
        "max_input_bytes": mlp_state.options.max_input_bytes,
        "max_concurrency": mlp_state.options.max_concurrency,
        "max_queue": mlp_state.options.max_queue,
        "request_timeout": mlp_state.options.request_timeout,
    }
    assert set(meta["boot"]) >= {"export", "verify", "session", "warmup"}
    assert meta["warmup"] == {
        "count": mlp_state.warmup_stats.count,
        "mean_ms": mlp_state.warmup_stats.mean_ms,
        "synthesized": mlp_state.warmup_stats.synthesized,
    }


def test_app_for_serves_predict_immediately():
    client = TestClient(app_for(clean_mlp.make_model(), clean_mlp.make_inputs(), warmup=1))
    resp = client.post("/predict", json=MLP_INPUT)
    assert resp.status_code == 200, resp.text
    assert client.get("/ready").json() == {"ready": True}


def test_app_for_verifies_a_pre_built_onnx_against_a_reference(exported_mlp):
    path, model, _ = exported_mlp
    client = TestClient(app_for(str(path), clean_mlp.make_inputs(), reference=model, warmup=1))
    meta = client.get("/metadata").json()
    assert meta["verdict"]["status"] == "CLEAN", meta["verdict"]["reason"]
    assert meta["backend"]["name"] == "onnxruntime"


def test_app_for_mounts_under_a_prefix():
    from fastapi import FastAPI

    outer = FastAPI()
    outer.mount("/model", app_for(clean_mlp.make_model(), clean_mlp.make_inputs(), warmup=1))
    client = TestClient(outer)
    assert client.get("/model/health").json() == {"status": "ok"}
    resp = client.post("/model/predict", json=MLP_INPUT)
    assert resp.status_code == 200, resp.text


def test_predict_batch(mlp_client):
    x = np.random.randn(3, 16).tolist()
    resp = mlp_client.post("/predict", json={"inputs": {"x": x}})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["shapes"]["output_0"] == [3, 4]
    assert np.asarray(body["outputs"]["output_0"]).shape == (3, 4)
    assert body["dtypes"]["output_0"] == "float32"


def test_predict_generates_a_request_id_and_echoes_it(mlp_client):
    resp = mlp_client.post("/predict", json=MLP_INPUT)
    assert resp.status_code == 200, resp.text
    assert resp.headers["x-request-id"]


def test_predict_echoes_a_client_supplied_request_id(mlp_client):
    resp = mlp_client.post(
        "/predict", json=MLP_INPUT, headers={"x-request-id": "caller-supplied-id"}
    )
    assert resp.status_code == 200, resp.text
    assert resp.headers["x-request-id"] == "caller-supplied-id"


def test_health_gets_its_own_request_id_too(mlp_client):
    resp = mlp_client.get("/health")
    assert resp.headers["x-request-id"]


def test_predict_reports_server_timing(mlp_client):
    resp = mlp_client.post("/predict", json=MLP_INPUT)
    assert resp.status_code == 200, resp.text
    timing = resp.headers["server-timing"]
    assert "parse;dur=" in timing
    assert "codec;dur=" in timing
    assert "infer;dur=" in timing


def test_predict_missing_input(mlp_client):
    resp = mlp_client.post("/predict", json={"inputs": {"y": [[0.0] * 16]}})
    assert resp.status_code == 400
    assert "x" in resp.json()["detail"]


def test_predict_malformed_json_body_is_422(mlp_client):
    # Bodies are parsed by orjson (OrjsonRoute); its decode error must still map to FastAPI's 422.
    resp = mlp_client.post(
        "/predict", content=b'{"inputs": ', headers={"content-type": "application/json"}
    )
    assert resp.status_code == 422, resp.text
    assert resp.json()["detail"][0]["msg"] == "JSON decode error"


def test_predict_wrong_feature_size(mlp_client):
    resp = mlp_client.post("/predict", json={"inputs": {"x": [[0.0] * 5]}})
    assert resp.status_code == 400
    assert resp.json()["detail"]


def test_predict_typed_input(mlp_client):
    payload = {"data": [0.0] * 32, "dtype": "float32", "shape": [2, 16]}
    resp = mlp_client.post("/predict", json={"inputs": {"x": payload}})
    assert resp.status_code == 200, resp.text
    assert resp.json()["shapes"]["output_0"] == [2, 4]


def test_predict_base64_input_matches_json_input(mlp_client):
    x = np.random.randn(3, 16).astype(np.float32)
    body = _assert_same_outputs(
        mlp_client, "/predict", {"inputs": {"x": x.tolist()}}, {"inputs": {"x": b64_input(x)}}
    )
    assert body["shapes"]["output_0"] == [3, 4]


def test_predict_base64_input_on_torch_backend_does_not_warn(serve_fixture):
    # np.frombuffer views are read-only; torch.from_numpy warns on those unless the backend
    # copies. Warnings are errors here so a regression fails instead of logging.
    client = TestClient(build_app(serve_fixture("clean_mlp", backend="torch")))
    x = np.random.randn(2, 16).astype(np.float32)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _assert_same_outputs(
            client, "/predict", {"inputs": {"x": x.tolist()}}, {"inputs": {"x": b64_input(x)}}
        )


@pytest.mark.parametrize(
    ("overrides", "expected_detail"),
    [
        ({"shape": None}, "needs dtype and shape"),
        ({"dtype": None}, "needs dtype and shape"),
        ({"shape": [2, 16]}, "expected 128 bytes"),
        ({"data": "not*base64"}, "invalid base64"),
        ({"data": "AAAA"}, "got 3"),
        ({"dtype": ">f4"}, "little-endian"),
        ({"dtype": "not-a-real-dtype"}, "unknown dtype"),
        ({"shape": [-1, 16]}, "negative dimension"),
    ],
)
def test_predict_base64_input_client_errors(mlp_client, overrides, expected_detail):
    x = np.zeros((3, 16), dtype=np.float32)
    resp = mlp_client.post("/predict", json={"inputs": {"x": b64_input(x, **overrides)}})
    assert resp.status_code == 400, resp.text
    detail = resp.json()["detail"]
    assert "'x'" in detail
    assert expected_detail in detail


def test_predict_base64_input_over_size_cap(mlp_state):
    client = TestClient(build_app(_with_options(mlp_state, max_input_bytes=64)))
    x = np.zeros((3, 16), dtype=np.float32)  # 192 bytes
    resp = client.post("/predict", json={"inputs": {"x": b64_input(x)}})
    assert resp.status_code == 400, resp.text
    detail = resp.json()["detail"]
    assert "'x'" in detail
    assert "192 bytes exceeds the server limit of 64 bytes" in detail

    # Just under the cap is fine.
    small = np.zeros((1, 16), dtype=np.float32)  # 64 bytes
    resp = client.post("/predict", json={"inputs": {"x": b64_input(small)}})
    assert resp.status_code == 200, resp.text


def test_predict_output_encoding_base64_per_request(mlp_client):
    request = {"inputs": {"x": np.random.randn(3, 16).tolist()}}
    body = _assert_same_outputs(
        mlp_client, "/predict", request, request | {"output_encoding": "base64"}
    )
    entry = body["outputs"]["output_0"]
    assert set(entry) == {"data", "dtype", "shape"}
    assert entry["dtype"] == "float32"
    assert entry["shape"] == [3, 4]
    assert body["shapes"]["output_0"] == [3, 4]
    assert body["dtypes"]["output_0"] == "float32"


def test_predict_graph_output_encoding_base64(gcn_client):
    request = {
        "x": np.random.randn(5, 8).tolist(),
        "edge_index": [[0, 1, 2, 3, 4, 0, 2], [1, 2, 3, 4, 0, 3, 4]],
    }
    body = _assert_same_outputs(
        gcn_client, "/predict/graph", request, request | {"output_encoding": "base64"}
    )
    assert body["outputs"]["output_0"]["shape"] == [5, 4]


def test_predict_graph_base64_inputs(gcn_client):
    x = np.random.randn(5, 8).astype(np.float32)
    edge_index = np.array([[0, 1, 2, 3, 4, 0, 2], [1, 2, 3, 4, 0, 3, 4]], dtype=np.int64)
    edge_attr = np.full((7, 1), 0.1, dtype=np.float32)
    body = _assert_same_outputs(
        gcn_client,
        "/predict/graph",
        {"x": x.tolist(), "edge_index": edge_index.tolist(), "edge_attr": edge_attr.tolist()},
        {
            "x": b64_input(x),
            "edge_index": b64_input(edge_index),
            "edge_attr": b64_input(edge_attr),
            "output_encoding": "base64",
        },
    )
    assert body["outputs"]["output_0"]["shape"] == [5, 4]


def test_predict_server_default_base64_and_per_request_json_override(mlp_state):
    client = TestClient(build_app(_with_options(mlp_state, output_encoding=OutputEncoding.base64)))
    request = {"inputs": {"x": np.random.randn(2, 16).tolist()}}
    # The per-request "json" override is the reference (the helper asserts it answers lists);
    # the plain request must fall back to the server default and answer base64.
    body = _assert_same_outputs(client, "/predict", request | {"output_encoding": "json"}, request)
    entry = body["outputs"]["output_0"]
    assert isinstance(entry, dict) and entry["shape"] == [2, 4]


def test_predict_base64_output_of_non_contiguous_and_scalar_arrays(mlp_state, monkeypatch):
    strided = np.arange(16, dtype=np.float32).reshape(4, 4)[:, ::2]
    scalar = np.asarray(2.5, dtype=np.float16)
    client = _client_emitting(mlp_state, monkeypatch, {"a": strided, "b": scalar})
    resp = client.post("/predict", json=MLP_INPUT | {"output_encoding": "base64"})
    assert resp.status_code == 200, resp.text
    outputs = resp.json()["outputs"]
    np.testing.assert_array_equal(b64_output(outputs["a"]), strided)
    assert outputs["b"]["shape"] == []
    assert outputs["b"]["dtype"] == "float16"
    assert b64_output(outputs["b"]) == np.float16(2.5)


def test_predict_rejects_unknown_output_encoding(mlp_client):
    resp = mlp_client.post("/predict", json=MLP_INPUT | {"output_encoding": "hex"})
    assert resp.status_code == 422
    assert "output_encoding" in resp.text


def test_openapi_documents_predict_contract(mlp_client):
    schema = mlp_client.get("/openapi.json").json()
    assert "PredictResponse" in schema["components"]["schemas"]
    for path in ("/predict", "/predict/graph"):
        ok = schema["paths"][path]["post"]["responses"]["200"]
        assert ok["content"]["application/json"]["schema"]["$ref"].endswith("/PredictResponse")
    for model in ("PredictRequest", "GraphPredictRequest"):
        prop = schema["components"]["schemas"][model]["properties"]["output_encoding"]
        assert "base64" in prop["description"]


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


def test_predict_graph_rejects_an_out_of_range_edge_index(gcn_client):
    x = np.random.randn(5, 8).tolist()
    resp = gcn_client.post("/predict/graph", json={"x": x, "edge_index": [[0, 99], [1, 2]]})
    assert resp.status_code == 400
    assert "outside the node range [0, 5)" in resp.json()["detail"]


def test_predict_graph_rejects_a_negative_edge_index(gcn_client):
    x = np.random.randn(5, 8).tolist()
    resp = gcn_client.post("/predict/graph", json={"x": x, "edge_index": [[-1, 2], [1, 2]]})
    assert resp.status_code == 400
    assert "outside the node range [0, 5)" in resp.json()["detail"]


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
    client = _client_emitting(mlp_state, monkeypatch, {"output_0": output})
    resp = client.post("/predict", json=MLP_INPUT)
    assert resp.status_code == 200, resp.text
    assert "null" in resp.text
    assert "NaN" not in resp.text and "Infinity" not in resp.text
    assert resp.json()["outputs"]["output_0"] == [[None, None, 1.0]]


@pytest.mark.parametrize(
    "output",
    [
        np.array([[0.1, 1.5, -2.25], [65504.0, 3.14159, 0.0]], dtype=np.float16),
        np.arange(16, dtype=np.float32).reshape(4, 4)[:, ::2],  # strided view: contiguity copy
        np.asarray(2.5, dtype=np.float32),  # 0-d: orjson rejects it, so the tolist() fallback
    ],
    ids=["float16", "non_contiguous", "scalar"],
)
def test_predict_output_serializes(mlp_state, monkeypatch, output):
    client = _client_emitting(mlp_state, monkeypatch, {"output_0": output})
    resp = client.post("/predict", json=MLP_INPUT)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["shapes"]["output_0"] == list(output.shape)
    assert body["dtypes"]["output_0"] == output.dtype.name
    np.testing.assert_allclose(body["outputs"]["output_0"], output.tolist(), rtol=1e-3)


def test_predict_unsupported_dtype_falls_back_per_array(mlp_state, monkeypatch):
    # Simulate a dtype this orjson cannot write natively (e.g. float128 on Linux) without
    # depending on the platform: drop float32 from the supported set for this test only.
    monkeypatch.setattr(serve_predict, "_ORJSON_DTYPES", serve_predict._ORJSON_DTYPES - {"float32"})
    fallback = np.array([[1.0, 2.0]], dtype=np.float32)
    native = np.array([[3, 4]], dtype=np.int64)
    client = _client_emitting(mlp_state, monkeypatch, {"a": fallback, "b": native})
    resp = client.post("/predict", json=MLP_INPUT)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["outputs"] == {"a": [[1.0, 2.0]], "b": [[3, 4]]}
    assert body["dtypes"] == {"a": "float32", "b": "int64"}


def test_backends_agree_on_same_contract():
    loaded = load_model(LoadSpec("tests.models.clean_mlp:make_model"))
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


def test_ready_and_predict_503_until_the_loader_lands(mlp_state):
    """Plan 3.3 U3: bind first, load on a background thread, 503 until the verdict is in,
    200 after, no restart in between."""
    release = threading.Event()

    def loader():
        assert release.wait(timeout=5)
        return mlp_state

    app = build_app(loader=loader)
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200

        ready = client.get("/ready")
        assert ready.status_code == 503
        assert ready.json() == {"ready": False, "phase": "load"}

        for resp in (
            client.get("/metadata"),
            client.get("/schema"),
            client.post("/predict", json=MLP_INPUT),
        ):
            assert resp.status_code == 503, resp.text
            assert resp.headers["retry-after"] == "2"
            assert resp.json() == {"detail": "model is not ready"}

        release.set()
        app.state.loader_thread.join(timeout=5)

        assert client.get("/ready").json() == {"ready": True}
        assert client.get("/metadata").status_code == 200
        assert client.get("/schema").status_code == 200
        assert client.post("/predict", json=MLP_INPUT).status_code == 200


def test_build_app_needs_exactly_one_of_state_or_loader(mlp_state):
    with pytest.raises(ValueError, match="exactly one"):
        build_app()
    with pytest.raises(ValueError, match="exactly one"):
        build_app(mlp_state, loader=lambda: mlp_state)


def test_predict_torch_backend_shape_error_is_400(serve_fixture):
    client = TestClient(build_app(serve_fixture("clean_mlp", backend="torch")))
    resp = client.post("/predict", json={"inputs": {"x": [[0.0] * 5]}})
    assert resp.status_code == 400, resp.text
    assert "'x'" not in resp.json()["detail"]  # torch's own message, not to_numpy's


def test_predict_backend_bug_maps_to_500_without_leaking_details(mlp_state, monkeypatch, caplog):
    def boom(inputs):
        raise RuntimeError("some internal bug: /etc/secret/path")

    monkeypatch.setattr(mlp_state.backend, "infer", boom)
    client = TestClient(build_app(mlp_state), raise_server_exceptions=False)

    with caplog.at_level(logging.ERROR, logger="downshift.serve"):
        resp = client.post("/predict", json=MLP_INPUT)

    assert resp.status_code == 500
    body = resp.json()
    assert body["detail"] == "inference failed on the server; see the server log"
    assert body["request_id"] == resp.headers["x-request-id"]
    assert "some internal bug" not in resp.text
    assert "/etc/secret/path" not in resp.text
    assert "unhandled exception" in caplog.text
    assert body["request_id"] in caplog.text


def test_predict_torch_backend_out_of_memory_maps_to_500(serve_fixture):
    state = serve_fixture("clean_mlp", backend="torch")

    def boom(*args, **kwargs):
        raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")

    state.backend.module.forward = boom
    client = TestClient(build_app(state), raise_server_exceptions=False)

    resp = client.post("/predict", json={"inputs": {"x": [[0.0] * 16]}})
    assert resp.status_code == 500
    assert "out of memory" not in resp.text


def test_predict_body_over_content_length_limit_is_413(mlp_state):
    client = TestClient(build_app(_with_options(mlp_state, max_body_bytes=100)))
    resp = client.post(
        "/predict",
        content=b'{"inputs": {"x": [[0.0]]}}',
        headers={"content-length": "300000000", "content-type": "application/json"},
    )
    assert resp.status_code == 413, resp.text
    assert resp.json()["detail"] == (
        "request body is 300000000 bytes; the server limit is 100 bytes (--max-body-bytes)"
    )


def test_predict_chunked_body_over_limit_is_413(mlp_state):
    client = TestClient(build_app(_with_options(mlp_state, max_body_bytes=10)))

    def chunks():
        yield b'{"inputs"'
        yield b': {"x": [[0.0]]}}'

    resp = client.post("/predict", content=chunks(), headers={"content-type": "application/json"})
    assert resp.status_code == 413, resp.text
    assert "the server limit is 10 bytes (--max-body-bytes)" in resp.json()["detail"]


def test_predict_body_within_limit_succeeds(mlp_state):
    client = TestClient(build_app(_with_options(mlp_state, max_body_bytes=10_000)))
    resp = client.post("/predict", json=MLP_INPUT)
    assert resp.status_code == 200, resp.text


def test_predict_concurrency_default_serializes_inferences(mlp_state, monkeypatch):
    active = 0
    max_active = 0
    guard = threading.Lock()

    def slow_infer(inputs):
        nonlocal active, max_active
        with guard:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.1)
        with guard:
            active -= 1
        return {"output_0": np.zeros((1, 4), dtype=np.float32)}

    serial_state = _with_options(mlp_state, max_concurrency=1)
    monkeypatch.setattr(serial_state.backend, "infer", slow_infer)
    client = TestClient(build_app(serial_state))

    with ThreadPoolExecutor(2) as pool:
        futures = [pool.submit(client.post, "/predict", json=MLP_INPUT) for _ in range(2)]
        results = [f.result() for f in futures]

    assert all(r.status_code == 200 for r in results)
    assert max_active == 1


def test_predict_concurrency_option_allows_overlapping_inferences(mlp_state, monkeypatch):
    active = 0
    max_active = 0
    guard = threading.Lock()

    def slow_infer(inputs):
        nonlocal active, max_active
        with guard:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.1)
        with guard:
            active -= 1
        return {"output_0": np.zeros((1, 4), dtype=np.float32)}

    parallel_state = _with_options(mlp_state, max_concurrency=2)
    monkeypatch.setattr(parallel_state.backend, "infer", slow_infer)
    client = TestClient(build_app(parallel_state))

    with ThreadPoolExecutor(2) as pool:
        futures = [pool.submit(client.post, "/predict", json=MLP_INPUT) for _ in range(2)]
        results = [f.result() for f in futures]

    assert all(r.status_code == 200 for r in results)
    assert max_active == 2


def _wait_until(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "condition never became true"
        time.sleep(0.005)


def test_predict_past_capacity_is_a_fast_503(mlp_state, monkeypatch):
    release = threading.Event()

    def blocking_infer(inputs):
        release.wait(timeout=5)
        return {"output_0": np.zeros((1, 4), dtype=np.float32)}

    state = _with_options(mlp_state, max_concurrency=1, max_queue=1)
    monkeypatch.setattr(state.backend, "infer", blocking_infer)
    client = TestClient(build_app(state))

    try:
        with ThreadPoolExecutor(3) as pool:
            admitted = [pool.submit(client.post, "/predict", json=MLP_INPUT) for _ in range(2)]
            _wait_until(lambda: state.in_flight >= 2)

            start = time.monotonic()
            over = client.post("/predict", json=MLP_INPUT)
            elapsed = time.monotonic() - start

            release.set()
            results = [f.result() for f in admitted]
    finally:
        release.set()

    assert over.status_code == 503
    assert over.headers["retry-after"] == "1"
    assert "server is at capacity" in over.json()["detail"]
    assert "1 running" in over.json()["detail"]
    assert "1 queued" in over.json()["detail"]
    assert elapsed < 0.5  # never touches the executor
    assert all(r.status_code == 200 for r in results)


def test_predict_queued_past_request_timeout_is_503_and_never_infers(mlp_state, monkeypatch):
    calls: list[int] = []

    def slow_infer(inputs):
        calls.append(1)
        time.sleep(0.3)
        return {"output_0": np.zeros((1, 4), dtype=np.float32)}

    state = _with_options(mlp_state, max_concurrency=1, max_queue=5, request_timeout=0.05)
    monkeypatch.setattr(state.backend, "infer", slow_infer)
    client = TestClient(build_app(state))

    with ThreadPoolExecutor(2) as pool:
        first = pool.submit(client.post, "/predict", json=MLP_INPUT)
        _wait_until(lambda: state.in_flight >= 1)
        second = pool.submit(client.post, "/predict", json=MLP_INPUT)

        first_result = first.result()
        second_result = second.result()

    assert first_result.status_code == 200
    assert second_result.status_code == 503
    assert "--request-timeout" in second_result.json()["detail"]
    assert len(calls) == 1  # the queued predict never reached infer()


def test_health_stays_fast_while_predicts_are_queued(mlp_state, monkeypatch):
    release = threading.Event()

    def blocking_infer(inputs):
        release.wait(timeout=5)
        return {"output_0": np.zeros((1, 4), dtype=np.float32)}

    state = _with_options(mlp_state, max_concurrency=1, max_queue=50)
    monkeypatch.setattr(state.backend, "infer", blocking_infer)
    client = TestClient(build_app(state))

    try:
        with ThreadPoolExecutor(10) as pool:
            predicts = [pool.submit(client.post, "/predict", json=MLP_INPUT) for _ in range(10)]
            _wait_until(lambda: state.in_flight >= 10)

            start = time.monotonic()
            health = client.get("/health")
            elapsed = time.monotonic() - start

            release.set()
            for f in predicts:
                assert f.result().status_code == 200
    finally:
        release.set()

    assert health.status_code == 200
    # Generously below the 5+ second stalls a shared sync threadpool used to cause; the
    # point is "never blocks behind the predict queue", not a tight latency bound.
    assert elapsed < 1.0


def test_axis_max_num_nodes_rejects_a_larger_graph(serve_fixture):
    client = TestClient(build_app(serve_fixture("gnn_gcn", axis_max={"num_nodes": 500})))
    edge_index = [[0, 1], [1, 0]]

    ok = client.post(
        "/predict/graph", json={"x": np.random.randn(500, 8).tolist(), "edge_index": edge_index}
    )
    too_big = client.post(
        "/predict/graph", json={"x": np.random.randn(501, 8).tolist(), "edge_index": edge_index}
    )

    assert ok.status_code == 200, ok.text
    assert too_big.status_code == 400
    assert "500" in too_big.json()["detail"]
