"""The error codes: each bad request gets the status and the code that docs/http-api.md lists."""

import asyncio
import dataclasses
import re
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from downshift.adapters.text import TextTooLongError
from downshift.serve.app import build_app
from downshift.serve.errors import ERROR_CODES, http_error_handler
from downshift.serve.openai import served_name

GOOD = {"inputs": {"x": [[0.0] * 16]}}
ST = "application/vnd.safetensors"
KEY = "s3cret"
REPO = Path(__file__).resolve().parent.parent


def _state(state, **options):
    return dataclasses.replace(state, options=dataclasses.replace(state.options, **options))


def _text_state(state, **recipe):
    def encode(text):
        raise TextTooLongError("text longer than this model reads")

    embedding = SimpleNamespace(prompts={"query": "q "}, default_prompt=None, **recipe)
    return dataclasses.replace(state, text=SimpleNamespace(encode=encode), embedding=embedding)


def _client(state, **kwargs):
    return TestClient(build_app(state, api_key=kwargs.pop("api_key", None)), **kwargs)


def _invalid_request(state, monkeypatch):
    return _client(state).post("/predict", content=b"{", headers={"Content-Type": "text/json"})


def _invalid_input(state, monkeypatch):
    return _client(state).post("/predict", json={"inputs": {"x": [[0.0] * 3]}})


def _missing_input(state, monkeypatch):
    return _client(state).post("/predict", json={"inputs": {"y": [[0.0] * 16]}})


def _text_too_long(state, monkeypatch):
    return _client(_text_state(state)).post("/predict", json={"text": ["hello"]})


def _unknown_prompt(state, monkeypatch):
    body = {"text": ["hello"], "prompt_name": "nope"}
    return _client(_text_state(state)).post("/predict", json=body)


def _text_not_supported(state, monkeypatch):
    return _client(state).post("/predict", json={"text": ["hello"]})


def _not_graph_model(state, monkeypatch):
    body = {"x": [[0.0] * 16], "edge_index": [[0], [0]]}
    return _client(state).post("/predict/graph", json=body)


def _unauthorized(state, monkeypatch):
    return _client(state, api_key=KEY).post("/predict", json=GOOD)


def _body_too_large(state, monkeypatch):
    return _client(_state(state, max_body_bytes=10)).post("/predict", json=GOOD)


def _unsupported_media_type(state, monkeypatch):
    return _client(state).post("/predict", content=b"abc", headers={"Content-Type": "text/csv"})


def _not_found(state, monkeypatch):
    return _client(state).get("/nowhere")


def _method_not_allowed(state, monkeypatch):
    return _client(state).delete("/health")


def _not_ready(state, monkeypatch):
    app = build_app(state, api_key=None)
    app.state.serving = None
    return TestClient(app).post("/predict", json=GOOD)


def _at_capacity(state, monkeypatch):
    full = _state(state, max_concurrency=1, max_queue=0)
    full.in_flight = 1
    return _client(full).post("/predict", json=GOOD)


def _request_timeout(state, monkeypatch):
    def slow(inputs):
        time.sleep(0.5)

    slow_state = _state(state, request_timeout=0.05)
    monkeypatch.setattr(slow_state.backend, "infer", slow)
    return _client(slow_state).post("/predict", json=GOOD)


def _internal_error(state, monkeypatch):
    def boom(inputs):
        raise RuntimeError("bug")

    monkeypatch.setattr(state.backend, "infer", boom)
    return _client(state, raise_server_exceptions=False).post("/predict", json=GOOD)


def _embedder(state):
    return _text_state(state, pooling="mean")


def _model_not_found(state, monkeypatch):
    body = {"model": "other", "input": "hello"}
    return _client(_embedder(state)).post("/v1/embeddings", json=body)


def _not_embedding_model(state, monkeypatch):
    return _client(state).post("/v1/embeddings", json={"model": "m", "input": "hello"})


def _dimensions_not_supported(state, monkeypatch):
    embedder = _embedder(state)
    body = {"model": served_name(embedder), "input": "hello", "dimensions": 8}
    return _client(embedder).post("/v1/embeddings", json=body)


OPENAI_CODES = {"model_not_found", "not_embedding_model", "dimensions_not_supported"}

CASES = {
    "invalid_request": (422, _invalid_request),
    "invalid_input": (400, _invalid_input),
    "missing_input": (400, _missing_input),
    "text_too_long": (400, _text_too_long),
    "unknown_prompt": (400, _unknown_prompt),
    "text_not_supported": (400, _text_not_supported),
    "not_graph_model": (400, _not_graph_model),
    "unauthorized": (401, _unauthorized),
    "body_too_large": (413, _body_too_large),
    "unsupported_media_type": (415, _unsupported_media_type),
    "not_found": (404, _not_found),
    "method_not_allowed": (405, _method_not_allowed),
    "not_ready": (503, _not_ready),
    "at_capacity": (503, _at_capacity),
    "request_timeout": (503, _request_timeout),
    "internal_error": (500, _internal_error),
    "model_not_found": (404, _model_not_found),
    "not_embedding_model": (400, _not_embedding_model),
    "dimensions_not_supported": (400, _dimensions_not_supported),
}


@pytest.mark.parametrize("code", CASES)
def test_a_bad_request_gets_its_status_and_code(code, mlp_state, monkeypatch):
    status, send = CASES[code]

    response = send(mlp_state, monkeypatch)

    assert response.status_code == status, response.text
    body = response.json()
    if code in OPENAI_CODES:  # these cases use a /v1 path
        assert body["error"]["code"] == code
        assert "message" in body["error"]
    else:
        assert body["code"] == code
        assert "detail" in body


def test_the_500_keeps_request_id_and_the_401_keeps_its_challenge(mlp_state, monkeypatch):
    bad = _unauthorized(mlp_state, monkeypatch)
    assert bad.headers["www-authenticate"] == "Bearer"

    boom = _internal_error(mlp_state, monkeypatch)
    assert boom.json()["request_id"] == boom.headers["x-request-id"]


def test_the_422_keeps_the_detail_list_of_fastapi(mlp_state):
    body = _client(mlp_state).post("/predict", json={"inputs": 1}).json()

    assert body["code"] == "invalid_request"
    assert body["detail"][0]["loc"][0] == "body"


def test_an_error_with_no_code_gets_one_from_its_status():
    from starlette.exceptions import HTTPException

    error = HTTPException(418, "teapot")
    response = asyncio.run(http_error_handler(SimpleNamespace(scope={"path": "/x"}), error))  # type: ignore[arg-type]

    assert response.body == b'{"detail":"teapot","code":"http_error"}'


def test_every_error_code_is_tested_and_listed_in_the_docs():
    text = (REPO / "docs" / "http-api.md").read_text(encoding="utf-8")
    listed = set(re.findall(r"^\| `([a-z_]+)` \| (?:\d{3}|any) \|", text, flags=re.MULTILINE))

    assert listed == set(ERROR_CODES)
    assert set(CASES) | {"http_error"} == set(ERROR_CODES)
