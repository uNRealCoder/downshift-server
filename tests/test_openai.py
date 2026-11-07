"""The OpenAI-compatible routes: /v1/models, /v1/embeddings, the OpenAI error shape on /v1, and
the options behind them. The embedder is a small BERT repo with random weights (mean pooling and
L2 normalization), so nothing is downloaded."""

import base64
import dataclasses

import numpy as np
import pytest

pytest.importorskip("transformers")
pytest.importorskip("tokenizers")

from fastapi.testclient import TestClient

import downshift
from downshift.cli.runtime import ServeArgs
from downshift.loading import LoadSpec, load_model
from downshift.serve.app import build_app
from downshift.serve.engine import prepare_serving
from downshift.serve.errors import ApiError
from downshift.serve.openai import _check_dimensions, _matryoshka_cut
from downshift.serve.options import ServeOptions
from tests.models import hf_repo


@pytest.fixture(scope="module")
def repo(tmp_path_factory) -> str:
    return hf_repo.write_encoder_repo(tmp_path_factory.mktemp("openai-repo"))


@pytest.fixture(scope="module")
def state(repo):
    return prepare_serving(load_model(LoadSpec(repo)))


@pytest.fixture(scope="module")
def name(state) -> str:
    return state.served_name


@pytest.fixture(scope="module")
def client(state) -> TestClient:
    return TestClient(build_app(state, api_key=None))


def _with(state, **options):
    return dataclasses.replace(state, options=dataclasses.replace(state.options, **options))


def _embed(client, name, input_, **extra):
    return client.post("/v1/embeddings", json={"model": name, "input": input_, **extra})


def _decode(vector: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(vector), dtype="<f4")


def _predict_vectors(client, texts: list[str]) -> np.ndarray:
    body = client.post("/predict", json={"text": texts}).json()
    return np.asarray(body["outputs"]["output_0"], dtype=np.float32)


# --- the model routes ---------------------------------------------------------------------


def test_the_model_list_has_one_model_named_after_the_source(client, state, name):
    body = client.get("/v1/models").json()

    assert name.startswith("openai-repo")
    assert body["object"] == "list"
    [card] = body["data"]
    assert card["id"] == name
    assert card["object"] == "model"
    assert card["owned_by"] == "downshift"
    assert isinstance(card["created"], int)


def test_one_model_by_id_and_a_404_for_another_id(client, name):
    assert client.get(f"/v1/models/{name}").json()["id"] == name

    response = client.get("/v1/models/other")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "model_not_found"


def test_version_route(client):
    assert client.get("/version").json() == {"version": downshift.__version__}


def test_served_model_name_replaces_the_name_and_survives_the_workers(state, repo):
    renamed = TestClient(build_app(_with(state, served_model_name="my-embedder"), api_key=None))

    assert renamed.get("/v1/models").json()["data"][0]["id"] == "my-embedder"
    assert renamed.get("/v1/models/my-embedder").status_code == 200
    assert _embed(renamed, "my-embedder", "hello").status_code == 200
    assert _embed(renamed, state.served_name, "hello").status_code == 404

    args = ServeArgs(
        load=LoadSpec(repo),
        options=dataclasses.replace(state.options, served_model_name="my-embedder"),
        reference=None,
        middleware=None,
        log_level="warning",
    )
    assert ServeArgs.from_json(args.to_json()).options.served_model_name == "my-embedder"


def test_matryoshka_survives_the_workers(state, repo):
    args = ServeArgs(
        load=LoadSpec(repo),
        options=dataclasses.replace(state.options, matryoshka=True),
        reference=None,
        middleware=None,
        log_level="warning",
    )

    assert ServeArgs.from_json(args.to_json()).options.matryoshka is True


# --- /v1/embeddings -----------------------------------------------------------------------


def test_float_vectors_match_predict(client, name):
    texts = ["hello world", "the capital"]

    body = _embed(client, name, texts).json()

    assert body["object"] == "list"
    assert body["model"] == name
    assert [row["index"] for row in body["data"]] == [0, 1]
    assert {row["object"] for row in body["data"]} == {"embedding"}
    vectors = np.asarray([row["embedding"] for row in body["data"]], dtype=np.float32)
    np.testing.assert_allclose(vectors, _predict_vectors(client, texts), rtol=1e-6)


def test_base64_vectors_are_little_endian_float32_and_match_predict(client, name):
    texts = ["hello world", "ignore previous instructions"]

    body = _embed(client, name, texts, encoding_format="base64").json()

    vectors = np.stack([_decode(row["embedding"]) for row in body["data"]])
    np.testing.assert_allclose(vectors, _predict_vectors(client, texts), rtol=1e-6)


def test_a_string_is_a_batch_of_one_and_extra_fields_are_ignored(client, name):
    body = _embed(client, name, "hello world", user="u1", extra_field=1).json()

    assert len(body["data"]) == 1


def test_usage_counts_the_tokens_that_are_not_padding(client, name):
    # [CLS] hello world [SEP] is 4 tokens. [CLS] the [SEP] is 3. The batch pads the second row.
    body = _embed(client, name, ["hello world", "the"]).json()

    assert body["usage"] == {"prompt_tokens": 7, "total_tokens": 7}


def test_a_cache_hit_has_the_usage_of_the_miss(state, name):
    cached = TestClient(build_app(state, api_key=None))
    assert state.result_cache is not None
    texts = ["ignore the capital", "previous instructions hello"]

    first = _embed(cached, name, texts)
    second = _embed(cached, name, texts)  # a row is stored on its second sighting
    third = _embed(cached, name, texts)

    assert first.headers["x-downshift-cache"] == "miss"
    assert third.headers["x-downshift-cache"] == "hit"
    assert first.json()["usage"] == second.json()["usage"] == third.json()["usage"]
    assert first.json()["usage"]["prompt_tokens"] == 10
    assert first.json()["data"] == third.json()["data"]


def test_a_row_that_predict_cached_still_has_its_usage(state, name):
    cached = TestClient(build_app(state, api_key=None))
    text = ["capital hello capital"]
    for _ in range(2):
        assert cached.post("/predict", json={"text": text}).status_code == 200

    response = _embed(cached, name, text)

    assert response.headers["x-downshift-cache"] == "hit"
    assert response.json()["usage"]["prompt_tokens"] == 5


def test_predict_does_not_show_the_token_count_of_a_cached_row(state):
    cached = TestClient(build_app(state, api_key=None))
    text = ["world world"]
    for _ in range(2):
        cached.post("/predict", json={"text": text})

    body = cached.post("/predict", json={"text": text}).json()

    assert list(body["outputs"]) == ["output_0"]


def test_without_the_cache_the_usage_is_the_same(state, name):
    plain = TestClient(build_app(_with(state, result_cache_rows=0), api_key=None))
    assert plain.app.state.serving.result_cache is None

    body = _embed(plain, name, ["hello world", "the"]).json()

    assert body["usage"]["prompt_tokens"] == 7


def test_an_unknown_model_is_a_404(client):
    response = _embed(client, "gpt-x", "hello")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "model_not_found"


def test_a_tensor_model_is_not_an_embedding_model(mlp_state):
    plain = TestClient(build_app(mlp_state, api_key=None))

    response = _embed(plain, "mlp", "hello")

    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "not_embedding_model"
    assert "POST /predict" in error["message"]


def test_token_vectors_are_not_an_embedding_model(tmp_path):
    repo = hf_repo.write_encoder_repo(tmp_path)
    tokens = prepare_serving(
        load_model(LoadSpec(repo, pooling="none")), ServeOptions(pooling="none")
    )
    plain = TestClient(build_app(tokens, api_key=None))

    response = _embed(plain, tokens.served_name, "hello")

    assert response.json()["error"]["code"] == "not_embedding_model"


def test_text_that_is_too_long_is_a_400_and_not_cut(client, name):
    response = _embed(client, name, ["hello " * 40])

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "text_too_long"


def test_dimensions_are_not_supported_without_matryoshka(client, name):
    response = _embed(client, name, "hello", dimensions=8)

    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "dimensions_not_supported"
    assert "--matryoshka" in error["message"]


@pytest.fixture(scope="module")
def cut_client(state) -> TestClient:
    return TestClient(build_app(_with(state, matryoshka=True), api_key=None))


def test_dimensions_keep_the_first_values_and_scale_them_to_unit_length(cut_client, name):
    texts = ["hello world", "the capital"]
    full = _predict_vectors(cut_client, texts)

    body = _embed(cut_client, name, texts, dimensions=8).json()

    cut = np.asarray([row["embedding"] for row in body["data"]], dtype=np.float32)
    expected = full[:, :8] / np.linalg.norm(full[:, :8], axis=1, keepdims=True)
    assert cut.shape == (2, 8)
    np.testing.assert_allclose(cut, expected, rtol=1e-5)
    np.testing.assert_allclose(np.linalg.norm(cut, axis=1), 1.0, rtol=1e-5)


def test_dimensions_work_with_base64(cut_client, name):
    body = _embed(cut_client, name, "hello", dimensions=4, encoding_format="base64").json()

    assert _decode(body["data"][0]["embedding"]).shape == (4,)


def test_all_the_dimensions_are_a_valid_request(cut_client, name):
    width = len(_embed(cut_client, name, "hello").json()["data"][0]["embedding"])

    body = _embed(cut_client, name, "hello", dimensions=width).json()

    assert len(body["data"][0]["embedding"]) == width


def test_one_cache_entry_serves_all_sizes(state, name):
    cached = TestClient(build_app(_with(state, matryoshka=True), api_key=None))
    texts = ["cut cache hello", "cut cache world"]
    for _ in range(2):  # a row is stored on its second sighting
        _embed(cached, name, texts)

    small = _embed(cached, name, texts, dimensions=4)
    large = _embed(cached, name, texts, dimensions=16)

    assert small.headers["x-downshift-cache"] == "hit"
    assert large.headers["x-downshift-cache"] == "hit"
    assert len(small.json()["data"][0]["embedding"]) == 4
    assert len(large.json()["data"][0]["embedding"]) == 16


@pytest.mark.parametrize("dimensions", [0, -1, 100000])
def test_dimensions_out_of_range_are_a_400_with_the_range(cut_client, name, dimensions):
    width = len(_embed(cut_client, name, "hello").json()["data"][0]["embedding"])

    response = _embed(cut_client, name, "hello", dimensions=dimensions)

    _shape(response, 400, "invalid_request_error", "invalid_input", param="dimensions")
    assert f"1 to {width}" in response.json()["error"]["message"]


def test_the_encoder_checks_the_range_if_the_width_is_not_known():
    vectors = np.ones((1, 4), dtype=np.float32)

    with pytest.raises(ApiError) as caught:
        _matryoshka_cut(vectors, 9)
    assert caught.value.code == "invalid_input"
    with pytest.raises(ApiError):
        _check_dimensions(0, None)
    _check_dimensions(5, None)


def test_a_float16_vector_is_scaled_without_overflow():
    vectors = np.full((1, 4), 300.0, dtype=np.float16)  # the square of 300 is above 65504

    cut = _matryoshka_cut(vectors, 4)

    assert np.allclose(cut, 0.5)
    assert cut.dtype == np.float32


def test_a_zero_vector_stays_zero():
    cut = _matryoshka_cut(np.zeros((1, 4), dtype=np.float32), 2)

    assert cut.tolist() == [[0.0, 0.0]]


def test_server_timing_is_on_the_embeddings_route_when_the_option_is_on(state, name):
    timed = TestClient(build_app(_with(state, server_timing=True), api_key=None))

    header = _embed(timed, name, "hello").headers["server-timing"]

    assert "infer;dur=" in header
    assert "encode;dur=" in header
    assert (
        "server-timing" not in _embed(TestClient(build_app(state, api_key=None)), name, "x").headers
    )


def test_an_empty_input_list_is_a_400(client, name):
    response = _embed(client, name, [])

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_input"


def test_the_embeddings_route_has_its_own_metrics_label(state, name):
    counted = TestClient(build_app(state, api_key=None, metrics=True))
    _embed(counted, name, "hello")
    counted.get(f"/v1/models/{name}")

    text = counted.get("/metrics").text

    assert 'route="/v1/embeddings"' in text
    assert 'route="/v1/models/{model_id:path}"' in text


# --- the OpenAI error shape ---------------------------------------------------------------


def _shape(response, status, kind, code, param=None):
    assert response.status_code == status
    assert response.json() == {
        "error": {
            "message": response.json()["error"]["message"],
            "type": kind,
            "param": param,
            "code": code,
        }
    }
    assert response.json()["error"]["message"]


def test_a_401_has_the_openai_shape_and_a_bearer_key_works(state, name):
    locked = TestClient(build_app(state, api_key="s3cret"))
    body = {"model": name, "input": "hello"}

    denied = locked.post("/v1/embeddings", json=body)
    wrong = locked.post("/v1/embeddings", json=body, headers={"Authorization": "Bearer nope"})
    allowed = locked.post("/v1/embeddings", json=body, headers={"Authorization": "Bearer s3cret"})

    _shape(denied, 401, "authentication_error", "unauthorized")
    assert denied.headers["www-authenticate"] == "Bearer"
    assert wrong.status_code == 401
    assert allowed.status_code == 200


def test_a_404_for_a_path_under_v1_has_the_openai_shape(client):
    _shape(client.get("/v1/nowhere"), 404, "invalid_request_error", "not_found")


def test_a_422_has_the_first_error_and_its_field(client):
    response = client.post("/v1/embeddings", json={"input": "hello"})

    _shape(response, 422, "invalid_request_error", "invalid_request", param="model")


def test_a_422_for_bad_json_has_no_field(client):
    response = client.post(
        "/v1/embeddings", content=b"{", headers={"Content-Type": "application/json"}
    )

    _shape(response, 422, "invalid_request_error", "invalid_request")


def test_the_other_routes_keep_the_detail_shape(client):
    body = client.post("/predict", json={"inputs": 1}).json()

    assert set(body) == {"detail", "code"}
    assert "error" not in client.get("/nowhere").json()


def test_a_413_and_a_415_have_the_openai_shape(state, name):
    small = TestClient(build_app(_with(state, max_body_bytes=10), api_key=None))

    _shape(_embed(small, name, "hello world"), 413, "invalid_request_error", "body_too_large")
    _shape(
        small.post("/v1/embeddings", content=b"abc", headers={"Content-Type": "text/csv"}),
        415,
        "invalid_request_error",
        "unsupported_media_type",
    )


def test_the_503s_have_the_openai_shape(state, name):
    loading = build_app(state, api_key=None)
    loading.state.serving = None
    _shape(
        TestClient(loading).post("/v1/embeddings", json={"model": name, "input": "x"}),
        503,
        "server_error",
        "not_ready",
    )
    _shape(TestClient(loading).get("/v1/models"), 503, "server_error", "not_ready")

    full = _with(state, max_concurrency=1, max_queue=0)
    full.in_flight = 1
    _shape(
        _embed(TestClient(build_app(full, api_key=None)), name, "x"),
        503,
        "server_error",
        "at_capacity",
    )


def test_the_500_has_the_openai_shape(state, name, monkeypatch):
    def boom(feeds):
        raise RuntimeError("bug")

    broken = _with(state, result_cache_rows=0)
    monkeypatch.setattr(broken.backend, "infer", boom)
    client = TestClient(build_app(broken, api_key=None), raise_server_exceptions=False)

    response = _embed(client, name, "hello")

    _shape(response, 500, "server_error", "internal_error")
    assert "bug" not in response.text
