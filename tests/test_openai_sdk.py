"""The `openai` Python SDK against the app, with only `base_url` and `api_key` changed. The SDK
client uses an ASGI transport, so no port is open."""

import asyncio
import dataclasses

import numpy as np
import pytest

openai = pytest.importorskip("openai")
httpx2 = pytest.importorskip("httpx2")
pytest.importorskip("transformers")
pytest.importorskip("tokenizers")

from fastapi.testclient import TestClient

from downshift.loading import LoadSpec, load_model
from downshift.serve.app import build_app
from downshift.serve.engine import prepare_serving
from tests.models import hf_repo

TEXTS = ["hello world", "the capital"]


@pytest.fixture(scope="module")
def state(tmp_path_factory):
    repo = hf_repo.write_encoder_repo(tmp_path_factory.mktemp("openai-sdk-repo"))
    loaded = prepare_serving(load_model(LoadSpec(repo)))
    options = dataclasses.replace(loaded.options, matryoshka=True)
    return dataclasses.replace(loaded, options=options)


@pytest.fixture(scope="module")
def app(state):
    return build_app(state, api_key=None)


def _call(app, make):
    """Run `make(client)` with an SDK client for `app`."""

    async def run():
        transport = httpx2.ASGITransport(app=app)
        http = httpx2.AsyncClient(transport=transport, base_url="http://testserver/v1")
        client = openai.AsyncOpenAI(base_url="http://testserver/v1", api_key="k", http_client=http)
        async with client:
            return await make(client)

    return asyncio.run(run())


def _predict_vectors(app, texts):
    body = TestClient(app).post("/predict", json={"text": texts}).json()
    return np.asarray(body["outputs"]["output_0"], dtype=np.float32)


def test_the_sdk_default_asks_for_base64_and_gets_vectors_like_predict(app, state):
    response = _call(app, lambda c: c.embeddings.create(model=state.served_name, input=TEXTS))

    vectors = np.asarray([row.embedding for row in response.data], dtype=np.float32)
    np.testing.assert_allclose(vectors, _predict_vectors(app, TEXTS), rtol=1e-5, atol=1e-6)
    assert response.model == state.served_name
    assert response.usage.prompt_tokens == 8


def test_float_format_gives_the_same_vectors(app, state):
    response = _call(
        app,
        lambda c: c.embeddings.create(
            model=state.served_name, input=TEXTS, encoding_format="float"
        ),
    )

    vectors = np.asarray([row.embedding for row in response.data], dtype=np.float32)
    np.testing.assert_allclose(vectors, _predict_vectors(app, TEXTS), rtol=1e-5, atol=1e-6)


def test_dimensions_through_the_sdk(app, state):
    response = _call(
        app,
        lambda c: c.embeddings.create(model=state.served_name, input=TEXTS, dimensions=8),
    )

    vectors = np.asarray([row.embedding for row in response.data], dtype=np.float32)
    full = _predict_vectors(app, TEXTS)[:, :8]
    expected = full / np.linalg.norm(full, axis=1, keepdims=True)
    np.testing.assert_allclose(vectors, expected, rtol=1e-4, atol=1e-6)


def test_a_wrong_model_name_raises_not_found_with_our_code(app):
    with pytest.raises(openai.NotFoundError) as caught:
        _call(app, lambda c: c.embeddings.create(model="gpt-x", input="hello"))

    assert caught.value.code == "model_not_found"
    assert caught.value.body["code"] == "model_not_found"
