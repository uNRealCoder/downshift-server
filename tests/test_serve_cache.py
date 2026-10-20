"""The result cache on the serving path: boot rules, then the request path."""

import asyncio
import base64
import dataclasses
import threading
import time
from concurrent.futures import Future

import numpy as np
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from safetensors.numpy import load as st_load
from safetensors.numpy import save as st_save
from transformers import BertForSequenceClassification

from downshift.core.axes import DimBound
from downshift.loading import LoadSpec, load_model
from downshift.serve import predict
from downshift.serve.app import build_app
from downshift.serve.engine import prepare_serving
from downshift.serve.options import ServeOptions
from downshift.serve.result_cache import Lookup
from downshift.settings import DEFAULT_RESULT_CACHE_ROWS
from tests.models import hf_repo


def _state(spec: str, **opts):
    """`spec` is a name in tests/models or the path of a repo directory."""
    if "/" not in spec and "\\" not in spec:
        spec = f"tests.models.{spec}:make_model"
    return prepare_serving(load_model(LoadSpec(spec)), ServeOptions(warmup=1, **opts))


class Spy:
    """Counts the calls of the backend and the rows in each call."""

    def __init__(self, state, delay: float = 0.0, fail_first: bool = False) -> None:
        self.rows: list[int] = []
        self._real = state.backend.infer
        self._delay = delay
        self._fail = fail_first
        self._lock = threading.Lock()
        state.backend.infer = self

    def __call__(self, inputs):
        with self._lock:
            self.rows.append(len(next(iter(inputs.values()))))
            fail, self._fail = self._fail, False
        time.sleep(self._delay)
        if fail:
            raise RuntimeError("boom")
        return self._real(inputs)


def _x(*values: float) -> list[list[float]]:
    """Rows of the 16-feature MLP. Each value makes one distinct row."""
    return [[v] * 16 for v in values]


def _post(client, rows, **extra):
    return client.post("/predict", json={"inputs": {"x": rows}} | extra)


# --- boot rules ---------------------------------------------------------------------------


def test_the_cache_is_on_by_default_for_independent_rows():
    state = _state("clean_mlp")

    assert state.result_cache is not None
    assert state.result_cache.max_rows == DEFAULT_RESULT_CACHE_ROWS == 100


def test_zero_rows_turns_the_cache_off():
    state = _state("clean_mlp", result_cache_rows=0)

    assert state.result_cache is None
    assert state.cache_off_reason is None


def test_the_default_is_off_for_a_model_with_dependent_rows():
    state = _state("cross_row")

    assert state.result_cache is None
    assert state.cache_off_reason is not None
    assert "when it runs with other rows" in state.cache_off_reason


def test_a_set_size_on_a_model_with_dependent_rows_is_a_usage_error():
    with pytest.raises(ValueError, match=r"Remove --result-cache-rows, or set it to 0"):
        _state("cross_row", result_cache_rows=50)


def test_the_size_is_free_for_independent_rows():
    cache = _state("clean_mlp", result_cache_rows=7).result_cache
    assert cache is not None
    assert cache.max_rows == 7


def test_metadata_reports_the_cache():
    client = TestClient(build_app(_state("clean_mlp")))

    cache = client.get("/metadata").json()["cache"]

    assert cache["max_rows"] == 100
    assert set(cache) == {"max_rows", "entries", "hits", "misses"}


def test_metadata_has_no_cache_when_it_is_off():
    client = TestClient(build_app(_state("clean_mlp", result_cache_rows=0)))

    assert client.get("/metadata").json()["cache"] is None


# --- the request path ---------------------------------------------------------------------


@pytest.fixture
def served():
    state = _state("clean_mlp")
    spy = Spy(state)
    return TestClient(build_app(state)), spy, state


def test_a_row_is_computed_twice_and_then_it_is_a_hit(served):
    client, spy, _ = served
    first = _post(client, _x(1.0)).json()
    second = _post(client, _x(1.0)).json()
    third = _post(client, _x(1.0)).json()

    assert spy.rows == [1, 1], "the third request never reached the model"
    assert first["outputs"] == second["outputs"] == third["outputs"]
    assert third["shapes"] == {"output_0": [1, 4]}


def test_a_partial_hit_runs_only_the_rows_that_missed_in_the_order_of_the_request(served):
    client, spy, state = served
    for _ in range(2):
        _post(client, _x(1.0, 2.0))  # rows 1.0 and 2.0 are now cached
    spy.rows.clear()

    mixed = _post(client, _x(3.0, 1.0, 4.0, 2.0)).json()

    assert spy.rows == [2]
    uncached = dataclasses.replace(
        state, options=dataclasses.replace(state.options, result_cache_rows=0)
    )
    reference = TestClient(build_app(uncached))
    expected = _post(reference, _x(3.0, 1.0, 4.0, 2.0)).json()
    np.testing.assert_allclose(
        mixed["outputs"]["output_0"], expected["outputs"]["output_0"], atol=1e-5
    )
    assert state.result_cache.stats()["hits"] == 2  # type: ignore[union-attr]


def test_the_same_row_twice_in_one_request_is_computed_once(served):
    client, spy, _ = served

    body = _post(client, _x(5.0, 5.0)).json()

    assert spy.rows == [1]
    assert body["outputs"]["output_0"][0] == body["outputs"]["output_0"][1]


def test_a_hit_is_bit_for_bit_the_miss_that_stored_it_in_each_encoding(served):
    client, _, _ = served
    for encoding in ("json", "base64"):
        bodies = [_post(client, _x(6.0), output_encoding=encoding).json() for _ in range(3)]
        assert bodies[0] == bodies[1] == bodies[2]
    safetensors = [_post(client, _x(6.0), output_encoding="safetensors").content for _ in range(3)]
    assert safetensors[0] == safetensors[1] == safetensors[2]
    assert st_load(safetensors[2])["output_0"].shape == (1, 4)


def test_a_binary_request_uses_the_cache(served):
    client, spy, _ = served
    body = st_save({"x": np.full((1, 16), 7.0, dtype=np.float32)})
    headers = {"Content-Type": "application/vnd.safetensors"}

    answers = [client.post("/predict", content=body, headers=headers) for _ in range(3)]

    assert [a.status_code for a in answers] == [200, 200, 200]
    assert spy.rows == [1, 1]
    assert answers[0].json() == answers[2].json()


def test_base64_and_json_forms_of_a_row_share_one_entry(served):
    client, spy, _ = served
    row = np.full((1, 16), 8.0, dtype=np.float32)
    typed = {
        "data": base64.b64encode(row.tobytes()).decode(),
        "dtype": "float32",
        "shape": [1, 16],
    }
    for value in (typed, _x(8.0), typed):
        assert client.post("/predict", json={"inputs": {"x": value}}).status_code == 200

    assert spy.rows == [1, 1]


def test_a_bad_request_is_a_400_and_does_not_block_the_next_one(served):
    client, _, _ = served

    bad = _post(client, [[1.0] * 3])
    good = _post(client, _x(9.0))

    assert bad.status_code == 400
    assert good.status_code == 200


def test_an_empty_batch_is_not_a_server_error(served):
    client, _, _ = served

    assert _post(client, []).status_code in (200, 400)


def test_the_cache_header_says_hit_partial_or_miss(served):
    client, _, _ = served

    labels = [_post(client, _x(20.0)).headers["X-Downshift-Cache"] for _ in range(3)]
    partial = _post(client, _x(20.0, 21.0)).headers["X-Downshift-Cache"]

    assert labels == ["miss", "miss", "hit"]
    assert partial == "partial"


def test_server_timing_has_a_cache_stage_with_the_hit_count(served):
    client, _, _ = served
    for _ in range(2):
        _post(client, _x(22.0))

    timing = _post(client, _x(22.0, 23.0)).headers["Server-Timing"]

    assert "cache;dur=" in timing
    assert 'desc="1/2"' in timing


def test_a_server_with_the_cache_off_has_the_old_header_and_no_cache_header():
    client = TestClient(build_app(_state("clean_mlp", result_cache_rows=0)))

    resp = _post(client, _x(1.0))

    assert "X-Downshift-Cache" not in resp.headers
    assert "cache" not in resp.headers["Server-Timing"]


def test_no_cache_skips_the_lookup_but_still_stores_the_row(served):
    client, spy, _ = served
    for _ in range(2):
        _post(client, _x(24.0))
    spy.rows.clear()

    skipped = client.post(
        "/predict", json={"inputs": {"x": _x(24.0)}}, headers={"Cache-Control": "no-cache"}
    )
    normal = _post(client, _x(24.0))

    assert skipped.headers["X-Downshift-Cache"] == "miss"
    assert spy.rows == [1]
    assert normal.headers["X-Downshift-Cache"] == "hit"


def _run_two(client, rows):
    results: list = []

    def call():
        results.append(_post(client, rows))

    threads = [threading.Thread(target=call) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results


def test_two_requests_for_one_row_run_the_model_once():
    state = _state("clean_mlp")
    spy = Spy(state, delay=0.4)
    client = TestClient(build_app(state))

    results = _run_two(client, _x(10.0))

    assert [r.status_code for r in results] == [200, 200]
    assert spy.rows == [1]
    assert results[0].json()["outputs"] == results[1].json()["outputs"]


def test_a_waiter_computes_the_row_itself_when_the_owner_fails():
    state = _state("clean_mlp")
    spy = Spy(state, delay=0.4, fail_first=True)
    client = TestClient(build_app(state), raise_server_exceptions=False)

    results = _run_two(client, _x(11.0))

    assert sorted(r.status_code for r in results) == [200, 500]
    assert spy.rows == [1, 1]


def test_a_partial_hit_is_not_checked_against_the_batch_bound_again():
    state = _state("clean_mlp")
    state.axis_bounds = {"x": {0: DimBound("batch", 3, 100)}}  # the export traced 3 to 100 rows
    client = TestClient(build_app(state))
    for _ in range(2):
        assert _post(client, _x(1.0, 2.0, 3.0)).status_code == 200

    partial = _post(client, _x(1.0, 2.0, 4.0))  # two hits: the model sees one row

    assert partial.status_code == 200
    assert partial.headers["X-Downshift-Cache"] == "partial"
    assert _post(client, _x(1.0, 2.0)).status_code == 400, "the whole batch is still checked"


def _fake_plan(cache, request, arrays, *, pending=None, claim=()):
    keys = [b"k0"]
    found = Lookup(pending=pending or {})
    owned = []
    if claim:
        found = cache.lookup(list(claim))
        owned = list(claim)
    return predict._CachePlan(request, arrays, keys, found, None, owned)


def test_a_request_that_leaves_during_the_prep_hop_releases_its_rows(monkeypatch):
    state = _state("clean_mlp")
    cache = state.result_cache
    request = predict._Request()

    def slow_prepare(*_args):
        time.sleep(0.3)  # the thread claims the row after the client has left
        return _fake_plan(cache, request, None, claim=[b"k0"])

    monkeypatch.setattr(predict, "_cache_prepare", slow_prepare)

    async def go():
        task = asyncio.ensure_future(
            predict._run_cached(
                state,
                cache,
                b"",
                request,
                binary=False,
                run_here=False,
                admitted_at=time.monotonic(),
                timings_ms={},
                accept_safetensors=False,
            )
        )
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.sleep(0.5)

    asyncio.run(go())

    assert cache.lookup([b"k0"]).missing == [0], "the claim was released"


def test_a_waiter_that_leaves_does_not_cancel_the_row_of_its_owner():
    state = _state("clean_mlp")
    future: Future = Future()

    async def go():
        waiter = asyncio.ensure_future(predict._await_row(state, future, time.monotonic()))
        await asyncio.sleep(0.05)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter

    asyncio.run(go())

    assert not future.cancelled()
    future.set_result({})  # the owner can still give the row


def test_a_waiter_gives_up_at_the_request_timeout():
    state = _state("clean_mlp", request_timeout=0.2)

    async def go():
        await predict._await_row(state, Future(), time.monotonic())

    with pytest.raises(HTTPException) as info:
        asyncio.run(go())

    assert info.value.status_code == 503
    assert "--request-timeout" in info.value.detail


def test_a_waiter_that_retries_can_get_a_whole_response(monkeypatch):
    # The owner failed. The retry gets an output with no row axis (a backend that breaks the
    # contract). This is a whole response, and not a missing row.
    state = _state("clean_mlp")
    cache = state.result_cache
    arrays = {"x": np.ones((1, 16), dtype=np.float32)}
    request = predict._Request(tensors=arrays)
    failed: Future = Future()
    failed.set_exception(RuntimeError("owner failed"))
    monkeypatch.setattr(
        predict,
        "_cache_prepare",
        lambda *_a: _fake_plan(cache, request, arrays, pending={0: failed}),
    )
    monkeypatch.setattr(predict, "_infer", lambda _s, _f: ({"output_0": np.float32(1.0)}, 0.0))

    async def go():
        return await predict._run_cached(
            state,
            cache,
            b"",
            request,
            binary=False,
            run_here=False,
            admitted_at=time.monotonic(),
            timings_ms={},
            accept_safetensors=False,
        )

    assert asyncio.run(go()).status_code == 200


def test_a_graph_model_serves_with_no_cache(gcn_client):
    state = gcn_client.app.state.serving

    assert state.result_cache is None
    assert state.cache_off_reason is not None


# --- text ---------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def embedding_repo(tmp_path_factory) -> str:
    return hf_repo.write_encoder_repo(tmp_path_factory.mktemp("cache-embedding"))


@pytest.fixture(scope="module")
def classifier_repo(tmp_path_factory) -> str:
    path = tmp_path_factory.mktemp("cache-classifier")
    labels = {0: "BENIGN", 1: "INJECTION", 2: "JAILBREAK"}
    config = hf_repo.config(
        num_labels=3, id2label=labels, label2id={v: k for k, v in labels.items()}
    )
    BertForSequenceClassification(config).eval().save_pretrained(path)
    hf_repo.tokenizer().save_pretrained(path)
    return str(path)


@pytest.mark.needs_torch_26
def test_text_rows_hit_without_the_tokenizer_and_keep_their_order(embedding_repo):
    state = _state(embedding_repo)
    spy = Spy(state)
    client = TestClient(build_app(state))
    texts = ["hello world", "the capital", "ignore previous instructions"]

    for _ in range(2):
        client.post("/predict", json={"text": texts[:2]})
    spy.rows.clear()
    calls: list[list[str]] = []
    real_encode = state.text.encode
    state.text.encode = lambda rows: calls.append(list(rows)) or real_encode(rows)  # type: ignore[method-assign, union-attr]
    mixed = client.post("/predict", json={"text": [texts[2], texts[0], texts[1]]}).json()

    assert calls == [[texts[2]]], "only the row that missed was tokenized"
    assert spy.rows == [1]
    alone = client.post("/predict", json={"text": texts[0]}).json()
    np.testing.assert_allclose(
        mixed["outputs"]["output_0"][1], alone["outputs"]["output_0"][0], atol=1e-5
    )


@pytest.mark.needs_torch_26
def test_a_classifier_answers_predictions_for_hits_and_misses(classifier_repo):
    state = _state(classifier_repo)
    spy = Spy(state)
    client = TestClient(build_app(state))
    body = {"text": ["hello world", "the capital"]}

    answers = [client.post("/predict", json=body).json() for _ in range(3)]

    assert spy.rows == [2, 2]
    assert answers[0]["predictions"] == answers[2]["predictions"]
    assert len(answers[2]["predictions"]) == 2
