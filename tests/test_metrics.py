"""GET /metrics: the Prometheus text, the counters, and the --metrics option."""

import time

import pytest
from fastapi.testclient import TestClient
from prometheus_client.parser import text_string_to_metric_families

import downshift
from downshift import settings
from downshift.cli import render
from downshift.loading import LoadSpec, load_model
from downshift.serve.app import RequestIdMiddleware, build_app
from downshift.serve.engine import prepare_serving
from downshift.serve.errors import error_response
from downshift.serve.metrics import BUCKETS, CONTENT_TYPE, Histogram, Metrics
from downshift.serve.metrics import render as render_text
from downshift.serve.options import ServeOptions
from tests.conftest import with_options
from tests.test_cli import _serve_captured
from tests.test_render import _serving_state, _verdict

GOOD = {"inputs": {"x": [[0.0] * 16]}}


def _state(**opts):
    loaded = load_model(LoadSpec("tests.models.clean_mlp:make_model"))
    return prepare_serving(loaded, ServeOptions(warmup=1, **opts))


@pytest.fixture(scope="module")
def plain_state():
    return _state(result_cache_rows=0)


def _client(state, **kwargs):
    return TestClient(build_app(state, api_key=kwargs.pop("api_key", None), metrics=True), **kwargs)


def _samples(client, **kwargs):
    """{(sample name, sorted labels): value} of GET /metrics."""
    response = client.get("/metrics", **kwargs)
    assert response.status_code == 200
    assert response.headers["content-type"] == CONTENT_TYPE
    found = {}
    for family in text_string_to_metric_families(response.text):
        for sample in family.samples:
            found[(sample.name, tuple(sorted(sample.labels.items())))] = sample.value
    return found


def _value(samples, name, **labels):
    labels = {"worker": "0", **labels}
    return samples[(name, tuple(sorted(labels.items())))]


def test_the_text_parses_and_has_every_name(plain_state):
    client = _client(plain_state)
    client.post("/predict", json=GOOD)

    response = client.get("/metrics")
    families = {f.name: f for f in text_string_to_metric_families(response.text)}

    assert set(families) == {
        "downshift_requests",
        "downshift_request_seconds",
        "downshift_stage_seconds",
        "downshift_rejected",
        "downshift_cache_requests",
        "downshift_cache_rows",
        "downshift_ready",
        "downshift_inflight",
        "downshift_queue_depth",
        "downshift_queue_limit",
        "downshift_model_info",
        "downshift_boot_seconds",
    }
    assert families["downshift_request_seconds"].type == "histogram"
    assert all("worker" in s.labels for f in families.values() for s in f.samples)


def test_a_predict_is_counted_by_route_and_code(plain_state):
    client = _client(plain_state)
    client.post("/predict", json=GOOD)
    client.post("/predict", json={"inputs": {"x": [[0.0] * 3]}})
    client.get("/nowhere/12345")

    samples = _samples(client)

    assert _value(samples, "downshift_requests_total", route="/predict", code="200") == 1
    assert _value(samples, "downshift_requests_total", route="/predict", code="400") == 1
    assert _value(samples, "downshift_requests_total", route="other", code="404") == 1
    assert not any("12345" in str(key) for key in samples)
    assert _value(samples, "downshift_request_seconds_count", route="/predict") == 2
    inf = _value(samples, "downshift_request_seconds_bucket", route="/predict", le="+Inf")
    assert inf == 2


def test_the_stage_histograms_count_a_predict(plain_state):
    client = _client(plain_state)
    client.post("/predict", json=GOOD)

    samples = _samples(client)

    for stage in ("queue", "prep", "infer", "encode"):
        assert _value(samples, "downshift_stage_seconds_count", stage=stage) == 1


def test_the_gauges_come_from_the_state(plain_state):
    state = with_options(plain_state, max_queue=7)
    state.ready = True
    samples = _samples(_client(state))

    assert _value(samples, "downshift_ready") == 1
    assert _value(samples, "downshift_inflight") == 0
    assert _value(samples, "downshift_queue_depth") == 0
    assert _value(samples, "downshift_queue_limit") == 7
    info = [k for k in samples if k[0] == "downshift_model_info"]
    assert len(info) == 1
    labels = dict(info[0][1])
    assert labels["downshift_version"] == downshift.__version__
    assert labels["backend"] == state.backend.name
    assert labels["verdict"] == "CLEAN"
    assert labels["family"] == state.verdict.model_family
    assert any(k[0] == "downshift_boot_seconds" for k in samples)


def test_the_cache_rows_agree_with_the_cache_headers():
    state = _state()
    client = _client(state)
    rows = lambda *values: {"inputs": {"x": [[v] * 16 for v in values]}}  # noqa: E731

    # A row is stored at its second sighting.
    results = [
        client.post("/predict", json=rows(*values)).headers[settings.CACHE_HEADER]
        for values in ((1.0, 2.0), (1.0, 2.0), (1.0, 2.0), (1.0, 3.0))
    ]
    samples = _samples(client)

    assert results == ["miss", "miss", "hit", "partial"]
    for result in ("hit", "partial", "miss"):
        expected = results.count(result)
        assert _value(samples, "downshift_cache_requests_total", result=result) == expected
    assert _value(samples, "downshift_cache_rows_total", result="hit") == 3
    assert _value(samples, "downshift_cache_rows_total", result="miss") == 5


def test_a_full_queue_counts_a_capacity_rejection(plain_state):
    full = with_options(plain_state, max_queue=0)
    full.in_flight = full.options.max_concurrency
    client = _client(full)

    response = client.post("/predict", json=GOOD)
    samples = _samples(client)

    assert response.status_code == 503
    assert _value(samples, "downshift_rejected_total", reason="capacity") == 1
    assert _value(samples, "downshift_rejected_total", reason="timeout") == 0
    assert _value(samples, "downshift_requests_total", route="/predict", code="503") == 1
    assert _value(samples, "downshift_queue_depth") == 0


def test_a_slow_request_counts_a_timeout_rejection(plain_state, monkeypatch):
    slow = with_options(plain_state, request_timeout=0.05)
    monkeypatch.setattr(slow.backend, "infer", lambda inputs: time.sleep(0.5))
    client = _client(slow)

    assert client.post("/predict", json=GOOD).status_code == 503

    assert _value(_samples(client), "downshift_rejected_total", reason="timeout") == 1


def test_a_503_that_a_middleware_builds_is_counted():
    async def refuse(scope, receive, send):
        response = error_response(scope, 503, "at_capacity", "full")
        await response(scope, receive, send)

    metrics = Metrics()
    client = TestClient(RequestIdMiddleware(refuse, metrics=metrics))
    assert client.get("/x").status_code == 503
    assert metrics.rejected["capacity"] == 1


def test_metrics_answers_while_the_model_loads(plain_state):
    app = build_app(plain_state, api_key=None, metrics=True)
    app.state.serving = None
    client = TestClient(app)

    assert client.post("/predict", json=GOOD).status_code == 503
    samples = _samples(client)

    assert _value(samples, "downshift_ready") == 0
    assert _value(samples, "downshift_rejected_total", reason="not_ready") == 1
    assert not any(k[0] == "downshift_model_info" for k in samples)


def test_metrics_is_a_404_and_nothing_is_counted_when_it_is_off(plain_state):
    app = build_app(plain_state, api_key=None)
    client = TestClient(app)

    assert client.get("/metrics").status_code == 404
    assert "metrics" not in client.get("/").json()["routes"]
    assert client.post("/predict", json=GOOD).status_code == 200


def test_the_route_shows_in_the_index_when_it_is_on(plain_state):
    routes = _client(plain_state).get("/").json()["routes"]

    assert routes["metrics_text"] == "GET /metrics"


def test_the_api_key_applies_to_metrics(plain_state):
    client = _client(plain_state, api_key="s3cret")

    assert client.get("/metrics").status_code == 401
    assert client.get("/metrics", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert _samples(client, headers={"Authorization": "Bearer s3cret"})


def test_the_worker_label_is_zero(plain_state):
    assert 'worker="0"' in _client(plain_state).get("/metrics").text


def test_a_label_value_is_escaped():
    metrics = Metrics(worker='a"b\\c')

    assert 'worker="a\\"b\\\\c"' in render_text(metrics, None)


def test_a_histogram_puts_a_value_in_the_first_bucket_that_holds_it():
    hist = Histogram()
    for seconds in (0.0004, 0.0005, 0.0006, 99):
        hist.observe(seconds)

    assert hist.counts[0] == 2
    assert hist.counts[1] == 1
    assert hist.counts[len(BUCKETS)] == 1
    assert hist.total == pytest.approx(99.0015)


def test_an_unknown_error_code_is_not_a_rejection():
    metrics = Metrics()
    metrics.record_rejected("invalid_input")
    metrics.record_cache_result(None)
    metrics.record_cache_result("other")

    assert sum(metrics.rejected.values()) == 0
    assert sum(metrics.cache_requests.values()) == 0


def test_the_option_reaches_the_app_and_the_banner(monkeypatch, caplog):
    _, captured = _serve_captured(monkeypatch, "--metrics")
    assert captured["app"].state.serving.options.metrics is True
    assert TestClient(captured["app"]).get("/metrics").status_code == 200

    _, captured = _serve_captured(monkeypatch)
    assert captured["app"].state.serving.options.metrics is False
    assert TestClient(captured["app"]).get("/metrics").status_code == 404


def test_the_env_variable_turns_the_option_on(monkeypatch):
    import importlib

    monkeypatch.setenv("DOWNSHIFT_METRICS", "1")
    assert importlib.reload(settings).METRICS is True
    monkeypatch.delenv("DOWNSHIFT_METRICS")
    assert importlib.reload(settings).METRICS is False


@pytest.mark.parametrize("on", [True, False])
def test_the_banner_shows_the_metrics_row_only_when_it_is_on(caplog, on):
    state = _serving_state(_verdict(), options=ServeOptions(metrics=on))
    render.print_banner(state, "127.0.0.1", 8000)

    assert ("GET /metrics" in caplog.text) is on
