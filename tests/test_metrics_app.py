"""/metrics on the app: what it counts, what it never labels, and how it is gated."""

import logging
import os
import subprocess
import sys
import threading
import uuid

import pytest
from fastapi.testclient import TestClient
from prometheus_client.parser import text_string_to_metric_families

from downshift.cli import main
from downshift.serve.app import build_app
from tests.conftest import subprocess_env

MLP_INPUT = {"inputs": {"x": [[0.5] * 16]}}
KEY = "s3cret"


def samples(client: TestClient, **kw) -> dict[str, dict[tuple, float]]:
    """{sample name: {sorted label items: value}} from one scrape."""
    resp = client.get("/metrics", **kw)
    assert resp.status_code == 200, resp.text
    out: dict[str, dict[tuple, float]] = {}
    for family in text_string_to_metric_families(resp.text):
        for s in family.samples:
            out.setdefault(s.name, {})[tuple(sorted(s.labels.items()))] = s.value
    return out


def value(found: dict, name: str, **labels: str) -> float:
    return found[name][tuple(sorted(labels.items()))]


def test_counters_match_the_access_log(mlp_state, serve_fixture, caplog):
    small = serve_fixture("clean_mlp", max_body_bytes=200)
    client = TestClient(build_app(small, api_key=None))
    big = {"inputs": {"x": [[0.5] * 16] * 50}}
    with caplog.at_level(logging.DEBUG, logger="downshift.access"):
        for _ in range(3):
            assert client.post("/predict", json=MLP_INPUT).status_code == 200
        assert client.post("/predict", json={"inputs": {}}).status_code == 400
        assert client.post("/predict", json=big).status_code == 413
        assert client.get("/nope").status_code == 404
        assert client.get("/health").status_code == 200
        found = samples(client)

    logged: dict[int, int] = {}
    for record in caplog.records:
        if record.name == "downshift.access" and record.path != "/metrics":
            logged[record.status] = logged.get(record.status, 0) + 1
    counted: dict[int, int] = {}
    for key, count in found["downshift_requests_total"].items():
        labels = dict(key)
        if labels["route"] != "/metrics":
            counted[int(labels["status"])] = counted.get(int(labels["status"]), 0) + int(count)
    assert counted == logged == {200: 4, 400: 1, 413: 1, 404: 1}
    assert value(found, "downshift_rejected_total", reason="body_too_large") == 1.0
    assert value(found, "downshift_requests_total", route="unmatched", status="404") == 1.0


def test_random_404_paths_add_one_series(mlp_state):
    client = TestClient(build_app(mlp_state, api_key=None))
    client.get("/health")
    samples(client)  # a scrape is counted only after it answers
    before = len(samples(client)["downshift_requests_total"])
    for _ in range(1000):
        client.get(f"/{uuid.uuid4().hex}/{uuid.uuid4().hex}")
    after = samples(client)["downshift_requests_total"]
    assert len(after) == before + 1
    assert after[(("route", "unmatched"), ("status", "404"))] == 1000.0
    assert not any("route" in dict(k) and len(dict(k)["route"]) > 20 for k in after)


def test_metrics_needs_the_api_key_when_one_is_set(mlp_state):
    client = TestClient(build_app(mlp_state, api_key=KEY))
    assert client.get("/metrics").status_code == 401
    assert client.get("/metrics", headers={"Authorization": "Bearer nope"}).status_code == 401
    assert client.get("/health").status_code == 200
    found = samples(client, headers={"Authorization": f"Bearer {KEY}"})
    assert value(found, "downshift_rejected_total", reason="auth") == 2.0


def test_metrics_answers_during_boot_with_ready_zero(mlp_state):
    release = threading.Event()

    def loader():
        assert release.wait(timeout=10)
        return mlp_state

    app = build_app(loader=loader, api_key=None)
    with TestClient(app) as client:
        assert client.post("/predict", json=MLP_INPUT).status_code == 503
        found = samples(client)
        assert found["downshift_ready"][()] == 0.0
        assert value(found, "downshift_rejected_total", reason="not_ready") == 1.0
        assert "downshift_info" not in found or not found["downshift_info"]

        release.set()
        app.state.loader_thread.join(timeout=10)
        found = samples(client)
        assert found["downshift_ready"][()] == 1.0
        assert found["downshift_concurrency"][()] == float(mlp_state.options.max_concurrency)


def test_info_labels(mlp_state):
    import downshift

    found = samples(TestClient(build_app(mlp_state, api_key=None)))
    (labels,) = found["downshift_info"]
    labels = dict(labels)
    assert labels["version"] == downshift.__version__
    assert labels["backend"] == mlp_state.backend.name
    assert labels["verdict"] == mlp_state.verdict.status
    assert labels["execution"] == mlp_state.execution.value
    assert labels["adapter"] == mlp_state.verdict.model_family
    assert "/" not in labels["model"] and "\\" not in labels["model"]


def test_stage_and_batch_histograms(mlp_state):
    client = TestClient(build_app(mlp_state, api_key=None))
    x = [[0.5] * 16] * 3
    assert client.post("/predict", json={"inputs": {"x": x}}).status_code == 200
    found = samples(client)
    for stage in ("parse", "prep", "infer", "encode"):
        assert value(found, "downshift_stage_duration_seconds_count", stage=stage) == 1.0
    assert found["downshift_batch_size_sum"][()] == 3.0
    assert found["downshift_in_flight"][()] == 0.0
    assert found["downshift_queued"][()] == 0.0


def test_graphs_per_request(gcn_client):
    from tests.test_graph_batching import _graphs

    app_client = TestClient(build_app(gcn_client.app.state.serving, api_key=None))
    assert app_client.post("/predict/graph", json={"graphs": _graphs()}).status_code == 200
    found = samples(app_client)
    assert found["downshift_graphs_per_request_sum"][()] == 3.0


def test_worker_metrics_dir_is_private_and_overrides_the_inherited_one(tmp_path, monkeypatch):
    monkeypatch.setenv("PROMETHEUS_MULTIPROC_DIR", str(tmp_path))
    made = main._worker_metrics_dir()
    try:
        assert made is not None and made != str(tmp_path)
        assert os.environ["PROMETHEUS_MULTIPROC_DIR"] == made
        if os.name == "posix":
            assert os.stat(made).st_mode & 0o777 == 0o700
    finally:
        os.environ.pop("PROMETHEUS_MULTIPROC_DIR", None)
        if made:
            os.rmdir(made)


_WORKER = (
    "from downshift.serve.metrics import Metrics;"
    "m = Metrics(); m.ready.set(1); m.observe_request('/predict', 200, 0.01);"
    "m.observe_stages({'infer': 5.0}); m.batch_size.observe(2)"
)


def test_two_worker_processes_aggregate_through_the_shared_dir(tmp_path, monkeypatch):
    """Each child is a separate interpreter with the env set before import, as a --workers
    worker is; a third process scrapes and sees both."""
    env = subprocess_env()
    env["PROMETHEUS_MULTIPROC_DIR"] = str(tmp_path)
    for _ in range(2):
        subprocess.run([sys.executable, "-c", _WORKER], env=env, check=True, timeout=60)

    monkeypatch.setenv("PROMETHEUS_MULTIPROC_DIR", str(tmp_path))
    from downshift.serve.metrics import Metrics

    text = Metrics().render()[0].decode()
    for _ in range(2):  # repeated scrapes keep reporting both
        assert 'downshift_requests_total{route="/predict",status="200"} 2.0' in text
        assert 'downshift_stage_duration_seconds_count{stage="infer"} 2.0' in text
        assert "downshift_batch_size_count 2.0" in text
        assert text.count("downshift_ready{pid=") == 2
        text = Metrics().render()[0].decode()


@pytest.mark.parametrize("workers", [1])
def test_single_process_serve_drops_an_inherited_dir(monkeypatch, tmp_path, workers):
    from tests.test_cli import _serve_captured

    monkeypatch.setenv("PROMETHEUS_MULTIPROC_DIR", str(tmp_path))
    _serve_captured(monkeypatch)
    assert "PROMETHEUS_MULTIPROC_DIR" not in os.environ
