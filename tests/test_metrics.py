import pytest
from prometheus_client import values

from downshift.serve.metrics import LATENCY_BUCKETS, Metrics


def _info(m: Metrics, source: str) -> None:
    m.set_info(
        version="0.5.0",
        source=source,
        backend="onnx",
        verdict="pass",
        execution="inline",
        adapter="generic",
    )


def test_two_instances_in_one_process():
    a, b = Metrics(), Metrics()
    a.reject("capacity")
    assert b"capacity" in a.render()[0]
    assert b"capacity" not in b.render()[0]


def test_buckets_include_submillisecond_and_are_sorted():
    assert 0.0005 in LATENCY_BUCKETS and 0.001 in LATENCY_BUCKETS
    assert list(LATENCY_BUCKETS) == sorted(LATENCY_BUCKETS)


@pytest.mark.parametrize("source", ["/srv/models/bert", "C:\\models\\bert", "models/bert/"])
def test_model_label_is_a_bare_name(source):
    m = Metrics()
    _info(m, source)
    text = m.render()[0].decode()
    line = next(x for x in text.splitlines() if x.startswith("downshift_info{"))
    assert 'model="bert"' in line
    assert "/" not in line and "\\" not in line


def test_render_text_format():
    m = Metrics()
    _info(m, "/m/bert")
    m.ready.set(1)
    m.boot_seconds.labels("export").set(1.5)
    m.observe_request("/predict", 200, 0.002)
    m.observe_stages({"parse": 1.0, "infer": 2.0, "bogus": 5.0})
    m.reject("auth")
    m.batch_size.observe(4)
    m.graphs_per_request.observe(2)
    m.concurrency.set(1)
    body, ctype = m.render()
    text = body.decode()
    assert ctype.startswith("text/plain")
    for series in (
        "downshift_info{",
        "downshift_ready 1.0",
        'downshift_boot_seconds{phase="export"} 1.5',
        'downshift_requests_total{route="/predict",status="200"} 1.0',
        'downshift_request_duration_seconds_count{route="/predict"} 1.0',
        'downshift_stage_duration_seconds_count{stage="parse"} 1.0',
        'downshift_rejected_total{reason="auth"} 1.0',
        "downshift_batch_size_count 1.0",
        "downshift_graphs_per_request_count 1.0",
        "downshift_in_flight",
        "downshift_queued",
        "downshift_concurrency 1.0",
    ):
        assert series in text
    assert "bogus" not in text


def test_multiprocess_render(tmp_path, monkeypatch):
    monkeypatch.setenv("PROMETHEUS_MULTIPROC_DIR", str(tmp_path))
    monkeypatch.setattr(values, "ValueClass", values.MultiProcessValue())
    m = Metrics()
    _info(m, "/m/bert")
    m.observe_request("/predict", 200, 0.01)
    m.ready.set(1)
    text = m.render()[0].decode()
    assert 'downshift_requests_total{route="/predict",status="200"} 1.0' in text
    assert 'downshift_request_duration_seconds_count{route="/predict"} 1.0' in text
    assert "downshift_ready" in text
    assert list(tmp_path.glob("*.db"))
