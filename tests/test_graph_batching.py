import numpy as np
import pytest
from fastapi.testclient import TestClient
from safetensors.numpy import load as st_load
from safetensors.numpy import save as st_save

from downshift.loading import LoadSpec, load_model
from downshift.serve import graphs as graph_mod
from downshift.serve.app import build_app
from downshift.serve.engine import prepare_serving
from downshift.serve.options import ServeOptions

ST = "application/vnd.safetensors"
KINDS = "tests.models.gnn_output_kinds"


def _graphs(sizes=((5, 7), (4, 3), (6, 9)), seed=0) -> list[dict[str, list]]:
    rng = np.random.default_rng(seed)
    out = []
    for n, e in sizes:
        out.append(
            {
                "x": rng.standard_normal((n, 8)).astype(np.float32).tolist(),
                "edge_index": rng.integers(0, n, (2, e)).tolist(),
            }
        )
    return out


def _tolerances(state) -> tuple[float, float]:
    numerics = state.verdict.numerics
    return (numerics.tolerance_abs, numerics.tolerance_rel) if numerics else (1e-4, 1e-4)


def _kinds_client(model: str, **opts) -> TestClient:
    loaded = load_model(LoadSpec(f"{KINDS}:{model}", inputs=f"{KINDS}:make_inputs"))
    return TestClient(build_app(prepare_serving(loaded, ServeOptions(warmup=1, **opts))))


@pytest.fixture(scope="module")
def edge_client() -> TestClient:
    return _kinds_client("make_edge_model")


@pytest.fixture(scope="module")
def fixed_client() -> TestClient:
    return _kinds_client("make_fixed_model")


@pytest.fixture(scope="module")
def torch_gcn_client(serve_fixture) -> TestClient:
    return TestClient(build_app(serve_fixture("gnn_gcn", backend="torch")))


def test_a_failed_export_still_classifies_outputs_for_the_torch_fallback(
    serve_fixture, monkeypatch
):
    # torch 2.5 can't export this GCN; the fallback must batch like --backend torch does.
    from downshift.core import verdict as verdict_mod
    from downshift.core.capture import CaptureResult

    failed = CaptureResult(success=False, capture_strategy=None, exception=RuntimeError("no"))
    monkeypatch.setattr(verdict_mod, "capture", lambda *a, **k: failed)
    state = serve_fixture("gnn_gcn")

    assert state.verdict.status == "FAILED"
    assert state.backend.name == "torch"
    assert state.verdict.output_axes == ["node"]
    resp = TestClient(build_app(state)).post("/predict/graph", json={"graphs": _graphs()})
    assert resp.status_code == 200, resp.text


@pytest.mark.parametrize("which", ["gcn_client", "torch_gcn_client"])
def test_batch_matches_single_requests(which, request):
    client = request.getfixturevalue(which)
    graphs = _graphs()
    resp = client.post("/predict/graph", json={"graphs": graphs})
    assert resp.status_code == 200, resp.text
    body = resp.json()["graphs"]
    assert len(body) == 3
    atol, rtol = _tolerances(client.app.state.serving)
    for graph, got in zip(graphs, body, strict=True):
        single = client.post("/predict/graph", json=graph).json()
        assert got["shapes"] == single["shapes"]
        assert got["dtypes"] == single["dtypes"]
        np.testing.assert_allclose(
            got["outputs"]["output_0"], single["outputs"]["output_0"], atol=atol, rtol=rtol
        )


def test_batch_layout_counts(gcn_client):
    from downshift.serve.predict import _prepare_feeds, _Request

    state = gcn_client.app.state.serving
    _, _, layout = _prepare_feeds(state, _Request(graphs=_graphs()))
    assert layout is not None
    nodes, edges = layout[0], layout[1]
    assert (len(nodes), sum(nodes), sum(edges)) == (3, 15, 19)


def test_batch_offsets_edge_indices():
    items = [
        {"x": np.zeros((2, 1), np.float32), "edge_index": np.array([[0, 1], [1, 0]])},
        {"x": np.zeros((3, 1), np.float32), "edge_index": np.array([[0, 2], [2, 1]])},
    ]
    feeds, nodes, edges = graph_mod.batch_graphs(items)
    assert nodes == [2, 3] and edges == [2, 2]
    assert feeds["edge_index"].tolist() == [[0, 1, 2, 4], [1, 0, 4, 3]]


def test_batch_response_as_base64_and_safetensors(gcn_client):
    graphs = _graphs()
    want = gcn_client.post("/predict/graph", json={"graphs": graphs}).json()["graphs"]
    resp = gcn_client.post("/predict/graph", json={"graphs": graphs, "output_encoding": "base64"})
    assert "data" in resp.json()["graphs"][1]["outputs"]["output_0"]
    resp = gcn_client.post("/predict/graph", json={"graphs": graphs}, headers={"Accept": ST})
    assert resp.headers["content-type"] == ST
    tensors = st_load(resp.content)
    assert set(tensors) == {f"graphs.{i}.output_0" for i in range(3)}
    np.testing.assert_allclose(
        tensors["graphs.2.output_0"], want[2]["outputs"]["output_0"], rtol=1e-5, atol=1e-6
    )


def test_edge_level_output_is_split_by_edges(edge_client):
    graphs = _graphs(((5, 7), (4, 3)))
    resp = edge_client.post("/predict/graph", json={"graphs": graphs})
    assert resp.status_code == 200, resp.text
    body = resp.json()["graphs"]
    assert [g["shapes"]["output_0"] for g in body] == [[7, 1], [3, 1]]
    for graph, got in zip(graphs, body, strict=True):
        single = edge_client.post("/predict/graph", json=graph).json()
        np.testing.assert_allclose(
            got["outputs"]["output_0"], single["outputs"]["output_0"], atol=1e-4, rtol=1e-4
        )


def test_bad_edge_index_names_the_graph(gcn_client):
    graphs = _graphs()
    graphs[2]["edge_index"][0][0] = 99
    resp = gcn_client.post("/predict/graph", json={"graphs": graphs})
    assert resp.status_code == 400
    assert "graphs[2]" in resp.json()["detail"]
    assert "outside the node range [0, 6)" in resp.json()["detail"]


def test_graphs_and_top_level_are_exclusive(gcn_client):
    graph = _graphs()[0]
    assert gcn_client.post("/predict/graph", json={"graphs": [graph], **graph}).status_code == 422
    assert gcn_client.post("/predict/graph", json={}).status_code == 422
    assert gcn_client.post("/predict/graph", json={"graphs": []}).status_code == 422


def test_fixed_output_needs_one_graph(fixed_client):
    graphs = _graphs(((5, 7), (4, 3)))
    resp = fixed_client.post("/predict/graph", json={"graphs": graphs})
    assert resp.status_code == 400
    assert "can't be split per graph" in resp.json()["detail"]
    assert "batch" in resp.json()["detail"]
    one = fixed_client.post("/predict/graph", json={"graphs": graphs[:1]})
    assert one.status_code == 200, one.text
    assert one.json()["graphs"][0]["shapes"]["output_0"] == [1, 4]


def test_unknown_kinds_refuse_a_multi_graph_batch(gcn_client):
    state = gcn_client.app.state.serving
    saved = state.verdict.output_axes
    state.verdict.output_axes = []
    try:
        assert gcn_client.post("/predict/graph", json={"graphs": _graphs()}).status_code == 400
        assert gcn_client.post("/predict/graph", json={"graphs": _graphs()[:1]}).status_code == 200
    finally:
        state.verdict.output_axes = saved


def _binary(graphs, **overrides):
    items = [{k: np.array(v) for k, v in g.items()} for g in graphs]
    tensors = {
        "x": np.concatenate([i["x"] for i in items]).astype(np.float32),
        "edge_index": np.concatenate([i["edge_index"] for i in items], axis=1).astype(np.int64),
        "num_nodes": np.array([len(i["x"]) for i in items], dtype=np.int64),
        "num_edges": np.array([i["edge_index"].shape[1] for i in items], dtype=np.int64),
    }
    tensors.update(overrides)
    return tensors


def _post_binary(client, tensors):
    return client.post("/predict/graph", content=st_save(tensors), headers={"Content-Type": ST})


def test_binary_batch_equals_json_batch(gcn_client):
    graphs = _graphs()
    want = gcn_client.post("/predict/graph", json={"graphs": graphs}).json()
    got = _post_binary(gcn_client, _binary(graphs))
    assert got.status_code == 200, got.text
    assert got.json() == want


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"num_nodes": np.array([[5, 4, 6]], dtype=np.int64)}, "1-D"),
        ({"num_edges": np.array([7, 3], dtype=np.int64)}, "must match"),
        ({"num_nodes": np.array([5, 0, 10], dtype=np.int64)}, "at least 1"),
        ({"num_edges": np.array([7, -3, 19], dtype=np.int64)}, "negative"),
        ({"num_nodes": np.array([5, 4, 7], dtype=np.int64)}, "sum(num_nodes)"),
        ({"num_edges": np.array([7, 3, 8], dtype=np.int64)}, "sum(num_edges)"),
        ({"num_nodes": np.array([5, 4, 6], dtype=np.int32)}, "int64"),
        (
            {"num_nodes": np.array([], dtype=np.int64), "num_edges": np.array([], dtype=np.int64)},
            "empty",
        ),
    ],
)
def test_binary_batch_mismatches_are_400(gcn_client, override, message):
    resp = _post_binary(gcn_client, _binary(_graphs(), **override))
    assert resp.status_code == 400, resp.text
    assert message in resp.json()["detail"]


def test_binary_batch_names_the_graph_with_a_bad_local_node_id(gcn_client):
    graphs = _graphs()
    graphs[1]["edge_index"][1][0] = len(graphs[1]["x"])  # one past graph 1's own nodes
    resp = _post_binary(gcn_client, _binary(graphs))
    assert resp.status_code == 400
    assert resp.json()["detail"].startswith("graphs[1]: edge_index contains")


def test_binary_batch_needs_both_counts(gcn_client):
    tensors = _binary(_graphs())
    del tensors["num_edges"]
    resp = _post_binary(gcn_client, tensors)
    assert resp.status_code == 400
    assert "num_edges" in resp.json()["detail"]
