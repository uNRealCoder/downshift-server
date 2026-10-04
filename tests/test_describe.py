"""GET /schema: does it tell a caller enough to build a request that actually works?

The load-bearing assertion in here is that the example body /schema hands out, posted back
to /predict unchanged, returns 200. Everything else describes that body.
"""

import pytest
from fastapi.testclient import TestClient

from downshift import sources
from downshift.serve.backends import IOSpec
from downshift.serve.describe import (
    DYNAMIC_AXIS,
    MAX_EXAMPLE_ELEMENTS,
    _axis,
    _fill_value,
    _nested,
    describe,
)
from tests.models import clean_mlp


def test_schema_describes_the_inputs_and_the_example_actually_works(mlp_client, mlp_state):
    body = mlp_client.get("/schema").json()

    assert body["model"] == "tests.models.clean_mlp:make_model"
    assert body["family"] == mlp_state.verdict.model_family
    assert body["backend"] == "onnxruntime"
    assert body["endpoint"] == "/predict"
    assert body["graph_endpoint"] is None

    (x,) = body["inputs"]
    assert x["name"] == "x"
    assert x["dtype"] == "float32"  # not ORT's own "tensor(float)"
    assert x["required"] is True
    assert x["shape"][1] == 16 and x["example_shape"] == [1, 16]

    assert body["outputs"][0]["name"] == "output_0"
    assert body["outputs"][0]["dtype"] == "float32"

    assert mlp_client.post("/predict", json=body["example_request"]).status_code == 200
    assert body["example_curl"].startswith("curl -s http://testserver/predict")


def test_schema_says_the_model_is_one_already_on_this_machine(mlp_client):
    source = mlp_client.get("/schema").json()["source"]

    assert source["spec"] == "tests.models.clean_mlp:make_model"
    assert source["kind"] == sources.IMPORT_SPEC
    assert source["description"] == sources.SOURCE_KIND_HELP[sources.IMPORT_SPEC]
    assert source["fetched_at_runtime"] is False


def test_schema_reports_the_wire_formats_and_the_request_limits(mlp_client, mlp_state):
    body = mlp_client.get("/schema").json()

    assert [f["name"] for f in body["input_formats"]] == ["nested list", "typed object", "base64"]
    assert body["output_encodings"] == ["json", "base64"]
    assert body["default_output_encoding"] == mlp_state.options.output_encoding.value
    assert body["limits"] == mlp_client.get("/metadata").json()["limits"]
    assert body["limits"]["max_body_bytes"] == mlp_state.options.max_body_bytes


def test_schema_points_a_graph_model_at_the_graph_route(gcn_client):
    body = gcn_client.get("/schema").json()

    assert body["graph_endpoint"] == "/predict/graph"
    assert {i["name"] for i in body["inputs"]} == {"x", "edge_index"}
    assert [i["dtype"] for i in body["inputs"] if i["name"] == "edge_index"] == ["int64"]
    assert gcn_client.post("/predict", json=body["example_request"]).status_code == 200
    assert body["graph_batching"] == {"output_0": "node"}


def test_schema_has_no_graph_batching_for_a_non_graph_model(mlp_client):
    assert mlp_client.get("/schema").json()["graph_batching"] is None


def test_a_large_fixed_shape_input_gets_a_note_instead_of_an_inlined_example(serve_fixture):
    """dynamic_batch_cnn takes (batch, 3, 16, 16): 768 elements even at batch 1."""
    state = serve_fixture("dynamic_batch_cnn")
    body = describe(state, "http://testserver/predict")

    assert body.example_request is None
    assert body.example_curl is None
    assert body.inputs[0].example_shape == [1, 3, 16, 16]
    assert f"past {MAX_EXAMPLE_ELEMENTS} elements" in body.notes[0]


def test_an_input_with_no_declared_dtype_is_called_out(mlp_state, monkeypatch):
    meta = mlp_state.backend.metadata()
    monkeypatch.setattr(meta, "inputs", [IOSpec("x", None, [1, 16])])
    monkeypatch.setattr(mlp_state.backend, "metadata", lambda: meta)

    body = describe(mlp_state, "http://testserver/predict")

    assert body.inputs[0].dtype is None
    assert any("declare no dtype" in note for note in body.notes)


def test_outputs_unknown_on_a_torch_backend_without_example_inputs(mlp_state, monkeypatch):
    meta = mlp_state.backend.metadata()
    monkeypatch.setattr(meta, "outputs", [])
    monkeypatch.setattr(mlp_state.backend, "metadata", lambda: meta)

    body = describe(mlp_state, "http://testserver/predict")

    assert body.outputs == []
    assert any("output names and dtypes are not known yet" in note for note in body.notes)


@pytest.mark.parametrize(
    ("dim", "expected"),
    [
        (16, 16),
        ("batch", "batch"),
        ("seq", "seq"),
        ("s77", DYNAMIC_AXIS),  # torch.export's own symbol names mean nothing to a caller
        ("u3", DYNAMIC_AXIS),
        (None, DYNAMIC_AXIS),
        (0, DYNAMIC_AXIS),
        (-1, DYNAMIC_AXIS),
    ],
)
def test_axis_rendering(dim, expected):
    assert _axis(dim) == expected


def test_an_attention_mask_is_filled_with_ones_not_zeros():
    """All-zero would be a well-formed request that hands back NaNs."""
    assert _fill_value("attention_mask", "int64") == 1
    assert _fill_value("decoder_attention_mask", "int64") == 1
    assert _fill_value("input_ids", "int64") == 0
    assert _fill_value("x", "float32") == 0.0
    assert _fill_value("flags", "bool") is False


def test_nested_builds_the_declared_shape():
    assert _nested([2, 3], 0.0) == [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    assert _nested([], 1) == 1


def test_openapi_documents_the_schema_route(mlp_client):
    spec = mlp_client.get("/openapi.json").json()
    assert "/schema" in spec["paths"]
    assert "SchemaResponse" in spec["components"]["schemas"]


def test_serving_state_reports_the_kind_of_its_source(mlp_state):
    assert mlp_state.source_kind == sources.IMPORT_SPEC


def test_schema_falls_back_when_the_backend_declares_no_shape(mlp_state, monkeypatch):
    """A torch backend built without example inputs declares names only."""
    meta = mlp_state.backend.metadata()
    monkeypatch.setattr(meta, "inputs", [])
    monkeypatch.setattr(mlp_state.backend, "metadata", lambda: meta)

    body = describe(mlp_state, "http://testserver/predict")

    assert [i.name for i in body.inputs] == list(mlp_state.input_names)
    assert body.inputs[0].shape is None
    assert body.example_request is None


def test_app_for_exposes_schema_for_an_in_memory_module():
    from downshift.serve import app_for

    client = TestClient(app_for(clean_mlp.make_model(), clean_mlp.make_inputs(), warmup=1))
    body = client.get("/schema").json()

    assert body["source"]["kind"] == sources.IN_PROCESS_MODULE
    assert body["model"] == "model"  # an nn.Module has no path to name
    assert client.post("/predict", json=body["example_request"]).status_code == 200


def test_app_for_names_an_onnx_file_by_its_file_name(exported_mlp):
    from downshift.serve import app_for

    path, model, _ = exported_mlp
    client = TestClient(app_for(str(path), clean_mlp.make_inputs(), reference=model, warmup=1))
    body = client.get("/schema").json()

    assert body["source"]["kind"] == sources.ONNX_FILE
    assert body["source"]["spec"] == path.name
    assert client.post("/predict", json=body["example_request"]).status_code == 200


def test_app_for_still_honours_an_explicit_source_label():
    from downshift.serve import app_for

    client = TestClient(
        app_for(clean_mlp.make_model(), clean_mlp.make_inputs(), source="tutorial", warmup=1)
    )
    assert client.get("/schema").json()["source"]["spec"] == "tutorial"


def test_schema_and_metadata_report_the_axes(mlp_client, mlp_state):
    (fact,) = mlp_state.verdict.axes
    expected = {
        "input": "x",
        "axis": 0,
        "name": fact.name,
        "served_min": fact.served_min,
        "served_max": fact.served_max,
        "sampled_min": fact.sampled_min,
        "sampled_max": fact.sampled_max,
    }

    assert mlp_client.get("/schema").json()["axes"] == [expected]
    assert mlp_client.get("/metadata").json()["axes"] == [expected]


def test_describe_has_no_axes_for_a_verdict_without_them(mlp_state, monkeypatch):
    monkeypatch.setattr(mlp_state.verdict, "axes", [])

    assert describe(mlp_state, "http://testserver/predict").axes == []


def test_embedding_block_lists_the_named_prompts_and_the_default():
    from types import SimpleNamespace

    from downshift.adapters.embedding import EmbeddingRecipe
    from downshift.serve import describe as describe_mod

    recipe = EmbeddingRecipe(
        "lasttoken", True, None, "modules.json", {"query": "Q: ", "document": ""}, "query"
    )
    outputs = [describe_mod.TensorSchema(name="output_0", dtype="float32", shape=["batch", 32])]

    info = describe_mod._embedding(SimpleNamespace(embedding=recipe), outputs)

    assert info is not None
    assert info.prompts == {"query": "Q: ", "document": ""}
    assert info.default_prompt == "query"
    assert info.dimension == 32
