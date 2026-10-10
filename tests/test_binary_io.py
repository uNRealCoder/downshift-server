import dataclasses
import json
import re
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from safetensors.numpy import load as st_load
from safetensors.numpy import save as st_save

from downshift.loading import LoadSpec, load_model
from downshift.serve import codec
from downshift.serve.app import build_app
from downshift.serve.codec import decode_safetensors, encode_safetensors
from downshift.serve.engine import prepare_serving
from downshift.serve.options import ExecutionChoice
from tests.models import hf_repo

BIG = 1 << 30


def _body(header: dict | str | list, data: bytes = b"") -> bytes:
    raw = (header if isinstance(header, str) else json.dumps(header)).encode()
    return len(raw).to_bytes(8, "little") + raw + data


def _f32(name="x", shape=(2,), begin=0, end=8):
    return {name: {"dtype": "F32", "shape": list(shape), "data_offsets": [begin, end]}}


def test_decode_reference_output():
    arrays = {
        "a": np.arange(6, dtype=np.float32).reshape(2, 3),
        "b": np.array([1, 2, 3], dtype=np.int64),
        "c": np.array([True, False]),
        "d": np.float16([1.5, 2.5]),
        "e": np.zeros((0, 4), dtype=np.float32),
        "s": np.array(7, dtype=np.int32),
    }
    out, meta = decode_safetensors(
        st_save(arrays, metadata={"output_encoding": "safetensors"}), max_input_bytes=BIG
    )
    assert meta == {"output_encoding": "safetensors"}
    assert set(out) == set(arrays)
    for k, v in arrays.items():
        assert out[k].dtype == v.dtype and out[k].shape == v.shape
        np.testing.assert_array_equal(out[k], v)


def test_decode_is_a_view():
    body = st_save({"a": np.arange(4, dtype=np.float32)})
    out, _ = decode_safetensors(body, max_input_bytes=BIG)
    assert out["a"].base is not None and not out["a"].flags.owndata


def test_encode_read_by_reference():
    arrays = {
        "a": np.arange(6, dtype=np.float32).reshape(2, 3),
        "b": np.array([1, 2, 3], dtype=np.int64),
        "c": np.array([True, False]),
        "t": np.arange(6, dtype=np.float64).reshape(3, 2).T,
    }
    body = encode_safetensors(arrays, {"k": "v"})
    assert int.from_bytes(body[:8], "little") % 8 == 0
    got = st_load(body)
    assert set(got) == set(arrays)
    for k, v in arrays.items():
        np.testing.assert_array_equal(got[k], v)
    assert decode_safetensors(body, max_input_bytes=BIG)[1] == {"k": "v"}


def test_encode_roundtrip_and_no_metadata():
    arrays = {"y": np.ones((2, 2), dtype=np.float32)}
    out, meta = decode_safetensors(encode_safetensors(arrays, None), max_input_bytes=BIG)
    assert meta == {}
    np.testing.assert_array_equal(out["y"], arrays["y"])


def test_encode_rejects_object_dtype():
    with pytest.raises(ValueError, match="not on the wire"):
        encode_safetensors({"o": np.array([object()])}, None)


DATA = bytes(16)
CASES = {
    "duplicate": (
        '{"x":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},"x":{"dtype":"F32",'
        '"shape":[2],"data_offsets":[8,16]}}',
        DATA,
        "repeats",
    ),
    "overflow": (_f32(shape=(1 << 62, 4)), DATA, "does not match"),
    "bf16": ({"x": {"dtype": "BF16", "shape": [2], "data_offsets": [0, 4]}}, bytes(4), "float32"),
    "unknown_dtype": (
        {"x": {"dtype": "C64", "shape": [1], "data_offsets": [0, 8]}},
        bytes(8),
        "unsupported dtype",
    ),
    "object_dtype": (
        {"x": {"dtype": "O", "shape": [1], "data_offsets": [0, 8]}},
        bytes(8),
        "unsupported dtype",
    ),
    "overlap": ({**_f32("a", (2,), 0, 8), **_f32("b", (2,), 4, 12)}, bytes(12), "overlaps"),
    "gap": ({**_f32("a", (2,), 0, 8), **_f32("b", (2,), 12, 20)}, bytes(20), "gap"),
    "out_of_bounds": (_f32(shape=(4,), begin=0, end=16), bytes(8), "outside"),
    "negative_dim": (_f32(shape=(-2,), begin=0, end=8), DATA, "negative"),
    "non_dict_header": ("[1, 2]", b"", "JSON object"),
    "bad_json": ("{nope", b"", "not valid JSON"),
    "bad_metadata_type": ({"__metadata__": {"k": 1}, **_f32()}, bytes(8), "__metadata__"),
    "metadata_not_dict": ({"__metadata__": ["k"], **_f32()}, bytes(8), "__metadata__"),
    "short_data": ({**_f32("a", (2,), 0, 8)}, bytes(12), "not fully covered"),
    "bool_in_shape": (_f32(shape=(True,), begin=0, end=4), bytes(4), "malformed shape"),
    "mismatch": (_f32(shape=(3,), begin=0, end=8), bytes(8), "does not match"),
}


@pytest.mark.parametrize("case", CASES)
def test_malformed_is_value_error(case):
    header, data, msg = CASES[case]
    with pytest.raises(ValueError, match=msg):
        decode_safetensors(_body(header, data), max_input_bytes=BIG)


def test_header_length_past_body():
    body = (1000).to_bytes(8, "little") + b"{}"
    with pytest.raises(ValueError, match="past the end"):
        decode_safetensors(body, max_input_bytes=BIG)


def test_body_shorter_than_prefix():
    with pytest.raises(ValueError, match="shorter"):
        decode_safetensors(b"abc", max_input_bytes=BIG)


def test_per_tensor_input_limit():
    body = st_save({"a": np.zeros(100, dtype=np.float32)})
    with pytest.raises(ValueError, match="input limit"):
        decode_safetensors(body, max_input_bytes=399)
    decode_safetensors(body, max_input_bytes=400)


def test_error_messages_are_single_line_and_capped():
    name = "evil\n" + "x" * 500
    header = {name: {"dtype": "O\nZ", "shape": [1], "data_offsets": [0, 8]}}
    with pytest.raises(ValueError) as e:
        decode_safetensors(_body(header, bytes(8)), max_input_bytes=BIG)
    assert "\n" not in str(e.value)
    assert len(str(e.value)) < 200
    header = {name: {"dtype": "F32", "shape": [-1], "data_offsets": [0, 8]}}
    with pytest.raises(ValueError) as e:
        decode_safetensors(_body(header, bytes(8)), max_input_bytes=BIG)
    assert "\n" not in str(e.value) and len(str(e.value)) < 200


def test_codec_source_has_no_pickle_or_np_load():
    source = Path(codec.__file__).read_text()
    assert "pickle" not in source
    assert not re.search(r"\bnp\.load\b|\bnumpy\.load\b", source)


# --- the HTTP routes ---------------------------------------------------------------------------

ST = "application/vnd.safetensors"


def _post(client, path, tensors, meta=None, content_type=ST, headers=None):
    return client.post(
        path,
        content=st_save(tensors, metadata=meta),
        headers={"Content-Type": content_type, **(headers or {})},
    )


def _outputs(resp) -> dict[str, np.ndarray]:
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"] == ST
    assert "Server-Timing" in resp.headers
    return st_load(resp.content)


def _with(state, **overrides):
    return dataclasses.replace(state, options=dataclasses.replace(state.options, **overrides))


def test_binary_request_matches_json(mlp_client):
    x = np.random.default_rng(0).standard_normal((3, 16)).astype(np.float32)
    want = mlp_client.post("/predict", json={"inputs": {"x": x.tolist()}}).json()
    for ct in (ST, "application/octet-stream; charset=binary"):
        resp = _post(mlp_client, "/predict", {"x": x}, content_type=ct)
        assert resp.status_code == 200, resp.text
        got = resp.json()
        assert got["shapes"] == want["shapes"]
        np.testing.assert_allclose(got["outputs"]["output_0"], want["outputs"]["output_0"])


def test_binary_request_and_response(mlp_client):
    x = np.ones((2, 16), dtype=np.float32)
    want = mlp_client.post("/predict", json={"inputs": {"x": x.tolist()}}).json()
    got = _outputs(_post(mlp_client, "/predict", {"x": x}, headers={"Accept": ST}))
    np.testing.assert_allclose(got["output_0"], want["outputs"]["output_0"], rtol=1e-6)


def test_output_encoding_in_metadata_and_in_json_body(mlp_client):
    x = np.ones((1, 16), dtype=np.float32)
    assert _outputs(_post(mlp_client, "/predict", {"x": x}, {"output_encoding": "safetensors"}))
    resp = mlp_client.post(
        "/predict", json={"inputs": {"x": x.tolist()}, "output_encoding": "safetensors"}
    )
    assert "output_0" in _outputs(resp)
    resp = _post(mlp_client, "/predict", {"x": x}, {"output_encoding": "base64"})
    assert "data" in resp.json()["outputs"]["output_0"]


@pytest.mark.parametrize("accept", ["*/*", "application/json", None])
def test_other_accept_stays_json(mlp_client, accept):
    headers = {"Accept": accept} if accept else {}
    resp = mlp_client.post("/predict", json={"inputs": {"x": [[0.0] * 16]}}, headers=headers)
    assert resp.headers["content-type"] == "application/json"


def test_unsupported_content_type_is_415(mlp_client):
    for path in ("/predict", "/predict/graph"):
        resp = mlp_client.post(path, content=b"abc", headers={"Content-Type": "application/x-npy"})
        assert resp.status_code == 415
        assert ST in resp.json()["detail"] and "JSON" in resp.json()["detail"]


def test_malformed_binary_is_400(mlp_client):
    resp = mlp_client.post("/predict", content=b"abc", headers={"Content-Type": ST})
    assert resp.status_code == 400 and "shorter" in resp.json()["detail"]


def test_unknown_tensor_name_is_400_and_truncated(mlp_client):
    resp = _post(mlp_client, "/predict", {"y" * 200: np.zeros(1, dtype=np.float32)})
    assert resp.status_code == 400
    assert "unknown tensor name" in resp.json()["detail"]
    assert len(resp.json()["detail"]) < 200


def test_missing_tensor_and_wrong_dtype_are_400(mlp_client):
    assert _post(mlp_client, "/predict", {}).status_code == 400
    resp = _post(mlp_client, "/predict", {"x": np.zeros((1, 16), dtype=np.float64)})
    assert resp.status_code == 400 and "float64" in resp.json()["detail"]


def test_text_in_metadata_is_400(mlp_client):
    resp = _post(mlp_client, "/predict", {"x": np.zeros((1, 16), np.float32)}, {"text": "hi"})
    assert resp.status_code == 400 and "JSON only" in resp.json()["detail"]


def test_body_limit_and_max_input_bytes(mlp_state):
    x = np.zeros((4, 16), dtype=np.float32)
    body_len = len(st_save({"x": x}))
    client = TestClient(build_app(_with(mlp_state, max_body_bytes=body_len - 1)))
    assert _post(client, "/predict", {"x": x}).status_code == 413
    client = TestClient(build_app(_with(mlp_state, max_input_bytes=255)))
    resp = _post(client, "/predict", {"x": x})
    assert resp.status_code == 400 and "input limit" in resp.json()["detail"]


def test_binary_never_goes_inline(mlp_state):
    client = TestClient(build_app(_with(mlp_state, execution=ExecutionChoice.inline)))
    resp = _post(client, "/predict", {"x": np.zeros((1, 16), np.float32)})
    assert resp.status_code == 200
    assert "Server-Timing" in resp.headers


def test_graph_route_binary(gcn_client):
    x = np.random.default_rng(1).standard_normal((5, 8)).astype(np.float32)
    edge_index = np.array([[0, 1, 2, 3, 4, 0, 2], [1, 2, 3, 4, 0, 3, 4]], dtype=np.int64)
    want = gcn_client.post(
        "/predict/graph", json={"x": x.tolist(), "edge_index": edge_index.tolist()}
    ).json()
    resp = _post(gcn_client, "/predict/graph", {"x": x, "edge_index": edge_index})
    np.testing.assert_allclose(resp.json()["outputs"]["output_0"], want["outputs"]["output_0"])
    accept = {"Accept": ST}
    got = _outputs(
        _post(gcn_client, "/predict/graph", {"x": x, "edge_index": edge_index}, headers=accept)
    )
    np.testing.assert_allclose(got["output_0"], want["outputs"]["output_0"], rtol=1e-5)
    bad = _post(gcn_client, "/predict/graph", {"x": x, "nope": edge_index})
    assert bad.status_code == 400 and "unknown tensor name" in bad.json()["detail"]
    assert _post(gcn_client, "/predict/graph", {"x": x}).status_code == 400


def test_text_request_with_accept_returns_embeddings_as_safetensors(tmp_path):
    repo = hf_repo.write_encoder_repo(tmp_path)
    client = TestClient(build_app(prepare_serving(load_model(LoadSpec(repo)))))
    text = {"text": ["hello world", "hello"]}
    want = client.post("/predict", json=text).json()
    resp = client.post("/predict", json=text, headers={"Accept": f"{ST}, application/json;q=0.5"})
    got = _outputs(resp)
    name = next(iter(want["outputs"]))
    np.testing.assert_allclose(got[name], want["outputs"][name], rtol=1e-5, atol=1e-6)
    meta = decode_safetensors(resp.content, max_input_bytes=BIG)[1]
    assert json.loads(meta["downshift.embedding"])["pooling"] == "mean"


def test_an_unknown_output_encoding_in_metadata_is_a_400(mlp_client):
    x = np.random.randn(1, 16).astype(np.float32)
    resp = _post(mlp_client, "/predict", {"x": x}, {"output_encoding": "pickle"})
    assert resp.status_code == 400
    assert "output_encoding" in resp.json()["detail"]


def test_an_empty_predict_body_is_the_usual_422(mlp_client):
    resp = mlp_client.post("/predict", content=b"", headers={"Content-Type": "application/json"})
    assert resp.status_code == 422
    assert resp.json()["detail"][0]["type"] == "missing"
