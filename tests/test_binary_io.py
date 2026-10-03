import json
import re
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import load as st_load
from safetensors.numpy import save as st_save

from downshift.serve import codec
from downshift.serve.codec import decode_safetensors, encode_safetensors

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
