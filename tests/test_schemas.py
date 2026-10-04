"""normalize_dtype() and to_numpy() edge cases not reached through the HTTP layer."""

import numpy as np
import pytest

from downshift.serve.codec import BASE64_CODEC
from downshift.serve.schemas import normalize_dtype, to_numpy
from tests.conftest import b64_input


def test_normalize_dtype_none_is_none():
    assert normalize_dtype(None) is None


def test_normalize_dtype_rejects_unknown_name():
    assert normalize_dtype("not-a-real-dtype") is None


def test_to_numpy_typed_array_with_unknown_dtype_raises():
    with pytest.raises(ValueError, match="unknown dtype"):
        to_numpy("x", {"data": [1, 2], "dtype": "not-a-real-dtype"})


def test_to_numpy_ragged_input_raises():
    with pytest.raises(ValueError, match="x"):
        to_numpy("x", [[1, 2], [3]])


def test_to_numpy_non_numeric_array_raises():
    with pytest.raises(ValueError, match="ragged or non-numeric"):
        to_numpy("x", [{"a": 1}])


def test_to_numpy_cast_failure_raises():
    with pytest.raises(ValueError, match="cannot cast"):
        to_numpy("x", ["not", "numbers"])


def test_to_numpy_explicit_shape_mismatch_raises():
    with pytest.raises(ValueError, match="x"):
        to_numpy("x", {"data": [1, 2, 3], "dtype": "float32", "shape": [2, 2]})


def test_to_numpy_base64_is_a_zero_copy_view():
    arr = np.arange(12, dtype=np.int16).reshape(3, 4)
    out = to_numpy("x", b64_input(arr))
    np.testing.assert_array_equal(out, arr)
    assert out.dtype == np.int16
    assert not out.flags.owndata  # frombuffer view over the decoded buffer, nothing copied
    assert out.flags.c_contiguous
    # pybase64 decodes into a bytearray, so torch can wrap the view without copying it.
    assert out.flags.writeable == (BASE64_CODEC == "pybase64")


def test_to_numpy_base64_explicit_dtype_wins_over_declared():
    # The backend declared float, the client sent int64; no silent cast either way.
    out = to_numpy("x", b64_input(np.arange(4, dtype=np.int64)), "tensor(float)")
    assert out.dtype == np.int64


@pytest.mark.parametrize("spec", ["tensor(float)", "<f4", "float32"])
def test_to_numpy_base64_accepts_ort_style_and_short_dtype_names(spec):
    out = to_numpy("x", b64_input(np.arange(4, dtype=np.float32), dtype=spec))
    assert out.dtype == np.float32


def test_to_numpy_base64_scalar_shape():
    out = to_numpy("x", b64_input(np.asarray(1.5, dtype=np.float64)))
    assert out.shape == ()
    assert out == 1.5


def test_to_numpy_base64_respects_max_bytes_keyword():
    payload = b64_input(np.zeros(8, dtype=np.float32))  # 32 bytes
    assert to_numpy("x", payload, max_bytes=32).shape == (8,)
    with pytest.raises(ValueError, match=r"'x'.*32 bytes exceeds the server limit of 31 bytes"):
        to_numpy("x", payload, max_bytes=31)


def test_to_numpy_base64_big_endian_is_rejected_before_decoding():
    with pytest.raises(ValueError, match=r"'x'.*little-endian"):
        to_numpy("x", {"data": "definitely not base64", "dtype": ">i4", "shape": [1]})


def test_axis_info_sampled_defaults_to_none():
    from downshift.serve.schemas import AxisInfo

    info = AxisInfo(input="x", axis=0, name="dim0", served_min=1, served_max=9)

    assert info.sampled_min is None and info.sampled_max is None
    assert AxisInfo.model_validate(info.model_dump()) == info


def test_openapi_lists_the_binary_content_types(mlp_client):
    paths = mlp_client.get("/openapi.json").json()["paths"]
    for route in ("/predict", "/predict/graph"):
        op = paths[route]["post"]
        request = set(op["requestBody"]["content"])
        assert {
            "application/json",
            "application/vnd.safetensors",
            "application/octet-stream",
        } <= request
        response = set(op["responses"]["200"]["content"])
        assert {"application/json", "application/vnd.safetensors"} <= response


def test_server_wide_output_encoding_stays_json_or_base64():
    from downshift.serve.schemas import OutputEncoding, RequestOutputEncoding

    assert {e.value for e in OutputEncoding} == {"json", "base64"}
    assert "safetensors" in {e.value for e in RequestOutputEncoding}
