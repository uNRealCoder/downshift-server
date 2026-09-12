"""normalize_dtype() and to_numpy() edge cases not reached through the HTTP layer."""

import pytest

from downshift.serve.schemas import normalize_dtype, to_numpy


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
