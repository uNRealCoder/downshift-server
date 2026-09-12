"""Request/response models for the HTTP layer, plus JSON <-> numpy conversion."""

import re
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

# ORT reports dtypes as "tensor(float)"; map those names onto numpy ones.
_ORT_DTYPES = {
    "float": "float32",
    "float16": "float16",
    "double": "float64",
    "int8": "int8",
    "int16": "int16",
    "int32": "int32",
    "int64": "int64",
    "uint8": "uint8",
    "bool": "bool",
}
_TENSOR_RE = re.compile(r"^tensor\((\w+)\)$")


def normalize_dtype(dtype: str | None) -> str | None:
    """Turn "tensor(float)" or "float32" into a numpy dtype name; None if unknown."""
    if dtype is None:
        return None
    match = _TENSOR_RE.match(dtype)
    if match:
        return _ORT_DTYPES.get(match.group(1))
    try:
        return np.dtype(dtype).name
    except TypeError:
        return None


class TypedArray(BaseModel):
    """Explicit form of an input: {"data": [...], "dtype": "float32", "shape": [2, 3]}."""

    data: Any
    dtype: str | None = None
    shape: list[int] | None = None


class PredictRequest(BaseModel):
    """`inputs` maps input name -> nested list, or a TypedArray object for explicit typing."""

    inputs: dict[str, Any]


class PredictResponse(BaseModel):
    outputs: dict[str, Any]
    shapes: dict[str, list[int]]
    dtypes: dict[str, str]


class GraphPredictRequest(BaseModel):
    """One graph per request: node features, COO edge index, optional edge attributes.

    No batching; concatenate graphs client-side (with offset edge indices) if needed.
    """

    x: list
    edge_index: list
    edge_attr: list | None = None


class MetadataResponse(BaseModel):
    model: str
    family: str
    verdict: dict
    backend: dict
    input_names: list[str]
    notes: list[str] = Field(default_factory=list)
    version: str


class HealthResponse(BaseModel):
    status: str = "ok"


class ReadyResponse(BaseModel):
    ready: bool


def to_numpy(name: str, value: Any, expected_dtype: str | None = None) -> np.ndarray:
    """Convert a JSON input to an ndarray.

    `expected_dtype` is whatever the backend declared (an ORT "tensor(...)" name, a numpy
    name, or None). An explicit dtype in the payload wins. With nothing declared, integer
    lists become int64 and everything else float32, which is what torch models expect.
    """
    explicit_dtype: str | None = None
    explicit_shape: list[int] | None = None
    if isinstance(value, dict):
        typed = TypedArray.model_validate(value)
        value = typed.data
        explicit_shape = typed.shape
        if typed.dtype:
            explicit_dtype = normalize_dtype(typed.dtype)
            if explicit_dtype is None:
                raise ValueError(f"input {name!r}: unknown dtype {typed.dtype!r}")

    try:
        arr = np.asarray(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"input {name!r}: {exc}") from exc
    if arr.dtype == object:
        raise ValueError(f"input {name!r}: ragged or non-numeric array")

    dtype = explicit_dtype or normalize_dtype(expected_dtype)
    if dtype is None:
        dtype = "int64" if np.issubdtype(arr.dtype, np.integer) else "float32"
    try:
        arr = arr.astype(dtype, copy=False)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"input {name!r}: cannot cast to {dtype}: {exc}") from exc

    if explicit_shape is not None:
        try:
            arr = arr.reshape(explicit_shape)
        except ValueError as exc:
            raise ValueError(f"input {name!r}: {exc}") from exc
    return arr
