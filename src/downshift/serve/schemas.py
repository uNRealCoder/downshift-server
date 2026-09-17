"""Request/response models for the HTTP layer, plus JSON <-> numpy conversion."""

import math
import re
from enum import StrEnum
from typing import Annotated, Any

import numpy as np
from pydantic import BaseModel, Field

from downshift.serve.codec import b64decode
from downshift.settings import DEFAULT_MAX_INPUT_BYTES


class OutputEncoding(StrEnum):
    """How response tensors are written: nested lists, or {data, dtype, shape} base64 dicts."""

    json = "json"
    base64 = "base64"


OutputEncodingField = Annotated[
    OutputEncoding | None,
    Field(
        None,
        description=(
            "How response tensors are encoded: 'json' for nested lists, 'base64' for "
            '{"data": <base64 little-endian bytes>, "dtype": ..., "shape": [...]} per output. '
            "Omit to use the server default (--output-encoding / DOWNSHIFT_OUTPUT_ENCODING)."
        ),
    ),
]

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
    """Explicit form of an input: {"data": [...], "dtype": "float32", "shape": [2, 3]}.

    `data` may instead be a base64 string of the raw little-endian buffer, in which case
    `dtype` and `shape` are required.
    """

    data: Any
    dtype: str | None = None
    shape: list[int] | None = None


class PredictRequest(BaseModel):
    """`inputs` maps input name -> nested list, or a TypedArray object for explicit typing."""

    inputs: dict[str, Any]
    output_encoding: OutputEncodingField = None


class PredictResponse(BaseModel):
    outputs: dict[str, Any]
    shapes: dict[str, list[int]]
    dtypes: dict[str, str]


class GraphPredictRequest(BaseModel):
    """One graph per request: node features, COO edge index, optional edge attributes.

    No batching; concatenate graphs client-side (with offset edge indices) if needed.
    Each tensor is a nested list or a TypedArray dict (which is how base64 arrives).
    """

    x: list | dict
    edge_index: list | dict
    edge_attr: list | dict | None = None
    output_encoding: OutputEncodingField = None


class MetadataResponse(BaseModel):
    model: str
    family: str
    verdict: dict
    backend: dict
    input_names: list[str]
    notes: list[str] = Field(default_factory=list)
    version: str
    limits: dict = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: str = "ok"


class ReadyResponse(BaseModel):
    ready: bool


def _from_base64(
    name: str, typed: TypedArray, dtype_name: str | None, max_bytes: int
) -> np.ndarray:
    """A view over the decoded bytes: no cast, no copy. The caller's dtype wins."""
    if dtype_name is None or typed.shape is None:
        raise ValueError(
            f"input {name!r}: base64 input needs dtype and shape "
            "(the shape cannot be inferred from bytes)"
        )
    # np.dtype(">f4").name is plain "float32", so byte order is checked on the client's spec.
    if str(typed.dtype).startswith(">"):
        raise ValueError(
            f"input {name!r}: dtype {typed.dtype!r} is big-endian; base64 input must be little-endian"
        )
    dtype = np.dtype(dtype_name)
    if any(dim < 0 for dim in typed.shape):
        raise ValueError(f"input {name!r}: shape {typed.shape} has a negative dimension")
    expected = math.prod(typed.shape) * dtype.itemsize
    # Checked before decoding: the shape says how much would be handed to the backend.
    if expected > max_bytes:
        raise ValueError(
            f"input {name!r}: {expected} bytes exceeds the server limit of {max_bytes} bytes"
        )
    try:
        raw = b64decode(typed.data)
    except ValueError as exc:  # binascii.Error is a ValueError
        raise ValueError(f"input {name!r}: invalid base64 data: {exc}") from exc
    if len(raw) != expected:
        raise ValueError(
            f"input {name!r}: expected {expected} bytes for shape {typed.shape} and dtype "
            f"{dtype.name}, got {len(raw)}"
        )
    return np.frombuffer(raw, dtype=dtype).reshape(typed.shape)


def to_numpy(
    name: str,
    value: Any,
    expected_dtype: str | None = None,
    *,
    max_bytes: int = DEFAULT_MAX_INPUT_BYTES,
) -> np.ndarray:
    """Convert a JSON input to an ndarray.

    `expected_dtype` is whatever the backend declared (an ORT "tensor(...)" name, a numpy
    name, or None). An explicit dtype in the payload wins. With nothing declared, integer
    lists become int64 and everything else float32, which is what torch models expect.

    A TypedArray whose `data` is a string is base64 of the raw little-endian buffer and
    must carry dtype and shape; it decodes to at most `max_bytes`.
    """
    explicit_dtype: str | None = None
    explicit_shape: list[int] | None = None
    if isinstance(value, dict):
        typed = TypedArray.model_validate(value)
        if typed.dtype:
            explicit_dtype = normalize_dtype(typed.dtype)
            if explicit_dtype is None:
                raise ValueError(f"input {name!r}: unknown dtype {typed.dtype!r}")
        if isinstance(typed.data, str):
            return _from_base64(name, typed, explicit_dtype, max_bytes)
        value = typed.data
        explicit_shape = typed.shape

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
