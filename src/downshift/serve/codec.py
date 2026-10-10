"""The wire codecs for binary tensor I/O: base64 and safetensors.

Base64 uses pybase64 (SIMD) if it is installed. Otherwise, it uses the standard library.

The binascii of CPython runs at about 230 MB/s. It is most of the cost of a base64 request
after the JSON codec is out of the way. pybase64 is about 12x faster in both directions. The
call signatures are the same in both cases, so callers never know which one they got. If the
codec allows it, decoding gives a writable buffer. Torch can then wrap it without a copy.
"""

import json
import math
from collections.abc import Callable
from typing import Any

import numpy as np

try:
    import pybase64

    BASE64_CODEC = "pybase64"
    b64encode: Callable[..., bytes] = pybase64.b64encode

    def b64decode(data: str | bytes) -> bytearray:
        out: bytearray = pybase64.b64decode_as_bytearray(data, validate=True)
        return out

except ImportError:  # pragma: no cover - runs only where pybase64 is absent
    import base64

    BASE64_CODEC = "stdlib"
    b64encode = base64.b64encode

    def b64decode(data: str | bytes) -> bytes:  # type: ignore[misc]
        return base64.b64decode(data, validate=True)


_SAFETENSORS_DTYPES: dict[str, np.dtype[Any]] = {
    "F64": np.dtype("<f8"),
    "F32": np.dtype("<f4"),
    "F16": np.dtype("<f2"),
    "I64": np.dtype("<i8"),
    "I32": np.dtype("<i4"),
    "I16": np.dtype("<i2"),
    "I8": np.dtype("i1"),
    "U8": np.dtype("u1"),
    "BOOL": np.dtype("?"),
}
_SAFETENSORS_NAMES = {dt: name for name, dt in _SAFETENSORS_DTYPES.items()}


def _q(value: object) -> str:
    """A client-supplied value as a one-line, length-capped literal for an error message."""
    return repr(value)[:64]


def _no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"safetensors header repeats the name {_q(key)}")
        out[key] = value
    return out


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def decode_safetensors(
    body: bytes, *, max_input_bytes: int
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """Parse a safetensors body into numpy views without a copy, plus its `__metadata__`.

    We wrote this reader by hand, by design. The header is JSON and the payload is raw bytes.
    Nothing here can execute code. It raises ValueError for all malformed input.
    """
    if len(body) < 8:
        raise ValueError("safetensors body is shorter than its 8-byte header length")
    header_len = int.from_bytes(body[:8], "little")
    if header_len > len(body) - 8:
        raise ValueError("safetensors header length runs past the end of the body")
    data_start = 8 + header_len
    data_size = len(body) - data_start
    try:
        header = json.loads(bytes(body[8:data_start]), object_pairs_hook=_no_duplicates)
    except json.JSONDecodeError as e:
        raise ValueError(f"safetensors header is not valid JSON: {e.msg}") from None
    except UnicodeDecodeError:
        raise ValueError("safetensors header is not valid UTF-8") from None
    if not isinstance(header, dict):
        raise ValueError("safetensors header must be a JSON object")

    metadata = header.pop("__metadata__", {})
    if not isinstance(metadata, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in metadata.items()
    ):
        raise ValueError("safetensors __metadata__ must map strings to strings")

    entries: list[tuple[int, int, str, np.dtype[Any], tuple[int, ...]]] = []
    for name, info in header.items():
        if not isinstance(info, dict):
            raise ValueError(f"safetensors entry {_q(name)} must be an object")
        dtype_name = info.get("dtype")
        if dtype_name == "BF16":
            raise ValueError(
                f"tensor {_q(name)} is BF16, which is not accepted; float32 is the wire type"
            )
        dtype = _SAFETENSORS_DTYPES.get(dtype_name) if isinstance(dtype_name, str) else None
        if dtype is None:
            raise ValueError(f"tensor {_q(name)} has unsupported dtype {_q(dtype_name)}")
        shape = info.get("shape")
        if not isinstance(shape, list) or not all(_is_int(d) for d in shape):
            raise ValueError(f"tensor {_q(name)} has a malformed shape")
        if any(d < 0 for d in shape):
            raise ValueError(f"tensor {_q(name)} has a negative dimension")
        offsets = info.get("data_offsets")
        if not isinstance(offsets, list) or len(offsets) != 2 or not all(map(_is_int, offsets)):
            raise ValueError(f"tensor {_q(name)} has malformed data_offsets")
        begin, end = offsets
        if begin < 0 or end < begin or end > data_size:
            raise ValueError(f"tensor {_q(name)} has data_offsets outside the data buffer")
        count = math.prod(shape)
        if count * dtype.itemsize != end - begin:
            raise ValueError(f"tensor {_q(name)} has a shape that does not match its data_offsets")
        if end - begin > max_input_bytes:
            raise ValueError(
                f"tensor {_q(name)} is {end - begin} bytes, over the {max_input_bytes}-byte input limit"
            )
        entries.append((begin, end, name, dtype, tuple(shape)))

    entries.sort(key=lambda e: (e[0], e[1]))
    cursor = 0
    for begin, end, name, _, _ in entries:
        if begin != cursor:
            kind = "overlaps another tensor" if begin < cursor else "leaves a gap before it"
            raise ValueError(f"tensor {_q(name)} {kind}; data must be contiguous")
        cursor = end
    if cursor != data_size:
        raise ValueError("safetensors data buffer is not fully covered by the tensors")

    arrays: dict[str, np.ndarray] = {}
    for begin, end, name, dtype, shape in entries:
        if begin == end:
            arrays[name] = np.empty(shape, dtype=dtype)
        else:
            flat = np.frombuffer(
                body, dtype=dtype, count=(end - begin) // dtype.itemsize, offset=data_start + begin
            )
            arrays[name] = flat.reshape(shape)
    return {name: arrays[name] for name in header}, metadata


def encode_safetensors(
    arrays: dict[str, np.ndarray], metadata: dict[str, str] | None = None
) -> bytes:
    """Serialize `arrays` as safetensors: little-endian, C-contiguous, header space-padded to 8."""
    header: dict[str, Any] = {}
    if metadata:
        header["__metadata__"] = metadata
    # Byte views of each array (contiguous, little-endian). The join below is then the only copy.
    chunks: list[np.ndarray] = []
    offset = 0
    # The reference writer sorts by descending itemsize and then by name. This keeps the tensors aligned.
    for name, array in sorted(arrays.items(), key=lambda kv: (-kv[1].dtype.itemsize, kv[0])):
        dtype = array.dtype.newbyteorder("<") if array.dtype.itemsize > 1 else array.dtype
        wire = _SAFETENSORS_NAMES.get(dtype)
        if wire is None:
            raise ValueError(f"tensor {_q(name)} has dtype {_q(str(array.dtype))}, not on the wire")
        raw = np.ascontiguousarray(array, dtype=dtype).reshape(-1).view(np.uint8)
        header[name] = {
            "dtype": wire,
            "shape": list(array.shape),
            "data_offsets": [offset, offset + raw.nbytes],
        }
        chunks.append(raw)
        offset += raw.nbytes
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    return b"".join([len(encoded).to_bytes(8, "little"), encoded, *chunks])
