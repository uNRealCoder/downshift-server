"""The base64 codec behind binary tensor I/O: pybase64 (SIMD) when installed, stdlib otherwise.

CPython's binascii runs at roughly 230 MB/s and is most of the cost of a base64 request once
the JSON codec is out of the way; pybase64 is about 12x faster on both directions. Same call
signatures either way, so callers never know which one they got. Decoding yields a writable
buffer where the codec allows it, so torch can wrap it without copying.
"""

from collections.abc import Callable

try:
    import pybase64

    BASE64_CODEC = "pybase64"
    b64encode: Callable[..., bytes] = pybase64.b64encode

    def b64decode(data: str | bytes) -> bytearray:
        return pybase64.b64decode_as_bytearray(data, validate=True)

except ImportError:  # pragma: no cover - exercised only where pybase64 is absent
    import base64

    BASE64_CODEC = "stdlib"
    b64encode = base64.b64encode

    def b64decode(data: str | bytes) -> bytes:  # type: ignore[misc]
        return base64.b64decode(data, validate=True)
