"""Provenance sidecar written next to every exported .onnx."""

import hashlib
import json
from datetime import UTC, datetime
from enum import IntEnum
from pathlib import Path

import onnx
import onnxruntime
import torch

from downshift.export.verdict import ExportVerdict


class _DTYPE_NAMES(IntEnum):
    """ONNX TensorProto dtype codes we can label, keyed by their short name."""

    fp32 = onnx.TensorProto.FLOAT
    fp16 = onnx.TensorProto.FLOAT16
    fp64 = onnx.TensorProto.DOUBLE
    bf16 = onnx.TensorProto.BFLOAT16


# A plain value->name dict, so a miss is a dict lookup rather than an IntEnum ValueError;
# most initializers (int64 indices, bools, ...) are misses, and this runs in a loop.
_DTYPE_LOOKUP: dict[int, str] = {member.value: member.name for member in _DTYPE_NAMES}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dtype_name(code: int) -> str | None:
    return _DTYPE_LOOKUP.get(code)


def observed_dtype(onnx_path: Path) -> str | None:
    """Float dtype of the graph's weights, i.e. what the export actually produced."""
    proto = onnx.load(str(onnx_path), load_external_data=False)
    for init in proto.graph.initializer:
        name = _dtype_name(init.data_type)
        if name is not None:
            return name
    for inp in proto.graph.input:
        name = _dtype_name(inp.type.tensor_type.elem_type)
        if name is not None:
            return name
    return None


def build_manifest(
    onnx_path: Path, verdict: ExportVerdict, source_path: Path | None, package_version: str
) -> dict:
    return {
        "downshift_version": package_version,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "onnx_file": onnx_path.name,
        "onnx_sha256": _sha256(onnx_path),
        "source_model": str(source_path) if source_path else None,
        "source_sha256": _sha256(source_path) if source_path and source_path.is_file() else None,
        "versions": {
            "torch": torch.__version__,
            "onnx": onnx.__version__,
            "onnxruntime": onnxruntime.__version__,
        },
        "opset": verdict.opset,
        "observed_dtype": observed_dtype(onnx_path),
        "execution_providers_available": onnxruntime.get_available_providers(),
        "verdict": verdict.to_dict(),
    }


def manifest_path_for(onnx_path: Path) -> Path:
    return onnx_path.with_suffix(".manifest.json")


def write_manifest(
    onnx_path: Path, verdict: ExportVerdict, source_path: Path | None, package_version: str
) -> Path:
    path = manifest_path_for(onnx_path)
    path.write_text(
        json.dumps(build_manifest(onnx_path, verdict, source_path, package_version), indent=2)
    )
    return path
