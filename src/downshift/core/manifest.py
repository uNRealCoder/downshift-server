"""Provenance sidecar written next to every exported .onnx."""

import hashlib
import json
from datetime import UTC, datetime
from enum import IntEnum
from pathlib import Path

import onnx
import onnxruntime
import torch
from onnx.external_data_helper import ExternalDataInfo, uses_external_data

from downshift.core.verdict import ExportVerdict
from downshift.sources import hide_paths, path_basename


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


def external_data_files(onnx_path: Path) -> list[str]:
    """Every external-data location the graph's initializers reference, in first-use order."""
    proto = onnx.load(str(onnx_path), load_external_data=False)
    locations: dict[str, None] = {}
    for init in proto.graph.initializer:
        if uses_external_data(init):
            locations[ExternalDataInfo(init).location] = None
    return list(locations)


def build_manifest(
    onnx_path: Path, verdict: ExportVerdict, source_path: Path | None, package_version: str
) -> dict:
    # The manifest travels with the exported file, so it names files and never records where
    # they sat on this machine (which would carry the exporter's directory layout, and their
    # username, to whoever receives the artifact). The hashes identify the files.
    paths = (onnx_path, source_path)
    verdict_dict = verdict.to_dict()
    verdict_dict["onnx_path"] = onnx_path.name if verdict_dict["onnx_path"] else None
    verdict_dict["reason"] = hide_paths(verdict_dict["reason"], paths)
    verdict_dict["warnings"] = [hide_paths(w, paths) for w in verdict_dict["warnings"]]
    return {
        "downshift_version": package_version,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "onnx_file": onnx_path.name,
        "onnx_sha256": _sha256(onnx_path),
        "external_data": [
            {"file": location, "sha256": _sha256(onnx_path.parent / location)}
            for location in external_data_files(onnx_path)
        ],
        "source_model": path_basename(str(source_path)) if source_path else None,
        "source_sha256": _sha256(source_path) if source_path and source_path.is_file() else None,
        "versions": {
            "torch": torch.__version__,
            "onnx": onnx.__version__,
            "onnxruntime": onnxruntime.__version__,
        },
        "opset": verdict.opset,
        "observed_dtype": observed_dtype(onnx_path),
        "execution_providers_available": onnxruntime.get_available_providers(),
        "verdict": verdict_dict,
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
