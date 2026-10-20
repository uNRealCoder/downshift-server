"""The provenance file that downshift writes next to each exported .onnx file."""

import json
from datetime import UTC, datetime
from pathlib import Path

import onnx
import onnxruntime
import torch
from onnx.external_data_helper import ExternalDataInfo, uses_external_data

from downshift.core.memo import sha256_file
from downshift.core.verdict import ExportVerdict
from downshift.sources import path_basename

# The ONNX float dtype codes that the manifest names. Other codes (int64 indices, bools) have no name.
_FLOAT_DTYPES = {
    onnx.TensorProto.FLOAT: "fp32",
    onnx.TensorProto.FLOAT16: "fp16",
    onnx.TensorProto.DOUBLE: "fp64",
    onnx.TensorProto.BFLOAT16: "bf16",
}


def observed_dtype(onnx_path: Path) -> str | None:
    """Float dtype of the graph's weights, i.e. what the export actually produced."""
    proto = onnx.load(str(onnx_path), load_external_data=False)
    codes = [init.data_type for init in proto.graph.initializer]
    codes += [inp.type.tensor_type.elem_type for inp in proto.graph.input]
    return next((_FLOAT_DTYPES[code] for code in codes if code in _FLOAT_DTYPES), None)


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
    # The manifest goes with the exported file. It names files and never records where they
    # were on this machine. A path would give the directory layout and the user name of the
    # exporter to each person who receives the artifact. The hashes identify the files.
    verdict_dict = verdict.redacted_dict((onnx_path, source_path))
    verdict_dict["onnx_path"] = onnx_path.name if verdict_dict["onnx_path"] else None
    return {
        "downshift_version": package_version,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "onnx_file": onnx_path.name,
        "onnx_sha256": sha256_file(onnx_path),
        "external_data": [
            {"file": location, "sha256": sha256_file(onnx_path.parent / location)}
            for location in external_data_files(onnx_path)
        ],
        "source_model": path_basename(str(source_path)) if source_path else None,
        "source_sha256": sha256_file(source_path)
        if source_path and source_path.is_file()
        else None,
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
