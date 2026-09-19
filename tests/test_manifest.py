"""The provenance manifest written next to every exported .onnx."""

import hashlib
import json
from enum import IntEnum

import onnx
import pytest
from onnx import TensorProto, helper

import downshift
from downshift.core.manifest import _DTYPE_NAMES, manifest_path_for, observed_dtype
from tests.models import clean_mlp, data_dependent_branch


def test_dtype_names_is_an_intenum_keyed_by_onnx_dtype_code():
    assert issubclass(_DTYPE_NAMES, IntEnum)
    assert _DTYPE_NAMES.fp32 == TensorProto.FLOAT
    assert _DTYPE_NAMES.fp16 == TensorProto.FLOAT16
    assert _DTYPE_NAMES.fp64 == TensorProto.DOUBLE
    assert _DTYPE_NAMES.bf16 == TensorProto.BFLOAT16


@pytest.fixture(scope="module")
def manifest(exported_mlp) -> dict:
    onnx_path, _, _ = exported_mlp
    return json.loads(manifest_path_for(onnx_path).read_text())


def test_export_writes_onnx_and_manifest_side_by_side(exported_mlp):
    onnx_path, _, verdict = exported_mlp
    assert verdict.status == "CLEAN", verdict.reason
    assert verdict.onnx_path == onnx_path
    assert onnx_path.is_file()
    assert manifest_path_for(onnx_path) == onnx_path.with_name("m.manifest.json")
    assert manifest_path_for(onnx_path).is_file()


def test_manifest_records_provenance(manifest):
    assert manifest["downshift_version"] == downshift.__version__
    assert manifest["created_utc"]
    assert manifest["source_sha256"] is None
    assert set(manifest["versions"]) == {"torch", "onnx", "onnxruntime"}
    assert isinstance(manifest["opset"], int)
    assert manifest["observed_dtype"] == "fp32"
    assert manifest["verdict"]["status"] == "CLEAN"
    assert manifest["verdict"]["input_names"] == ["x"]


def test_manifest_sha_matches_onnx_file(exported_mlp, manifest):
    onnx_path, _, _ = exported_mlp
    assert manifest["onnx_sha256"] == hashlib.sha256(onnx_path.read_bytes()).hexdigest()


def test_fp16_export_records_observed_dtype(tmp_path):
    out = tmp_path / "h.onnx"
    verdict = downshift.export(clean_mlp.make_model(), out, clean_mlp.make_inputs(), fp16=True)

    assert verdict.status != "FAILED", verdict.reason
    assert out.is_file()
    manifest = json.loads(manifest_path_for(out).read_text())
    assert manifest["observed_dtype"] == "fp16"
    assert manifest["verdict"]["status"] == verdict.status


def _onnx_model_without_initializers(elem_type: int) -> onnx.ModelProto:
    """A graph with no weights at all, so observed_dtype must fall back to its input."""
    inp = helper.make_tensor_value_info("x", elem_type, [1, 4])
    out = helper.make_tensor_value_info("y", elem_type, [1, 4])
    node = helper.make_node("Identity", ["x"], ["y"])
    graph = helper.make_graph([node], "g", [inp], [out])
    return helper.make_model(graph)


def test_observed_dtype_falls_back_to_graph_input_when_no_initializer(tmp_path):
    path = tmp_path / "no_init.onnx"
    onnx.save(_onnx_model_without_initializers(TensorProto.FLOAT16), str(path))
    assert observed_dtype(path) == "fp16"


def test_observed_dtype_returns_none_for_unmapped_dtype(tmp_path):
    path = tmp_path / "unmapped.onnx"
    onnx.save(_onnx_model_without_initializers(TensorProto.STRING), str(path))
    assert observed_dtype(path) is None


def test_failed_export_writes_nothing(tmp_path):
    out = tmp_path / "bad.onnx"
    verdict = downshift.export(
        data_dependent_branch.make_model(), out, data_dependent_branch.make_inputs()
    )

    assert verdict.status == "FAILED"
    assert verdict.onnx_path is None
    assert list(tmp_path.iterdir()) == []
