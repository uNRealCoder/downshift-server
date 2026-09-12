"""Direct unit coverage for verify.py's small helpers, exercised only indirectly elsewhere."""

import onnxruntime as ort
import pytest
import torch

from downshift.export.verify import _as_tensor_list, _resize_dim0, _to_session, verify
from tests.models import clean_mlp


def test_resize_dim0_is_a_noop_for_scalars_and_matching_sizes():
    scalar = torch.tensor(5.0)
    assert _resize_dim0(scalar, 3) is scalar

    tensor = torch.randn(4, 3)
    assert _resize_dim0(tensor, 4) is tensor


def test_to_session_accepts_raw_bytes(exported_mlp):
    onnx_path, _, _ = exported_mlp
    session = _to_session(onnx_path.read_bytes())
    assert isinstance(session, ort.InferenceSession)


def test_as_tensor_list_filters_a_tuple_and_rejects_other_types():
    t1, t2 = torch.zeros(1), torch.ones(1)
    assert _as_tensor_list((t1, "not a tensor", t2)) == [t1, t2]

    with pytest.raises(TypeError, match="can't compare"):
        _as_tensor_list({"a": 1})


def test_verify_requires_dynamic_shapes_or_vary_fn():
    with pytest.raises(ValueError, match="dynamic_shapes or an explicit vary_fn"):
        verify(clean_mlp.make_model(), "unused.onnx", clean_mlp.make_inputs())
