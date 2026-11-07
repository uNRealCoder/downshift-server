"""The serving check on a real GPU. These tests skip without CUDA torch and the ORT CUDA
provider, so CI and the CPU venv skip them."""

import onnxruntime as ort
import pytest
import torch

from downshift.loading import LoadSpec, load_model
from downshift.serve.engine import ServeOptions, prepare_serving

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or "CUDAExecutionProvider" not in ort.get_available_providers(),
    reason="needs CUDA torch and the ONNX Runtime CUDA provider",
)


def _serve(**opts):
    loaded = load_model(LoadSpec("tests.models.clean_mlp:make_model"))
    return prepare_serving(loaded, ServeOptions(warmup=1, device="cuda", **opts))


@pytest.mark.parametrize("tf32", [False, True])
def test_ort_cuda_serving_check_passes_and_the_provider_options_reach_the_session(tf32):
    state = _serve(tf32=tf32)
    check = state.serving_numerics
    assert check is not None
    assert check.ok
    assert check.provider == "CUDAExecutionProvider"
    assert check.tf32 is tf32
    options = state.backend.session.get_provider_options()["CUDAExecutionProvider"]
    assert options["use_tf32"] == ("1" if tf32 else "0")
    print(f"tf32={tf32} max_abs_err={check.max_abs_err:.3e}")


def test_torch_cuda_serving_check_sets_the_tf32_flags():
    state = _serve(backend="torch", tf32=False)
    check = state.serving_numerics
    assert check is not None
    assert check.ok
    assert check.provider == "torch"
    assert torch.backends.cuda.matmul.allow_tf32 is False
    assert torch.backends.cudnn.allow_tf32 is False
    state = _serve(backend="torch", tf32=True)
    assert torch.backends.cuda.matmul.allow_tf32 is True
    assert torch.backends.cudnn.allow_tf32 is True
    torch.backends.cuda.matmul.allow_tf32 = False
