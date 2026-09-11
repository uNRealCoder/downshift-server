"""Every fixture in tests/models/ must construct and forward-pass cleanly.

This does not check export/ONNX behavior (that's the Saturday export-core work) — it only
guarantees the fixtures themselves are valid, runnable PyTorch models, which is the
foundation everything else in the sprint plan is built on.
"""

import pytest
import torch

from tests.models import (
    clean_mlp,
    custom_autograd,
    data_dependent_branch,
    dict_input,
    dropout_model,
    dynamic_batch_cnn,
    scatter_include_self_false,
    tied_weights,
)

FIXTURE_MODULES = [
    clean_mlp,
    dynamic_batch_cnn,
    data_dependent_branch,
    custom_autograd,
    tied_weights,
    dict_input,
    dropout_model,
    scatter_include_self_false,
]


@pytest.mark.parametrize("module", FIXTURE_MODULES, ids=lambda m: m.__name__.rsplit(".", 1)[-1])
def test_fixture_forward_pass(module) -> None:
    model = module.make_model()
    inputs = module.make_inputs()

    with torch.no_grad():
        output = model(*inputs)

    assert isinstance(output, torch.Tensor)
    assert output.numel() > 0
    assert torch.isfinite(output).all()
