"""Each fixture in tests/models/ must construct and run a forward pass.

There is no export here. This test only makes sure that the fixtures are valid PyTorch models.
A failure in test_export is then about the exporter and not about the fixture.
"""

import pytest
import torch

from tests.models import (
    bf16_weights,
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
    bf16_weights,
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
