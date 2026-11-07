"""The models of tests/models/ and examples/ build and run a forward pass."""

import importlib

import pytest
import torch
from typer.testing import CliRunner

from downshift.cli.main import app
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


@pytest.mark.parametrize(
    ("name", "shape"),
    [
        ("clean_mlp", (1, 4)),
        ("scatter_include_self_false", (4, 8)),
        ("data_dependent_branch", (1, 8)),
    ],
)
def test_example_forward_pass(name, shape):
    module = importlib.import_module(f"examples.{name}")
    assert module.make_model()(*module.make_inputs()).shape == shape


def test_data_dependent_branch_both_branches():
    model = importlib.import_module("examples.data_dependent_branch").make_model()
    assert model(torch.ones(1, 8)).shape == (1, 8)
    assert model(-torch.ones(1, 8)).shape == (1, 8)


def test_check_examples_clean_mlp_via_cli():
    result = CliRunner().invoke(app, ["check", "examples.clean_mlp:make_model", "--json"])
    assert result.exit_code == 0, result.output
