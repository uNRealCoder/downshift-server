"""examples/: the models behind the README's own commands."""

import torch
from typer.testing import CliRunner

from downshift.cli.main import app
from examples import clean_mlp, data_dependent_branch, scatter_include_self_false

runner = CliRunner()


def test_clean_mlp_make_model_and_inputs():
    model = clean_mlp.make_model()
    inputs = clean_mlp.make_inputs()
    out = model(*inputs)
    assert out.shape == (1, 4)


def test_scatter_include_self_false_make_model_and_inputs():
    model = scatter_include_self_false.make_model()
    inputs = scatter_include_self_false.make_inputs()
    out = model(*inputs)
    assert out.shape == (4, 8)


def test_data_dependent_branch_make_model_and_inputs():
    model = data_dependent_branch.make_model()
    inputs = data_dependent_branch.make_inputs()
    out = model(*inputs)
    assert out.shape == (1, 8)


def test_data_dependent_branch_both_branches():
    model = data_dependent_branch.make_model()
    assert model(torch.ones(1, 8)).shape == (1, 8)
    assert model(-torch.ones(1, 8)).shape == (1, 8)


def test_check_examples_clean_mlp_via_cli():
    result = runner.invoke(app, ["check", "examples.clean_mlp:make_model", "--json"])
    assert result.exit_code == 0, result.output
