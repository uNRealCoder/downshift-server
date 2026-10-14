"""--output-tol: one pair of bounds for each output (core/verify.py, settings.py, the CLI)."""

import pytest
import torch
from typer.testing import CliRunner

from downshift import settings
from downshift.cli import main
from downshift.core import memo
from downshift.core import verify as verify_mod
from downshift.core.verify import NumericsReport, verify
from tests.models import clean_mlp

# The base bounds of float32 are atol 1e-4 and rtol 1e-3.
SMALL, LARGE = 1e-6, 5e-3


class TwoOutputs(torch.nn.Module):
    def forward(self, x):
        return x * 2.0, x + 1.0


class NoisySession:
    """Stands in for an ONNX Runtime session. It gives the torch values plus a fixed error on
    each output, so the error of each output is known."""

    def __init__(self, errors: tuple[float, float]) -> None:
        self.errors = errors

    def get_inputs(self):
        return [type("Input", (), {"name": "x"})()]

    def run(self, _, feeds):
        x = torch.from_numpy(feeds["x"])
        return [(x * 2.0 + self.errors[0]).numpy(), (x + 1.0 + self.errors[1]).numpy()]


@pytest.fixture
def noisy(monkeypatch):
    def run(errors=(SMALL, LARGE), **kwargs) -> NumericsReport:
        monkeypatch.setattr(verify_mod, "load_session", lambda _: NoisySession(errors))
        base = (torch.ones(2, 3),)
        return verify(TwoOutputs(), "stub", base, vary_fn=lambda i: base, k=2, **kwargs)

    return run


def test_one_pair_of_bounds_fails_the_loose_output(noisy):
    report = noisy()

    assert not report.passed
    assert [o.failures for o in report.outputs] == [0, 2]
    assert [o.overridden for o in report.outputs] == [False, False]


def test_a_named_output_passes_with_its_own_bound_and_the_other_keeps_the_base(noisy):
    report = noisy(output_tol={"output_1": (1e-2, None)})

    assert report.passed
    first, second = report.outputs
    assert (first.atol, first.rtol, first.overridden) == (1e-4, 1e-3, False)
    assert (second.atol, second.rtol, second.overridden) == (1e-2, 1e-3, True)
    assert report.tolerance_abs == 1e-4, "the top-level bound is the base"


def test_the_error_of_each_output_is_in_the_report_and_the_top_level_is_the_max(noisy):
    report = noisy(output_tol={"output_1": (1e-2, None)})

    first, second = report.outputs
    assert first.max_abs_err == pytest.approx(SMALL, rel=0.05)
    assert second.max_abs_err == pytest.approx(LARGE, rel=0.05)
    assert report.max_abs_err == pytest.approx(max(first.max_abs_err, second.max_abs_err))


def test_a_tight_bound_makes_a_small_error_fail(noisy):
    report = noisy(output_tol={"output_0": (1e-9, 0.0), "output_1": (1e-2, 0.0)})

    assert [o.failures for o in report.outputs] == [2, 0]
    assert not report.passed


def test_an_unknown_output_name_is_a_usage_error_with_the_names_of_the_model(noisy):
    with pytest.raises(ValueError) as error:
        noisy(output_tol={"output_3": (1e-3, None)})

    message = str(error.value)
    assert "unknown output `output_3`" in message
    assert "`output_0`, `output_1`" in message
    assert message.endswith("Change the name in --output-tol.")


def test_the_report_round_trips_with_its_outputs(noisy):
    report = noisy(output_tol={"output_1": (1e-2, 1e-2)})

    assert NumericsReport.from_dict(report.to_dict()) == report


def test_an_old_report_has_no_outputs():
    data = {
        "samples_tested": 1,
        "max_abs_err": 0.0,
        "max_rel_err": 0.0,
        "failures": 0,
        "shape_generalization": True,
        "tolerance_abs": 1e-4,
        "tolerance_rel": 1e-3,
    }

    assert NumericsReport.from_dict(data).outputs == []


def test_a_real_model_gets_one_record_for_its_output():
    import downshift

    verdict = downshift.check(clean_mlp.make_model(), clean_mlp.make_inputs(2), k=3)

    assert verdict.numerics is not None
    assert [o.name for o in verdict.numerics.outputs] == ["output_0"]


# --- the option -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("output_1=1e-3", {"output_1": (1e-3, None)}),
        ("output_1=1e-3:1e-2", {"output_1": (1e-3, 1e-2)}),
        ("output_0=1e-5,output_1=1e-3:1e-2", {"output_0": (1e-5, None), "output_1": (1e-3, 1e-2)}),
    ],
)
def test_parse_output_tol(text, expected):
    assert settings.parse_output_tol([text]) == expected


@pytest.mark.parametrize(
    "text", ["output_1", "=1e-3", "output_1=abc", "output_1=-1", "x=1:-2", "x=nan"]
)
def test_a_bad_form_is_a_usage_error(text):
    with pytest.raises(ValueError, match="expected NAME=ATOL or NAME=ATOL:RTOL"):
        settings.parse_output_tol([text])


def test_the_env_variable_sets_the_default(monkeypatch):
    import importlib

    monkeypatch.setenv("DOWNSHIFT_OUTPUT_TOL", "output_0=1e-5,output_1=1e-3:1e-2")
    try:
        reloaded = importlib.reload(settings)
        assert reloaded.OUTPUT_TOL == {"output_0": (1e-5, None), "output_1": (1e-3, 1e-2)}
    finally:
        monkeypatch.delenv("DOWNSHIFT_OUTPUT_TOL")
        importlib.reload(settings)


def test_the_cli_check_takes_the_option_and_names_an_unknown_output():
    result = CliRunner().invoke(
        main.app,
        ["check", "tests.models.clean_mlp:make_model", "--output-tol", "output_5=1e-3", "--json"],
    )

    assert result.exit_code == 4, result.output
    assert "unknown output `output_5`" in result.output


def test_a_change_of_the_option_is_a_new_export_key():
    plain = memo.common_parts(
        adapter=None, dynamic=None, k=8, seed=0, atol=None, rtol=None, vary=None, axis_max=None
    )
    tuned = memo.common_parts(
        adapter=None,
        dynamic=None,
        k=8,
        seed=0,
        atol=None,
        rtol=None,
        vary=None,
        axis_max=None,
        output_tol={"output_0": (1e-3, None)},
    )

    assert plain != tuned
