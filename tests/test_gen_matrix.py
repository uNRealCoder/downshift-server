"""scripts/gen_matrix.py: a fixture whose make_model() raises must be skipped. It must not
crash the whole matrix run (tests.models.broken_factory exists for exactly this case)."""

from scripts.gen_matrix import COLUMNS, _run_fixture


def test_a_fixture_whose_factory_raises_is_skipped_not_a_crash():
    row = _run_fixture("broken_factory")

    assert row.model == "broken_factory"
    assert len(row.cells) == len(COLUMNS) - 2  # Model and Hazard are separate fields
    assert "skipped" in row.cells[1]  # Export column
    assert "RuntimeError" in row.cells[1]
