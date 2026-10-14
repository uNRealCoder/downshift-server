"""The row check (core/rows.py): a row must not change when it runs with other rows."""

import pytest

pytest.importorskip("transformers")

from downshift.adapters.embedding import (  # noqa: E402
    EMBEDDING_ATTR,
    PADDING_SIDE_ATTR,
    EmbeddingRecipe,
)
from downshift.core import rows  # noqa: E402
from downshift.core.verdict import build_verdict, prepare_model  # noqa: E402
from tests.models import clean_mlp, cross_row, tiny_bert  # noqa: E402


def _rows_of(model, example=None):
    prepared = prepare_model(model, example)
    verdict = build_verdict(prepared)
    return prepared, verdict.numerics, verdict.take_session()


def test_an_mlp_is_row_independent():
    prepared, numerics, session = _rows_of(clean_mlp.make_model(), clean_mlp.make_inputs(2))

    assert rows.batch_axis(prepared, numerics) is None
    assert rows.check_rows(prepared, numerics, session) == (True, None)


def test_a_model_that_reads_the_batch_mean_is_not_independent():
    prepared, numerics, session = _rows_of(cross_row.make_model(), cross_row.make_inputs())

    independent, reason = rows.check_rows(prepared, numerics, session)

    assert independent is False
    assert reason is not None
    assert "when it runs with other rows" in reason


def test_token_outputs_have_no_batch_axis():
    prepared, numerics, session = _rows_of(tiny_bert.make_model(), tiny_bert.make_inputs())

    independent, reason = rows.check_rows(prepared, numerics, session)

    assert independent is False
    assert reason is not None
    assert "output_0 axis 1 changes" in reason


@pytest.mark.parametrize(("side", "independent"), [("right", True), ("left", False)])
def test_padding_is_part_of_the_check(side, independent):
    # BERT has absolute position ids. Left padding moves a row to other positions, so the
    # joined batch really gives another output. The check must see this.
    model = tiny_bert.make_model()
    setattr(model, EMBEDDING_ATTR, EmbeddingRecipe("mean", True, None, "--pooling"))
    setattr(model, PADDING_SIDE_ATTR, side)
    prepared, numerics, session = _rows_of(model, tiny_bert.make_inputs())

    sample = rows.two_row_sample(prepared)

    assert not isinstance(sample, str)
    joined, (row_a, row_b) = sample
    assert joined[0].shape[0] == 2
    assert row_a[0].shape[1] != row_b[0].shape[1], "one row must be padded in the batch"
    result, reason = rows.check_rows(prepared, numerics, session)
    assert result is independent
    assert (reason is None) is independent


def test_a_failure_of_the_check_is_not_checked_and_not_a_crash():
    prepared, numerics, _ = _rows_of(clean_mlp.make_model(), clean_mlp.make_inputs(2))

    independent, reason = rows.check_rows(prepared, numerics, session=object())

    assert independent is None
    assert reason is not None and reason.startswith("not checked: ")


def test_rows_agree_names_the_first_difference():
    def run(feeds):
        out = feeds[0].numpy().copy()
        if len(out) > 1:
            out[1] += 0.5  # the second row of a joined batch changes
        return [out]

    import torch

    joined = (torch.zeros(2, 3),)
    alone = [(torch.zeros(1, 3),), (torch.zeros(1, 3),)]

    reason = rows.rows_agree(run, joined, alone, atol=1e-6, rtol=1e-6)

    assert reason == "output 0 row 1 differs by 5.0e-01 when it runs with other rows"
    assert rows.rows_agree(lambda f: [f[0].numpy()], joined, alone, 1e-6, 1e-6) is None


def test_generic_rows_need_one_row_shape():
    import torch

    prepared, _, _ = _rows_of(clean_mlp.make_model(), clean_mlp.make_inputs(2))
    prepared.vary_fn = lambda i: (torch.zeros(2, 16),) if i == 0 else (torch.zeros(2, 5),)

    assert rows.two_row_sample(prepared) == "no two samples with the same row shape"
