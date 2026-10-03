"""AxisFact: sampled vs served ranges per dynamic axis (core/axes.py)."""

import pytest

import downshift
from downshift.core.axes import AxisFact, axis_facts
from downshift.core.verdict import ExportVerdict
from tests.models import clean_mlp, gnn_gcn, tiny_bert


def test_axis_facts_pairs_served_bounds_with_sampled_sizes() -> None:
    bounds = {"x": {0: ("dim0", 1, 100)}, "mask": {1: ("seq", 1, 64)}}
    samples = [[(2, 5), (2, 8)], [(7, 5), (7, 3)]]

    facts = axis_facts(bounds, ("x", "mask"), samples)

    assert facts == [
        AxisFact("x", 0, "dim0", 1, 100, 2, 7),
        AxisFact("mask", 1, "seq", 1, 64, 3, 8),
    ]


@pytest.mark.parametrize("samples", [None, []])
def test_axis_facts_sampled_is_none_without_verify(samples) -> None:
    (fact,) = axis_facts({"x": {0: ("dim0", 1, 9)}}, ("x",), samples)

    assert (fact.sampled_min, fact.sampled_max) == (None, None)
    assert (fact.served_min, fact.served_max) == (1, 9)


def test_axis_facts_ignores_inputs_without_dynamic_axes() -> None:
    assert axis_facts({}, ("x",), [[(1, 2)]]) == []


def test_axis_fact_round_trips_through_dict() -> None:
    for fact in (AxisFact("x", 0, "dim0", 1, 9, 2, 5), AxisFact("x", 1, "seq", 1, 9, None, None)):
        assert AxisFact.from_dict(fact.to_dict()) == fact


def test_clean_mlp_has_one_batch_like_axis() -> None:
    verdict = downshift.check(clean_mlp.make_model(), clean_mlp.make_inputs(), k=8)

    (fact,) = verdict.axes
    assert (fact.input, fact.axis, fact.name) == ("x", 0, "dim0")
    assert fact.sampled_min is not None and fact.sampled_max is not None
    assert fact.served_min <= fact.sampled_min <= fact.sampled_max <= fact.served_max


def test_clean_mlp_without_verify_has_unsampled_facts() -> None:
    verdict = downshift.check(
        clean_mlp.make_model(), clean_mlp.make_inputs(), verify_numerics=False
    )

    (fact,) = verdict.axes
    assert (fact.sampled_min, fact.sampled_max) == (None, None)
    assert fact.served_max > 1


@pytest.mark.needs_torch_26
def test_tiny_bert_has_batch_and_seq_facts() -> None:
    verdict = downshift.check(tiny_bert.make_model(), tiny_bert.make_inputs(), k=8)

    by_name = {fact.name: fact for fact in verdict.axes}
    assert {"batch", "seq"} <= set(by_name)
    assert by_name["seq"].served_max == 64  # max_position_embeddings of the fixture
    for fact in by_name.values():
        assert fact.sampled_max is not None and fact.sampled_max <= fact.served_max


@pytest.mark.needs_torch_26
def test_gcn_has_independent_node_and_edge_facts() -> None:
    verdict = downshift.check(gnn_gcn.make_model(), gnn_gcn.make_inputs(), k=8)

    by_name = {fact.name: fact for fact in verdict.axes}
    assert {"num_nodes", "num_edges"} <= set(by_name)
    assert by_name["num_nodes"].input == "x" and by_name["num_nodes"].axis == 0
    assert by_name["num_edges"].input == "edge_index" and by_name["num_edges"].axis == 1
    assert by_name["num_nodes"].served_max > 2 * (by_name["num_nodes"].sampled_max or 0)


def test_a_verdict_without_prepared_has_no_axes_and_survives_from_dict() -> None:
    data = {
        "status": "UNVERIFIED",
        "model_family": "onnx",
        "recommended_backend": "onnxruntime",
    }

    assert ExportVerdict.from_dict(data).axes == []
