"""PyG adapter integration tests: GCN/SAGE/GAT through the same verdict pipeline as the
generic fixtures, covering the N/E-independent dynamic shapes and the PyG-aware vary_fn
(edge_index redrawn against the resampled node count).

All three come back CLEAN on torch 2.14 / torch_geometric 2.8. GATConv's attention
aggregation doesn't hit the scatter_reduce(include_self=False) translation problem, so
the scatter_include_self_false fixture in tests/test_export.py, not GAT, is where the
silent-wrong-answer case is demonstrated.
"""

import pytest

import downshift
from downshift.core.verdict import ExportVerdict
from tests.models import gnn_gat, gnn_gcn, gnn_output_kinds, gnn_sage

FIXTURES = [gnn_gcn, gnn_sage, gnn_gat]


@pytest.mark.needs_torch_26
@pytest.mark.parametrize("module", FIXTURES, ids=lambda m: m.__name__.rsplit(".", 1)[-1])
def test_gnn_fixture_is_clean(module) -> None:
    model = module.make_model()
    inputs = module.make_inputs()

    verdict = downshift.check(model, inputs, k=8)

    assert verdict.status == "CLEAN", verdict.reason
    assert verdict.model_family == "pyg"
    assert verdict.numerics is not None
    assert verdict.numerics.shape_generalization


@pytest.mark.needs_torch_26
def test_gnn_verdict_survives_node_edge_count_mismatch() -> None:
    """If N and E were tied to one Dim, varying the edge count while the node count stays
    put would blow up during verification."""
    model = gnn_gcn.make_model()
    inputs = gnn_gcn.make_inputs(num_nodes=6, num_edges=10)

    verdict = downshift.check(model, inputs, k=8)

    assert verdict.status == "CLEAN"


@pytest.mark.needs_torch_26
def test_axis_max_num_nodes_pins_a_sample_at_that_size() -> None:
    verdict = downshift.check(
        gnn_gcn.make_model(), gnn_gcn.make_inputs(), axis_max={"num_nodes": 500}, k=4
    )

    assert verdict.status == "CLEAN", verdict.reason
    assert {fact.name: fact.served_max for fact in verdict.axes}["num_nodes"] == 500
    assert verdict.numerics is not None
    assert verdict.numerics.sample_shapes[1][0][0] == 500


@pytest.mark.needs_torch_26
def test_gcn_output_is_node_level() -> None:
    verdict = downshift.check(gnn_gcn.make_model(), gnn_gcn.make_inputs(), k=6)

    assert verdict.output_axes == ["node"]
    assert ExportVerdict.from_dict(verdict.to_dict()).output_axes == ["node"]


@pytest.mark.needs_torch_26
def test_edge_model_output_is_edge_level() -> None:
    verdict = downshift.check(
        gnn_output_kinds.make_edge_model(), gnn_output_kinds.make_inputs(), k=6
    )

    assert verdict.status == "CLEAN", verdict.reason
    assert verdict.output_axes == ["edge"]


@pytest.mark.needs_torch_26
def test_fixed_size_readout_output_is_fixed() -> None:
    verdict = downshift.check(
        gnn_output_kinds.make_fixed_model(), gnn_output_kinds.make_inputs(), k=6
    )

    assert verdict.status == "CLEAN", verdict.reason
    assert verdict.output_axes == ["fixed"]
