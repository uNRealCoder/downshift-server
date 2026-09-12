"""PyG adapter integration tests (sprint plan H9-H10): GCN/SAGE/GAT through the same
verdict pipeline as the generic fixtures, exercising the N/E-independent dynamic shapes
and the PyG-aware vary_fn (edge_index redrawn against the resampled node count, not the
original tensor's numeric range).

Observed result, worth being honest about: all three come back CLEAN on torch 2.14 /
torch_geometric 2.8 — GATConv's attention aggregation does NOT hit the same
scatter_reduce(include_self=False) translation bug that scatter_include_self_false
(tests/test_export.py) demonstrates directly. IMPLEMENTATION_PLAN.md hoped GAT itself
would come back DEGRADED as the headline "silent wrong answer" result; in this
environment it doesn't. The scatter_include_self_false fixture is the real demonstration
of that failure mode instead — see tests/test_export.py.
"""

import pytest

import downshift
from tests.models import gnn_gat, gnn_gcn, gnn_sage

FIXTURES = [gnn_gcn, gnn_sage, gnn_gat]


@pytest.mark.parametrize("module", FIXTURES, ids=lambda m: m.__name__.rsplit(".", 1)[-1])
def test_gnn_fixture_is_clean(module) -> None:
    model = module.make_model()
    inputs = module.make_inputs()

    verdict = downshift.check(model, inputs, k=8)

    assert verdict.status == "CLEAN", verdict.reason
    assert verdict.model_family == "pyg"
    assert verdict.numerics is not None
    assert verdict.numerics.shape_generalization


def test_gnn_verdict_survives_node_edge_count_mismatch() -> None:
    """Regression guard for the N/E-independence bug IMPLEMENTATION_PLAN.md §5.3 warns
    about: if N and E were wrongly linked to the same Dim, varying edge count alone
    (while node count stays put) would blow up during verification."""
    model = gnn_gcn.make_model()
    inputs = gnn_gcn.make_inputs(num_nodes=6, num_edges=10)

    verdict = downshift.check(model, inputs, k=8)

    assert verdict.status == "CLEAN"
