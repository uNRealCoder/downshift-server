"""Graph batches for /predict/graph: many graphs become one disjoint-union graph (one
inference), and the outputs are cut again for each graph by what their axis 0 follows (nodes
or edges, ExportVerdict.output_axes).
"""

import logging
from collections.abc import Sequence
from typing import Any

import numpy as np

logger = logging.getLogger("downshift.serve")

NODE, EDGE, FIXED, UNKNOWN = "node", "edge", "fixed", "unknown"

GRAPH_TENSORS = ("x", "edge_index", "edge_attr")
# The inputs that make a model "graph-shaped", so /predict/graph serves it.
GRAPH_INPUTS = frozenset(GRAPH_TENSORS[:2])
# The count vectors for each graph that a batched safetensors body adds.
GRAPH_COUNTS = ("num_nodes", "num_edges")

_FIXED_HELP = (
    "has a fixed size (its axis 0 follows neither nodes nor edges), so it can't be split per "
    "graph. Downshift does not support graph-level batching (a model that takes a `batch` "
    "vector). Send one graph per request"
)
_UNKNOWN_HELP = (
    "could not be classified as per-node or per-edge when the model was verified, so it can't "
    "be split per graph. Send one graph per request"
)


def eager_output_axes(prepared: Any, samples: int = 4) -> list[str]:
    """The output kinds of a PyG model that was not verified, from a few eager passes over
    the adapter's samples. [] if they cannot be found."""
    from downshift.adapters.base import Family
    from downshift.core.axes import classify_outputs

    if prepared.family != Family.pyg or prepared.vary_fn is None:
        return []
    try:
        import torch

        from downshift.core.verify import as_tensor_list

        names = list(prepared.input_names)
        sample_shapes: list[list[tuple[int, ...]]] = []
        output_shapes: list[list[tuple[int, ...]]] = []
        prepared.model.eval()
        with torch.random.fork_rng(devices=[]), torch.inference_mode():
            torch.manual_seed(0)
            for i in range(samples):
                sample = prepared.vary_fn(i)
                outs = as_tensor_list(prepared.model(*sample))
                sample_shapes.append([tuple(t.shape) for t in sample])
                output_shapes.append([tuple(t.shape) for t in outs])
        return classify_outputs(
            sample_shapes, output_shapes, names.index("x"), names.index("edge_index")
        )
    except Exception:  # noqa: BLE001 - an unclassified model serves, but cannot batch
        logger.debug("could not classify the model's outputs", exc_info=True)
        return []


def index_range_violation(name: str, arr: np.ndarray | None, upper: int, what: str) -> str | None:
    """Why index tensor `arr` has a value outside [0, upper), or None. A backend would wrap a
    negative index or read out of bounds instead of refusing it."""
    if arr is None or arr.size == 0:
        return None
    lo, hi = int(arr.min()), int(arr.max())
    if lo >= 0 and hi < upper:
        return None
    return f"{name} contains {lo if lo < 0 else hi}, outside the {what} [0, {upper})"


def edge_index_violation(feeds: dict[str, np.ndarray]) -> str | None:
    """edge_index must index into x's node dimension."""
    x = feeds.get("x")
    if x is None:
        return None
    return index_range_violation("edge_index", feeds.get("edge_index"), x.shape[0], "node range")


def batch_graphs(
    items: Sequence[dict[str, np.ndarray]],
) -> tuple[dict[str, np.ndarray], list[int], list[int]]:
    """Join graphs ({x, edge_index, edge_attr?}, local node IDs) into one feed dict with
    offset node IDs. Returns (feeds, node_counts, edge_counts). A ValueError names the graph."""
    node_counts: list[int] = []
    edge_counts: list[int] = []
    with_attr = "edge_attr" in items[0]
    for i, item in enumerate(items):
        where = f"graphs[{i}]"
        x, edge_index = item["x"], item["edge_index"]
        if x.ndim < 1 or x.shape[0] < 1:
            raise ValueError(f"{where}: x must have at least one node")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError(
                f"{where}: edge_index must have shape [2, E], got {list(edge_index.shape)}"
            )
        if x.shape[1:] != items[0]["x"].shape[1:]:
            raise ValueError(
                f"{where}: x has trailing shape {list(x.shape[1:])}; graphs[0] has "
                f"{list(items[0]['x'].shape[1:])}"
            )
        violation = edge_index_violation({"x": x, "edge_index": edge_index})
        if violation is not None:
            raise ValueError(f"{where}: {violation}")
        if ("edge_attr" in item) != with_attr:
            raise ValueError(f"{where}: edge_attr must be given for every graph or for none")
        if with_attr:
            attr = item["edge_attr"]
            if attr.ndim < 1 or attr.shape[0] != edge_index.shape[1]:
                raise ValueError(
                    f"{where}: edge_attr has {attr.shape[0] if attr.ndim else 0} rows; "
                    f"edge_index has {edge_index.shape[1]} edges"
                )
            if attr.shape[1:] != items[0]["edge_attr"].shape[1:]:
                raise ValueError(
                    f"{where}: edge_attr has trailing shape {list(attr.shape[1:])}; graphs[0] "
                    f"has {list(items[0]['edge_attr'].shape[1:])}"
                )
        node_counts.append(int(x.shape[0]))
        edge_counts.append(int(edge_index.shape[1]))

    offsets = np.concatenate(([0], np.cumsum(node_counts[:-1]))).astype(np.int64)
    edge_index = np.concatenate([item["edge_index"] for item in items], axis=1)
    edge_index += np.repeat(offsets, edge_counts).astype(edge_index.dtype)  # a new array
    feeds = {"x": np.concatenate([item["x"] for item in items]), "edge_index": edge_index}
    if with_attr:
        feeds["edge_attr"] = np.concatenate([item["edge_attr"] for item in items])
    return feeds, node_counts, edge_counts


def binary_batch(
    tensors: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], list[int], list[int]]:
    """batch_graphs() for a safetensors batch: joined tensors with local IDs plus the int64
    vectors `num_nodes` and `num_edges`."""
    if "num_nodes" not in tensors or "num_edges" not in tensors:
        raise ValueError("a batched body needs both 'num_nodes' and 'num_edges'")
    num_nodes, num_edges = tensors["num_nodes"], tensors["num_edges"]
    for name, counts in (("num_nodes", num_nodes), ("num_edges", num_edges)):
        if counts.ndim != 1:
            raise ValueError(f"{name} must be a 1-D vector, got shape {list(counts.shape)}")
        if counts.dtype != np.int64:
            raise ValueError(f"{name} is {counts.dtype.name}. It must be int64")
    if num_nodes.shape != num_edges.shape:
        raise ValueError(
            f"num_nodes has {num_nodes.shape[0]} entries and num_edges {num_edges.shape[0]}. "
            "They must match"
        )
    if num_nodes.shape[0] == 0:
        raise ValueError("num_nodes is empty. A batch needs at least one graph")
    if int(num_nodes.min()) < 1:
        raise ValueError("num_nodes must be at least 1 for every graph")
    if int(num_edges.min()) < 0:
        raise ValueError("num_edges must not be negative")
    x, edge_index = tensors["x"], tensors["edge_index"]
    total_nodes, total_edges = int(num_nodes.sum()), int(num_edges.sum())
    if x.ndim < 1 or total_nodes != x.shape[0]:
        raise ValueError(
            f"sum(num_nodes) is {total_nodes}. x has {x.shape[0] if x.ndim else 0} rows"
        )
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(f"edge_index must have shape [2, E], got {list(edge_index.shape)}")
    if total_edges != edge_index.shape[1]:
        raise ValueError(
            f"sum(num_edges) is {total_edges}. edge_index has {edge_index.shape[1]} columns"
        )
    attr = tensors.get("edge_attr")
    if attr is not None and (attr.ndim < 1 or attr.shape[0] != total_edges):
        raise ValueError(
            f"sum(num_edges) is {total_edges}. edge_attr has "
            f"{attr.shape[0] if attr.ndim else 0} rows"
        )

    # Range-check the local IDs and offset them to batch IDs in one vectorized pass.
    edge_nodes = np.repeat(num_nodes, num_edges)
    bad = (edge_index < 0) | (edge_index >= edge_nodes)
    if bad.any():
        column = int(np.argmax(bad.any(axis=0)))
        graph = int(np.searchsorted(np.cumsum(num_edges), column, side="right"))
        value = int(edge_index[:, column][bad[:, column]][0])
        raise ValueError(
            f"graphs[{graph}]: edge_index contains {value}, outside the node range "
            f"[0, {int(num_nodes[graph])})"
        )
    starts = np.repeat(np.cumsum(num_nodes) - num_nodes, num_edges)
    feeds = {"x": x, "edge_index": edge_index + starts.astype(edge_index.dtype)}
    if attr is not None:
        feeds["edge_attr"] = attr
    return feeds, num_nodes.tolist(), num_edges.tolist()


def split_refusal(kinds: Sequence[str], names: Sequence[str], graph_count: int) -> str | None:
    """Why a batch of `graph_count` graphs cannot be split, or None. One graph always can."""
    if graph_count <= 1:
        return None
    for i in range(max(len(kinds), len(names))):
        kind = kinds[i] if i < len(kinds) else UNKNOWN
        if kind in (NODE, EDGE):
            continue
        name = names[i] if i < len(names) else f"output_{i}"
        return f"output {name!r} {_FIXED_HELP if kind == FIXED else _UNKNOWN_HELP}"
    return None


def split_outputs(
    outputs: dict[str, np.ndarray],
    kinds: Sequence[str],
    node_counts: Sequence[int],
    edge_counts: Sequence[int],
) -> list[dict[str, np.ndarray]]:
    """One outputs dict for each graph, in request order, cut by node or edge counts."""
    if len(node_counts) == 1:
        return [outputs]
    refusal = split_refusal(kinds, list(outputs), len(node_counts))
    if refusal is not None:
        raise ValueError(refusal)
    per_graph: list[dict[str, np.ndarray]] = [{} for _ in node_counts]
    for i, (name, arr) in enumerate(outputs.items()):
        counts = node_counts if kinds[i] == NODE else edge_counts
        if arr.ndim < 1 or arr.shape[0] != sum(counts):
            raise ValueError(
                f"output {name!r} has {arr.shape[0] if arr.ndim else 0} rows on axis 0; "
                f"expected {sum(counts)} to split per graph"
            )
        bounds = np.concatenate(([0], np.cumsum(counts))).tolist()
        for g, graph in enumerate(per_graph):
            graph[name] = arr[bounds[g] : bounds[g + 1]]
    return per_graph
