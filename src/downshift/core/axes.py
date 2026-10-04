"""AxisFact: per dynamic axis, what the server accepts next to what verification exercised.

`served_*` is the export `Dim`'s bounds (what the request check enforces); `sampled_*` is the
range of sizes verify() actually ran through both backends, None when verify never ran.
Torch-free so the CLI and the schemas can import it.
"""

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import NamedTuple


class DimBound(NamedTuple):
    """One dynamic axis's Dim name and the (min, max) torch.export traced it for (U2)."""

    name: str
    min: int
    max: int


# input -> axis -> its DimBound
AxisBounds = dict[str, dict[int, DimBound]]


def axis_bounds_to_json(bounds: AxisBounds) -> dict[str, list[list]]:
    """[[axis, name, min, max], ...] per input: the form the export cache's serving.json and
    the `serve --workers N` handoff carry, for a process that has no Prepared to derive them."""
    return {
        name: [[axis, b.name, b.min, b.max] for axis, b in axes.items()]
        for name, axes in bounds.items()
    }


def axis_bounds_from_json(raw: dict[str, list[list]]) -> AxisBounds:
    return {
        name: {int(axis): DimBound(str(dim), int(lo), int(hi)) for axis, dim, lo, hi in rows}
        for name, rows in raw.items()
    }


@dataclass(frozen=True)
class AxisFact:
    input: str
    axis: int
    name: str
    served_min: int
    served_max: int
    sampled_min: int | None
    sampled_max: int | None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "AxisFact":
        sampled_min, sampled_max = data.get("sampled_min"), data.get("sampled_max")
        return cls(
            input=str(data["input"]),
            axis=int(data["axis"]),
            name=str(data["name"]),
            served_min=int(data["served_min"]),
            served_max=int(data["served_max"]),
            sampled_min=None if sampled_min is None else int(sampled_min),
            sampled_max=None if sampled_max is None else int(sampled_max),
        )


def classify_outputs(
    sample_shapes: Sequence[Sequence[Sequence[int]]],
    output_shapes: Sequence[Sequence[Sequence[int]]],
    x_index: int,
    edge_index_index: int,
) -> list[str]:
    """Per output: "node" (axis 0 follows x's node count across samples), "edge" (follows
    edge_index's edge count), "fixed" (axis 0 never changes) or "unknown". Telling node from
    edge needs at least two samples where N != E; with fewer, everything is "unknown"."""
    if not output_shapes or len(sample_shapes) != len(output_shapes):
        return []
    n_outputs = len(output_shapes[0])
    counts = [(shapes[x_index][0], shapes[edge_index_index][1]) for shapes in sample_shapes]
    if sum(n != e for n, e in counts) < 2:
        return ["unknown"] * n_outputs
    kinds: list[str] = []
    for out in range(n_outputs):
        if any(len(outs) <= out or not outs[out] for outs in output_shapes):
            kinds.append("unknown")
            continue
        sizes = [outs[out][0] for outs in output_shapes]
        tracks_node = all(size == n for size, (n, _) in zip(sizes, counts, strict=True))
        tracks_edge = all(size == e for size, (_, e) in zip(sizes, counts, strict=True))
        if tracks_node != tracks_edge:
            kinds.append("node" if tracks_node else "edge")
        elif len(set(sizes)) == 1:
            kinds.append("fixed")
        else:
            kinds.append("unknown")
    return kinds


def axis_facts(
    bounds: AxisBounds,
    input_names: Sequence[str],
    sample_shapes: Sequence[Sequence[Sequence[int]]] | None,
) -> list[AxisFact]:
    """One fact per dynamic axis, in input order. `sample_shapes` is
    NumericsReport.sample_shapes ([sample][input] -> shape); None or empty means verify
    didn't run, so every sampled_* is None."""
    facts: list[AxisFact] = []
    for index, name in enumerate(input_names):
        for axis, (dim_name, served_min, served_max) in sorted(bounds.get(name, {}).items()):
            sizes = [
                shapes[index][axis]
                for shapes in sample_shapes or ()
                if index < len(shapes) and axis < len(shapes[index])
            ]
            facts.append(
                AxisFact(
                    input=name,
                    axis=axis,
                    name=dim_name,
                    served_min=served_min,
                    served_max=served_max,
                    sampled_min=min(sizes) if sizes else None,
                    sampled_max=max(sizes) if sizes else None,
                )
            )
    return facts
