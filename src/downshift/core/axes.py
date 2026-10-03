"""AxisFact: per dynamic axis, what the server accepts next to what verification exercised.

`served_*` is the export `Dim`'s bounds (what the request check enforces); `sampled_*` is the
range of sizes verify() actually ran through both backends, None when verify never ran.
Torch-free so the CLI and the schemas can import it.
"""

from collections.abc import Sequence
from dataclasses import dataclass

# input -> axis -> (Dim name, served min, served max)
AxisBounds = dict[str, dict[int, tuple[str, int, int]]]


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
        return {
            "input": self.input,
            "axis": self.axis,
            "name": self.name,
            "served_min": self.served_min,
            "served_max": self.served_max,
            "sampled_min": self.sampled_min,
            "sampled_max": self.sampled_max,
        }

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
