"""Entities: adapters."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from torch import nn


class Family(StrEnum):
    """The built-in adapter names. A custom adapter's `name` is a plain string."""

    hf = "hf"
    pyg = "pyg"
    generic = "generic"


VaryFn = Callable[[int], tuple]


@dataclass
class Prepared:
    model: nn.Module  # the module that is ready for export. forward takes flat tensors
    inputs: tuple  # flat example inputs, one for each entry of input_names
    input_names: tuple[str, ...]
    dynamic_shapes: tuple  # for each input: {axis: torch.export.Dim} or None
    vary_fn: VaryFn | None  # sample i -> inputs. None means: use the shared-axis-0 default
    family: str  # the `name` of the adapter. ExportVerdict.model_family reports it
    embedding: bool = False  # True: the first output is a pooled embedding, one row for each text

    @property
    def dynamic_dims(self) -> dict[str, list[int]]:
        return {
            name: sorted(spec)
            for name, spec in zip(self.input_names, self.dynamic_shapes, strict=True)
            if spec
        }


class PoolingChoice(StrEnum):
    """--pooling: how the token vectors of an encoder-only repo become one embedding for each
    text. "none" serves the token vectors as they are. It ignores each recipe that the repo
    declares."""

    mean = "mean"
    cls = "cls"
    maximum = "max"  # not `max`, because that would hide the builtin inside the class body
    mean_sqrt_len = "mean_sqrt_len"
    lasttoken = "lasttoken"
    weightedmean = "weightedmean"
    none = "none"


@dataclass(frozen=True)
class EmbeddingRecipe:
    pooling: str
    normalize: bool
    max_seq_length: int | None  # the training length of the model. None if the repo does not say
    origin: str  # "modules.json", or the flag that overrode it
    prompts: dict[str, str] = field(default_factory=dict)  # name -> text for the start of a row
    default_prompt: str | None = None  # the name that downshift applies if a request names none

    def describe(self) -> str:
        return f"{self.pooling} pooling" + (", L2-normalised" if self.normalize else "")

    def info(self, dimension: int | None) -> dict[str, Any]:
        """The recipe as /schema and the safetensors metadata report it (without prompts)."""
        return {
            "pooling": self.pooling,
            "normalized": self.normalize,
            "dimension": dimension,
            "max_seq_length": self.max_seq_length,
            "from": self.origin,
        }


SOFTMAX = "softmax"


class TextTooLongError(ValueError):
    """A text row has more tokens than the model reads."""


@dataclass
class TextIO:
    tokenizer: Any
    max_length: int
    id2label: dict[int, str] | None = None  # set for a sequence classifier
    activation: str | None = None  # SOFTMAX or SIGMOID. None if the output is not a class score

    def encode(self, text: list[str]) -> dict[str, np.ndarray]:
        """The tokenizer outputs for a batch, padded to the longest row. A row longer than the
        model reads is a TextTooLongError: it is never cut."""
        encoded = self.tokenizer(text)
        lengths = [len(ids) for ids in encoded["input_ids"]]
        over = {i: n for i, n in enumerate(lengths) if n > self.max_length}
        if over:
            rows = ", ".join(f"row {i}: {n}" for i, n in over.items())
            raise TextTooLongError(
                f"text longer than this model reads ({self.max_length} tokens, a limit from the "
                f"model files) is refused and not cut. Token counts: {rows}. Send shorter "
                "text, or split it into more rows"
            )
        padded = self.tokenizer.pad(encoded, return_tensors="np")
        return {name: np.asarray(value) for name, value in padded.items()}

    def predictions(self, logits: np.ndarray) -> list[dict[str, Any]] | None:
        """One {label, score, probabilities} for each row, or None if this is not a classifier
        output ([batch, num_labels])."""
        if self.activation is None or self.id2label is None:
            return None
        if logits.ndim != 2 or logits.shape[1] != len(self.id2label):
            return None
        scores = logits.astype(np.float64)
        if self.activation == SOFTMAX:
            scores = np.exp(scores - scores.max(axis=1, keepdims=True))
            scores /= scores.sum(axis=1, keepdims=True)
        else:
            scores = 1.0 / (1.0 + np.exp(-scores))
        rows = []
        for row in scores:
            top = int(row.argmax())
            rows.append(
                {
                    "label": self.id2label[top],
                    "score": float(row[top]),
                    "probabilities": {self.id2label[i]: float(p) for i, p in enumerate(row)},
                }
            )
        return rows
