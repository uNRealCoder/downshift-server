"""`{"text": ...}` on /predict: tokenize and pad the batch for the graph, and turn a
classifier's logits into `predictions` (softmax, or sigmoid for multi-label). No transformers
import: the tokenizer comes from hf_repo.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

SOFTMAX = "softmax"
SIGMOID = "sigmoid"


@dataclass
class TextIO:
    tokenizer: Any
    max_length: int
    id2label: dict[int, str] | None = None  # set for a sequence classifier
    activation: str | None = None  # SOFTMAX or SIGMOID. None if the output is not a class score

    def encode(self, text: list[str]) -> dict[str, np.ndarray]:
        """The tokenizer outputs for a batch, padded to the longest row. A row longer than the
        model reads is a ValueError: it is never cut."""
        encoded = self.tokenizer(text)
        lengths = [len(ids) for ids in encoded["input_ids"]]
        over = {i: n for i, n in enumerate(lengths) if n > self.max_length}
        if over:
            rows = ", ".join(f"row {i}: {n}" for i, n in over.items())
            raise ValueError(
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
