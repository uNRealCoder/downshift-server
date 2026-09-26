"""Text in, class probabilities out, for a downloaded Hugging Face repo.

`/predict` normally takes tensors. When the served directory also holds tokenizer files,
`{"text": ...}` is accepted too: the server tokenizes and pads the batch (refusing a row longer
than the model's own limit), then feeds the same graph the tensor path does. A sequence classifier also
gets a `predictions` block (softmax, or sigmoid for a multi-label config) next to the raw
logits. Nothing here imports transformers: the tokenizer is whatever object the hf adapter
loaded, used through its call signature.

This is the contract between the hf adapter, which builds a TextIO, and the server, which
uses it; it lives with the adapters so they do not depend on the serve package.
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
    activation: str | None = None  # SOFTMAX or SIGMOID; None when the output is not a class score

    def encode(self, text: list[str]) -> dict[str, np.ndarray]:
        """Tokenizer outputs for a batch of texts, padded to the longest row; the caller picks
        the ones the graph takes. A row longer than the model can read is refused with a
        ValueError, never cut: the only cap on input size is the request body limit.
        """
        encoded = self.tokenizer(text)
        lengths = [len(ids) for ids in encoded["input_ids"]]
        over = {i: n for i, n in enumerate(lengths) if n > self.max_length}
        if over:
            rows = ", ".join(f"row {i}: {n}" for i, n in over.items())
            raise ValueError(
                f"text longer than this model reads ({self.max_length} tokens, from the "
                f"model's own files) is refused rather than cut; token counts {rows}"
            )
        padded = self.tokenizer.pad(encoded, return_tensors="np")
        return {name: np.asarray(value) for name, value in padded.items()}

    def predictions(self, logits: np.ndarray) -> list[dict[str, Any]] | None:
        """One {label, score, probabilities} per row, or None when this is not a
        classifier output ([batch, num_labels] with labels to name)."""
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
