"""Text in, class probabilities out, for a downloaded Hugging Face repo.

`/predict` normally takes tensors. When the served directory also holds tokenizer files,
`{"text": ...}` is accepted too: the server tokenizes, truncates to the model's own limit and
pads the batch, then feeds the same graph the tensor path does. A sequence classifier also
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

    def encode(self, text: list[str]) -> tuple[dict[str, np.ndarray], list[bool]]:
        """(tokenizer outputs, per-row truncated flags) for a batch of texts.

        Rows longer than the model's limit are cut, not rejected, and reported back: a
        classifier that quietly reads only the head of a long input is a hazard the caller
        should be told about. The caller picks the outputs the graph actually takes.
        """
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=True,
            return_tensors="np",
        )
        if encoded.encodings is not None:
            # Fast tokenizer: one pass. `overflowing` holds the cut-off remainder as its own
            # Encoding, non-empty exactly when this row was truncated (checked against
            # transformers 5.17 without return_overflowing_tokens=True: it is populated).
            truncated = [bool(e.overflowing) for e in encoded.encodings]
        else:
            # Slow tokenizer: no Encoding objects to inspect, so measure with a second pass.
            # verbose=False: this call only measures. Without it transformers warns that the
            # text "will result in indexing errors", which is not true of the encoded above.
            full = self.tokenizer(text, truncation=False, padding=False, verbose=False)
            truncated = [len(ids) > self.max_length for ids in full["input_ids"]]
        return {name: np.asarray(value) for name, value in encoded.items()}, truncated

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
