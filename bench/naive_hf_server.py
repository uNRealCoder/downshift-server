"""The FastAPI + transformers server people write by hand. `python -m bench.naive_hf_server <repo_dir> <port> <sync|async>`

Written the charitable way, like bench/servers.py: plain dict body, no pydantic, no
response_model, truncation instead of a 400, so it does strictly less work per request than
downshift. Embedding repos get mean pooling + L2 normalise (what all-MiniLM-L6-v2 declares);
classifier repos get softmax + argmax label.
"""

from __future__ import annotations

import sys
import warnings

warnings.filterwarnings("ignore")

import torch  # noqa: E402
import uvicorn  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from transformers import (  # noqa: E402
    AutoConfig,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


def build(repo: str, mode: str) -> FastAPI:
    config = AutoConfig.from_pretrained(repo, local_files_only=True)
    is_classifier = any(
        a.endswith("ForSequenceClassification") for a in (config.architectures or [])
    )
    cls = AutoModelForSequenceClassification if is_classifier else AutoModel
    model = cls.from_pretrained(repo, local_files_only=True).eval()
    tok = AutoTokenizer.from_pretrained(repo, local_files_only=True)
    max_len = 256 if not is_classifier else 512

    def run(text) -> dict:
        texts = [text] if isinstance(text, str) else text
        enc = tok(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        with torch.inference_mode():
            out = model(**enc)
        if is_classifier:
            probs = torch.softmax(out.logits, dim=-1)
            labels = [config.id2label[int(i)] for i in probs.argmax(-1)]
            return {
                "outputs": {"output_0": out.logits.tolist()},
                "predictions": [
                    {"label": lab, "score": float(p.max())}
                    for lab, p in zip(labels, probs, strict=True)
                ],
            }
        mask = enc["attention_mask"].unsqueeze(-1).to(out.last_hidden_state.dtype)
        pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        return {"outputs": {"output_0": pooled.tolist()}}

    for _ in range(3):
        run("warm up")
    app = FastAPI()

    @app.get("/health")
    async def health() -> dict:  # the usual tutorial form; runs on the event loop
        return {"ok": True}

    if mode == "async":

        @app.post("/predict")
        async def predict(body: dict) -> dict:  # blocks the event loop
            return run(body["text"])
    else:

        @app.post("/predict")
        def predict(body: dict) -> dict:
            return run(body["text"])

    return app


if __name__ == "__main__":
    repo, port, mode = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    uvicorn.run(
        build(repo, mode), host="127.0.0.1", port=port, log_level="warning", access_log=False
    )
