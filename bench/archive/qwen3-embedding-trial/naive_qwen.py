"""The FastAPI server people write by hand for Qwen3-Embedding, following the model card.

usage: python naive_qwen.py REPO PORT torch|onnx sync|async [model.onnx]
Last-token pooling with left padding, L2 normalise, the repo's query prompt by name, no truncation.
"""
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import uvicorn  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from transformers import AutoModel, AutoTokenizer  # noqa: E402

repo, port, backend, mode = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
PROMPTS = {"query": "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"}
tok = AutoTokenizer.from_pretrained(repo, padding_side="left")

if backend == "torch":
    model = AutoModel.from_pretrained(repo).eval()

    def embed(enc):
        with torch.inference_mode():
            h = model(**enc).last_hidden_state[:, -1].float()
        return torch.nn.functional.normalize(h, dim=-1).tolist()
else:
    import onnxruntime as ort

    sess = ort.InferenceSession(sys.argv[5], providers=["CPUExecutionProvider"])

    def embed(enc):
        h = sess.run(None, {k: enc[k].numpy() for k in ("input_ids", "attention_mask")})[0][:, -1]
        return (h / np.linalg.norm(h, axis=-1, keepdims=True)).tolist()


def run(body: dict) -> dict:
    texts = body["text"]
    prefix = PROMPTS.get(body.get("prompt_name"), "")
    enc = tok([prefix + t for t in texts], padding=True, return_tensors="pt")
    return {"outputs": {"output_0": embed(enc)}}


for _ in range(3):
    run({"text": ["warm up"]})
app = FastAPI()


@app.get("/ready")
async def ready() -> dict:
    return {"ready": True}


if mode == "async":
    @app.post("/predict")
    async def predict(body: dict) -> dict:
        return run(body)
else:
    @app.post("/predict")
    def predict(body: dict) -> dict:
        return run(body)

uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
