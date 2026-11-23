"""Compare downshift's /predict embeddings with a plain-transformers fp32 reference."""
import json
import sys
import time
import urllib.request

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

REPO = sys.argv[1]
URL = "http://127.0.0.1:8765/predict"
QUERY = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"
texts = [
    "What is the capital of China?",
    "Explain gravity",
    "The capital of China is Beijing. It has been the political centre for centuries.",
    "Gravity is a force that attracts two bodies towards each other.",
]


def post(body):
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"content-type": "application/json"})
    t = time.perf_counter()
    with urllib.request.urlopen(req) as r:
        out = json.load(r)
    return np.array(out["outputs"]["output_0"], dtype=np.float32), time.perf_counter() - t


tok = AutoTokenizer.from_pretrained(REPO, padding_side="left")
for dtype in (torch.float32, torch.bfloat16):
    model = AutoModel.from_pretrained(REPO, dtype=dtype).eval()
    print("loaded dtype", next(model.parameters()).dtype)

    def ref(rows):
        b = tok(rows, padding=True, return_tensors="pt")
        with torch.no_grad():
            h = model(**b).last_hidden_state[:, -1].float()
        return torch.nn.functional.normalize(h, dim=-1).numpy()

    r = ref(texts)
    srv, dt = post({"text": texts})
    print(f"  batch of 4: cos(server, ref) = {np.sum(srv * r, 1).round(5)}  ({dt:.2f}s)")
    singles = np.vstack([post({"text": [t]})[0] for t in texts])
    print(f"  singles vs batch cos = {np.sum(singles * srv, 1).round(5)}")
    q = [QUERY + t for t in texts[:2]]
    rq = ref(q)
    sq, _ = post({"text": texts[:2], "prompt_name": "query"})
    print(f"  prompt_name=query cos(server, ref) = {np.sum(sq * rq, 1).round(5)}")
    print(f"  retrieval scores (query x doc), server:\n{(sq @ srv[2:].T).round(4)}")
    print(f"  retrieval scores (query x doc), ref:\n{(rq @ r[2:].T).round(4)}")
