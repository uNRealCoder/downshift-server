"""Eager CPU latency of the backbone in fp32 vs bf16."""
import sys
import time

import torch
from transformers import AutoModel

REPO = sys.argv[1]
for dtype in (torch.float32, torch.bfloat16):
    model = AutoModel.from_pretrained(REPO, dtype=dtype).eval()
    for b, s in ((1, 16), (1, 500), (1, 2000), (8, 128)):
        ids = torch.randint(0, 150000, (b, s))
        mask = torch.ones_like(ids)
        with torch.no_grad():
            model(input_ids=ids, attention_mask=mask)
            t = time.perf_counter()
            for _ in range(3):
                model(input_ids=ids, attention_mask=mask)
        print(f"{str(dtype):15} batch {b} seq {s:5}: {(time.perf_counter() - t) / 3 * 1000:8.1f} ms")
