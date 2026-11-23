"""What a user does by hand: export the Qwen3 embedding backbone to ONNX in fp32 (ORT CPU has no bf16)."""
import sys
import time

import torch
from transformers import AutoModel

repo, out = sys.argv[1], sys.argv[2]
model = AutoModel.from_pretrained(repo, dtype=torch.float32).eval()
model.config.use_cache = False


class Backbone(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, input_ids, attention_mask):
        return self.m(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).last_hidden_state


ids = torch.randint(0, 1000, (2, 16))
mask = torch.ones_like(ids)
batch, seq = torch.export.Dim("batch"), torch.export.Dim("seq")
t = time.perf_counter()
torch.onnx.export(
    Backbone(model), (ids, mask), out, dynamo=True, external_data=True,
    input_names=["input_ids", "attention_mask"], output_names=["last_hidden_state"],
    dynamic_shapes={"input_ids": {0: batch, 1: seq}, "attention_mask": {0: batch, 1: seq}},
)
print(f"exported in {time.perf_counter() - t:.0f} s")
