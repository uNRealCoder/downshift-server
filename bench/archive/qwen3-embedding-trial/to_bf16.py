"""Option 2 trial: save a bf16 copy of an fp32 embedding repo, and measure what bf16 does to its vectors.

usage: python to_bf16.py SRC_REPO DST_REPO mean|lasttoken
"""
import shutil
import sys
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

src, dst, pooling = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
dst.mkdir(parents=True, exist_ok=True)
for item in src.iterdir():  # tokenizer, pooling and sentence-transformers files, not the weights
    if item.name.startswith(("tokenizer", "vocab", "merges", "special_tokens", "modules", "config_sentence",
                             "sentence_bert")):
        shutil.copy(item, dst / item.name)
    elif item.is_dir() and item.name[0].isdigit():
        shutil.copytree(item, dst / item.name, dirs_exist_ok=True)

fp32 = AutoModel.from_pretrained(src, dtype=torch.float32).eval()
bf16 = AutoModel.from_pretrained(src, dtype=torch.bfloat16).eval()
bf16.save_pretrained(dst, safe_serialization=True)
print("saved", dst, "config dtype:", bf16.config.dtype)

tok = AutoTokenizer.from_pretrained(src, padding_side="left" if pooling == "lasttoken" else "right")
sentences = [
    "How do I reset my password?", "Steps to change the account password", "Best pizza in Naples",
    "Where can I eat good pizza in Italy?", "The stock market fell sharply today", "Shares dropped on Wall Street",
    "Photosynthesis converts light into chemical energy", "How plants make food from sunlight",
    "A dog chased the ball across the park", "Puppies love playing fetch outside",
    "Python is a popular programming language", "Learning to code in Python for beginners",
    "The capital of France is Paris", "Paris is France's largest city and capital",
    "Heavy rain is expected tomorrow", "Weather forecast: storms on Thursday",
]


def embed(model):
    enc = tok(sentences, padding=True, return_tensors="pt")
    with torch.no_grad():
        h = model(**enc).last_hidden_state.float()
    if pooling == "lasttoken":
        v = h[:, -1]
    else:
        m = enc["attention_mask"].unsqueeze(-1).float()
        v = (h * m).sum(1) / m.sum(1)
    return torch.nn.functional.normalize(v, dim=-1)


a, b = embed(fp32), embed(bf16)
cos = (a * b).sum(-1)
sa, sb = a @ a.T, b @ b.T
# topk(2)[1]: the nearest other sentence (the first is the sentence itself)
same_top1 = sum(int(sa[i].topk(2).indices[1] == sb[i].topk(2).indices[1]) for i in range(len(sentences)))
print(f"cos(fp32, bf16) per sentence: min {cos.min():.5f}  mean {cos.mean():.5f}")
print(f"max |similarity change| over all pairs: {(sa - sb).abs().max():.4f}")
print(f"nearest neighbour unchanged for {same_top1}/{len(sentences)} sentences")
