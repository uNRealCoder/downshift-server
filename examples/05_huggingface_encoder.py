# %% [markdown]
# # Lesson 5: Hugging Face encoders
#
# Requires the `hf` extra: `pip install -e ".[hf]"`.
#
# The `hf` adapter builds `input_ids` / `attention_mask` dummies straight from the
# model's config — no `optimum` dependency. This lesson uses a randomly initialized,
# two-layer BERT so it runs without downloading anything; a real checkpoint works the
# same way, just pass a Hugging Face repo id (`downshift check bert-base-uncased`) or a
# loaded model.

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from transformers import BertConfig, BertModel

import downshift

config = BertConfig(
    vocab_size=100,
    hidden_size=16,
    num_hidden_layers=2,
    num_attention_heads=2,
    intermediate_size=32,
    max_position_embeddings=64,
)
model = BertModel(config).eval()

# %% [markdown]
# ## Letting the adapter guess the inputs
#
# No `example_inputs` here — the adapter reads `vocab_size` off the config and builds a
# small `input_ids` / `attention_mask` pair itself.

# %%
verdict = downshift.check(model)
print("status: ", verdict.status)
print("family: ", verdict.model_family)
print("inputs: ", verdict.input_names)
print("dynamic:", verdict.dynamic_dims)

# %% [markdown]
# `dynamic_dims` marks both the batch axis and the sequence axis dynamic for both
# tensors, bounded by `max_position_embeddings` — export would fail its own guards past
# that bound, so the adapter reads it out of the config rather than guessing.

# %%
print("numerics:", verdict.numerics.max_abs_err, "over", verdict.numerics.samples_tested, "samples")
print("shape_generalization:", verdict.numerics.shape_generalization)

# %% [markdown]
# ## Supplying real input_ids
#
# Same call, with explicit tensors this time — the shapes you pass are always honored
# over anything the adapter would have guessed.

# %%
input_ids = torch.randint(0, config.vocab_size, (2, 8))
attention_mask = torch.ones_like(input_ids)

verdict = downshift.check(model, (input_ids, attention_mask))
print("status:", verdict.status, "| reason:", verdict.reason)

# %% [markdown]
# ## A real checkpoint
#
# Outside this tutorial, the same command works against a downloaded model:
#
# ```bash
# downshift check bert-base-uncased
# downshift serve bert-base-uncased --port 8000
# ```
#
# `downshift` calls `AutoModel.from_pretrained` under the hood — encoder-only models
# only in this release; a causal LM path (`onnxruntime-genai`, KV cache, streaming) isn't
# built yet.

# %% [markdown]
# Next: [lesson 6](06_custom_adapter.py) teaches `downshift` a model family of its own.
