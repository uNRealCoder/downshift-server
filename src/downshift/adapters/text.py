"""`{"text": ...}` on /predict: tokenize and pad the batch for the graph, and turn a
classifier's logits into `predictions` (softmax, or sigmoid for multi-label). No transformers
import: the tokenizer comes from hf_repo.
"""

SIGMOID = "sigmoid"
