# Compatibility matrix

Generated 2026-10-12 by `scripts/gen_matrix.py` with torch 2.14.1+cpu, onnx 1.23.2, onnxruntime 1.31.0, torch_geometric 2.8.1, transformers 5.19.0.
Downshift checked each model with `downshift.check(..., samples=8)`.

**CLEAN**: The model exports, matches PyTorch on every sample, and works on shapes that the exporter did not see. Served by ONNX Runtime. **DEGRADED**: The model exports without error, but the numbers differ from PyTorch by more than the tolerance on at least one sample. Served by eager PyTorch, unless you set `--force-onnx`. **FAILED**: The model does not export, or it exports but ONNX Runtime cannot load or run the graph. Served by eager PyTorch in both cases. **UNVERIFIED**: A `.onnx` file with no reference model. Downshift never checked the numbers. Served by ONNX Runtime and labelled as unverified.

| Model | Hazard | Family | Export | Capture | Numerics | Shape-general | Backend |
|---|---|---|---|---|---|---|---|
| `bf16_weights` | bfloat16 weights: ONNX Runtime CPU has no bf16 Gemm kernel | generic | FAILED | strict=False | — | — | torch |
| `broken_factory` | Not an export hazard fixture: raises as soon as it is instantiated | — | skipped (RuntimeError) | — | — | — | — |
| `clean_mlp` | Control fixture: no export hazards | generic | CLEAN | strict=False | 1.8e-07 | ✓ | onnxruntime |
| `custom_autograd` | custom autograd.Function with no symbolic override | generic | CLEAN | strict=False | 1.5e-07 | ✓ | onnxruntime |
| `data_dependent_branch` | data-dependent control flow | generic | FAILED | — | — | — | torch |
| `dict_input` | dataclass container input | generic | CLEAN | strict=False | 1.2e-07 | ✓ | onnxruntime |
| `dropout_model` | stochastic layer | generic | CLEAN | strict=False | 2.4e-07 | ✓ | onnxruntime |
| `dynamic_batch_cnn` | batch-dim generalization | generic | CLEAN | strict=False | 3.0e-08 | ✓ | onnxruntime |
| `gnn_gat` | GNN fixture: 3-layer GAT node classifier | pyg | CLEAN | strict=False | 2.4e-07 | ✓ | onnxruntime |
| `gnn_gcn` | GNN fixture: 2-layer GCN node classifier | pyg | CLEAN | strict=False | 3.6e-07 | ✓ | onnxruntime |
| `gnn_sage` | GNN fixture: 2-layer GraphSAGE node classifier | pyg | CLEAN | strict=False | 2.4e-07 | ✓ | onnxruntime |
| `scatter_include_self_false` | scatter_reduce(include_self=False) has no faithful ONNX translation | generic | DEGRADED | strict=False | 2.2e+00 | — | torch |
| `tied_weights` | tied embedding/output weight (GPT-2/OPT-style) | generic | CLEAN | strict=False | 1.9e-06 | ✓ | onnxruntime |
| `tiny_bert` | HF fixture: a BERT encoder with two layers and random initialization | hf | CLEAN | strict=False | 7.2e-07 | ✓ | onnxruntime |

## How to read this

CLEAN means that the ONNX graph agrees with PyTorch. It does not mean that the graph is fast. DEGRADED means that the graph runs, but the numbers are wrong on at least one of the K samples. The Numerics column shows the largest absolute error. Downshift still serves FAILED models, through eager PyTorch behind the same endpoint. UNVERIFIED does not appear here, because every model in the corpus has a PyTorch reference.

<!-- real-models:start -->
## Tested real models

Generated 2026-10-12 by `scripts/gen_matrix.py` with torch 2.14.1+cpu, onnx 1.23.2, onnxruntime 1.31.0, torch_geometric 2.8.1, transformers 5.19.0.
Downshift ran `downshift check --json` on each model, on the CPU.
The Rows column is the number of verification samples. The script is in the repository: to run it again, use `python scripts/check_real_models.py`.

| Model | Family | Verdict | Rows | Max abs err | Min cosine | Backend | Seconds |
|---|---|---|---|---|---|---|---|
| `sentence-transformers/all-MiniLM-L6-v2` | hf | CLEAN | 8 | 2.4e-07 | 1.000000 | onnxruntime | 19.2 |
| `BAAI/bge-small-en-v1.5` | hf | CLEAN | 8 | 3.0e-07 | 1.000000 | onnxruntime | 21.9 |
| `intfloat/e5-small-v2` | hf | CLEAN | 8 | 2.4e-07 | 1.000000 | onnxruntime | 21.8 |
| `distilbert/distilbert-base-uncased-finetuned-sst-2-english` | hf | CLEAN | 8 | 1.3e-05 | — | onnxruntime | 20.4 |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | hf | CLEAN | 8 | 6.7e-06 | — | onnxruntime | 18.6 |
| `meta-llama/Prompt-Guard-86M` | hf | CLEAN | 8 | 6.0e-05 | — | onnxruntime | 39.0 |
| `torchvision/resnet18` | generic | CLEAN | 8 | 3.8e-06 | — | onnxruntime | 10.6 |
| `pyg/GCN (tests/models/gnn_gcn)` | pyg | CLEAN | 8 | 1.2e-07 | — | onnxruntime | 17.6 |
<!-- real-models:end -->
