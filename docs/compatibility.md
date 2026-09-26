# Compatibility matrix

Generated 2026-09-26 by `scripts/gen_matrix.py` with torch 2.14.0, onnx 1.22.0, onnxruntime 1.30.0, torch_geometric 2.8.0.post1, transformers 5.17.0.
Each model was checked with `downshift.check(..., k=8)`.

**CLEAN** exports, matches PyTorch on every sample, and survives shapes the exporter never saw; served via ONNX Runtime. **DEGRADED** exports without error but produces numbers that differ from PyTorch beyond tolerance on at least one sample; served via eager PyTorch unless `--force-onnx`. **FAILED** does not export, or exports but ONNX Runtime can't load or run the graph; served via eager PyTorch either way. **UNVERIFIED** is a `.onnx` file with no reference model, so numerics were never checked; served via ONNX Runtime and labelled as such.

| Model | Hazard | Family | Export | Capture | Numerics | Shape-general | Backend |
|---|---|---|---|---|---|---|---|
| `bf16_weights` | bfloat16 weights: ONNX Runtime CPU has no bf16 Gemm kernel | generic-torch | FAILED | strict=False | — | — | torch |
| `broken_factory` | Not an export hazard fixture: raises as soon as it's instantiated | — | skipped (RuntimeError) | — | — | — | — |
| `clean_mlp` | Control fixture: no export hazards | generic-torch | CLEAN | strict=False | 1.2e-07 | ✓ | onnxruntime |
| `custom_autograd` | custom autograd.Function with no symbolic override | generic-torch | CLEAN | strict=False | 1.8e-07 | ✓ | onnxruntime |
| `data_dependent_branch` | data-dependent control flow | generic-torch | FAILED | — | — | — | torch |
| `dict_input` | dataclass container input | generic-torch | CLEAN | strict=False | 2.4e-07 | ✓ | onnxruntime |
| `dropout_model` | stochastic layer | generic-torch | CLEAN | strict=False | 1.2e-07 | ✓ | onnxruntime |
| `dynamic_batch_cnn` | batch-dim generalization | generic-torch | CLEAN | strict=False | 3.0e-08 | ✓ | onnxruntime |
| `gnn_gat` | GNN fixture: 3-layer GAT node classifier | pyg | CLEAN | strict=False | 1.0e-07 | ✓ | onnxruntime |
| `gnn_gcn` | GNN fixture: 2-layer GCN node classifier | pyg | CLEAN | strict=False | 2.1e-07 | ✓ | onnxruntime |
| `gnn_sage` | GNN fixture: 2-layer GraphSAGE node classifier | pyg | CLEAN | strict=False | 8.9e-08 | ✓ | onnxruntime |
| `scatter_include_self_false` | scatter_reduce(include_self=False) has no faithful ONNX translation | generic-torch | DEGRADED | strict=False | 1.2e+00 | — | torch |
| `tied_weights` | tied embedding/output weight (GPT-2/OPT-style) | generic-torch | CLEAN | strict=False | 1.9e-06 | ✓ | onnxruntime |
| `tiny_bert` | HF fixture: a randomly initialised two-layer BERT encoder | hf-transformers | CLEAN | strict=False | 7.2e-07 | ✓ | onnxruntime |

## How to read this

CLEAN means the ONNX graph agrees with PyTorch, not that it is fast. DEGRADED means the graph runs and returns numbers that are wrong on at least one of the K samples; the Numerics column is the worst absolute error seen. FAILED models are still served, via eager PyTorch behind the same endpoint. UNVERIFIED never appears here because every corpus model has a PyTorch reference.
