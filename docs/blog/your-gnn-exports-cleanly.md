Status: draft

# Your scatter-based model exports to ONNX cleanly. It's also wrong.

Here is the output of `downshift check` on a PyTorch module small enough to quote in full. No error was raised during export. No warning was printed.

```
downshift v0.4.0
  Model           tests.models.scatter_include_self_false:make_model
  Family          generic-torch
  Export          DEGRADED  (strict=False, opset 20)
  Numerics        max abs err 1.22e+00 over 8 samples  6/8 failed
  Tolerance       atol 1e-04, rtol 1e-03 (float32)
  Worst           output_0[1, 4]: torch 0.6410, onnxruntime 1.8568  (sample 5, x (12,8), segment_ids (12))
  Samples         x: (6,8) (12,8) (7,8) (7,8) (1,8) (12,8) (1,8) (7,8)
                  segment_ids: (6) (12) (7) (7) (1) (12) (1) (7)
  Shape-general   n/a (baseline fails)
  Dynamic dims    x[0], segment_ids[0]
  Backend         torch
  Reason          exported via strict=False but numerics diverge on 6/8 samples (max abs err 1.22e+00)
```

Six of eight random inputs come back with different numbers from ONNX Runtime than from PyTorch — including sample 0, the exact input this graph was traced on. The outputs are roughly unit scale, so a max absolute error above one means the answer is not slightly off. It is a different answer.

## The model

This is the whole forward pass:

```python
def forward(self, x, segment_ids):
    x = self.linear(x)
    out = torch.zeros(self.num_segments, x.shape[-1], dtype=x.dtype)
    index = segment_ids.unsqueeze(-1).expand_as(x)
    return out.scatter_reduce(0, index, x, reduce="mean", include_self=False)
```

Project each row, then average the rows that share a segment id. That is message-passing aggregation with the framework stripped away. Every hand-written GNN layer, every "pool the token embeddings by document" step, every segment-mean in a recommender does some version of this. `include_self=False` is the part that says: the zeros I started with are not data, do not count them in the mean.

## What the exporter did with it

ONNX has no scatter operator with a mean reduction that excludes the destination. So the exporter emitted what it could: two plain `ScatterElements` nodes with no reduction at all.

I checked what that graph actually computes. For each segment, ONNX Runtime returns the last row that was scattered into it. Not the mean, not the sum, not the mean including the zeros. The last row. A segment that happens to receive exactly one row comes out correct, because the mean of one value is that value. Every segment with two or more rows is wrong, and how wrong depends on which row the scatter happened to write last. That is why the reported error moves around between runs: the inputs are random, and so is the damage.

Nothing in that process is a bug you could file. The exporter translated what it was given. The checker validated a well-formed graph. ONNX Runtime executed it. Each step did its job. The number at the end is still wrong.

## Export success is not correctness

The general point is not about scatter. It is that "the export succeeded" is a statement about the exporter, not about the model. A successful export tells you a graph was produced and that it type-checks. It does not tell you the graph computes the same function.

The only way to know that is to run both and compare. Not once, on the example input you traced with, because that input is the one case the exporter has effectively seen. You need several samples, and some of them need to have shapes the exporter never saw, because a graph that silently froze a dimension will pass on the trace shape and fail on the next one. This model fails even harder than that: `Shape-general: n/a (baseline fails)` in the table above means the very first sample — the exact shape it was traced on — already disagrees with PyTorch, so shape generalization was never even evaluated.

Once you do that comparison, you need somewhere to put the result, and a pass/fail flag is not enough. This model did not fail. It also did not pass. It exported and lied. That is a third state, and most export tooling has no name for it, which is exactly how graphs like this reach production. In `downshift` the state is called DEGRADED, and it sits between CLEAN and FAILED with its own exit code so a CI job can refuse to ship it.

## The GNN result, honestly

I built this expecting the headline to be a PyTorch Geometric model. GAT in particular does attention-weighted scatter aggregation, and that pattern has a history of translation trouble.

It didn't happen. On torch 2.14 with torch_geometric 2.8, a two-layer GCN, a two-layer GraphSAGE, and a three-layer GAT all came back CLEAN: max absolute error around 2e-7 across eight samples with independently varied node and edge counts. PyG's own aggregation path takes a route the exporter translates faithfully. That is good news, and I am reporting it as such.

It is also the reason a compatibility matrix that gets regenerated weekly matters more than any single blog post. The result above is true for one set of versions on one date. The scatter fixture might be fixed by a future opset. The GAT fixture might regress on a future torch. A claim written once and a table rebuilt every Monday by CI are different kinds of evidence, and only one of them ages well.

## What the tool does with it

When `downshift serve` gets a DEGRADED verdict, it serves the PyTorch model in eager mode behind the same `/predict` endpoint it would have used for ONNX Runtime. The banner says why: a warning with the failing-sample count and the max error, and an `Override` row naming `--force-onnx`. Clients see the same request and response shape; they just get correct numbers at eager speed instead of wrong numbers at ONNX speed.

If you disagree with the call, `--force-onnx` serves the graph anyway. The banner then notes `--force-onnx: serving a DEGRADED graph; outputs may be wrong`, and `/metadata` carries the full numerics report. `downshift export` still writes the `.onnx` for a DEGRADED model, with a manifest sidecar recording the max error, the failing sample count, the torch and onnxruntime versions, and the SHA-256 of the source checkpoint when there was one. The artifact is not withheld. It is labelled.

## Try it

```bash
pip install "downshift-server[all]"
downshift check your_pkg.models:build
```

Exit code 0 means CLEAN, 2 means DEGRADED, 1 means it did not export at all. The full matrix, with the versions it was generated under, is in [docs/compatibility.md](../compatibility.md).
