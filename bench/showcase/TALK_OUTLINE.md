# Talk outline (10 minutes)

For a conference talk or a blog post. Every number comes from the data files named in [RESULTS.md](RESULTS.md). Keep the limits slide.

## 1. The problem (1 min)
- `torch.onnx.export` can finish with no error and give a graph with wrong numbers.
- Show: `scatter_include_self_false`. Verdict DEGRADED. exported via strict=False but numerics diverge on 6/8 samples (max abs err 1.59e+00).
- A hand-rolled ONNX Runtime server answers with an error of 1.2e+00. (Page: correctness)

## 2. What downshift does (1 min)
- Show the flow diagram (`index.html`): load, export, verify, verdict, backend.
- 8 samples, some with shapes the exporter did not see. Four verdicts. The verdict selects the backend.

## 3. Does the gate agree with the truth? (2 min)
- 14 hazard fixtures with known verdicts. 8 real models: 8 of 8 CLEAN, worst error 6.0e-05 against a tolerance of 1e-4. (Page: compatibility)
- Limit: the gate is a sample of 8. The data has no false-negative rate.

## 4. What does the gate cost? (2 min)
- Boot: 19.2 s against 11.8 s on MiniLM. (Page: boot time)
- Request path: on a tiny model, 66% of a request is HTTP and hand-offs. A hand-rolled server wins on `clean_mlp`: 3216 against 1428 requests/s at peak. (Pages: throughput, time of one request)

## 5. What does the gate give? (2 min)
- ONNX Runtime on a real encoder: MiniLM, one sentence, 8 clients: 486 against 141 requests/s. p50 at one client: 4.9 ms against 9.2 ms. (Page: torch vs ORT)
- The gain falls with the input size. Show the three payloads.
- Wire format: safetensors helps wide inputs, hurts tiny ones. (Page: wire formats)

## 6. Honest notes (1 min)
- 1.0.0rc1 is not faster than 0.5.1. The data shows no gain. (Page: releases)
- The result cache: a hit takes 0.255 ms of server time against 4.62 ms. The load-run numbers are a best case. (Page: result cache)
- One machine, one run, no repeats, CPU only. Not measured: GPU, Linux, other servers.

## 7. Q and A: likely questions
- "Why not export once and trust it?" The fixtures show exports that finish with no error and give wrong numbers.
- "Why is a hand-rolled server faster on small models?" It does less work for each request. The time page shows the split. The data does not say more.
- "Is it faster on a GPU?" Not measured.
- "How noisy is this?" Not measured. No file holds repeats.
