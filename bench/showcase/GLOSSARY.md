# Glossary

Terms of the benchmark pages. Sources: `README.md`, `docs/architecture.md` and `docs/cli.md` of the repo, and `bench/README.md`.

## The gate and the verdict

| Term | Meaning |
|---|---|
| Gate | The step before the first request. downshift exports the model, then compares ONNX Runtime with PyTorch. |
| Verdict | The result of the gate. It holds the status, a reason, the numbers and the axis facts. |
| CLEAN | The model exports. It matches PyTorch on every sample, also on shapes that the exporter did not see. Served by ONNX Runtime. |
| DEGRADED | The model exports, but the numbers differ from PyTorch by more than the tolerance on at least one sample. Served by eager PyTorch. `--force-onnx` overrides this. |
| FAILED | The model does not export, or ONNX Runtime cannot load or run the graph. Served by eager PyTorch. |
| UNVERIFIED | A `.onnx` file with no reference model, or `--no-verify`. Served by ONNX Runtime and labelled as never checked. |
| Backend | The engine that runs the model for each request: ONNX Runtime or eager PyTorch. |
| Eager PyTorch | The model run as normal PyTorch code, with no export. |
| ONNX Runtime (ORT) | The engine that runs an exported ONNX graph. |
| Sample | One set of inputs for the check. The default is 8. Samples have different shapes. |
| Tolerance | How far ONNX Runtime may differ from PyTorch. A value passes when `abs(got - expected) <= atol + rtol * abs(expected)`. For float32: atol 1e-4, rtol 1e-3. |
| atol, rtol | Absolute and relative tolerance. |
| Max abs err | The largest absolute difference between two outputs. |
| Cosine check | For an embedding model: the smallest cosine similarity of a row must be at least 0.9999. |
| Row check | The check that the rows of a batch do not change each other. It uses two rows, so it cannot prove that a small effect is absent. |
| Rows independent | The result of the row check. If it is true, the result cache may be on by default. |
| Dynamic axis | An axis of an input whose size can change, such as batch or sequence length. "Sampled" is what the check ran. "Served" is what the server accepts. |
| Capture strategy | How the graph is traced. `strict=False` is the `strict` argument of `torch.export`. |
| Opset | The version of the ONNX operator set (20 in the data). |
| Hazard fixture | A small model in `tests/models/` that isolates one export hazard and has a known verdict. |

## The benchmark

| Term | Meaning |
|---|---|
| Hand-rolled (naive) server | A short FastAPI file that puts a model behind HTTP. In the harness it does less work for each request than downshift. |
| `async def` server | A hand-rolled server whose endpoint blocks the event loop. A warning row. |
| Closed loop | Each client sends its next request when the last one returns. |
| Concurrency (c) | The number of clients that send requests at the same time. |
| Requests/s (rps) | Completed requests in the window, divided by the window length. |
| Rows/s | Requests/s times the batch size. |
| Batch | The number of rows in one request. |
| p50, p99 | The median latency and the latency that 99% of requests are below. |
| Window | The time that one measurement lasts. 3 s on fixtures, 5 s on MiniLM. |
| Peak | The best requests/s over the concurrency sweep. |
| Load generator ceiling | The requests/s of `GET /health` on a do-nothing server. A result near it measures the client. |
| Server-Timing | A response header. downshift writes the time of each stage in it. |
| Stage | One part of a request: parse, prep wait, prep, infer wait, infer, encode. "HTTP + hand-offs" is the client total minus all stages. |
| In-process | Measured with no server and no HTTP. |
| Safetensors | A binary tensor format. A request body can use it in place of JSON. |
| Result cache | Keeps the output of a row. A repeated row is answered from memory. |
| `--workers N` | Starts N server processes. |
| Boot time (`ready_s`) | Seconds from process start to the first 200 answer. |
| Best case | A result where every request is a cache hit. Not a normal workload. |
