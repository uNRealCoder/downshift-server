"""No-HTTP baselines: what the model costs before any server touches it.

Four levels per (case, batch), so the serving tax can be decomposed rather than asserted:

  compute         backend.infer(feeds) on numpy already in memory
  compute+json    parse the request body, build arrays, infer, .tolist(), serialize the response
  compute+binary  same request/response round trip with base64 tensor bodies: json.loads,
                  b64decode + np.frombuffer, infer, b64encode the output bytes, json.dumps
  (the rest)      measured server latency at concurrency 1 minus compute+json (or
                  compute+binary for the base64 variant) = ASGI/socket cost

Run as a subprocess: `python -m bench.inproc <case> <batch> <iters>`, JSON on stdout.
"""

from __future__ import annotations

import base64
import json
import statistics
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

from bench._path import ROOT  # noqa: E402,F401  (must be first: fixes sys.path)
from bench.cases import make_inputs, prepare, to_payload  # noqa: E402


def _parse_binary(body: bytes, order: tuple[str, ...]) -> dict[str, np.ndarray]:
    """What the server does with a base64 request: json.loads, b64decode, zero-copy frombuffer."""
    inputs = json.loads(body)["inputs"]
    fed = {}
    for n in order:
        spec = inputs[n]
        raw = base64.b64decode(spec["data"])
        fed[n] = np.frombuffer(raw, dtype=np.dtype(spec["dtype"])).reshape(spec["shape"])
    return fed


def _dump_binary(outs: list[np.ndarray]) -> str:
    """What the server does for `output_encoding: base64`: b64encode the contiguous bytes."""
    outputs = {}
    for i, o in enumerate(outs):
        o = np.require(o, requirements="C")
        outputs[f"output_{i}"] = {
            "data": base64.b64encode(o.tobytes()).decode("ascii"),
            "dtype": o.dtype.name,
            "shape": list(o.shape),
        }
    return json.dumps({"outputs": outputs})


def _timeit(fn, iters: int, warmup: int = 20) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000)
    samples.sort()
    return {
        "mean_ms": round(statistics.fmean(samples), 4),
        "p50_ms": round(samples[len(samples) // 2], 4),
        "p99_ms": round(samples[min(len(samples) - 1, int(len(samples) * 0.99))], 4),
        "min_ms": round(samples[0], 4),
        "iters": iters,
    }


def main() -> None:
    import torch

    case_name, batch, iters = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    case = prepare(case_name, export=True)
    feeds = make_inputs(case_name, batch)
    body = json.dumps(to_payload(feeds)).encode()
    body_binary = json.dumps(to_payload(feeds, encoding="base64")).encode()

    result: dict = {"case": case_name, "batch": batch, "verdict": case.verdict_status}

    # --- eager torch -----------------------------------------------------------------
    module = case.module.eval()
    order = case.input_names

    def torch_compute() -> None:
        args = [torch.from_numpy(np.ascontiguousarray(feeds[n])) for n in order]
        with torch.inference_mode():
            module(*args)

    def torch_json() -> None:
        inputs = json.loads(body)["inputs"]
        args = [torch.from_numpy(np.asarray(inputs[n], dtype=case.dtypes[n])) for n in order]
        with torch.inference_mode():
            out = module(*args)
        tensors = (
            [out]
            if isinstance(out, torch.Tensor)
            else [t for t in out if isinstance(t, torch.Tensor)]
        )
        json.dumps({"outputs": {f"output_{i}": t.numpy().tolist() for i, t in enumerate(tensors)}})

    def torch_binary() -> None:
        fed = _parse_binary(body_binary, order)
        # frombuffer is read-only; torch.from_numpy needs a writable buffer, so copy (as the
        # server's torch path must).
        args = [torch.from_numpy(np.array(fed[n], copy=True)) for n in order]
        with torch.inference_mode():
            out = module(*args)
        tensors = (
            [out]
            if isinstance(out, torch.Tensor)
            else [t for t in out if isinstance(t, torch.Tensor)]
        )
        _dump_binary([t.numpy() for t in tensors])

    result["torch"] = {
        "compute": _timeit(torch_compute, iters),
        "compute_json": _timeit(torch_json, iters),
        "compute_binary": _timeit(torch_binary, iters),
    }

    # --- onnx runtime ----------------------------------------------------------------
    if case.onnx_bytes is not None:
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(
            case.onnx_bytes, sess_options=options, providers=["CPUExecutionProvider"]
        )
        out_names = [o.name for o in session.get_outputs()]

        def ort_compute() -> None:
            session.run(out_names, feeds)

        def ort_json() -> None:
            inputs = json.loads(body)["inputs"]
            fed = {n: np.asarray(inputs[n], dtype=case.dtypes[n]) for n in order}
            outs = session.run(out_names, fed)
            json.dumps({"outputs": {f"output_{i}": o.tolist() for i, o in enumerate(outs)}})

        def ort_binary() -> None:
            fed = _parse_binary(body_binary, order)
            outs = session.run(out_names, fed)
            _dump_binary(outs)

        result["onnxruntime"] = {
            "compute": _timeit(ort_compute, iters),
            "compute_json": _timeit(ort_json, iters),
            "compute_binary": _timeit(ort_binary, iters),
        }

    result["request_bytes"] = len(body)
    result["request_bytes_binary"] = len(body_binary)
    sys.stdout.write(json.dumps(result))


if __name__ == "__main__":
    main()
