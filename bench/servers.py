"""The hand-rolled baselines, one process each. `python -m bench.servers <variant> <case> <port>`.

The downshift servers are not here: bench/run.py launches the real `downshift serve` CLI for
those, so every version is driven exactly as a user drives it.

Variants
  naive_torch        hand-rolled FastAPI + eager PyTorch, sync endpoint (threadpool)
  naive_onnx         hand-rolled FastAPI + ONNX Runtime, sync endpoint (threadpool)
  naive_torch_async  same as naive_torch but `async def`, which blocks the event loop

The naive variants are written the charitable way: a plain `dict` body with no pydantic
validation and a plain dict response with no response_model. Both of those are cheaper than
what downshift does, so any overhead the harness attributes to downshift is real overhead
and not a strawman baseline.
"""

from __future__ import annotations

import sys
import warnings
from typing import Any

import numpy as np

warnings.filterwarnings("ignore")

from bench._path import ROOT  # noqa: E402,F401  (must be first: fixes sys.path)
from bench.cases import prepare  # noqa: E402

WARMUP = 5


def _feeds(case, inputs: dict[str, Any]) -> dict[str, np.ndarray]:
    return {n: np.asarray(inputs[n], dtype=case.dtypes[n]) for n in case.input_names}


def build_naive_torch(case, blocking: bool):
    """What you write when you just need the model behind HTTP."""
    import torch
    from fastapi import FastAPI

    model = case.module.eval()
    app = FastAPI()

    def run(inputs: dict[str, Any]) -> dict[str, Any]:
        args = [torch.from_numpy(v) for v in _feeds(case, inputs).values()]
        with torch.inference_mode():
            out = model(*args)
        tensors = (
            [out]
            if isinstance(out, torch.Tensor)
            else [t for t in out if isinstance(t, torch.Tensor)]
        )
        return {"outputs": {f"output_{i}": t.numpy().tolist() for i, t in enumerate(tensors)}}

    if blocking:

        @app.post("/predict")
        async def predict(body: dict) -> dict:  # blocks the event loop: the classic mistake
            return run(body["inputs"])
    else:

        @app.post("/predict")
        def predict(body: dict) -> dict:  # sync def -> FastAPI runs it in a threadpool
            return run(body["inputs"])

    @app.get("/health")
    def health() -> dict:
        return {"ok": True}

    for _ in range(WARMUP):
        run(_json_example(case))
    return app


def build_naive_onnx(case):
    """What you write when someone told you ONNX Runtime is faster. No verification anywhere."""
    import onnxruntime as ort
    from fastapi import FastAPI

    if case.onnx_bytes is None:
        raise SystemExit(f"{case.name}: no ONNX graph; this variant cannot serve it")
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        case.onnx_bytes, sess_options=options, providers=["CPUExecutionProvider"]
    )
    out_names = [o.name for o in session.get_outputs()]
    app = FastAPI()

    def run(inputs: dict[str, Any]) -> dict[str, Any]:
        outs = session.run(out_names, _feeds(case, inputs))
        return {"outputs": {f"output_{i}": o.tolist() for i, o in enumerate(outs)}}

    @app.post("/predict")
    def predict(body: dict) -> dict:
        return run(body["inputs"])

    @app.get("/health")
    def health() -> dict:
        return {"ok": True}

    for _ in range(WARMUP):
        run(_json_example(case))
    return app


def _json_example(case) -> dict[str, Any]:
    from bench.cases import make_inputs

    return {k: v.tolist() for k, v in make_inputs(case.name, 1).items()}


def main() -> None:
    variant, case_name, port = sys.argv[1], sys.argv[2], int(sys.argv[3])
    import uvicorn

    from bench._path import verify

    verify()  # naive_onnx exports through the checkout, the same graph the orchestrator saw

    case = prepare(case_name, export=variant == "naive_onnx")
    if variant == "naive_torch":
        app = build_naive_torch(case, blocking=False)
    elif variant == "naive_torch_async":
        app = build_naive_torch(case, blocking=True)
    elif variant == "naive_onnx":
        app = build_naive_onnx(case)
    else:
        raise SystemExit(f"unknown variant {variant!r}")

    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning", access_log=False)


if __name__ == "__main__":
    main()
