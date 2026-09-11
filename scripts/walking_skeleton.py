"""Walking skeleton: torch -> onnx -> onnxruntime -> FastAPI -> HTTP, end to end.

Ugly and hardcoded on purpose (per SPRINT_PLAN.md H2-H3) -- this proves the integration
assumptions the whole rest of the weekend is built on. Not part of the src/downshift package.

Run:
    python scripts/walking_skeleton.py

Then, in another shell:
    curl -X POST http://127.0.0.1:8000/predict \
         -H "Content-Type: application/json" \
         -d "{\"x\": [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]]}"
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import onnxruntime as ort
import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from tests.models.clean_mlp import make_inputs, make_model

ONNX_PATH = Path(__file__).parent / "walking_skeleton_model.onnx"


def export_model(model: torch.nn.Module, example_inputs: tuple) -> None:
    torch.onnx.export(
        model,
        example_inputs,
        str(ONNX_PATH),
        dynamo=True,
        input_names=["x"],
        output_names=["y"],
        # torch.onnx.export treats verbose=None (its default) as verbose=True, which
        # prints unicode checkmarks that crash on Windows consoles using cp1252.
        verbose=False,
    )
    print(f"Exported ONNX model to {ONNX_PATH}")


class PredictRequest(BaseModel):
    x: list[list[float]]


class PredictResponse(BaseModel):
    output: list[list[float]]


app = FastAPI(title="downshift walking skeleton")
_session: ort.InferenceSession | None = None


def get_session() -> ort.InferenceSession:
    global _session
    if _session is None:
        if not ONNX_PATH.exists():
            export_model(make_model(), make_inputs(batch=1))
        _session = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    return _session


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    session = get_session()
    x = np.array(req.x, dtype=np.float32)
    (y,) = session.run(None, {"x": x})
    return PredictResponse(output=y.tolist())


if __name__ == "__main__":
    # Build one model/input pair and reuse it for both export and the sanity check below —
    # make_model() is unseeded, so two separate calls would yield different random weights
    # and make the torch-vs-onnxruntime comparison meaningless.
    model = make_model()
    example_inputs = make_inputs(batch=1)
    export_model(model, example_inputs)

    # Sanity check before serving: ONNX Runtime output must match a direct torch forward pass.
    with torch.no_grad():
        torch_out = model(*example_inputs).numpy()

    session = get_session()
    (ort_out,) = session.run(None, {"x": example_inputs[0].numpy()})

    max_diff = np.abs(torch_out - ort_out).max()
    print(f"torch vs onnxruntime max abs diff: {max_diff:.2e}")
    assert max_diff < 1e-5, "ONNX output diverges from torch output — pipe is broken"
    print("torch and onnxruntime agree. Starting server on http://127.0.0.1:8000 ...")

    uvicorn.run(app, host="127.0.0.1", port=8000)
