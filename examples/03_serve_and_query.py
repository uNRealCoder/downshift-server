# %% [markdown]
# # Lesson 3: serving over HTTP
#
# `downshift.serve.app_for()` is the one-line path from a model to a FastAPI app: it
# wraps `LoadedModel`, `prepare_serving()` and `build_app()`, runs the export-and-verify
# gate, picks a backend from the verdict, warms it up, and hands back an app that's ready
# to serve — the same three steps `downshift serve` does before it starts uvicorn for
# real. This lesson queries it in-process with `TestClient`, so nothing binds a real port.

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from fastapi.testclient import TestClient
from torch import nn

from downshift.loading import LoadedModel
from downshift.serve import app_for
from downshift.serve.app import build_app
from downshift.serve.engine import ServeOptions, prepare_serving


class CleanMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


model = CleanMLP().eval()
inputs = (torch.randn(1, 16),)

# %%
app = app_for(model, inputs, source="tutorial-model", warmup=2)
client = TestClient(app)

print(client.get("/health").json())
print(client.get("/ready").json())

# %% [markdown]
# `app_for` already ran the verdict, picked ONNX Runtime (because the verdict is CLEAN),
# and warmed it up with 2 dummy inferences — `/ready` is already `200`, no loader thread
# involved.
#
# ## `/predict`
#
# Send a batch of 3 rows, each 16 features — a different batch size than the example
# input the model was traced on. That's the point of the CLEAN verdict: it generalizes.

# %%
response = client.post("/predict", json={"inputs": {"x": [[0.0] * 16, [1.0] * 16, [-1.0] * 16]}})
print(response.status_code)
body = response.json()
print("output shape:", [len(row) for row in body["outputs"]["output_0"]])
print("dtypes:", body["dtypes"])

# %% [markdown]
# ## `/metadata`
#
# Everything a client needs to know before calling `/predict`: the verdict, the backend,
# and the input/output signature.

# %%
meta = client.get("/metadata").json()
print("family: ", meta["family"])
print("backend:", meta["backend"]["name"], meta["backend"]["device"])
print("inputs: ", [i["name"] for i in meta["backend"]["inputs"]])

# %% [markdown]
# ## What `app_for` actually does
#
# `app_for` is `LoadedModel` + `prepare_serving()` + `build_app()` in one call. Spelled
# out, so the pieces are visible — this is what to reach for when `app_for`'s keyword
# options aren't enough: a `ServeOptions` you build once and reuse, or a `ServingState`
# you want to inspect before wrapping it in an app.

# %%
loaded = LoadedModel(source="tutorial-model", model=model, example_inputs=inputs)
state = prepare_serving(loaded, ServeOptions(warmup=2))
same_app = build_app(state)

print("verdict:", state.verdict.status)
print("backend:", state.backend.name)
print("ready:  ", state.ready)

# %% [markdown]
# ## The same contract for a model that never exported
#
# Serve the branching model from lesson 1 with `app_for` too. The verdict is FAILED, the
# backend is torch, and `/predict` still works — a client can't tell the difference except
# by reading `/metadata`.


# %%
class DataDependentBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.pos = nn.Linear(8, 8)
        self.neg = nn.Linear(8, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.sum() > 0:
            return self.pos(x)
        return self.neg(x)


branch_client = TestClient(
    app_for(DataDependentBranch().eval(), (torch.randn(1, 8),), source="branch-model", warmup=1)
)

meta = branch_client.get("/metadata").json()
print("verdict:", meta["verdict"]["status"], "backend:", meta["backend"]["name"])
response = branch_client.post("/predict", json={"inputs": {"x": [[0.1] * 8]}})
print(response.status_code, "output shape:", len(response.json()["outputs"]["output_0"][0]))

# %% [markdown]
# ## Mounting it inside an existing service
#
# `app_for` returns a plain `FastAPI` app, so it mounts into one you already run instead
# of competing with it for the port:
#
# ```python
# from fastapi import FastAPI
# from downshift.serve import app_for
#
# app = FastAPI()
# app.mount("/model", app_for(my_model, example_inputs))
# ```
#
# `/model/predict`, `/model/health` and the rest work exactly like a standalone
# `downshift serve`.
#
# ## Running it for real
#
# Outside a tutorial, skip `TestClient` and let `downshift serve` start uvicorn:
#
# ```bash
# downshift serve mypackage.models:build_model --port 8000
# curl -X POST http://localhost:8000/predict \
#      -H "Content-Type: application/json" \
#      -d '{"inputs": {"x": [[0.1, 0.2, ...]]}}'
# ```
#
# `--backend`, `--force-onnx`, `--reference`, and `--middleware` all work the same way
# whether the model came from a spec string, a `.onnx` file, or a Hugging Face repo id.

# %% [markdown]
# Next: [lesson 4](04_pyg_graph_neural_networks.py) checks a graph neural network, where
# node count and edge count need to vary independently.
