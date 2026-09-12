# %% [markdown]
# # Lesson 3: serving over HTTP
#
# `downshift serve` on the command line does three things: pick a backend from the
# verdict, warm it up, and start a FastAPI app. This lesson does the same three things
# from Python and queries the app in-process with `TestClient`, so nothing binds a real
# port — the same `build_app()` call is what `downshift serve` uses to start uvicorn for
# real.

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from fastapi.testclient import TestClient
from torch import nn

from downshift.loading import LoadedModel
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

# %% [markdown]
# `LoadedModel` is the small dataclass `load_model()` normally builds for you from a
# model spec string (`"pkg.module:fn"`, a `.onnx` path, a Hugging Face repo id — see
# lesson 7). Since this model was built inline, we construct it directly.

# %%
loaded = LoadedModel(source="tutorial-model", model=model, example_inputs=inputs)
state = prepare_serving(loaded, ServeOptions(warmup=2))

print("verdict:", state.verdict.status)
print("backend:", state.backend.name)
print("ready:  ", state.ready)

# %% [markdown]
# `prepare_serving` already ran the verdict, picked ONNX Runtime (because the verdict is
# CLEAN), and warmed it up with 2 dummy inferences. `state.ready` is what `/ready`
# reports.

# %%
app = build_app(state)
client = TestClient(app)

print(client.get("/health").json())
print(client.get("/ready").json())

# %% [markdown]
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
# ## The same contract for a model that never exported
#
# Serve the branching model from lesson 1. The verdict is FAILED, the backend is torch,
# and `/predict` still works — a client can't tell the difference except by reading
# `/metadata`.

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


branch_loaded = LoadedModel(source="branch-model", model=DataDependentBranch().eval(),
                             example_inputs=(torch.randn(1, 8),))
branch_state = prepare_serving(branch_loaded, ServeOptions(warmup=1))
branch_client = TestClient(build_app(branch_state))

print("verdict:", branch_state.verdict.status, "backend:", branch_state.backend.name)
response = branch_client.post("/predict", json={"inputs": {"x": [[0.1] * 8]}})
print(response.status_code, "output shape:", len(response.json()["outputs"]["output_0"][0]))

# %% [markdown]
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
