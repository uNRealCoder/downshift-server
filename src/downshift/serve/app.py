"""FastAPI app over a ServingState. The same routes regardless of which backend is behind it."""

from collections.abc import Sequence
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

import downshift
from downshift.serve.engine import ServingState
from downshift.serve.middleware import load_middleware
from downshift.serve.schemas import (
    GraphPredictRequest,
    HealthResponse,
    MetadataResponse,
    PredictRequest,
    PredictResponse,
    ReadyResponse,
    to_numpy,
)


def _declared_dtypes(state: ServingState) -> dict[str, str | None]:
    return {spec.name: spec.dtype for spec in state.backend.metadata().inputs}


def run_predict(state: ServingState, inputs: dict[str, Any]) -> PredictResponse:
    """Validate, convert, infer. Raises HTTPException(400) for anything the client got wrong."""
    missing = [n for n in state.input_names if n not in inputs]
    if missing:
        raise HTTPException(400, f"missing inputs: {missing}")

    declared = _declared_dtypes(state)
    try:
        feeds = {n: to_numpy(n, inputs[n], declared.get(n)) for n in state.input_names}
        outputs = state.backend.infer(feeds)
    except HTTPException:
        raise
    except Exception as exc:  # shape/dtype errors from ORT or torch are the client's problem
        raise HTTPException(400, str(exc)) from exc

    arrays = {name: np.asarray(arr) for name, arr in outputs.items()}
    return PredictResponse(
        outputs={name: arr.tolist() for name, arr in arrays.items()},
        shapes={name: list(arr.shape) for name, arr in arrays.items()},
        dtypes={name: arr.dtype.name for name, arr in arrays.items()},
    )


def build_app(state: ServingState, middleware: Sequence[str] = ()) -> FastAPI:
    app = FastAPI(title="downshift", version=downshift.__version__)
    app.state.serving = state
    load_middleware(app, middleware)

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse()

    @app.get("/ready", response_model=ReadyResponse)
    def ready() -> JSONResponse:
        status = 200 if state.ready else 503
        return JSONResponse({"ready": state.ready}, status_code=status)

    @app.get("/metadata", response_model=MetadataResponse)
    def metadata() -> MetadataResponse:
        return MetadataResponse(
            model=state.source,
            family=state.verdict.model_family,
            verdict=state.verdict.to_dict(),
            backend=state.backend.metadata().to_dict(),
            input_names=list(state.input_names),
            notes=list(state.notes),
            version=downshift.__version__,
        )

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest) -> PredictResponse:
        return run_predict(state, req.inputs)

    @app.post("/predict/graph", response_model=PredictResponse)
    def predict_graph(req: GraphPredictRequest) -> PredictResponse:
        if not {"x", "edge_index"} <= set(state.input_names):
            raise HTTPException(
                400,
                f"model is not graph-shaped: inputs are {list(state.input_names)}, "
                "expected at least 'x' and 'edge_index'",
            )
        inputs: dict[str, Any] = {
            "x": req.x,
            "edge_index": {"data": req.edge_index, "dtype": "int64"},
        }
        if req.edge_attr is not None:
            inputs["edge_attr"] = req.edge_attr
        return run_predict(state, inputs)

    return app
