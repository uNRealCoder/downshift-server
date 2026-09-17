"""FastAPI app over a ServingState. The same routes regardless of which backend is behind it."""

from collections.abc import Sequence
from typing import Any

import numpy as np
import orjson
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

# dtypes orjson's OPT_SERIALIZE_NUMPY writes straight from the array buffer (orjson >= 3.9).
_ORJSON_DTYPES = frozenset(
    {
        "float16",
        "float32",
        "float64",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "bool",
    }
)


class NumpyJSONResponse(JSONResponse):
    """orjson with OPT_SERIALIZE_NUMPY: arrays are written from their buffers; NaN/Inf become null.

    Not built on fastapi's ORJSONResponse: older versions lack the numpy flag, newer ones
    deprecate the class and warn at import.
    """

    def render(self, content: Any) -> bytes:
        return orjson.dumps(content, option=orjson.OPT_SERIALIZE_NUMPY)


def _json_ready(arr: np.ndarray) -> np.ndarray | list:
    """The array as orjson can write it in one pass, or a Python list for that array only.

    orjson needs a C-contiguous array of a supported dtype with at least one dimension
    (`np.ascontiguousarray` would silently turn a 0-d array into a 1-d one).
    """
    if arr.dtype.name in _ORJSON_DTYPES and arr.ndim > 0:
        return np.ascontiguousarray(arr)
    return arr.tolist()


def run_predict(state: ServingState, inputs: dict[str, Any]) -> NumpyJSONResponse:
    """Validate, convert, infer. Raises HTTPException(400) for anything the client got wrong."""
    missing = [n for n in state.input_names if n not in inputs]
    if missing:
        raise HTTPException(400, f"missing inputs: {missing}")

    declared = {spec.name: spec.dtype for spec in state.backend.metadata().inputs}
    try:
        feeds = {n: to_numpy(n, inputs[n], declared.get(n)) for n in state.input_names}
        outputs = state.backend.infer(feeds)
    except HTTPException:
        raise
    except Exception as exc:  # shape/dtype errors from ORT or torch are the client's problem
        raise HTTPException(400, str(exc)) from exc

    arrays = {name: np.asarray(arr) for name, arr in outputs.items()}
    # Same keys as PredictResponse; built by hand so orjson serializes the buffers directly.
    body = {
        "outputs": {name: _json_ready(arr) for name, arr in arrays.items()},
        "shapes": {name: list(arr.shape) for name, arr in arrays.items()},
        "dtypes": {name: arr.dtype.name for name, arr in arrays.items()},
    }
    return NumpyJSONResponse(body)


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

    # No response_model: that would re-validate and re-walk the outputs. `responses` keeps
    # PredictResponse in the OpenAPI schema.
    @app.post(
        "/predict",
        response_class=NumpyJSONResponse,
        responses={200: {"model": PredictResponse}},
    )
    def predict(req: PredictRequest) -> NumpyJSONResponse:
        return run_predict(state, req.inputs)

    @app.post(
        "/predict/graph",
        response_class=NumpyJSONResponse,
        responses={200: {"model": PredictResponse}},
    )
    def predict_graph(req: GraphPredictRequest) -> NumpyJSONResponse:
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
