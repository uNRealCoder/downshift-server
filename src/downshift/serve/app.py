"""FastAPI app over a ServingState. The same routes regardless of which backend is behind it."""

from collections.abc import Callable, Coroutine, Sequence
from typing import Any

import numpy as np
import orjson
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute

import downshift
from downshift.serve.codec import b64encode
from downshift.serve.engine import ServingState
from downshift.serve.middleware import load_middleware
from downshift.serve.schemas import (
    GraphPredictRequest,
    HealthResponse,
    MetadataResponse,
    OutputEncoding,
    PredictRequest,
    PredictResponse,
    ReadyResponse,
    to_numpy,
)

# dtypes orjson's OPT_SERIALIZE_NUMPY writes straight from the array buffer (orjson >= 3.9).
_ORJSON_DTYPES = frozenset(
    "float16 float32 float64 int8 int16 int32 int64 uint8 uint16 uint32 uint64 bool".split()
)


class NumpyJSONResponse(JSONResponse):
    """orjson with OPT_SERIALIZE_NUMPY: arrays are written from their buffers; NaN/Inf become null.

    Not built on fastapi's ORJSONResponse: older versions lack the numpy flag, newer ones
    deprecate the class and warn at import.
    """

    def render(self, content: Any) -> bytes:
        return orjson.dumps(content, option=orjson.OPT_SERIALIZE_NUMPY)


class _OrjsonRequest(Request):
    async def json(self) -> Any:
        if not hasattr(self, "_json"):
            self._json = orjson.loads(await self.body())
        return self._json


class OrjsonRoute(APIRoute):
    """Parse request bodies with orjson: several times faster than stdlib on MiB-scale bodies.

    orjson's JSONDecodeError subclasses the stdlib one, so malformed JSON still maps to 422.
    """

    def get_route_handler(self) -> Callable[[Request], Coroutine[Any, Any, Response]]:
        handler = super().get_route_handler()

        async def route_handler(request: Request) -> Response:
            return await handler(_OrjsonRequest(request.scope, request.receive))

        return route_handler


# The predict routes: orjson response, and PredictResponse kept in the OpenAPI schema without
# the response_model re-validation walk over every output.
_PREDICT_ROUTE: dict[str, Any] = {
    "response_class": NumpyJSONResponse,
    "responses": {200: {"model": PredictResponse}},
}


def _json_ready(arr: np.ndarray) -> np.ndarray | list:
    """The contiguous array itself when orjson can write it in one pass, else a list."""
    return arr if arr.dtype.name in _ORJSON_DTYPES and arr.ndim else arr.tolist()  # 0-d: orjson rejects


def _base64_ready(arr: np.ndarray) -> dict[str, Any]:
    """{data, dtype, shape}: base64 straight off the contiguous buffer."""
    return {
        "data": b64encode(arr).decode("ascii"),
        "dtype": arr.dtype.name,
        "shape": list(arr.shape),
    }


def run_predict(
    state: ServingState, inputs: dict[str, Any], encoding: OutputEncoding | None
) -> NumpyJSONResponse:
    """Validate, convert, infer. Raises HTTPException(400) for anything the client got wrong."""
    missing = [n for n in state.input_names if n not in inputs]
    if missing:
        raise HTTPException(400, f"missing inputs: {missing}")

    declared = state.declared_dtypes
    max_bytes = state.options.max_input_bytes
    try:
        feeds = {
            n: to_numpy(n, inputs[n], declared.get(n), max_bytes=max_bytes)
            for n in state.input_names
        }
        outputs = state.backend.infer(feeds)
    except HTTPException:
        raise
    except Exception as exc:  # shape/dtype errors from ORT or torch are the client's problem
        raise HTTPException(400, str(exc)) from exc

    encoding = encoding or state.options.output_encoding
    encode = _base64_ready if encoding == OutputEncoding.base64 else _json_ready
    # C-contiguous once, up front (np.require keeps 0-d arrays 0-d; ascontiguousarray does not).
    arrays = {name: np.require(arr, requirements="C") for name, arr in outputs.items()}
    # Same keys as PredictResponse; built by hand so orjson serializes the buffers directly.
    body = {
        "outputs": {name: encode(arr) for name, arr in arrays.items()},
        "shapes": {name: list(arr.shape) for name, arr in arrays.items()},
        "dtypes": {name: arr.dtype.name for name, arr in arrays.items()},
    }
    return NumpyJSONResponse(body)


def build_app(state: ServingState, middleware: Sequence[str] = ()) -> FastAPI:
    app = FastAPI(title="downshift", version=downshift.__version__)
    app.router.route_class = OrjsonRoute
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

    @app.post("/predict", **_PREDICT_ROUTE)
    def predict(req: PredictRequest) -> NumpyJSONResponse:
        return run_predict(state, req.inputs, req.output_encoding)

    @app.post("/predict/graph", **_PREDICT_ROUTE)
    def predict_graph(req: GraphPredictRequest) -> NumpyJSONResponse:
        if not {"x", "edge_index"} <= set(state.input_names):
            raise HTTPException(
                400,
                f"model is not graph-shaped: inputs are {list(state.input_names)}, "
                "expected at least 'x' and 'edge_index'",
            )
        # No dtype hints needed: to_numpy takes the backend's declared dtype (int64 for
        # edge_index on both backends), and integer lists default to int64 anyway.
        inputs: dict[str, Any] = {"x": req.x, "edge_index": req.edge_index}
        if req.edge_attr is not None:
            inputs["edge_attr"] = req.edge_attr
        return run_predict(state, inputs, req.output_encoding)

    return app
