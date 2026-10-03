"""The serving hardening from the 0.4.0 review round: API key, request-body limits, admission
before the body is read, bf16 models, input range checks, /ready phases, torch error mapping."""

import dataclasses
import logging
import threading
import time

import numpy as np
import pytest
import torch
from fastapi import FastAPI
from fastapi.testclient import TestClient
from torch import nn

from downshift.core.phase import Phase, report
from downshift.loading import LoadSpec, load_model
from downshift.serve.app import build_app
from downshift.serve.backends import OnnxRuntimeBackend, TorchBackend
from downshift.serve.engine import ServeOptions, ServingState, prepare_serving
from downshift.serve.options import BackendChoice

MLP_INPUT = {"inputs": {"x": [[0.0] * 16]}}
NOT_SET = "DOWNSHIFT_SERVER_API_KEY is not set"
KEY = "s3cret"
AUTH = {"Authorization": f"Bearer {KEY}"}
BODY_LIMIT = 200


def _serve(spec: str, **opts) -> ServingState:
    return prepare_serving(load_model(LoadSpec(spec)), ServeOptions(warmup=1, **opts))


def _with_options(state: ServingState, **overrides) -> ServingState:
    return dataclasses.replace(state, options=dataclasses.replace(state.options, **overrides))


@pytest.fixture(scope="module")
def mlp_state() -> ServingState:
    return _serve("tests.models.clean_mlp:make_model")


@pytest.fixture(scope="module")
def small_body_state(mlp_state) -> ServingState:
    return _with_options(mlp_state, max_body_bytes=BODY_LIMIT)


# --- API key ---------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def keyed(mlp_state) -> TestClient:
    return TestClient(build_app(mlp_state, api_key=KEY))


def test_predict_without_the_key_is_401_with_a_bearer_challenge(keyed):
    resp = keyed.post("/predict", json=MLP_INPUT)

    assert resp.status_code == 401
    assert resp.headers["www-authenticate"] == "Bearer"
    assert resp.json() == {"detail": "Authorization header is not set or incorrect"}


@pytest.mark.parametrize("header", ["Bearer nope", "Bearer ", f"Basic {KEY}", KEY])
def test_a_wrong_credential_is_401(keyed, header):
    resp = keyed.post("/predict", json=MLP_INPUT, headers={"Authorization": header})
    assert resp.status_code == 401


def test_the_right_key_is_accepted(keyed):
    resp = keyed.post("/predict", json=MLP_INPUT, headers=AUTH)
    assert resp.status_code == 200, resp.text


def test_every_route_but_the_probes_needs_the_key(keyed):
    assert keyed.get("/health").status_code == 200
    assert keyed.get("/ready").status_code == 200
    assert keyed.get("/metadata").status_code == 401
    assert keyed.get("/schema").status_code == 401
    assert keyed.get("/metadata", headers=AUTH).status_code == 200


def test_a_non_ascii_authorization_header_is_401_not_500(keyed):
    resp = keyed.post(
        "/predict", json=MLP_INPUT, headers=[(b"authorization", b"Bearer caf\xe9\xff")]
    )
    assert resp.status_code == 401


def test_an_empty_key_counts_as_unset(mlp_state, caplog):
    with caplog.at_level(logging.WARNING, logger="downshift.serve"):
        client = TestClient(build_app(mlp_state, api_key=""))

    assert [r.getMessage() for r in caplog.records if NOT_SET in r.getMessage()]
    assert client.post("/predict", json=MLP_INPUT).status_code == 200
    bearer_only = client.post("/predict", json=MLP_INPUT, headers={"Authorization": "Bearer "})
    assert bearer_only.status_code == 200


def test_no_key_warns_once_at_build(mlp_state, caplog):
    with caplog.at_level(logging.WARNING, logger="downshift.serve"):
        client = TestClient(build_app(mlp_state, api_key=None))
        client.post("/predict", json=MLP_INPUT)
        client.get("/health")

    warned = [r for r in caplog.records if NOT_SET in r.getMessage()]
    assert len(warned) == 1
    assert warned[0].levelno == logging.WARNING


def test_a_key_set_does_not_warn(mlp_state, caplog):
    with caplog.at_level(logging.WARNING, logger="downshift.serve"):
        build_app(mlp_state, api_key=KEY)
    assert not [r for r in caplog.records if NOT_SET in r.getMessage()]


def test_the_probes_stay_exempt_when_the_app_is_mounted_under_a_prefix(mlp_state):
    root = FastAPI()
    root.mount("/model", build_app(mlp_state, api_key=KEY))
    client = TestClient(root)

    assert client.get("/model/health").status_code == 200
    assert client.get("/model/ready").status_code == 200
    assert client.post("/model/predict", json=MLP_INPUT).status_code == 401
    assert client.post("/model/predict", json=MLP_INPUT, headers=AUTH).status_code == 200


# --- request body limit ----------------------------------------------------------------------


def _chunked(*parts: bytes):
    def body():
        yield from parts

    return body()


@pytest.fixture(scope="module")
def limited(small_body_state) -> TestClient:
    return TestClient(build_app(small_body_state, api_key=None))


def test_a_chunked_body_over_the_limit_is_413(limited):
    parts = [b" " * 100] * 5

    resp = limited.post(
        "/predict", content=_chunked(*parts), headers={"content-type": "application/json"}
    )

    assert "content-length" not in resp.request.headers  # the premise: it really was chunked
    assert resp.status_code == 413
    assert "--max-body-bytes" in resp.json()["detail"]


def test_a_chunked_body_under_the_limit_works(limited):
    payload = b'{"inputs": {"x": [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,'
    rest = b" 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]}}"

    resp = limited.post(
        "/predict", content=_chunked(payload, rest), headers={"content-type": "application/json"}
    )

    assert "content-length" not in resp.request.headers
    assert len(payload) + len(rest) < BODY_LIMIT
    assert resp.status_code == 200, resp.text


def test_a_content_length_over_the_limit_is_413_before_the_handler_runs(
    small_body_state, limited, monkeypatch
):
    calls: list[dict] = []
    monkeypatch.setattr(
        small_body_state.backend, "infer", lambda inputs: calls.append(inputs) or {}
    )

    resp = limited.post(
        "/predict",
        content=b"this is not json " + b"x" * BODY_LIMIT,
        headers={"content-type": "application/json"},
    )

    assert resp.status_code == 413  # not 422: the body was never parsed
    assert calls == []


# --- admission happens before the body is read -----------------------------------------------


@pytest.fixture
def tight_state(mlp_state) -> ServingState:
    return _with_options(mlp_state, max_concurrency=1, max_queue=0, max_body_bytes=BODY_LIMIT)


def test_a_full_server_answers_503_without_reading_the_body(tight_state):
    client = TestClient(build_app(tight_state, api_key=None))
    assert tight_state.try_admit()

    resp = client.post("/predict", content=b"x" * (BODY_LIMIT * 2))

    assert resp.status_code == 503  # not 413: admission comes first
    assert resp.headers["retry-after"] == "1"
    assert "at capacity" in resp.json()["detail"]
    assert tight_state.in_flight == 1  # a refused request never held a slot
    tight_state.release()
    assert tight_state.in_flight == 0


@pytest.mark.parametrize(
    ("kwargs", "status"),
    [
        ({"content": b"x" * (BODY_LIMIT * 2)}, 413),
        ({"json": {"inputs": {}}}, 400),
        ({"content": b"{not json", "headers": {"content-type": "application/json"}}, 422),
        ({"json": MLP_INPUT}, 200),
    ],
)
def test_every_exit_releases_the_admitted_slot(tight_state, kwargs, status):
    client = TestClient(build_app(tight_state, api_key=None))

    resp = client.post("/predict", **kwargs)

    assert resp.status_code == status, resp.text
    assert tight_state.in_flight == 0
    assert client.post("/predict", json=MLP_INPUT).status_code == 200  # and the slot is reusable


# --- bf16 ------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def bf16_client() -> TestClient:
    return TestClient(build_app(_serve("tests.models.bf16_weights:make_model"), api_key=None))


def test_a_bf16_model_boots_on_torch_and_predicts_float32(bf16_client):
    assert bf16_client.get("/ready").status_code == 200
    assert bf16_client.get("/metadata").json()["backend"]["name"] == "torch"

    resp = bf16_client.post("/predict", json={"inputs": {"x": [[0.5] * 8]}})

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["dtypes"] == {"output_0": "float32"}
    assert body["shapes"] == {"output_0": [1, 8]}


def test_a_bf16_model_schema_reports_float32_inputs(bf16_client):
    (entry,) = bf16_client.get("/schema").json()["inputs"]
    assert entry["dtype"] == "float32"


# Hand-built graphs pin ir_version: make_model stamps the installed onnx's newest IR version,
# which can be ahead of what the installed onnxruntime reads (onnx 1.20 writes IR 14 while
# onnxruntime 1.23 stops at 13). 8 is the lowest that allows opset 17.


def _bf16_output_onnx_bytes() -> bytes:
    """A graph ORT's CPU EP can actually execute (Cast, not Gemm - which has no bf16 CPU
    kernel, see tests/models/bf16_weights.py) but whose declared output is bfloat16: the
    case OnnxRuntimeBackend.infer's OrtValue/DLPack widening (B1) covers, since
    OrtValue.numpy() has no bfloat16/float16 numpy dtype to convert to."""
    import onnx
    from onnx import TensorProto, helper

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [None, 8])
    y = helper.make_tensor_value_info("y", TensorProto.BFLOAT16, [None, 8])
    cast = helper.make_node("Cast", ["x"], ["y"], to=TensorProto.BFLOAT16)
    graph = helper.make_graph([cast], "g", [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8)
    onnx.checker.check_model(model)
    return model.SerializeToString()


@pytest.mark.skipif(
    not hasattr(__import__("onnxruntime").OrtValue, "__dlpack__"),
    reason="this onnxruntime has no OrtValue DLPack, so it can't return bfloat16 at all",
)
def test_onnxruntime_backend_widens_a_bfloat16_output_to_float32():
    backend = OnnxRuntimeBackend(_bf16_output_onnx_bytes(), device="cpu")

    out = backend.infer({"x": np.ones((1, 8), dtype=np.float32)})

    assert out["output_0"].dtype == np.float32
    np.testing.assert_allclose(out["output_0"], np.ones((1, 8), dtype=np.float32))


def _fp16_output_onnx_bytes() -> bytes:
    import onnx
    from onnx import TensorProto, helper

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [None, 8])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT16, [None, 8])
    cast = helper.make_node("Cast", ["x"], ["y"], to=TensorProto.FLOAT16)
    graph = helper.make_graph([cast], "g", [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8)
    onnx.checker.check_model(model)
    return model.SerializeToString()


def test_onnxruntime_backend_widens_a_float16_output_without_dlpack(monkeypatch):
    """fp16 has a numpy dtype, so it must not depend on OrtValue.__dlpack__, which older
    onnxruntime releases (the 1.17 floor) do not have."""
    backend = OnnxRuntimeBackend(_fp16_output_onnx_bytes(), device="cpu")
    monkeypatch.setattr(torch, "from_dlpack", _no_dlpack)

    out = backend.infer({"x": np.ones((1, 8), dtype=np.float32)})

    assert out["output_0"].dtype == np.float32
    np.testing.assert_allclose(out["output_0"], np.ones((1, 8), dtype=np.float32))


def _no_dlpack(_value):
    raise AssertionError("fp16 must not go through DLPack")


def test_a_bfloat16_output_on_an_onnxruntime_without_dlpack_says_what_to_do():
    from downshift.serve import backends

    class NoDlpack:
        def data_type(self) -> str:
            return "tensor(bfloat16)"

    with pytest.raises(RuntimeError, match="--backend torch"):
        backends._widen_ort_value(NoDlpack())


@pytest.mark.parametrize(
    ("declared", "expected"),
    [
        ("tensor(float)", "float32"),  # ORT's spelling
        ("tensor(float32)", "float32"),  # the torch backend's spelling
        ("tensor(float64)", "float64"),
        ("tensor(bfloat16)", "float32"),  # bf16 is float32 on the wire
        ("tensor(int64)", "int64"),
        ("float16", "float16"),
        ("tensor(nonsense)", None),
        (None, None),
    ],
)
def test_normalize_dtype_reads_both_backends_spellings(declared, expected):
    from downshift.serve.schemas import normalize_dtype

    assert normalize_dtype(declared) == expected


# --- Hugging Face: token-id range and axis bounds --------------------------------------------

HF_MAX_POSITIONS = 32  # hf_repo.MAX_POSITIONS, kept here so the import stays lazy


@pytest.fixture(scope="module")
def hf_client(tmp_path_factory) -> TestClient:
    pytest.importorskip("transformers")
    pytest.importorskip("tokenizers")
    from tests.models import hf_repo

    assert hf_repo.MAX_POSITIONS == HF_MAX_POSITIONS
    repo = hf_repo.write_encoder_repo(tmp_path_factory.mktemp("bert"))
    state = prepare_serving(load_model(LoadSpec(repo)), ServeOptions(warmup=1))
    assert state.vocab_size == 100
    return TestClient(build_app(state, api_key=None))


def _ids(row: list[int]) -> dict:
    return {"inputs": {"input_ids": [row], "attention_mask": [[1] * len(row)]}}


@pytest.mark.parametrize("bad", [-5, 100, 12345])
def test_an_out_of_vocabulary_token_id_is_400_and_names_the_value(hf_client, bad):
    resp = hf_client.post("/predict", json=_ids([2, bad, 3]))

    assert resp.status_code == 400
    assert resp.json()["detail"] == f"input_ids contains {bad}, outside the vocabulary [0, 100)"


def test_the_first_and_last_valid_token_ids_are_accepted(hf_client):
    resp = hf_client.post("/predict", json=_ids([0, 99, 5]))
    assert resp.status_code == 200, resp.text


def test_a_sequence_longer_than_the_export_traced_is_400(hf_client):
    resp = hf_client.post("/predict", json=_ids([2] * (HF_MAX_POSITIONS + 8)))

    assert resp.status_code == 400
    assert resp.json()["detail"] == (
        f"input_ids axis 1 is {HF_MAX_POSITIONS + 8}; this model accepts 1 to {HF_MAX_POSITIONS}"
    )


def test_a_sequence_at_the_bound_is_accepted(hf_client):
    resp = hf_client.post("/predict", json=_ids([2] * HF_MAX_POSITIONS))
    assert resp.status_code == 200, resp.text


def test_schema_carries_the_adapters_axis_names_and_bounds(hf_client):
    inputs = {i["name"]: i for i in hf_client.get("/schema").json()["inputs"]}

    assert set(inputs) == {"input_ids", "attention_mask"}
    for entry in inputs.values():
        assert entry["shape"] == ["batch", "seq"]
        batch, seq = entry["bounds"]
        assert batch["min"] == 1
        assert seq == {"min": 1, "max": HF_MAX_POSITIONS}


# --- /ready phases ---------------------------------------------------------------------------


def test_ready_reports_the_phase_the_loader_is_in(mlp_state):
    reached_export, go_on, release = threading.Event(), threading.Event(), threading.Event()

    def loader() -> ServingState:
        report(Phase.export)
        reached_export.set()
        go_on.wait(30)
        report(Phase.session)
        release.wait(30)
        return mlp_state

    app = build_app(loader=loader, api_key=None)
    with TestClient(app) as client:
        try:
            assert reached_export.wait(30)
            resp = client.get("/ready")
            assert resp.status_code == 503
            assert resp.json() == {"ready": False, "phase": "export"}
            assert client.get("/health").status_code == 200
            predict = client.post("/predict", json=MLP_INPUT)
            assert predict.status_code == 503
            assert "retry-after" in predict.headers

            go_on.set()
            deadline = threading.Event()
            for _ in range(300):
                if client.get("/ready").json().get("phase") == "session":
                    break
                deadline.wait(0.1)
            assert client.get("/ready").json() == {"ready": False, "phase": "session"}
        finally:
            go_on.set()
            release.set()
        app.state.loader_thread.join(30)

        resp = client.get("/ready")
        assert resp.status_code == 200
        assert resp.json() == {"ready": True}
        assert client.post("/predict", json=MLP_INPUT).status_code == 200


# --- torch backend error mapping -------------------------------------------------------------


@pytest.fixture(scope="module")
def torch_state() -> ServingState:
    state = _serve("tests.models.clean_mlp:make_model", backend=BackendChoice.torch)
    assert state.backend.name == "torch"
    return state


class _Failing(nn.Module):
    def __init__(self, message: str) -> None:
        super().__init__()
        self.message = message

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise RuntimeError(self.message)


def test_a_client_shape_mistake_on_torch_is_a_400(torch_state):
    client = TestClient(build_app(torch_state, api_key=None))

    resp = client.post("/predict", json={"inputs": {"x": [[0.0] * 5]}})

    assert resp.status_code == 400
    assert "shape" in resp.json()["detail"]


@pytest.mark.parametrize(
    "message",
    [
        "CUDA out of memory. Tried to allocate a size",
        "CUDA error: an illegal memory access was encountered",
        "INTERNAL ASSERT FAILED at foo.cpp:1",
        "not implemented for 'BFloat16'",
        "[enforce fail at alloc_cpu.cpp:118] DefaultCPUAllocator: can't allocate memory: "
        "you tried to allocate 8589934592 bytes.",
        "Expected all tensors to be on the same device, but found at least two devices",
    ],
)
def test_a_server_side_runtime_error_in_forward_is_a_500(torch_state, monkeypatch, message):
    monkeypatch.setattr(torch_state.backend, "module", _Failing(message))
    client = TestClient(build_app(torch_state, api_key=None), raise_server_exceptions=False)

    resp = client.post("/predict", json=MLP_INPUT)

    assert resp.status_code == 500
    assert message not in resp.text  # the exception text stays in the server log
    assert resp.json()["request_id"] == resp.headers["x-request-id"]


def test_an_unrecognized_runtime_error_in_forward_is_presumed_client_input(
    torch_state, monkeypatch
):
    """Torch has no exception type for "the client's input was bad", only ever-varying
    RuntimeError/ValueError messages (a shape mismatch, an out-of-bounds target, ...);
    enumerating every client-input phrasing under-classifies, so anything that isn't one of
    the few known server-side markers (out of memory, a CUDA/cuDNN fault, an internal
    assert, a missing kernel) is presumed to be the client's fault instead of an opaque 500."""
    monkeypatch.setattr(torch_state.backend, "module", _Failing("kaboom: an unrelated failure"))
    client = TestClient(build_app(torch_state, api_key=None))

    resp = client.post("/predict", json=MLP_INPUT)

    assert resp.status_code == 400
    assert "kaboom" in resp.json()["detail"]


@pytest.mark.parametrize("device", ["cuda", "cuda:0"])
def test_device_cuda_without_cuda_fails_at_construction(monkeypatch, device):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(ValueError, match=f"--device {device}"):
        TorchBackend(nn.Linear(2, 2), ("x",), device)


# --- the prep pool keeps conversion off the inference slot -----------------------------------


def test_a_slow_text_encode_does_not_hold_the_inference_slot(tmp_path, monkeypatch):
    pytest.importorskip("transformers")
    pytest.importorskip("tokenizers")
    from downshift.adapters.text import TextIO
    from tests.models import hf_repo

    repo = hf_repo.write_encoder_repo(tmp_path)
    state = prepare_serving(
        load_model(LoadSpec(repo)),
        ServeOptions(warmup=1, max_concurrency=1, prep_threads=2),
    )
    client = TestClient(build_app(state, api_key=None))
    started = threading.Event()

    def slow_encode(self, text):
        started.set()
        time.sleep(2.0)
        raise ValueError("never mind")

    monkeypatch.setattr(TextIO, "encode", slow_encode)
    results: list[int] = []
    slow = threading.Thread(
        target=lambda: results.append(client.post("/predict", json={"text": "hi"}).status_code)
    )
    slow.start()
    assert started.wait(5)

    began = time.perf_counter()
    resp = client.post("/predict", json=_ids([2, 3, 4]))
    elapsed = time.perf_counter() - began
    slow.join()

    assert resp.status_code == 200, resp.text
    assert elapsed < 1.0
    assert results == [400]


def test_a_request_queued_for_inference_still_times_out(mlp_state, monkeypatch):
    calls: list[int] = []

    def slow_infer(inputs):
        calls.append(1)
        time.sleep(0.4)
        return {"output_0": np.zeros((1, 4), dtype=np.float32)}

    state = _with_options(mlp_state, max_concurrency=1, max_queue=5, request_timeout=0.1)
    monkeypatch.setattr(state.backend, "infer", slow_infer)
    client = TestClient(build_app(state, api_key=None))
    results: list[int] = []
    first = threading.Thread(
        target=lambda: results.append(client.post("/predict", json=MLP_INPUT).status_code)
    )
    first.start()
    time.sleep(0.05)
    second = client.post("/predict", json=MLP_INPUT)
    first.join()

    assert second.status_code == 503
    assert "--request-timeout" in second.json()["detail"]
    assert results == [200]
    assert len(calls) == 1


def test_server_timing_and_the_access_log_carry_the_wait_stages(mlp_state):
    resp = TestClient(build_app(mlp_state, api_key=None)).post("/predict", json=MLP_INPUT)

    assert resp.status_code == 200
    names = [part.split(";")[0] for part in resp.headers["server-timing"].split(", ")]
    assert names == ["parse", "prep_wait", "prep", "infer_wait", "infer", "encode"]
