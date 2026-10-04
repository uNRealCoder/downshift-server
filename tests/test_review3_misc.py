"""Small pieces added in the 0.4.0 review round: axis-bound and ServeArgs JSON hand-offs, RoBERTa
position limits, the load-phase reporter, stderr capture off the main thread, the new limit
defaults and the plain-text banner."""

import io
import json
import logging
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest
import torch

from downshift import settings
from downshift.cli import render
from downshift.cli.runtime import ArtifactHandoff, ServeArgs
from downshift.core import capture as capture_module
from downshift.core.axes import DimBound, axis_bounds_from_json, axis_bounds_to_json
from downshift.core.phase import CURRENT_PROGRESS, LoadProgress, Phase, report
from downshift.loading import LoadSpec, load_model
from downshift.serve.engine import ServeOptions, prepare_serving
from downshift.serve.options import BackendChoice
from downshift.serve.schemas import OutputEncoding
from tests.conftest import subprocess_env
from tests.models import clean_mlp

BOUNDS = {
    "input_ids": {0: DimBound("batch", 1, 4096), 1: DimBound("seq", 1, 510)},
    "attention_mask": {0: DimBound("batch", 1, 4096), 1: DimBound("seq", 1, 510)},
}


# --- axis bounds and ServeArgs hand-off ------------------------------------------------------


def test_axis_bounds_round_trip_through_json():
    encoded = axis_bounds_to_json(BOUNDS)

    assert encoded["input_ids"] == [[0, "batch", 1, 4096], [1, "seq", 1, 510]]
    restored = axis_bounds_from_json(json.loads(json.dumps(encoded)))
    assert restored == BOUNDS
    assert all(isinstance(axis, int) for axes in restored.values() for axis in axes)


def test_axis_bounds_round_trip_of_nothing():
    assert axis_bounds_from_json(axis_bounds_to_json({})) == {}


def _serve_args(**overrides) -> ServeArgs:
    fields = {
        "load": LoadSpec("some/model", pooling="mean", normalize=True),
        "options": ServeOptions(
            backend=BackendChoice.torch,
            output_encoding=OutputEncoding.base64,
            max_body_bytes=1234,
            request_timeout=2.5,
            dynamic={"x": [0, 1]},
        ),
        "reference": "ref.pt",
        "middleware": ["pkg.mod:Mw"],
        "log_level": "info",
        "artifact": ArtifactHandoff(
            verdict={"status": "CLEAN"},
            input_names=["input_ids", "attention_mask"],
            onnx_path="model.onnx",
            feeds_path="feeds.npz",
            axis_bounds=axis_bounds_to_json(BOUNDS),
            kind="onnx-file",
        ),
        "access_log": False,
        "tokenizer_from": "path/to/repo/dir",
    }
    fields.update(overrides)
    return ServeArgs(**fields)


def test_serve_args_round_trip_keeps_access_log_and_axis_bounds():
    args = _serve_args()

    restored = ServeArgs.from_json(args.to_json())

    assert restored == args
    assert restored.access_log is False
    assert restored.options.backend is BackendChoice.torch
    assert restored.options.output_encoding is OutputEncoding.base64
    assert restored.artifact is not None
    assert axis_bounds_from_json(restored.artifact.axis_bounds) == BOUNDS


def test_serve_args_round_trip_without_an_artifact_defaults_access_log_on():
    args = _serve_args(artifact=None, access_log=True)

    restored = ServeArgs.from_json(args.to_json())

    assert restored == args
    assert restored.artifact is None
    assert restored.access_log is True


# --- position_limit --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def hf():
    pytest.importorskip("transformers")
    from downshift import hf_repo as module

    return module


def test_position_limit_of_a_roberta_config_skips_the_padding_offset(hf):
    from transformers import RobertaConfig

    config = RobertaConfig(max_position_embeddings=514, pad_token_id=1)
    assert hf.position_limit(config) == 512


def test_position_limit_of_the_other_roberta_family_configs(hf):
    from transformers import XLMRobertaConfig

    assert hf.position_limit(XLMRobertaConfig(max_position_embeddings=514)) == 512
    assert (
        hf.position_limit(SimpleNamespace(model_type="camembert", max_position_embeddings=514))
        == 512
    )


def test_position_limit_uses_the_configs_own_pad_token_id(hf):
    config = SimpleNamespace(model_type="roberta", max_position_embeddings=514, pad_token_id=3)
    assert hf.position_limit(config) == 514 - 4


def test_position_limit_assumes_pad_id_one_when_a_roberta_config_has_none(hf):
    config = SimpleNamespace(model_type="roberta", max_position_embeddings=514, pad_token_id=None)
    assert hf.position_limit(config) == 512


def test_position_limit_never_drops_below_two(hf):
    config = SimpleNamespace(model_type="roberta", max_position_embeddings=3, pad_token_id=5)
    assert hf.position_limit(config) == 2


def test_position_limit_of_bert_is_unchanged(hf):
    from transformers import BertConfig

    assert hf.position_limit(BertConfig(max_position_embeddings=512)) == 512
    assert hf.position_limit(BertConfig(max_position_embeddings=512, pad_token_id=7)) == 512


def test_position_limit_is_none_without_a_declared_maximum(hf):
    assert hf.position_limit(SimpleNamespace(model_type="bert")) is None


# --- load-phase reporting --------------------------------------------------------------------


def test_report_is_a_no_op_when_nothing_is_listening():
    token = CURRENT_PROGRESS.set(None)
    try:
        report(Phase.export)
    finally:
        CURRENT_PROGRESS.reset(token)


def test_report_updates_the_current_progress():
    progress = LoadProgress()
    assert progress.phase == Phase.load
    token = CURRENT_PROGRESS.set(progress)
    try:
        report(Phase.export)
        assert progress.phase == Phase.export
        report(Phase.warmup)
        assert progress.phase == "warmup"
    finally:
        CURRENT_PROGRESS.reset(token)


# --- capture off the main thread -------------------------------------------------------------


@pytest.fixture
def stderr_seen_by_export(monkeypatch) -> list:
    """torch.export.export replaced by a spy that records sys.stderr and fails, so capture()
    runs its stderr handling without paying for a real export."""
    seen: list = []

    def spy(*args, **kwargs):
        seen.append(sys.stderr)
        raise RuntimeError("spy: no export")

    monkeypatch.setattr(torch.export, "export", spy)
    return seen


def test_capture_leaves_sys_stderr_alone_off_the_main_thread(stderr_seen_by_export):
    outcome: dict = {}

    def run() -> None:
        outcome["before"] = sys.stderr
        outcome["result"] = capture_module.capture(clean_mlp.make_model(), clean_mlp.make_inputs())
        outcome["after"] = sys.stderr

    worker = threading.Thread(target=run)
    worker.start()
    worker.join()

    assert not outcome["result"].success
    assert stderr_seen_by_export  # both strategies were tried
    assert all(seen is outcome["before"] for seen in stderr_seen_by_export)
    assert outcome["after"] is outcome["before"]


def test_capture_redirects_sys_stderr_on_the_main_thread(stderr_seen_by_export):
    assert threading.current_thread() is threading.main_thread()
    before = sys.stderr

    result = capture_module.capture(clean_mlp.make_model(), clean_mlp.make_inputs())

    assert not result.success
    assert stderr_seen_by_export
    assert all(seen is not before for seen in stderr_seen_by_export)
    assert sys.stderr is before


def test_off_thread_capture_still_collects_torch_logging():
    sink = io.StringIO()
    worker_saw = {}

    def run() -> None:
        with capture_module._capture_torch_output(sink):
            logging.getLogger("torch.some_module").warning("from torch")
            worker_saw["stderr"] = sys.stderr

    worker = threading.Thread(target=run)
    worker.start()
    worker.join()

    assert "from torch" in sink.getvalue()
    assert worker_saw["stderr"] is not sink


# --- limit defaults --------------------------------------------------------------------------


def test_the_body_and_timeout_defaults():
    assert settings.DEFAULT_MAX_BODY_BYTES == 32 * 1024 * 1024
    assert settings.DEFAULT_REQUEST_TIMEOUT == 30.0


def _serve_options_in_a_child(**env: str) -> dict:
    """ServeOptions() as a fresh interpreter sees it: its defaults are read from settings when
    downshift.serve.options is first imported, so an in-process reload cannot show them."""
    code = (
        "import json; from downshift.serve.options import ServeOptions; o = ServeOptions(); "
        "print(json.dumps({'max_body_bytes': o.max_body_bytes, 'request_timeout': o.request_timeout}))"
    )
    child_env = {k: v for k, v in subprocess_env().items() if not k.startswith("DOWNSHIFT_")}
    child_env.update(env)
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=child_env
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_serve_options_pick_up_the_new_defaults():
    assert _serve_options_in_a_child() == {
        "max_body_bytes": 32 * 1024 * 1024,
        "request_timeout": 30.0,
    }


def test_the_environment_overrides_reach_serve_options():
    options = _serve_options_in_a_child(
        DOWNSHIFT_MAX_BODY_BYTES="4096", DOWNSHIFT_REQUEST_TIMEOUT="2.5"
    )
    assert options == {"max_body_bytes": 4096, "request_timeout": 2.5}


# --- the report logger's plain text ----------------------------------------------------------


@pytest.fixture(scope="module")
def mlp_state():
    return prepare_serving(
        load_model(LoadSpec("tests.models.clean_mlp:make_model")), ServeOptions(warmup=1)
    )


def _report_messages(caplog: pytest.LogCaptureFixture) -> list[str]:
    records = [r for r in caplog.records if r.name == "downshift.report"]
    assert all(r.levelno == logging.INFO for r in records)
    return [r.getMessage() for r in records]


def _plain_ascii(text: str) -> bool:
    return "\x1b" not in text and all(ord(c) < 128 for c in text)


def test_print_ready_reports_the_boot_time_on_the_report_logger(caplog):
    with caplog.at_level(logging.INFO, logger="downshift.report"):
        render.print_ready(SimpleNamespace(timings={"load": 0.5, "export": 1.0, "warmup": 1.5}))
        render.print_ready(SimpleNamespace(timings={}))

    assert _report_messages(caplog) == ["ready in 3.0 s", "ready"]


def test_print_banner_is_one_plain_ascii_report_record(mlp_state, caplog):
    with caplog.at_level(logging.INFO, logger="downshift.report"):
        render.print_banner(mlp_state, "0.0.0.0", 8123)

    (banner,) = _report_messages(caplog)
    assert _plain_ascii(banner)
    rows = {
        line.split()[0] for line in banner.splitlines() if line.startswith("  ") and line[2] != " "
    }
    assert {"Model", "Capacity", "Endpoint"} <= rows
    (endpoint,) = [line for line in banner.splitlines() if line.strip().startswith("Endpoint")]
    assert "http://localhost:8123" in endpoint
    assert "0.0.0.0:8123" not in endpoint


def test_print_banner_names_a_specific_host_as_is(mlp_state, caplog):
    with caplog.at_level(logging.INFO, logger="downshift.report"):
        render.print_banner(mlp_state, "10.1.2.3", 9000)

    (banner,) = _report_messages(caplog)
    (endpoint,) = [line for line in banner.splitlines() if line.strip().startswith("Endpoint")]
    assert "http://10.1.2.3:9000" in endpoint
    assert "localhost" not in endpoint


def test_print_booting_is_plain_ascii(caplog):
    with caplog.at_level(logging.INFO, logger="downshift.report"):
        render.print_booting("some/model", "::", 8000)

    messages = _report_messages(caplog)
    assert len(messages) == 2
    assert all(_plain_ascii(m) for m in messages)
    assert "http://localhost:8000" in messages[1]
