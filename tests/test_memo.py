"""The in-process export memo (0.5.0 C1): what its key is made of, who reads and writes it, and
that a second boot of the same model skips export and verify (and, for a Hugging Face repo
directory, never loads the weights)."""

import importlib.util
import os
import sys
import textwrap
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient

import downshift
from downshift.core import capture, memo, verdict
from downshift.core.memo import MEMO, ExportEntry, ExportMemo
from downshift.serve import app_for
from tests.models import clean_mlp

RUN = {
    "adapter": None,
    "dynamic": None,
    "k": 2,
    "seed": 0,
    "atol": None,
    "rtol": None,
    "vary": None,
    "axis_max": None,
}


def boot(app) -> dict:
    return TestClient(app).get("/metadata").json()["boot"]


def mlp(seed: int = 0) -> torch.nn.Module:
    torch.manual_seed(seed)
    return clean_mlp.make_model()


def fake_entry(status: str = "CLEAN", **kwargs) -> ExportEntry:
    return ExportEntry(
        verdict={"status": status}, input_names=["x"], axis_bounds={}, onnx_bytes=b"x", **kwargs
    )


@pytest.fixture
def count_builds(monkeypatch) -> list[int]:
    """One item per export-and-verify the process actually ran."""
    calls: list[int] = []
    real = verdict.build_verdict

    def counting(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(verdict, "build_verdict", counting)
    return calls


def load_module(path: Path, name: str, monkeypatch):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


# --- the key ----------------------------------------------------------------------------------


def test_the_key_is_stable_and_follows_the_weights():
    model = mlp()
    inputs = clean_mlp.make_inputs()
    first = memo.model_key(model, inputs, **RUN)

    assert memo.model_key(model, inputs, **RUN) == first
    assert memo.model_key(mlp(), inputs, **RUN) == first  # same seed, same weights
    assert memo.model_key(mlp(seed=1), inputs, **RUN) != first


def test_the_key_changes_with_each_option():
    model = mlp()
    inputs = clean_mlp.make_inputs()
    base = memo.model_key(model, inputs, **RUN)

    for change in (
        {"k": 3},
        {"seed": 1},
        {"atol": 1e-3},
        {"axis_max": {"dim0": 8}},
        {"dynamic": {"x": [0]}},
        {"pooling": "mean"},
        {"normalize": True},
        {"adapter": "generic"},
        {"vary": "tests.models.clean_mlp:make_inputs"},
        {"fp16": True},
    ):
        assert memo.model_key(model, inputs, **(RUN | change)) != base, change
    assert memo.model_key(model, clean_mlp.make_inputs(2), **RUN) != base


def test_the_key_changes_with_a_tolerance_override(monkeypatch):
    model = mlp()
    base = memo.model_key(model, None, **RUN)
    monkeypatch.setitem(memo.settings.TOLERANCES, "float32", (0.5, 0.5))

    assert memo.model_key(model, None, **RUN) != base


def test_the_key_changes_with_a_toolchain_version(monkeypatch):
    model = mlp()
    base = memo.model_key(model, None, **RUN)
    monkeypatch.setattr(memo, "tool_versions", lambda: {"torch": "0.0.faked"})

    assert memo.model_key(model, None, **RUN) != base


def test_editing_forward_changes_the_key(tmp_path, monkeypatch):
    source = textwrap.dedent(
        """
        import torch

        class Doubler(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)

            def forward(self, x):
                return self.linear(x) * {scale}
        """
    )
    path = tmp_path / "doubler.py"
    path.write_text(source.format(scale=2))
    torch.manual_seed(0)
    first_module = load_module(path, "doubler_v1", monkeypatch)
    before = memo.model_key(first_module.Doubler(), None, **RUN)

    path.write_text(source.format(scale=3))
    torch.manual_seed(0)
    second_module = load_module(path, "doubler_v2", monkeypatch)
    after = memo.model_key(second_module.Doubler(), None, **RUN)

    assert before != after


def test_a_plain_attribute_changes_the_key():
    class Scaled(torch.nn.Module):
        def __init__(self, scale: float):
            super().__init__()
            self.scale = scale

        def forward(self, x):
            return x * self.scale

    assert memo.model_key(Scaled(1.0), None, **RUN) != memo.model_key(Scaled(2.0), None, **RUN)


def test_training_mode_is_not_part_of_the_key():
    model = mlp()
    model.train()
    training = memo.model_key(model, None, **RUN)
    model.eval()

    assert memo.model_key(model, None, **RUN) == training


def test_a_repo_key_follows_each_file_identity(tmp_path):
    pytest.importorskip("transformers")
    from tests.models.hf_repo import write_encoder_repo

    repo = write_encoder_repo(tmp_path / "repo")
    key = memo.repo_key(repo, None, memo.file_identity, **RUN)
    assert memo.repo_key(repo, None, memo.file_identity, **RUN) == key

    weights = Path(repo) / "model.safetensors"
    stat = weights.stat()
    os.utime(weights, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    assert memo.repo_key(repo, None, memo.file_identity, **RUN) != key

    other = write_encoder_repo(tmp_path / "other", max_seq_length=8)
    assert memo.repo_key(other, None, memo.file_identity, **RUN) != key


# --- the memo ---------------------------------------------------------------------------------


def test_the_memo_is_an_lru_of_two():
    memo_ = ExportMemo(max_entries=2)
    for name in ("a", "b"):
        memo_.put(name, fake_entry())
    assert memo_.get("a") is not None  # touching "a" makes "b" the oldest
    memo_.put("c", fake_entry())

    assert "b" not in memo_
    assert "a" in memo_ and "c" in memo_
    assert len(memo_) == 2


def test_the_memo_ignores_failed_and_external_data_verdicts():
    memo_ = ExportMemo()
    memo_.put("failed", fake_entry("FAILED"))
    memo_.put("unverified", fake_entry("UNVERIFIED"))
    memo_.put("external", ExportEntry({"status": "CLEAN"}, ["x"], {}, onnx_path=Path("m.onnx")))

    assert len(memo_) == 0
    memo_.put("degraded", fake_entry("DEGRADED"))
    assert "degraded" in memo_


def test_a_third_model_evicts_the_oldest():
    models = [mlp(seed) for seed in range(3)]
    inputs = clean_mlp.make_inputs()
    keys = [memo.model_key(m, inputs, **(RUN | {"k": 1})) for m in models]
    for m in models:
        downshift.check(m, inputs, k=1)

    assert len(MEMO) == 2
    assert keys[0] not in MEMO
    assert keys[1] in MEMO and keys[2] in MEMO


def test_an_external_data_verdict_is_not_memoised(monkeypatch):
    monkeypatch.setattr(capture, "EXTERNAL_DATA_THRESHOLD", 0)
    result = downshift.check(mlp(), clean_mlp.make_inputs(), k=1)

    assert result.status == "CLEAN"
    assert result.onnx_path is not None
    assert len(MEMO) == 0


# --- who reads and who writes -----------------------------------------------------------------


def test_check_stores_but_never_reads(monkeypatch, count_builds):
    model = mlp()
    inputs = clean_mlp.make_inputs()
    downshift.check(model, inputs, k=1)
    assert len(MEMO) == 1

    def refuse(key):
        raise AssertionError("check() read the memo")

    monkeypatch.setattr(MEMO, "get", refuse)
    downshift.check(model, inputs, k=1)

    assert len(count_builds) == 2


def test_export_reuses_a_check_in_the_same_process(tmp_path, count_builds):
    import onnxruntime as ort

    model = mlp()
    inputs = clean_mlp.make_inputs()
    checked = downshift.check(model, inputs, k=2)

    out = tmp_path / "out" / "m.onnx"
    exported = downshift.export(model, out, inputs, k=2)

    assert len(count_builds) == 1
    assert exported.status == checked.status == "CLEAN"
    assert exported.onnx_path == out
    assert (tmp_path / "out" / "m.manifest.json").exists()
    session = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    assert session.get_inputs()[0].name == "x"


def test_export_stores_for_a_later_export(tmp_path, count_builds):
    model = mlp()
    inputs = clean_mlp.make_inputs()
    downshift.export(model, tmp_path / "a.onnx", inputs, k=1)
    downshift.export(model, tmp_path / "b.onnx", inputs, k=1)

    assert len(count_builds) == 1
    assert (tmp_path / "a.onnx").read_bytes() == (tmp_path / "b.onnx").read_bytes()


def test_app_for_reuses_a_check(count_builds):
    model = mlp()
    inputs = clean_mlp.make_inputs()
    downshift.check(model, inputs, k=1)

    app = app_for(model, inputs, k=1, warmup=1)

    assert len(count_builds) == 1
    assert "export" not in boot(app) and "verify" not in boot(app)
    body = TestClient(app).post("/predict", json={"inputs": {"x": inputs[0].tolist()}}).json()
    assert body["shapes"]["output_0"] == [1, 4]


def test_cache_false_bypasses_both_ways(tmp_path, count_builds):
    model = mlp()
    inputs = clean_mlp.make_inputs()

    downshift.check(model, inputs, k=1, cache=False)
    assert len(MEMO) == 0

    downshift.check(model, inputs, k=1)
    downshift.export(model, tmp_path / "m.onnx", inputs, k=1, cache=False)
    assert len(count_builds) == 3  # export ignored the stored entry

    app = app_for(model, inputs, k=1, warmup=1, cache=False)
    assert {"export", "verify"} <= set(boot(app))


def test_a_second_app_for_of_a_repo_skips_export_and_never_loads_weights(tmp_path, monkeypatch):
    pytest.importorskip("transformers")
    from transformers import AutoModel

    from tests.models.hf_repo import write_encoder_repo

    repo = write_encoder_repo(tmp_path / "repo")
    first = app_for(repo, k=2, warmup=1)
    assert {"export", "verify"} <= set(boot(first))
    expected = TestClient(first).post("/predict", json={"text": ["hello world"]}).json()

    def refuse(*args, **kwargs):
        raise AssertionError("from_pretrained was called on a memo hit")

    monkeypatch.setattr(AutoModel, "from_pretrained", refuse)
    second_app = app_for(repo, k=2, warmup=1)

    timings = boot(second_app)
    assert "export" not in timings and "verify" not in timings
    client = TestClient(second_app)
    assert client.post("/predict", json={"text": ["hello world"]}).json() == expected
    assert client.get("/schema").json()["text_input"] is not None


def test_editing_a_repo_file_misses(tmp_path, monkeypatch):
    pytest.importorskip("transformers")
    from tests.models.hf_repo import write_encoder_repo

    repo = write_encoder_repo(tmp_path / "repo")
    app_for(repo, k=1, warmup=0)
    weights = Path(repo) / "model.safetensors"
    stat = weights.stat()
    os.utime(weights, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))

    assert {"export", "verify"} <= set(boot(app_for(repo, k=1, warmup=0)))


def test_the_banner_boot_row_says_reused(tmp_path):
    from downshift.cli import render
    from downshift.serve.engine import ServeOptions, ServingState

    state = ServingState.__new__(ServingState)
    state.timings = {"load": 0.0, "session": 0.1, "warmup": 0.1}
    state.reused = "memory"
    assert "reused (in-process)" in render._boot_text(state)
    state.reused = "disk"
    assert "reused from --export-cache-dir" in render._boot_text(state)
    assert ServeOptions().export_cache_dir is None
