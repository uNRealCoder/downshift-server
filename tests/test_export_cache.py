"""`--export-cache-dir` (0.5.0 C2): the optional disk tier behind the export memo. After a restart
(simulated by clearing the memo), downshift reuses the saved export and does not load the
weights. By default, it writes nothing. A damaged entry is exported again. Downshift computes the
digests one time for each file."""

import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from fastapi.testclient import TestClient
from typer.testing import CliRunner

import downshift
from downshift.cli.main import app as cli_app
from downshift.cli.runtime import ServeArgs
from downshift.core import capture, export_cache
from downshift.core.memo import MEMO
from downshift.loading import LoadSpec
from downshift.serve import app_for
from downshift.serve.options import ServeOptions
from tests.models import clean_mlp

runner = CliRunner()


def boot(app) -> dict:
    return TestClient(app).get("/metadata").json()["boot"]


def entries(cache: Path) -> list[Path]:
    return sorted(p for p in cache.iterdir() if p.is_dir() and ".tmp-" not in p.name)


def mlp() -> torch.nn.Module:
    torch.manual_seed(0)
    return clean_mlp.make_model()


@pytest.fixture
def cache(tmp_path) -> Path:
    path = tmp_path / "cache"
    path.mkdir()
    return path


@pytest.fixture
def repo(tmp_path) -> str:
    pytest.importorskip("transformers")
    from tests.models.hf_repo import write_encoder_repo

    return write_encoder_repo(tmp_path / "repo")


@pytest.fixture
def no_weights(monkeypatch):
    """Call it once the first boot is done: from then on, loading the repo's weights fails."""
    pytest.importorskip("transformers")
    from transformers import AutoModel

    def refuse(*args, **kwargs):
        raise AssertionError("from_pretrained was called on a cache hit")

    return lambda: monkeypatch.setattr(AutoModel, "from_pretrained", refuse)


def test_the_default_writes_nothing(tmp_path, repo):
    before = sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*"))

    app_for(repo, k=1, warmup=0)
    app_for(mlp(), clean_mlp.make_inputs(), k=1, warmup=0)

    assert sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*")) == before


@pytest.mark.needs_torch_26
def test_a_restart_reuses_the_export_and_never_loads_weights(cache, repo, no_weights):
    first = app_for(repo, k=2, warmup=1, export_cache_dir=str(cache))
    assert {"export", "verify"} <= set(boot(first))
    expected = TestClient(first).post("/predict", json={"text": ["hello world"]}).json()
    (entry,) = entries(cache)
    assert {p.name for p in entry.iterdir()} >= {
        "model.onnx",
        "model.manifest.json",
        "verdict.json",
        "feeds.npz",
        "serving.json",
    }

    MEMO.clear()
    no_weights()
    second = app_for(repo, k=2, warmup=1, export_cache_dir=str(cache))

    assert "export" not in boot(second) and "verify" not in boot(second)
    client = TestClient(second)
    assert client.post("/predict", json={"text": ["hello world"]}).json() == expected
    assert client.get("/metadata").json()["verdict"]["status"] == "CLEAN"
    assert len(entries(cache)) == 1


@pytest.mark.needs_torch_26
def test_the_boot_banner_says_where_the_export_came_from(cache, repo, no_weights):
    from downshift.cli import render
    from downshift.serve.reuse import prepare_serving_reusing

    options = ServeOptions(k=1, warmup=0, export_cache_dir=str(cache))

    def load():
        raise AssertionError("loaded on a hit")

    app_for(repo, k=1, warmup=0, export_cache_dir=str(cache))
    no_weights()
    memory = prepare_serving_reusing(load, options, repo=repo)
    MEMO.clear()
    disk = prepare_serving_reusing(load, options, repo=repo)

    assert (memory.reused, disk.reused) == ("memory", "disk")
    assert "reused (in-process)" in render._boot_text(memory)
    assert "reused from --export-cache-dir" in render._boot_text(disk)


@pytest.mark.needs_torch_26
def test_a_corrupted_graph_is_exported_again_and_overwritten(cache, repo, caplog):
    app_for(repo, k=1, warmup=0, export_cache_dir=str(cache))
    (entry,) = entries(cache)
    (entry / "model.onnx").write_bytes(b"not an onnx file")
    MEMO.clear()

    with caplog.at_level(logging.WARNING, logger="downshift.export_cache"):
        again = app_for(repo, k=1, warmup=0, export_cache_dir=str(cache))

    assert {"export", "verify"} <= set(boot(again))
    assert len([r for r in caplog.records if "unusable" in r.getMessage()]) == 1
    assert entries(cache) == [entry]
    assert (entry / "model.onnx").read_bytes() != b"not an onnx file"

    MEMO.clear()
    healed = app_for(repo, k=1, warmup=0, export_cache_dir=str(cache))
    assert "export" not in boot(healed)


def test_a_damaged_data_file_or_manifest_is_a_miss(cache):
    app_for(mlp(), clean_mlp.make_inputs(), k=1, warmup=0, export_cache_dir=str(cache))
    (entry,) = entries(cache)
    (entry / "serving.json").write_text("{")
    MEMO.clear()

    again = app_for(mlp(), clean_mlp.make_inputs(), k=1, warmup=0, export_cache_dir=str(cache))

    assert {"export", "verify"} <= set(boot(again))


@pytest.mark.needs_torch_26
def test_a_leftover_temp_directory_is_ignored(cache, repo, no_weights):
    leftover = cache / "deadbeef.tmp-1234-abcd"
    leftover.mkdir()
    (leftover / "model.onnx").write_bytes(b"half written")

    app_for(repo, k=1, warmup=0, export_cache_dir=str(cache))
    MEMO.clear()
    no_weights()
    again = app_for(repo, k=1, warmup=0, export_cache_dir=str(cache))

    assert "export" not in boot(again)
    assert len(entries(cache)) == 1
    assert (leftover / "model.onnx").read_bytes() == b"half written"


def test_a_failed_write_cleans_up_its_temp_directory(cache, monkeypatch):
    from downshift.core.memo import entry_from_verdict

    def crash(tmp, final):
        raise RuntimeError("crash before the rename")

    monkeypatch.setattr(export_cache.ExportCache, "_install", staticmethod(crash))
    entry = entry_from_verdict(downshift.check(mlp(), clean_mlp.make_inputs(), k=1))
    assert entry is not None

    with pytest.raises(RuntimeError):
        export_cache.ExportCache(cache).put("a" * 64, entry)

    assert list(cache.iterdir()) == []
    assert export_cache.ExportCache(cache).get("a" * 64) is None


def test_an_external_data_graph_is_on_disk_but_not_in_memory(cache, monkeypatch):
    monkeypatch.setattr(capture, "EXTERNAL_DATA_THRESHOLD", 0)
    inputs = clean_mlp.make_inputs()
    first = app_for(mlp(), inputs, k=1, warmup=1, export_cache_dir=str(cache))
    expected = TestClient(first).post("/predict", json={"inputs": {"x": inputs[0].tolist()}}).json()

    assert len(MEMO) == 0
    (entry,) = entries(cache)
    assert (entry / "model.onnx").exists()
    assert any(p.name.endswith(".data") for p in entry.iterdir())

    second = app_for(mlp(), inputs, k=1, warmup=1, export_cache_dir=str(cache))

    assert "export" not in boot(second)
    got = TestClient(second).post("/predict", json={"inputs": {"x": inputs[0].tolist()}}).json()
    assert got == expected


def test_export_to_a_directory_after_an_external_disk_hit(cache, tmp_path, monkeypatch):
    import onnxruntime as ort

    monkeypatch.setattr(capture, "EXTERNAL_DATA_THRESHOLD", 0)
    inputs = clean_mlp.make_inputs()
    downshift.export(mlp(), tmp_path / "a.onnx", inputs, k=1, export_cache_dir=str(cache))
    MEMO.clear()

    verdict = downshift.export(mlp(), tmp_path / "b.onnx", inputs, k=1, export_cache_dir=str(cache))

    assert verdict.status == "CLEAN"
    assert (tmp_path / "b.onnx.data").exists()
    manifest = json.loads((tmp_path / "b.manifest.json").read_text())
    assert [item["file"] for item in manifest["external_data"]] == ["b.onnx.data"]
    session = ort.InferenceSession(str(tmp_path / "b.onnx"), providers=["CPUExecutionProvider"])
    assert session.get_inputs()[0].name == "x"


@pytest.mark.needs_torch_26
def test_a_content_digest_is_computed_once_per_file_across_boots(
    cache, repo, no_weights, monkeypatch
):
    hashed: list[str] = []
    real = export_cache.sha256_file

    def counting(path):
        hashed.append(Path(path).name)
        return real(path)

    app_for(repo, k=1, warmup=0, export_cache_dir=str(cache))
    monkeypatch.setattr(export_cache, "sha256_file", counting)
    (cache / "index.json").unlink()
    MEMO.clear()
    no_weights()

    app_for(repo, k=1, warmup=0, export_cache_dir=str(cache))
    MEMO.clear()
    app_for(repo, k=1, warmup=0, export_cache_dir=str(cache))

    assert hashed.count("model.safetensors") == 1


def test_an_unwritable_directory_is_a_usage_error(tmp_path, repo):
    missing = tmp_path / "nope"
    a_file = tmp_path / "file"
    a_file.write_text("x")

    for target in (missing, a_file):
        served = runner.invoke(cli_app, ["serve", repo, "--export-cache-dir", str(target)])
        exported = runner.invoke(
            cli_app,
            ["export", repo, "-o", str(tmp_path / "out"), "--export-cache-dir", str(target)],
        )
        assert served.exit_code == 4, served.output
        assert exported.exit_code == 4, exported.output
    with pytest.raises(ValueError, match="export-cache-dir"):
        app_for(mlp(), clean_mlp.make_inputs(), export_cache_dir=str(missing))


def test_check_ignores_the_cache_directory(cache, monkeypatch):
    monkeypatch.setattr("downshift.settings.EXPORT_CACHE_DIR", str(cache))

    result = runner.invoke(cli_app, ["check", "tests.models.clean_mlp:make_model", "-k", "1"])
    assert result.exit_code == 0, result.output
    assert list(cache.iterdir()) == []

    rejected = runner.invoke(
        cli_app,
        ["check", "tests.models.clean_mlp:make_model", "--export-cache-dir", str(cache)],
    )
    assert rejected.exit_code == 2


def test_cache_false_skips_the_disk_tier_too(cache):
    app_for(mlp(), clean_mlp.make_inputs(), k=1, warmup=0, export_cache_dir=str(cache), cache=False)

    assert list(cache.iterdir()) == []


def test_the_option_round_trips_through_serve_args():
    args = ServeArgs(
        load=LoadSpec("m"),
        options=ServeOptions(export_cache_dir="/some/dir"),
        reference=None,
        middleware=None,
        log_level="warning",
    )

    assert ServeArgs.from_json(args.to_json()).options.export_cache_dir == "/some/dir"


def test_a_worker_with_an_artifact_never_opens_the_cache(cache, monkeypatch, tmp_path):
    from downshift.cli import runtime

    def refuse(*args, **kwargs):
        raise AssertionError("a worker opened the export cache")

    monkeypatch.setattr(export_cache, "ExportCache", refuse)
    from downshift.serve import reuse

    built = SimpleNamespace(timings={})
    monkeypatch.setattr(reuse, "state_from_entry", lambda *args, **kwargs: built)
    from downshift.cli.runtime import ArtifactHandoff

    args = ServeArgs(
        load=LoadSpec("m"),
        options=ServeOptions(export_cache_dir=str(cache)),
        reference=None,
        middleware=None,
        log_level="warning",
        artifact=ArtifactHandoff(verdict={}, input_names=["x"], kind="onnx-file"),
    )

    assert runtime._build_serving_state(args) is built
