"""Nothing that a client or a manifest reader receives names a directory on the disk of the server."""

import dataclasses
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import downshift
from downshift.core.manifest import manifest_path_for
from downshift.serve.app import build_app
from downshift.sources import (
    HF_REPO_DIR,
    IMPORT_SPEC,
    IN_PROCESS_MODULE,
    ONNX_FILE,
    TORCH_CHECKPOINT,
    display_source,
    hide_paths,
    path_basename,
)
from tests.models import clean_mlp

SECRET_DIR = "/srv/models/acme-fraud"
SECRET_PATH = f"{SECRET_DIR}/v7/model.onnx"


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("/srv/models/model.onnx", "model.onnx"),
        ("./bert-base-uncased", "bert-base-uncased"),
        ("models/bert/", "bert"),
        ("C:\\Users\\alice\\models\\model.onnx", "model.onnx"),
        ("C:\\Users\\alice\\models\\bert\\", "bert"),
        ("model.onnx", "model.onnx"),
        ("//srv//models//m.pt", "m.pt"),
    ],
)
def test_path_basename_cuts_either_separator(path, expected):
    assert path_basename(path) == expected


def test_path_basename_names_the_directory_a_dot_points_at(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert path_basename(".") == tmp_path.name
    assert path_basename("..") == tmp_path.parent.name


@pytest.mark.parametrize("kind", [ONNX_FILE, TORCH_CHECKPOINT, HF_REPO_DIR, "unknown"])
def test_display_source_is_the_name_for_anything_that_is_a_path(kind):
    assert display_source(SECRET_PATH, kind) == "model.onnx"


@pytest.mark.parametrize("kind", [IMPORT_SPEC, IN_PROCESS_MODULE])
def test_display_source_leaves_a_label_alone(kind):
    assert display_source("pkg.module:make_model", kind) == "pkg.module:make_model"
    assert display_source("tutorial", kind) == "tutorial"


def test_hide_paths_replaces_every_spelling_of_a_path():
    text = f"Load model from {SECRET_PATH} failed; also {SECRET_PATH.replace('/', chr(92))}"
    out = hide_paths(text, [SECRET_PATH, None])
    assert out == "Load model from model.onnx failed; also model.onnx"


def test_hide_paths_ignores_a_bare_name_and_a_dot():
    text = "Load model from model.onnx failed. Sizes are 1.5 x 2."
    assert hide_paths(text, ["model.onnx", ".", "..", "./"]) == text


def _located_state(mlp_state, kind: str = ONNX_FILE):
    verdict = dataclasses.replace(
        mlp_state.verdict,
        onnx_path=Path(SECRET_PATH),
        reason=f"pre-built ONNX cannot run in ONNX Runtime: Load model from {SECRET_PATH} failed",
        warnings=[f"could not read {SECRET_PATH}"],
    )
    return dataclasses.replace(
        mlp_state,
        source=SECRET_PATH,
        source_kind=kind,
        verdict=verdict,
        notes=[f"text input unavailable: OSError: can't load tokenizer for '{SECRET_PATH}'"],
    )


def test_metadata_names_the_file_and_nothing_above_it(mlp_state):
    client = TestClient(build_app(_located_state(mlp_state)))
    response = client.get("/metadata")
    body = response.json()

    assert body["model"] == "model.onnx"
    assert "onnx_path" not in body["verdict"]
    assert body["verdict"]["reason"].endswith("Load model from model.onnx failed")
    assert body["verdict"]["warnings"] == ["could not read model.onnx"]
    assert body["notes"] == [
        "text input unavailable: OSError: can't load tokenizer for 'model.onnx'"
    ]
    assert SECRET_DIR not in response.text


def test_metadata_leaves_an_import_spec_and_its_texts_alone(mlp_state):
    body = TestClient(build_app(mlp_state)).get("/metadata").json()
    assert body["model"] == "tests.models.clean_mlp:make_model"
    assert "onnx_path" not in body["verdict"]


def test_schema_names_the_file_and_nothing_above_it(mlp_state):
    client = TestClient(build_app(_located_state(mlp_state, HF_REPO_DIR)))
    response = client.get("/schema")
    body = response.json()

    assert body["model"] == "model.onnx"
    assert body["source"]["spec"] == "model.onnx"
    assert body["source"]["kind"] == HF_REPO_DIR
    assert SECRET_DIR not in response.text


def test_manifest_records_names_not_locations(tmp_path):
    source = tmp_path / "alice-home" / "checkpoints" / "mlp.pt"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"weights")
    out = tmp_path / "artifacts" / "m.onnx"

    verdict = downshift.export(
        clean_mlp.make_model(), out, clean_mlp.make_inputs(), source_path=source
    )
    assert verdict.status == "CLEAN", verdict.reason
    assert verdict.onnx_path == out  # the verdict in the process keeps the real location

    text = manifest_path_for(out).read_text()
    manifest = json.loads(text)
    assert manifest["onnx_file"] == "m.onnx"
    assert manifest["source_model"] == "mlp.pt"
    assert manifest["source_sha256"]
    assert manifest["verdict"]["onnx_path"] == "m.onnx"
    assert "alice-home" not in text
    assert str(tmp_path) not in text
