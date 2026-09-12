"""load_model(): every accepted CLI model form and the error for each rejected one."""

import sys

import pytest
import torch
from torch import nn

import downshift.adapters as adapters_pkg
from downshift.adapters import hf
from downshift.loading import LoadError, _instantiate, import_object, load_model
from tests.models import clean_mlp, tiny_bert

MLP_FACTORY = "tests.models.clean_mlp:make_model"
MLP_CLASS = "tests.models.clean_mlp:CleanMLP"


@pytest.mark.parametrize("spec", [MLP_FACTORY, MLP_CLASS], ids=["factory", "class"])
def test_import_spec_picks_up_sibling_make_inputs(spec):
    loaded = load_model(spec)

    assert isinstance(loaded.model, nn.Module)
    assert loaded.onnx_path is None
    assert isinstance(loaded.example_inputs, tuple)
    assert len(loaded.example_inputs) == 1
    assert isinstance(loaded.example_inputs[0], torch.Tensor)


def test_explicit_inputs_spec_overrides_sibling():
    loaded = load_model(MLP_FACTORY, inputs="tests.models.clean_mlp:make_inputs")
    assert isinstance(loaded.example_inputs, tuple)
    assert tuple(loaded.example_inputs[0].shape) == (1, 16)


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        ("tests.models.no_such_module:make_model", "can't import"),
        ("tests.models.clean_mlp:no_such_attr", "no attribute"),
        ("tests.models.clean_mlp:make_inputs", "not an nn.Module"),
    ],
    ids=["bad-module", "bad-attr", "not-a-module"],
)
def test_bad_import_specs_raise_load_error(spec, message):
    with pytest.raises(LoadError, match=message):
        load_model(spec)


def test_state_dict_requires_model_class(tmp_path):
    path = tmp_path / "w.pt"
    torch.save(clean_mlp.make_model().state_dict(), path)

    with pytest.raises(LoadError, match="--model-class"):
        load_model(str(path))


def test_state_dict_loads_into_model_class(tmp_path):
    source = clean_mlp.make_model()
    path = tmp_path / "w.pt"
    torch.save(source.state_dict(), path)

    loaded = load_model(str(path), model_class=MLP_CLASS)

    assert isinstance(loaded.model, nn.Module)
    assert loaded.onnx_path is None
    for (name, got), (_, want) in zip(
        loaded.model.named_parameters(), source.named_parameters(), strict=True
    ):
        assert torch.equal(got, want), name


def test_pickled_module_needs_unsafe_load(tmp_path):
    path = tmp_path / "full.pt"
    torch.save(clean_mlp.make_model(), path)

    with pytest.raises(LoadError, match="--unsafe-load"):
        load_model(str(path))

    loaded = load_model(str(path), unsafe_load=True)
    assert isinstance(loaded.model, clean_mlp.CleanMLP)


def test_missing_onnx_file_is_an_error(tmp_path):
    with pytest.raises(LoadError, match="does not exist"):
        load_model(str(tmp_path / "missing.onnx"))


def test_existing_onnx_file_is_passed_through(tmp_path):
    path = tmp_path / "m.onnx"
    path.write_bytes(b"")  # load_model only checks existence; intake parses later

    loaded = load_model(str(path))

    assert loaded.model is None
    assert loaded.onnx_path == path
    assert loaded.source_path == path


def test_import_object_rejects_a_non_import_spec():
    with pytest.raises(LoadError, match="not an import spec"):
        import_object("not a valid spec at all")


def test_instantiate_passes_through_an_existing_module():
    model = clean_mlp.make_model()
    assert _instantiate(model) is model


def test_instantiate_rejects_non_module_non_callable():
    with pytest.raises(LoadError, match="neither an nn.Module nor a callable"):
        _instantiate(42)


def test_checkpoint_load_failure_with_unsafe_load_raises_load_error(tmp_path):
    path = tmp_path / "bad.pt"
    path.write_bytes(b"not a real checkpoint")

    with pytest.raises(LoadError, match="failed to load"):
        load_model(str(path), unsafe_load=True)


def test_checkpoint_with_unexpected_payload_type_is_rejected(tmp_path):
    path = tmp_path / "list.pt"
    torch.save([1, 2, 3], path)

    with pytest.raises(LoadError, match="expected a state dict"):
        load_model(str(path))


def test_unrecognized_existing_file_suffix_is_rejected(tmp_path):
    path = tmp_path / "model.xyz"
    path.write_text("hi")

    with pytest.raises(LoadError, match="don't know how to load"):
        load_model(str(path))


def test_missing_path_with_a_suffix_is_reported_as_not_existing(tmp_path):
    missing = tmp_path / "missing.xyz"

    with pytest.raises(LoadError, match="does not exist"):
        load_model(str(missing))


def test_hf_extra_not_installed_reports_a_helpful_error(monkeypatch):
    monkeypatch.delattr(adapters_pkg, "hf", raising=False)
    monkeypatch.setitem(sys.modules, "downshift.adapters.hf", None)

    with pytest.raises(LoadError, match=r"\[hf\] extra"):
        load_model("org/repo")


def test_hf_hub_load_failure_is_wrapped_in_load_error(monkeypatch):
    def boom(repo_id):
        raise OSError("network unreachable")

    monkeypatch.setattr(hf, "load_pretrained", boom)

    with pytest.raises(LoadError, match="can't load"):
        load_model("org/repo")


def test_unknown_string_is_treated_as_hf_repo_id(monkeypatch):
    seen = []

    def fake_load(repo_id):
        seen.append(repo_id)
        return tiny_bert.make_model()

    monkeypatch.setattr(hf, "load_pretrained", fake_load)

    loaded = load_model("org/tiny-bert")

    assert seen == ["org/tiny-bert"]
    assert loaded.adapter_hint == "hf"
    assert isinstance(loaded.model, nn.Module)
    assert loaded.example_inputs is None
