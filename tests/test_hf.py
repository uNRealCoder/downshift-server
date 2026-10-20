"""The HF adapter in the full verdict pipeline, on a small BERT with random weights."""

import pytest

pytest.importorskip("transformers")

import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import downshift
from downshift.cli import main
from downshift.loading import LoadError, LoadSpec, load_model
from downshift.serve.engine import ServeOptions, prepare_serving
from tests.models import hf_repo, tiny_bert


@pytest.fixture(scope="module")
def verdict() -> downshift.ExportVerdict:
    return downshift.check(tiny_bert.make_model(), tiny_bert.make_inputs())


@pytest.mark.needs_torch_26
def test_bert_exports_clean(verdict):
    assert verdict.status == "CLEAN", verdict.reason
    assert verdict.model_family == "hf"
    assert verdict.recommended_backend == "onnxruntime"
    assert verdict.warnings == []


def test_bert_inputs_and_dynamic_dims(verdict):
    assert verdict.input_names == ("input_ids", "attention_mask")
    assert verdict.dynamic_dims == {"input_ids": [0, 1], "attention_mask": [0, 1]}


@pytest.mark.needs_torch_26
def test_bert_generalises_across_batch_and_sequence(verdict):
    assert verdict.numerics is not None
    assert verdict.numerics.passed
    assert verdict.numerics.shape_generalization is True


@pytest.mark.needs_torch_26
def test_bert_without_inputs_uses_config_derived_inputs():
    verdict = downshift.check(tiny_bert.make_model())
    assert verdict.status == "CLEAN", verdict.reason
    assert verdict.input_names == ("input_ids", "attention_mask")


@pytest.mark.needs_torch_26
def test_bert_at_max_position_embeddings_is_clean():
    """The example is at the own sequence-length ceiling of the model. The sampler must not push
    a varied sample above it. (This was an exit-4 crash and not a verdict.)"""
    model = tiny_bert.make_model()  # max_position_embeddings=64
    inputs = tiny_bert.make_inputs(batch=1, seq=64)

    verdict = downshift.check(model, inputs)

    assert verdict.status == "CLEAN", verdict.reason


@pytest.mark.needs_torch_26
def test_bert_at_max_position_embeddings_serves():
    from downshift.loading import LoadedModel

    model = tiny_bert.make_model()
    inputs = tiny_bert.make_inputs(batch=1, seq=64)
    loaded = LoadedModel(source="tiny_bert", model=model, example_inputs=inputs, adapter_hint="hf")

    state = prepare_serving(loaded)

    assert state.ready
    assert state.verdict.status == "CLEAN", state.verdict.reason


@pytest.mark.needs_torch_26
def test_axis_max_seq_pins_a_sample_and_lowers_the_served_bound():
    verdict = downshift.check(
        tiny_bert.make_model(), tiny_bert.make_inputs(), axis_max={"seq": 24}, k=4
    )

    assert verdict.status == "CLEAN", verdict.reason
    assert {fact.name: fact.served_max for fact in verdict.axes}["seq"] == 24
    assert verdict.numerics is not None
    assert verdict.numerics.sample_shapes[1][0] == (1, 24)


def test_axis_max_above_the_position_limit_is_a_value_error():
    with pytest.raises(ValueError, match="seq=65 is above the limit of 64"):
        downshift.check(tiny_bert.make_model(), tiny_bert.make_inputs(), axis_max={"seq": 65})


# --- refusals ---


def _cli(*args: str):
    from typer.testing import CliRunner

    return CliRunner().invoke(main.app, list(args))


def test_text_generator_without_a_recipe_is_refused(tmp_path):
    path = hf_repo.write_decoder_repo(tmp_path)

    with pytest.raises(LoadError, match="Qwen2ForCausalLM: this model generates text"):
        load_model(LoadSpec(path))
    result = _cli("check", path)

    assert result.exit_code == main.EXIT_USAGE
    assert "--pooling lasttoken" in result.output


def test_text_generator_with_pooling_none_is_refused(tmp_path):
    path = hf_repo.write_decoder_repo(tmp_path, recipe="lasttoken")

    with pytest.raises(LoadError, match="generates text"):
        load_model(LoadSpec(path, pooling="none"))


@pytest.mark.parametrize("recipe_file", [True, False])
def test_text_generator_with_a_recipe_loads_the_backbone(tmp_path, recipe_file):
    path = hf_repo.write_decoder_repo(tmp_path, recipe="lasttoken" if recipe_file else None)

    loaded = load_model(LoadSpec(path, pooling=None if recipe_file else "lasttoken"))

    assert type(loaded.model).__name__ == "Qwen2Model"


@pytest.mark.parametrize("filename", ["config.json", "tokenizer_config.json"])
def test_auto_map_is_refused_in_either_file(tmp_path, filename):
    path = hf_repo.write_decoder_repo(tmp_path, recipe="lasttoken")
    target = Path(path) / filename
    data = json.loads(target.read_text())
    data["auto_map"] = {"AutoModel": "modeling_custom.CustomModel"}
    target.write_text(json.dumps(data))

    with pytest.raises(LoadError, match=f"{filename} has an auto_map"):
        load_model(LoadSpec(path))
    assert _cli("check", path).exit_code == main.EXIT_USAGE


def test_decoder_fixture_auto_map_toggle(tmp_path):
    path = hf_repo.write_decoder_repo(tmp_path, recipe="lasttoken", auto_map=True)

    with pytest.raises(LoadError, match="auto_map"):
        load_model(LoadSpec(path))


class _Payload:
    def __init__(self, marker: Path) -> None:
        self.marker = marker

    def __reduce__(self):
        return (Path.write_text, (self.marker, "ran"))


def test_a_pickle_only_repo_does_not_run_its_payload(tmp_path):
    marker = tmp_path / "pwned"
    repo = tmp_path / "repo"
    repo.mkdir()
    hf_repo.write_encoder_repo(repo)
    for weights in repo.glob("*.safetensors"):
        weights.unlink()
    (repo / "pytorch_model.bin").write_bytes(pickle.dumps(_Payload(marker)))

    try:
        load_model(LoadSpec(str(repo)))
    except Exception:  # noqa: BLE001 - a refusal and a failure to load are both acceptable
        pass

    assert not marker.exists()


# --- a decoder embedder: export, left padding, releasing the torch model ---------------------


def _decoder_prepared(path: str):
    from downshift.core.verdict import prepare_model

    loaded = load_model(LoadSpec(path))
    return loaded, prepare_model(loaded.model, loaded.example_inputs, None, None)


@pytest.mark.needs_torch_26
def test_decoder_embedder_exports_clean_on_onnxruntime(tmp_path):
    path = hf_repo.write_decoder_repo(tmp_path, recipe="lasttoken")
    loaded = load_model(LoadSpec(path))

    verdict = downshift.check(loaded.model, loaded.example_inputs)

    assert verdict.status == "CLEAN", verdict.reason
    assert verdict.recommended_backend == "onnxruntime"
    assert verdict.numerics is not None and verdict.numerics.shape_generalization is True


def test_the_padding_side_is_read_from_the_tokenizer_config(tmp_path):
    from downshift.adapters.embedding import PADDING_SIDE_ATTR

    left = hf_repo.write_decoder_repo(tmp_path / "l", recipe="lasttoken", padding_side="left")
    right = hf_repo.write_decoder_repo(tmp_path / "r", recipe="lasttoken", padding_side="right")

    assert getattr(load_model(LoadSpec(left)).model, PADDING_SIDE_ATTR) == "left"
    assert getattr(load_model(LoadSpec(right)).model, PADDING_SIDE_ATTR) == "right"


def test_a_verify_sample_has_leading_mask_zeros_under_left_padding(tmp_path):
    (tmp_path / "l").mkdir()
    (tmp_path / "r").mkdir()
    left = hf_repo.write_decoder_repo(tmp_path / "l", recipe="lasttoken", padding_side="left")
    right = hf_repo.write_decoder_repo(tmp_path / "r", recipe="lasttoken", padding_side="right")

    _, prepared = _decoder_prepared(left)
    masks = [prepared.vary_fn(i)[1] for i in range(2, 12)]  # type: ignore[misc]
    assert any(m[:, 0].eq(0).any() for m in masks)
    # An attended token is always at the end of the row.
    assert all(m[:, -1].eq(1).all() for m in masks)

    _, prepared = _decoder_prepared(right)
    masks = [prepared.vary_fn(i)[1] for i in range(2, 12)]  # type: ignore[misc]
    assert all(m[:, 0].eq(1).all() for m in masks)
    assert any(m[:, -1].eq(0).any() for m in masks)


@pytest.mark.needs_torch_26
def test_the_torch_model_is_released_after_a_clean_onnx_verdict(tmp_path):
    import gc
    import weakref

    path = hf_repo.write_decoder_repo(tmp_path, recipe="lasttoken")
    loaded = load_model(LoadSpec(path))
    ref = weakref.ref(loaded.model)

    state = prepare_serving(loaded)
    gc.collect()

    assert state.backend.name == "onnxruntime"
    assert ref() is None
    assert loaded.model is None
    assert state.verdict.prepared is None
    assert state.axis_bounds  # read from the Prepared before downshift dropped it
    assert state.verdict.numerics is not None


def test_the_torch_backend_keeps_its_model(tmp_path):

    path = hf_repo.write_decoder_repo(tmp_path, recipe="lasttoken")

    state = prepare_serving(load_model(LoadSpec(path)), ServeOptions(backend="torch"))

    assert state.backend.name == "torch"
    assert state.verdict.prepared is not None


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
