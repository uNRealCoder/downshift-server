"""HF adapter through the full verdict pipeline on a tiny random BERT."""

import pytest

pytest.importorskip("transformers")

import downshift  # noqa: E402
from tests.models import tiny_bert  # noqa: E402


@pytest.fixture(scope="module")
def verdict() -> downshift.ExportVerdict:
    return downshift.check(tiny_bert.make_model(), tiny_bert.make_inputs())


def test_bert_exports_clean(verdict):
    assert verdict.status == "CLEAN", verdict.reason
    assert verdict.model_family == "hf-transformers"
    assert verdict.recommended_backend == "onnxruntime"
    assert verdict.warnings == []


def test_bert_inputs_and_dynamic_dims(verdict):
    assert verdict.input_names == ("input_ids", "attention_mask")
    assert verdict.dynamic_dims == {"input_ids": [0, 1], "attention_mask": [0, 1]}


def test_bert_generalises_across_batch_and_sequence(verdict):
    assert verdict.numerics is not None
    assert verdict.numerics.passed
    assert verdict.numerics.shape_generalization is True


def test_bert_without_inputs_uses_config_derived_inputs():
    verdict = downshift.check(tiny_bert.make_model())
    assert verdict.status == "CLEAN", verdict.reason
    assert verdict.input_names == ("input_ids", "attention_mask")


def test_bert_at_max_position_embeddings_is_clean():
    """The example sits at the model's own sequence-length ceiling; the sampler must not
    push a varied sample past it (that used to be an exit-4 crash, not a verdict)."""
    model = tiny_bert.make_model()  # max_position_embeddings=64
    inputs = tiny_bert.make_inputs(batch=1, seq=64)

    verdict = downshift.check(model, inputs)

    assert verdict.status == "CLEAN", verdict.reason


def test_bert_at_max_position_embeddings_serves():
    from downshift.loading import LoadedModel
    from downshift.serve.engine import prepare_serving

    model = tiny_bert.make_model()
    inputs = tiny_bert.make_inputs(batch=1, seq=64)
    loaded = LoadedModel(source="tiny_bert", model=model, example_inputs=inputs, adapter_hint="hf")

    state = prepare_serving(loaded)

    assert state.ready
    assert state.verdict.status == "CLEAN", state.verdict.reason
