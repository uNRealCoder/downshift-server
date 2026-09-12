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
