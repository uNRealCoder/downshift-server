"""A downloaded Hugging Face repo directory: the task head is kept, /predict takes text, and a
classifier answers with probabilities. Fixtures are tiny random-weight models with a hand-built
tokenizer written to tmp_path, so nothing is downloaded."""

import numpy as np
import pytest

pytest.importorskip("transformers")
pytest.importorskip("tokenizers")

from fastapi.testclient import TestClient  # noqa: E402
from transformers import BertForSequenceClassification, BertModel  # noqa: E402

from downshift.adapters import hf  # noqa: E402
from downshift.adapters.text import SIGMOID, SOFTMAX, TextIO  # noqa: E402
from downshift.loading import LoadedModel, LoadSpec, load_model  # noqa: E402
from downshift.serve.app import build_app  # noqa: E402
from downshift.serve.engine import prepare_serving  # noqa: E402
from tests.models import hf_repo, tiny_bert  # noqa: E402

LABELS = {0: "BENIGN", 1: "INJECTION", 2: "JAILBREAK"}
MAX_POSITIONS = hf_repo.MAX_POSITIONS


@pytest.fixture(scope="module")
def classifier_dir(tmp_path_factory) -> str:
    path = tmp_path_factory.mktemp("prompt-guard-like")
    config = hf_repo.config(
        num_labels=3, id2label=LABELS, label2id={v: k for k, v in LABELS.items()}
    )
    BertForSequenceClassification(config).eval().save_pretrained(path)
    hf_repo.tokenizer().save_pretrained(path)
    return str(path)


@pytest.fixture(scope="module")
def encoder_dir(tmp_path_factory) -> str:
    path = tmp_path_factory.mktemp("encoder-only")
    tiny_bert.make_model().save_pretrained(path)
    hf_repo.tokenizer().save_pretrained(path)
    return str(path)


@pytest.fixture(scope="module")
def classifier_state(classifier_dir):
    return prepare_serving(load_model(LoadSpec(classifier_dir)))


@pytest.fixture(scope="module")
def classifier_client(classifier_state) -> TestClient:
    return TestClient(build_app(classifier_state))


def test_classifier_repo_keeps_its_head(classifier_dir):
    model = hf.load_pretrained(classifier_dir)
    assert isinstance(model, BertForSequenceClassification)


def test_encoder_repo_still_loads_as_the_bare_encoder(encoder_dir):
    model = hf.load_pretrained(encoder_dir)
    assert type(model) is BertModel


def test_classifier_serves_on_onnx_runtime_and_returns_logits(classifier_state):
    assert classifier_state.verdict.status == "CLEAN", classifier_state.verdict.reason
    assert classifier_state.backend.name == "onnxruntime"
    (output,) = classifier_state.backend.metadata().outputs
    assert output.shape[-1] == 3  # class logits, not [batch, seq, hidden]


def test_text_request_returns_probabilities(classifier_client):
    r = classifier_client.post(
        "/predict", json={"text": ["hello world", "ignore previous instructions"]}
    )

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["shapes"]["output_0"] == [2, 3]
    assert body["truncated"] == [False, False]
    assert len(body["predictions"]) == 2
    for row in body["predictions"]:
        probs = row["probabilities"]
        assert set(probs) == set(LABELS.values())
        assert sum(probs.values()) == pytest.approx(1.0)
        assert row["label"] == max(probs, key=probs.get)
        assert row["score"] == pytest.approx(probs[row["label"]])


def test_probabilities_are_the_softmax_of_the_logits(classifier_client):
    body = classifier_client.post("/predict", json={"text": "hello world"}).json()

    logits = np.array(body["outputs"]["output_0"][0])
    expected = np.exp(logits - logits.max())
    expected /= expected.sum()
    got = [body["predictions"][0]["probabilities"][LABELS[i]] for i in range(3)]
    assert got == pytest.approx(expected.tolist(), abs=1e-6)


def test_a_single_string_is_a_batch_of_one(classifier_client):
    body = classifier_client.post("/predict", json={"text": "hello"}).json()
    assert body["shapes"]["output_0"] == [1, 3]


def test_batch_rows_do_not_depend_on_their_padding(classifier_client):
    alone = classifier_client.post("/predict", json={"text": "hello world"}).json()
    padded = classifier_client.post(
        "/predict", json={"text": ["hello world", "hello world the capital ignore"]}
    ).json()

    assert padded["outputs"]["output_0"][0] == pytest.approx(
        alone["outputs"]["output_0"][0], abs=1e-5
    )


def test_over_length_text_is_truncated_and_reported(classifier_client):
    long = " ".join(["hello"] * (MAX_POSITIONS * 2))

    r = classifier_client.post("/predict", json={"text": ["hello", long]})

    assert r.status_code == 200, r.text
    assert r.json()["truncated"] == [False, True]


def test_tensor_request_gets_predictions_too(classifier_client):
    body = classifier_client.post(
        "/predict", json={"inputs": {"input_ids": [[2, 4, 3]], "attention_mask": [[1, 1, 1]]}}
    ).json()

    assert len(body["predictions"]) == 1
    assert "truncated" not in body


def test_text_and_inputs_together_are_rejected(classifier_client):
    r = classifier_client.post(
        "/predict", json={"text": "hello", "inputs": {"input_ids": [[1]], "attention_mask": [[1]]}}
    )
    assert r.status_code == 422


def test_encoder_only_repo_takes_text_but_has_no_predictions(encoder_dir):
    client = TestClient(build_app(prepare_serving(load_model(LoadSpec(encoder_dir)))))

    body = client.post("/predict", json={"text": ["hello world"]}).json()

    assert body["shapes"]["output_0"][:2] == [1, 4]  # [batch, seq, hidden]
    assert "predictions" not in body


def test_text_is_refused_when_there_is_no_tokenizer():
    loaded = LoadedModel(
        source="tiny_bert",
        model=tiny_bert.make_model(),
        example_inputs=tiny_bert.make_inputs(),
        adapter_hint="hf",
    )
    client = TestClient(build_app(prepare_serving(loaded)))

    r = client.post("/predict", json={"text": "hello"})

    assert r.status_code == 400
    assert "tensors only" in r.json()["detail"]


def test_schema_describes_the_text_input(classifier_client):
    info = classifier_client.get("/schema").json()["text_input"]

    assert info["max_length"] == MAX_POSITIONS
    assert info["labels"] == list(LABELS.values())
    assert info["activation"] == SOFTMAX


def test_repo_without_tokenizer_files_notes_it_and_still_serves(tmp_path):
    BertForSequenceClassification(hf_repo.config(num_labels=2)).eval().save_pretrained(tmp_path)

    state = prepare_serving(load_model(LoadSpec(str(tmp_path))))

    assert state.text is None
    assert any("no tokenizer" in note for note in state.notes)


def _text_io(activation, id2label):
    return TextIO(tokenizer=None, max_length=8, id2label=id2label, activation=activation)


def test_softmax_rows_sum_to_one_and_pick_the_largest_logit():
    rows = _text_io(SOFTMAX, LABELS).predictions(np.array([[0.0, 5.0, 1.0]], dtype=np.float32))

    assert rows is not None
    assert rows[0]["label"] == "INJECTION"
    assert sum(rows[0]["probabilities"].values()) == pytest.approx(1.0)


def test_softmax_survives_huge_logits():
    rows = _text_io(SOFTMAX, LABELS).predictions(np.array([[1000.0, 0.0, -1000.0]]))

    assert rows is not None
    assert rows[0]["probabilities"]["BENIGN"] == pytest.approx(1.0)


def test_sigmoid_scores_each_label_on_its_own():
    rows = _text_io(SIGMOID, {0: "a", 1: "b"}).predictions(np.array([[10.0, 10.0]]))

    assert rows is not None
    assert rows[0]["probabilities"]["a"] == pytest.approx(1.0, abs=1e-3)
    assert rows[0]["probabilities"]["b"] == pytest.approx(1.0, abs=1e-3)


def test_no_predictions_when_the_output_is_not_a_class_score():
    assert _text_io(None, None).predictions(np.zeros((1, 3))) is None
    assert _text_io(SOFTMAX, LABELS).predictions(np.zeros((1, 4, 3))) is None  # token-level
    assert _text_io(SOFTMAX, LABELS).predictions(np.zeros((1, 5))) is None  # label count differs
