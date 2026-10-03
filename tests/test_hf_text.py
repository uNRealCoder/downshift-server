"""A downloaded Hugging Face repo directory: the task head is kept, /predict takes text, and a
classifier answers with probabilities. Fixtures are tiny random-weight models with a hand-built
tokenizer written to tmp_path, so nothing is downloaded."""

import numpy as np
import pytest

pytest.importorskip("transformers")
pytest.importorskip("tokenizers")

import torch  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from transformers import BertForSequenceClassification, BertModel  # noqa: E402

import downshift  # noqa: E402
from downshift.adapters.text import SIGMOID, SOFTMAX, TextIO  # noqa: E402
from downshift.hf_repo import load_pretrained  # noqa: E402
from downshift.loading import LoadedModel, LoadSpec, load_model  # noqa: E402
from downshift.serve import app_for  # noqa: E402
from downshift.serve.app import build_app  # noqa: E402
from downshift.serve.engine import (  # noqa: E402
    ServeOptions,
    prepare_serving,
    serving_state_from_artifact,
)
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
def embedding_dir(tmp_path_factory) -> str:
    path = tmp_path_factory.mktemp("embedding-repo")
    return hf_repo.write_encoder_repo(path)


@pytest.fixture(scope="module")
def embedding_onnx(embedding_dir, tmp_path_factory) -> str:
    """embedding_dir's model, already exported to a standalone .onnx: what a user who ran
    optimum, Olive, or their own export script would hand to `serve --tokenizer-from`."""
    loaded = load_model(LoadSpec(embedding_dir))
    input_ids = torch.randint(0, 100, (2, 8))
    example = (input_ids, torch.ones_like(input_ids))
    out = tmp_path_factory.mktemp("embedding-onnx") / "model.onnx"
    verdict = downshift.export(loaded.model, out, example)
    assert verdict.status == "CLEAN", verdict.reason
    return str(out)


@pytest.fixture(scope="module")
def classifier_state(classifier_dir):
    return prepare_serving(load_model(LoadSpec(classifier_dir)))


@pytest.fixture(scope="module")
def classifier_client(classifier_state) -> TestClient:
    return TestClient(build_app(classifier_state))


def test_classifier_repo_keeps_its_head(classifier_dir):
    model = load_pretrained(classifier_dir)
    assert isinstance(model, BertForSequenceClassification)


def test_encoder_repo_still_loads_as_the_bare_encoder(encoder_dir):
    model = load_pretrained(encoder_dir)
    assert type(model) is BertModel


@pytest.mark.needs_torch_26
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


def test_over_length_text_is_refused_not_truncated(classifier_client):
    long = " ".join(["hello"] * (MAX_POSITIONS * 2))

    r = classifier_client.post("/predict", json={"text": ["hello", long]})

    assert r.status_code == 400, r.text
    assert "refused rather than cut" in r.text
    assert "row 1:" in r.text and "row 0:" not in r.text


def test_tensor_request_gets_predictions_too(classifier_client):
    body = classifier_client.post(
        "/predict", json={"inputs": {"input_ids": [[2, 4, 3]], "attention_mask": [[1, 1, 1]]}}
    ).json()

    assert len(body["predictions"]) == 1


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


# --- a bare .onnx served with --tokenizer-from pointing at its Hugging Face repo --------------


@pytest.mark.needs_torch_26
def test_onnx_plus_tokenizer_from_gets_pooling_and_text_input(embedding_onnx, embedding_dir):
    state = prepare_serving(load_model(LoadSpec(embedding_onnx)), tokenizer_from=embedding_dir)

    assert state.hf_source == embedding_dir
    assert state.embedding is not None
    assert state.embedding.pooling == "mean"
    assert state.text is not None
    # No --reference was given: --tokenizer-from alone does not verify numerics.
    assert state.verdict.status == "UNVERIFIED"

    client = TestClient(build_app(state))
    body = client.post("/predict", json={"text": ["hello world"]}).json()
    assert body["shapes"]["output_0"] == [1, hf_repo.HIDDEN]  # pooled, not token-level


@pytest.fixture(scope="module")
def encoder_onnx(encoder_dir, tmp_path_factory) -> str:
    """A bare encoder exported on its own: token vectors out, no pooling in the graph."""
    loaded = load_model(LoadSpec(encoder_dir))
    input_ids = torch.randint(0, 100, (2, 8))
    example = (input_ids, torch.ones_like(input_ids))
    out = tmp_path_factory.mktemp("encoder-onnx") / "model.onnx"
    verdict = downshift.export(loaded.model, out, example)
    assert verdict.status == "CLEAN", verdict.reason
    return str(out)


@pytest.mark.needs_torch_26
def test_tokenizer_from_does_not_claim_pooling_a_token_level_graph(encoder_onnx, embedding_dir):
    """embedding_dir declares mean pooling, but the served graph is the bare encoder: /schema
    must not report an embedding recipe the graph never applies, and should say why the
    output is token-level instead."""
    opts = ServeOptions(pooling="cls")
    state = prepare_serving(load_model(LoadSpec(encoder_onnx)), opts, tokenizer_from=embedding_dir)

    assert state.text is not None
    assert state.embedding is None
    assert any("--pooling/--normalize ignored" in note for note in state.notes)

    body = TestClient(build_app(state)).get("/schema").json()
    assert body["embedding"] is None
    assert any("--tokenizer-from only supplies the tokenizer" in n for n in body["notes"])


@pytest.mark.needs_torch_26
def test_reference_and_tokenizer_from_are_independent(embedding_onnx, embedding_dir):
    """--reference verifies numerics; --tokenizer-from supplies the tokenizer; passing both,
    naming the same directory, does both jobs at once without either implying the other."""
    state = prepare_serving(
        load_model(LoadSpec(embedding_onnx)),
        reference=load_model(LoadSpec(embedding_dir)),
        tokenizer_from=embedding_dir,
    )

    assert state.hf_source == embedding_dir
    assert state.text is not None
    assert state.verdict.status == "CLEAN", state.verdict.reason


@pytest.mark.needs_torch_26
def test_onnx_alone_has_no_text_input_and_is_unverified(embedding_onnx):
    state = prepare_serving(load_model(LoadSpec(embedding_onnx)))

    assert state.hf_source is None
    assert state.text is None
    assert state.embedding is None
    assert state.verdict.status == "UNVERIFIED"


@pytest.mark.needs_torch_26
def test_reference_alone_does_not_turn_on_text_input(embedding_onnx, embedding_dir):
    """--reference is purely numeric: naming an HF repo directory there, with no
    --tokenizer-from, verifies the graph but does not attach a tokenizer."""
    state = prepare_serving(
        load_model(LoadSpec(embedding_onnx)), reference=load_model(LoadSpec(embedding_dir))
    )

    assert state.hf_source is None
    assert state.text is None
    assert state.verdict.status == "CLEAN", state.verdict.reason


def test_hf_dir_as_model_wins_over_tokenizer_from(encoder_dir, embedding_dir):
    """When MODEL is itself a Hugging Face repo directory, its own tokenizer is used even if
    --tokenizer-from names a different one; --tokenizer-from's role only applies to a bare
    .onnx MODEL, which has no tokenizer of its own to prefer."""
    state = prepare_serving(load_model(LoadSpec(encoder_dir)), tokenizer_from=embedding_dir)

    assert state.hf_source == encoder_dir


def test_tokenizer_from_must_be_an_hf_repo_directory(tmp_path):
    from downshift.loading import LoadError, resolve_tokenizer_source

    not_a_dir = tmp_path / "weights.pt"
    not_a_dir.write_bytes(b"")
    with pytest.raises(LoadError, match="config.json"):
        resolve_tokenizer_source(str(not_a_dir))

    no_config = tmp_path / "empty-dir"
    no_config.mkdir()
    with pytest.raises(LoadError, match="config.json"):
        resolve_tokenizer_source(str(no_config))


@pytest.mark.needs_torch_26
def test_worker_rebuild_attaches_text_from_a_carried_hf_source(embedding_onnx, embedding_dir):
    """serving_state_from_artifact is what a `--workers N` worker calls to rebuild from the
    parent's already-verified onnx artifact; it takes no reference model or --tokenizer-from
    string to re-validate, so a worker does neither - the resolved hf_source path (shipped in
    ArtifactHandoff) is enough on its own."""
    base = prepare_serving(load_model(LoadSpec(embedding_onnx)))
    assert base.text is None  # no --tokenizer-from given here: nothing to carry yet

    rebuilt = serving_state_from_artifact(
        embedding_onnx,
        base.verdict.onnx_path,
        base.verdict,
        base.options,
        base.input_names,
        hf_source=embedding_dir,
    )

    assert rebuilt.text is not None
    assert rebuilt.embedding is not None
    assert rebuilt.embedding.pooling == "mean"


@pytest.mark.needs_torch_26
def test_app_for_takes_tokenizer_from_independent_of_reference(embedding_onnx, embedding_dir):
    """The library entry point offers the same split as the CLI: tokenizer_from= supplies
    text input, reference= verifies numerics, and neither implies the other."""
    client = TestClient(app_for(embedding_onnx, tokenizer_from=embedding_dir, warmup=1))

    r = client.get("/schema").json()
    assert r["text_input"] is not None
    assert r["embedding"]["pooling"] == "mean"

    body = client.post("/predict", json={"text": ["hello world"]}).json()
    assert body["shapes"]["output_0"] == [1, hf_repo.HIDDEN]


# --- decoder embedder: left and right padding, named prompts (0.5.0 E3, E5) -------------------

import json  # noqa: E402

TEXTS = ["hello", "the capital hello world", "world the"]


def _decoder_state(path):
    return prepare_serving(load_model(LoadSpec(str(path))), ServeOptions(warmup=0))


def _embed(client: TestClient, **body) -> np.ndarray:
    response = client.post("/predict", json=body)
    assert response.status_code == 200, response.text
    return np.asarray(response.json()["outputs"]["output_0"])


@pytest.mark.needs_torch_26
def test_decoder_text_gives_one_vector_per_row_equal_under_either_padding(tmp_path):
    (tmp_path / "left").mkdir()
    (tmp_path / "right").mkdir()
    left = hf_repo.write_decoder_repo(tmp_path / "left", recipe="lasttoken", padding_side="left")
    right = hf_repo.write_decoder_repo(tmp_path / "right", recipe="lasttoken", padding_side="right")
    left_state = _decoder_state(left)
    right_state = _decoder_state(right)
    assert left_state.backend.name == right_state.backend.name == "onnxruntime"
    assert left_state.text is not None and left_state.text.tokenizer.padding_side == "left"
    assert right_state.text is not None and right_state.text.tokenizer.padding_side == "right"

    on_left = _embed(TestClient(build_app(left_state)), text=TEXTS)
    on_right = _embed(TestClient(build_app(right_state)), text=TEXTS)

    assert on_left.shape == (len(TEXTS), hf_repo.DECODER_HIDDEN)
    assert np.allclose(on_left, on_right, atol=1e-4)
    assert np.allclose(np.linalg.norm(on_left, axis=1), 1.0, atol=1e-4)  # the recipe normalises


@pytest.fixture(scope="module")
def prompted_client(tmp_path_factory) -> TestClient:
    path = tmp_path_factory.mktemp("prompted")
    hf_repo.write_decoder_repo(
        path, recipe="lasttoken", prompts={"query": "the ", "document": "world "}
    )
    return TestClient(build_app(_decoder_state(path)))


@pytest.mark.needs_torch_26
def test_prompt_name_prepends_the_named_prompt_before_tokenizing(prompted_client):
    named = _embed(prompted_client, text=["capital"], prompt_name="query")
    typed = _embed(prompted_client, text=["the capital"])
    plain = _embed(prompted_client, text=["capital"])

    assert np.allclose(named, typed, atol=1e-6)
    assert not np.array_equal(named, plain)


@pytest.mark.needs_torch_26
def test_unknown_prompt_name_is_a_400_listing_the_names_with_a_cut_echo(prompted_client):
    response = prompted_client.post("/predict", json={"text": ["hi"], "prompt_name": "x" * 500})

    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "['document', 'query']" in detail
    assert "x" * 64 not in detail  # the repr, quotes included, is cut at 64 characters
    assert len(detail) < 200


@pytest.mark.needs_torch_26
def test_prompt_name_without_text_is_a_400(prompted_client):
    response = prompted_client.post(
        "/predict",
        json={"inputs": {"input_ids": [[1]], "attention_mask": [[1]]}, "prompt_name": "query"},
    )

    assert response.status_code == 400
    assert "text" in response.json()["detail"]


@pytest.mark.needs_torch_26
def test_default_prompt_name_applies_when_a_request_names_none(tmp_path):
    path = hf_repo.write_decoder_repo(tmp_path, recipe="lasttoken", prompts={"query": "the "})
    config_file = tmp_path / "config_sentence_transformers.json"
    config = json.loads(config_file.read_text())
    config["default_prompt_name"] = "query"
    config_file.write_text(json.dumps(config))
    client = TestClient(build_app(_decoder_state(path)))

    schema = client.get("/schema").json()["embedding"]
    assert schema["prompts"] == {"query": "the "}
    assert schema["default_prompt"] == "query"
    assert np.allclose(
        _embed(client, text=["capital"]), _embed(client, text=["capital"], prompt_name="query")
    )
