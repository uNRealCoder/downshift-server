"""Embedding models from a downloaded sentence-transformers repo. Downshift reads the recipe
from the own files of the repo and applies it inside the exported graph. The result matches the
definition of sentence-transformers (masked pooling, then L2 normalization). The fixtures are
small models with random weights."""

import json
import re
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("transformers")
pytest.importorskip("tokenizers")

import torch  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from transformers import AutoModel, BertForSequenceClassification  # noqa: E402

from downshift.adapters.embedding import (  # noqa: E402
    RecipeError,
    pool,
    read_recipe,
    resolve_recipe,
)
from downshift.cli import main, runtime  # noqa: E402
from downshift.loading import LoadError, LoadSpec, load_model  # noqa: E402
from downshift.serve.app import build_app  # noqa: E402
from downshift.serve.engine import prepare_serving  # noqa: E402
from downshift.serve.options import BackendChoice, ServeOptions  # noqa: E402
from tests.models import hf_repo  # noqa: E402

TEXTS = ["hello world", "ignore previous instructions the capital", "the"]


def _reference(path: str, texts: list[str], max_length: int, mode: str, normalize: bool):
    """sentence-transformers' own definition, spelled out: masked pooling then L2 normalise."""
    tokenizer = hf_repo.tokenizer()
    model = AutoModel.from_pretrained(path, local_files_only=True).eval()
    enc = tokenizer(
        texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
    )
    with torch.no_grad():
        hidden = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])[0]
    mask = enc["attention_mask"]
    out = []
    for row, keep in zip(hidden, mask, strict=True):
        tokens = row[keep.bool()]
        if mode == "mean":
            vec = tokens.mean(0)
        elif mode == "cls":
            vec = tokens[0]
        elif mode == "max":
            vec = tokens.max(0).values
        elif mode == "lasttoken":
            vec = tokens[-1]
        elif mode == "weightedmean":
            weights = torch.arange(1, len(tokens) + 1, dtype=tokens.dtype).unsqueeze(-1)
            vec = (tokens * weights).sum(0) / weights.sum()
        else:
            vec = tokens.sum(0) / len(tokens) ** 0.5
        out.append(torch.nn.functional.normalize(vec, dim=0) if normalize else vec)
    return torch.stack(out).numpy()


@pytest.fixture(scope="module")
def repo(tmp_path_factory) -> str:
    return hf_repo.write_encoder_repo(tmp_path_factory.mktemp("st-repo"))


@pytest.fixture(scope="module")
def state(repo):
    return prepare_serving(load_model(LoadSpec(repo)))


@pytest.fixture(scope="module")
def client(state) -> TestClient:
    return TestClient(build_app(state))


# --- reading the recipe -------------------------------------------------------------------


def test_recipe_is_read_from_the_repo(repo):
    recipe = read_recipe(Path(repo))

    assert recipe is not None
    assert (recipe.pooling, recipe.normalize) == ("mean", True)
    assert recipe.max_seq_length == 16
    assert recipe.origin == "modules.json"


def test_repo_without_modules_json_has_no_recipe(tmp_path):
    path = hf_repo.write_encoder_repo(tmp_path, modules=None, pooling=None, max_seq_length=None)

    assert resolve_recipe(path, None, None, has_head=False) is None


@pytest.mark.parametrize(
    ("flag", "mode"),
    [("pooling_mode_lasttoken", "lasttoken"), ("pooling_mode_weightedmean_tokens", "weightedmean")],
)
def test_last_token_and_weighted_mean_flags_are_read_from_the_recipe(tmp_path, flag, mode):
    pooling = {**hf_repo.MEAN_POOLING, "pooling_mode_mean_tokens": False, flag: True}
    path = hf_repo.write_encoder_repo(tmp_path, pooling=pooling)

    recipe = read_recipe(Path(path))

    assert recipe is not None
    assert recipe.pooling == mode


def test_two_pooling_modes_at_once_are_refused(tmp_path):
    pooling = {**hf_repo.MEAN_POOLING, "pooling_mode_cls_token": True}
    path = hf_repo.write_encoder_repo(tmp_path, pooling=pooling)

    with pytest.raises(LoadError, match="2 pooling modes"):
        load_model(LoadSpec(path))


def test_a_dense_module_is_refused_and_none_is_the_way_out(tmp_path):
    dense = {
        "idx": 2,
        "name": "2",
        "path": "2_Dense",
        "type": "sentence_transformers.models.Dense",
    }
    path = hf_repo.write_encoder_repo(tmp_path, modules=[*hf_repo.ST_MODULES[:2], dense])

    with pytest.raises(LoadError, match="Dense.*--pooling none"):
        load_model(LoadSpec(path))
    with pytest.raises(LoadError, match="Dense"):  # a named pooling does not skip the check
        load_model(LoadSpec(path, pooling="mean"))

    assert load_model(LoadSpec(path, pooling="none")).model is not None


# --- serving ------------------------------------------------------------------------------


@pytest.mark.needs_torch_26
def test_embedding_serves_on_onnx_runtime_with_the_pooling_in_the_graph(state):
    assert state.verdict.status == "CLEAN", state.verdict.reason
    assert state.backend.name == "onnxruntime"
    (output,) = state.backend.metadata().outputs
    assert output.shape[-1] == hf_repo.HIDDEN
    assert len(output.shape) == 2  # [batch, dim], not [batch, seq, dim]
    assert state.embedding is not None
    assert state.embedding.describe() == "mean pooling, L2-normalised"
    assert state.notes == []  # informational, so it is not shown as a warning


def test_served_embeddings_match_the_reference_definition(client, repo):
    body = client.post("/predict", json={"text": TEXTS}).json()

    got = np.array(body["outputs"]["output_0"])
    assert got.shape == (3, hf_repo.HIDDEN)
    assert np.linalg.norm(got, axis=1) == pytest.approx(1.0, abs=1e-5)
    expected = _reference(repo, TEXTS, max_length=16, mode="mean", normalize=True)
    assert got == pytest.approx(expected, abs=1e-5)
    assert "predictions" not in body


def test_a_row_does_not_depend_on_what_it_is_batched_with(client):
    alone = np.array(client.post("/predict", json={"text": TEXTS[0]}).json()["outputs"]["output_0"])
    batched = np.array(client.post("/predict", json={"text": TEXTS}).json()["outputs"]["output_0"])

    assert alone[0] == pytest.approx(batched[0], abs=1e-5)


def test_the_authors_max_seq_length_wins_over_the_position_embeddings(client):
    """The position embeddings allow 32. The repo says that it was trained at 16. A row can
    therefore have at most 16 tokens. One more token is refused and not cut without a message."""
    fifteen = " ".join(["hello"] * 14)  # + [CLS] [SEP] = 16: fits
    seventeen = " ".join(["hello"] * 15)  # 17: one over

    assert client.post("/predict", json={"text": fifteen}).status_code == 200
    r = client.post("/predict", json={"text": [fifteen, seventeen]})
    assert r.status_code == 400
    assert "16 tokens" in r.text


def test_schema_says_what_kind_of_vector_this_is(client):
    schema = client.get("/schema").json()

    assert schema["embedding"] == {
        "pooling": "mean",
        "normalized": True,
        "dimension": hf_repo.HIDDEN,
        "max_seq_length": 16,
        "from": "modules.json",
        "prompts": {},
        "default_prompt": None,
    }
    assert schema["text_input"]["max_length"] == 16


# --- overrides ----------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["cls", "max", "mean_sqrt_len", "lasttoken", "weightedmean"])
def test_pooling_override_changes_the_served_pooling(repo, mode):
    loaded = load_model(LoadSpec(repo, pooling=mode, normalize=False))
    client = TestClient(
        build_app(prepare_serving(loaded, ServeOptions(pooling=mode, normalize=False)))
    )

    body = client.post("/predict", json={"text": TEXTS}).json()

    got = np.array(body["outputs"]["output_0"])
    assert got == pytest.approx(_reference(repo, TEXTS, 16, mode, normalize=False), abs=1e-5)
    assert client.get("/schema").json()["embedding"]["from"] == "modules.json, --pooling changed"


def test_pooling_none_serves_token_vectors_and_says_so(repo):
    loaded = load_model(LoadSpec(repo, pooling="none"))
    state = prepare_serving(loaded, ServeOptions(pooling="none"))
    client = TestClient(build_app(state))

    body = client.post("/predict", json={"text": TEXTS}).json()
    schema = client.get("/schema").json()

    assert len(body["shapes"]["output_0"]) == 3  # [batch, seq, hidden]
    assert schema["embedding"] is None
    assert any("token-level" in note for note in schema["notes"])


def test_pooling_can_be_set_on_a_repo_that_declares_none(tmp_path):
    path = hf_repo.write_encoder_repo(tmp_path, modules=None, pooling=None, max_seq_length=None)

    recipe = resolve_recipe(path, "mean", True, has_head=False)

    assert recipe is not None
    assert (recipe.pooling, recipe.normalize, recipe.origin) == ("mean", True, "--pooling")


def test_normalize_alone_needs_a_recipe_to_change(tmp_path):
    path = hf_repo.write_encoder_repo(tmp_path, modules=None, pooling=None, max_seq_length=None)

    with pytest.raises(RecipeError, match="needs a pooling"):
        resolve_recipe(path, None, True, has_head=False)


def test_no_normalize_turns_off_what_the_repo_declared(repo):
    recipe = resolve_recipe(repo, None, False, has_head=False)

    assert recipe is not None
    assert (recipe.pooling, recipe.normalize) == ("mean", False)


def test_a_classifier_has_nothing_to_pool(tmp_path):
    BertForSequenceClassification(hf_repo.config(num_labels=2)).eval().save_pretrained(tmp_path)

    with pytest.raises(LoadError, match="classification head"):
        load_model(LoadSpec(str(tmp_path), pooling="mean"))


# --- the pooling itself -------------------------------------------------------------------


def _padded():
    hidden = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0], [100.0, -100.0]],  # last position is padding
            [[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]],
        ]
    )
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
    return hidden, mask


def test_mean_ignores_padding():
    hidden, mask = _padded()
    assert pool(hidden, mask, "mean")[0].tolist() == [2.0, 3.0]
    assert pool(hidden, mask, "mean")[1].tolist() == [7.0, 8.0]


def test_max_ignores_padding():
    hidden, mask = _padded()
    assert pool(hidden, mask, "max")[0].tolist() == [3.0, 4.0]  # not the padded 100


def test_cls_takes_the_first_token():
    hidden, mask = _padded()
    assert pool(hidden, mask, "cls").tolist() == [[1.0, 2.0], [5.0, 6.0]]


def test_mean_sqrt_len_divides_by_the_root_of_the_real_length():
    hidden, mask = _padded()
    assert pool(hidden, mask, "mean_sqrt_len")[0].tolist() == pytest.approx(
        [4.0 / 2**0.5, 6.0 / 2**0.5]
    )


def test_last_token_is_the_last_attended_position_under_either_padding():
    row = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    pad = torch.tensor([[100.0, -100.0]])
    right = torch.cat([row, pad])[None]
    left = torch.cat([pad, row])[None]
    right_mask = torch.tensor([[1, 1, 1, 0]])
    left_mask = torch.tensor([[0, 1, 1, 1]])

    assert pool(right, right_mask, "lasttoken").tolist() == [[5.0, 6.0]]
    assert pool(left, left_mask, "lasttoken").tolist() == [[5.0, 6.0]]


def test_last_token_in_a_mixed_batch():
    hidden = torch.arange(24, dtype=torch.float32).reshape(3, 4, 2)
    mask = torch.tensor([[1, 1, 0, 0], [0, 0, 1, 1], [1, 1, 1, 1]])

    assert pool(hidden, mask, "lasttoken").tolist() == [[2.0, 3.0], [14.0, 15.0], [22.0, 23.0]]


def test_weighted_mean_matches_the_reference_formula():
    hidden, mask = _padded()

    def reference(tokens):
        weights = [i + 1 for i in range(len(tokens))]
        return [
            sum(w * t[d] for w, t in zip(weights, tokens, strict=True)) / sum(weights)
            for d in range(2)
        ]

    got = pool(hidden, mask, "weightedmean")

    assert got[0].tolist() == pytest.approx(reference([[1.0, 2.0], [3.0, 4.0]]))
    assert got[1].tolist() == pytest.approx(reference([[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]))


@pytest.mark.needs_torch_26
@pytest.mark.parametrize("mode", ["lasttoken", "weightedmean"])
def test_last_token_and_weighted_mean_export_on_a_repo_without_a_recipe(tmp_path, mode):
    path = hf_repo.write_encoder_repo(tmp_path, modules=None, pooling=None, max_seq_length=None)
    loaded = load_model(LoadSpec(path, pooling=mode))
    state = prepare_serving(loaded, ServeOptions(pooling=mode))

    assert state.verdict.status == "CLEAN", state.verdict.reason
    assert state.embedding is not None
    assert state.embedding.origin == "--pooling"
    client = TestClient(build_app(state))
    got = np.array(client.post("/predict", json={"text": TEXTS}).json()["outputs"]["output_0"])
    assert got == pytest.approx(_reference(path, TEXTS, 32, mode, normalize=False), abs=1e-5)


# --- the CLI flags ------------------------------------------------------------------------


def _cli(*args: str):
    from typer.testing import CliRunner

    return CliRunner().invoke(main.app, list(args))


@pytest.fixture(scope="module")
def onnx_path(state, tmp_path_factory) -> str:
    """The model of the repo, already exported to a standalone .onnx file. It is the equivalent
    of serving the repo directly, for the --tokenizer-from tests below."""
    path = tmp_path_factory.mktemp("onnx") / "model.onnx"
    path.write_bytes(state.verdict.onnx_bytes)
    return str(path)


def _stub_uvicorn_server(monkeypatch) -> dict:
    """A single-worker `serve` binds through uvicorn.Server directly. This stand-in lets the CLI
    test run the real loader thread without a socket. It is a small local copy of
    _fake_uvicorn_server of test_cli.py. It is here and not imported across test modules."""
    import uvicorn

    captured: dict = {}

    def fake_init(self, config) -> None:
        self.config = config
        self.should_exit = False
        captured["app"] = config.app

    def fake_run(self) -> None:
        with TestClient(self.config.app):
            thread = getattr(self.config.app.state, "loader_thread", None)
            if thread is not None:
                thread.join(timeout=30)

    monkeypatch.setattr(uvicorn.Server, "__init__", fake_init)
    monkeypatch.setattr(uvicorn.Server, "run", fake_run)
    return captured


def test_serve_rejects_a_tokenizer_from_that_is_not_an_hf_repo(onnx_path, tmp_path):
    not_a_repo = tmp_path / "not-a-repo"
    not_a_repo.mkdir()

    result = _cli("serve", onnx_path, "--tokenizer-from", str(not_a_repo))

    assert result.exit_code == main.EXIT_USAGE, result.output
    assert "config.json" in result.output


@pytest.mark.needs_torch_26
def test_serve_tokenizer_from_attaches_text_input_end_to_end(monkeypatch, onnx_path, repo):
    captured = _stub_uvicorn_server(monkeypatch)

    result = _cli("serve", onnx_path, "--tokenizer-from", repo, "--warmup", "1")

    assert result.exit_code == 0, result.output
    serving = captured["app"].state.serving
    assert serving.hf_source == repo
    assert serving.text is not None
    assert serving.verdict.status == "UNVERIFIED"  # no --reference given: no numerics check


@pytest.mark.needs_torch_26
def test_serve_reference_alone_does_not_attach_text(monkeypatch, onnx_path, repo):
    captured = _stub_uvicorn_server(monkeypatch)

    result = _cli("serve", onnx_path, "--reference", repo, "--warmup", "1")

    assert result.exit_code == 0, result.output
    serving = captured["app"].state.serving
    assert serving.text is None
    assert serving.verdict.status == "CLEAN", serving.verdict.reason


@pytest.mark.needs_torch_26
def test_check_takes_the_pooling_flags(repo):
    result = _cli("check", repo, "--pooling", "cls", "--no-normalize", "--json")

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["status"] == "CLEAN"


def test_check_rejects_an_unknown_pooling(repo):
    result = _cli("check", repo, "--pooling", "median")

    assert result.exit_code == 2  # the own usage error of typer
    assert "median" in result.output


def test_check_reports_an_unusable_recipe_as_a_usage_error(tmp_path):
    pooling = {**hf_repo.MEAN_POOLING, "pooling_mode_lasttoken": True}
    path = hf_repo.write_encoder_repo(tmp_path, pooling=pooling)

    result = _cli("check", path)

    assert result.exit_code == main.EXIT_USAGE
    assert "2 pooling modes" in result.output


@pytest.mark.parametrize("command", ["check", "export", "serve"])
def test_the_flags_are_on_every_command_that_loads_a_model(command):
    result = _cli(command, "--help")

    plain = re.sub(r"\x1b\[[0-9;]*m", "", result.output)  # rich colors the option names
    assert "--pooling" in plain
    assert "--no-normalize" in plain


# --- the --workers builders keep the text and embedding metadata --------------------------


def _worker_args(repo: str, *, pooling=None, normalize=None, **artifact) -> main.ServeArgs:
    return main.ServeArgs(
        load=LoadSpec(repo, pooling=pooling, normalize=normalize),
        options=ServeOptions(warmup=1, device="cpu", pooling=pooling, normalize=normalize),
        reference=None,
        middleware=None,
        log_level="warning",
        artifact=main.ArtifactHandoff(**artifact) if artifact else None,
    )


def _rebuilt(monkeypatch, args: main.ServeArgs) -> TestClient:
    monkeypatch.setenv(main._SERVE_ARGS_ENV, args.to_json())
    api = runtime._serve_app_factory()
    with TestClient(api):
        api.state.loader_thread.join(timeout=60)
    return TestClient(api)


@pytest.mark.needs_torch_26
def test_onnx_artifact_worker_still_takes_text_and_reports_the_recipe(
    monkeypatch, tmp_path, repo, state, client
):
    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(state.verdict.onnx_bytes)
    args = _worker_args(
        repo,
        verdict=state.verdict.to_dict(),
        input_names=list(state.input_names),
        onnx_path=str(onnx_path),
        kind=state.source_kind,  # what serve_cmd of the parent sends
    )

    worker = _rebuilt(monkeypatch, args)

    body = worker.post("/predict", json={"text": TEXTS}).json()
    expected = client.post("/predict", json={"text": TEXTS}).json()
    assert body["outputs"] == expected["outputs"]
    assert worker.get("/schema").json()["embedding"]["pooling"] == "mean"


def test_torch_artifact_worker_still_takes_text_and_reports_the_recipe(monkeypatch, repo, client):
    torch_state = prepare_serving(
        load_model(LoadSpec(repo)), ServeOptions(backend=BackendChoice.torch)
    )
    args = _worker_args(
        repo,
        verdict=torch_state.verdict.to_dict(),
        input_names=list(torch_state.input_names),
        kind=torch_state.source_kind,
    )

    worker = _rebuilt(monkeypatch, args)

    assert worker.app.state.serving.backend.name == "torch"
    body = worker.post("/predict", json={"text": TEXTS}).json()
    expected = client.post("/predict", json={"text": TEXTS}).json()
    assert np.array(body["outputs"]["output_0"]) == pytest.approx(
        np.array(expected["outputs"]["output_0"]), abs=1e-5
    )
    assert worker.get("/schema").json()["embedding"]["pooling"] == "mean"


def test_worker_args_carry_the_overrides(monkeypatch, repo):
    """--pooling given to `serve` has to reach a worker that reloads the model itself."""
    args = _worker_args(repo, pooling="cls", normalize=False)

    worker = _rebuilt(monkeypatch, args)

    schema = worker.get("/schema").json()["embedding"]
    assert (schema["pooling"], schema["normalized"]) == ("cls", False)


def test_the_recipe_reads_named_prompts_and_the_default_prompt_name(tmp_path):
    path = hf_repo.write_encoder_repo(tmp_path)
    (tmp_path / "config_sentence_transformers.json").write_text(
        json.dumps(
            {"prompts": {"query": "Q: ", "document": "", "bad": 3}, "default_prompt_name": "query"}
        )
    )

    recipe = read_recipe(Path(path))

    assert recipe is not None
    assert recipe.prompts == {
        "query": "Q: ",
        "document": "",
    }  # values that are not strings are dropped
    assert recipe.default_prompt == "query"


def test_a_default_prompt_name_with_no_such_prompt_is_ignored(tmp_path):
    path = hf_repo.write_encoder_repo(tmp_path)
    (tmp_path / "config_sentence_transformers.json").write_text(
        json.dumps({"prompts": {"query": "Q: "}, "default_prompt_name": "nope"})
    )

    recipe = read_recipe(Path(path))

    assert recipe is not None and recipe.default_prompt is None


def test_a_repo_without_the_file_has_no_prompts(tmp_path):
    recipe = read_recipe(Path(hf_repo.write_encoder_repo(tmp_path)))

    assert recipe is not None and recipe.prompts == {} and recipe.default_prompt is None


def test_pooling_flag_on_a_bare_repo_still_reads_the_prompts(tmp_path):
    path = hf_repo.write_encoder_repo(tmp_path, modules=None, pooling=None)
    (tmp_path / "config_sentence_transformers.json").write_text(
        json.dumps({"prompts": {"query": "Q: "}})
    )

    recipe = resolve_recipe(path, "lasttoken", None, has_head=False)

    assert recipe is not None and recipe.prompts == {"query": "Q: "}


def test_the_pooled_output_is_float32_whatever_the_hidden_dtype():
    from downshift.adapters.embedding import EmbeddingRecipe, PoolingHead

    head = PoolingHead(EmbeddingRecipe("lasttoken", True, None, "--pooling"))
    hidden = torch.randn(2, 5, 8).bfloat16()

    assert head(hidden, torch.ones(2, 5, dtype=torch.long)).dtype == torch.float32
