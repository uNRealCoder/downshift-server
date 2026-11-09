# Walkthrough

This page is a guided tour. You run the commands, and you read the code that each command uses. It takes about one hour. For the big picture first, read [`architecture.md`](architecture.md).

Every command here ran on a clean clone. The numbers in your output can differ a little. The statuses and the exit codes do not.

## Step 0. Set up

Install a CPU build of torch, then install the package in editable mode.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev,gnn,hf,fast]"
downshift --version
```

If a non-editable `downshift-server` is also installed, set `PYTHONPATH=src`. If you do not, Python imports the installed copy and not your changes.

The example models are in `examples/`. Run all commands from the root of the repo, so that `examples.clean_mlp` is importable.

## Step 1. A CLEAN model

```bash
downshift check examples.clean_mlp:make_model
```

Read the report. These rows matter:

- `Export CLEAN`: the graph exported.
- `Numerics ... OK`: torch and ONNX Runtime agree, on 8 samples.
- `Samples`: the batch size changes from sample to sample. This is how downshift finds a graph that froze a shape.
- `Rows independent`: rows of a batch do not change each other.
- `Backend onnxruntime`: the server would use ONNX Runtime.

Check the exit code:

```bash
downshift check examples.clean_mlp:make_model > /dev/null 2>&1; echo $?
```

The output is `0`.

**Read the code.** `examples/clean_mlp.py` is a plain `nn.Module` and a `make_model` function. Then open `src/downshift/cli/main.py` and find `check_cmd`. Follow it into `downshift.check` in `src/downshift/core/verdict.py`.

## Step 2. A DEGRADED model

```bash
downshift check examples.scatter_include_self_false:make_model
echo $?
```

The export has no error. But `Numerics` shows `6/8 failed`, `Export` shows `DEGRADED`, and the `Worst` row shows the output where torch and ONNX Runtime differ. The backend is `torch`. The exit code is `2`.

This is the reason for the whole project. `scatter_reduce(include_self=False)` exports without an error and gives wrong numbers. Only the numerics check finds it.

**Read the code.** Open `src/downshift/core/verify.py` and read the module docstring. Then find `verify` and read how it makes samples with new shapes.

## Step 3. A FAILED model

```bash
downshift check examples.data_dependent_branch:make_model
echo $?
```

`Export` shows `FAILED`, and `Reason` shows a `GuardOnDataDependentSymNode` error from `torch.export`. A Python `if` on a tensor value cannot export. The backend is `torch`. The exit code is `1`.

To see the full torch trace, add `--log-level debug`.

**Read the code.** Open `src/downshift/core/capture.py` and find `capture`. It tries `strict=False` and then `strict=True`. Then find where `build_verdict` in `core/verdict.py` turns the error into a FAILED verdict instead of raising it.

## Step 4. Export a file

```bash
mkdir -p /tmp/ds
downshift export examples.clean_mlp:make_model -o /tmp/ds
ls /tmp/ds
```

The directory holds `clean_mlp.onnx` and `clean_mlp.manifest.json`. The manifest records where the file came from. For a FAILED verdict, `export` writes nothing.

**Read the code.** `export()` in `src/downshift/__init__.py` and `src/downshift/core/manifest.py`.

## Step 5. Serve a model

Start a server in one terminal:

```bash
downshift serve examples.clean_mlp:make_model --port 8765
```

The log shows the same report as `check`, then `ready`. Open a second terminal.

Check the probes and the model:

```bash
curl -s localhost:8765/health
curl -s localhost:8765/ready
curl -s localhost:8765/schema
```

`/health` answers `{"status":"ok"}`. `/ready` answers `200` when the model is ready. During the boot, `/ready` answers `503` and `/health` answers `200`.

Call the model. The body has an `inputs` object, with one key for each input:

```bash
curl -s -X POST localhost:8765/predict \
  -H 'content-type: application/json' \
  -d '{"inputs": {"x": [[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]]}}'
```

The answer has `outputs`, `shapes` and `dtypes`. The output keys are `output_0`, `output_1` and so on, for both backends.

If you send a body with no `inputs` key, the answer is `400`. `GET /schema` shows the right format, and it has an `example_request`.

Look at the verdict that the server holds:

```bash
curl -s localhost:8765/metadata
```

Stop the server with `Ctrl+C`.

**Read the code.** In this order:

1. `serve_cmd` in `src/downshift/cli/main.py`.
2. `_build_serving_state` in `src/downshift/cli/runtime.py`.
3. `prepare_serving` and `choose_backend` in `src/downshift/serve/engine.py`.
4. `build_app` in `src/downshift/serve/app.py`.
5. `PredictRoute` in `src/downshift/serve/app.py`, then `src/downshift/serve/predict.py`.
6. `src/downshift/serve/backends.py`.

## Step 6. See the fallback

Serve the DEGRADED model:

```bash
downshift serve examples.scatter_include_self_false:make_model --port 8766
```

The banner shows `Backend torch (eager) <- auto-selected`. Send a request. This model has two inputs, and the second input holds the segment index of each row:

```bash
curl -s -X POST localhost:8766/predict \
  -H 'content-type: application/json' \
  -d '{"inputs": {"x": [[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]], "segment_ids": [0, 1, 1]}}'
```

The client gets correct numbers, from PyTorch. To serve the wrong ONNX graph on purpose, add `--force-onnx`. Do this only to see the difference.

Stop the server.

**Read the code.** The rules of `choose_backend` in `src/downshift/serve/engine.py` (rule 3 is `force_onnx`; refer to [`serving.md`](serving.md#choose_backend)), and the `force_onnx` field of `ServeOptions`.

## Step 7. Call it from Python

```python
import downshift
from examples.clean_mlp import make_model

verdict = downshift.check(make_model())
print(verdict.status, verdict.recommended_backend)
print(verdict.numerics.max_abs_err)
```

`check` does not write to disk. Refer to [`python-api.md`](python-api.md) for `export`, `intake` and `app_for`.

## Step 8. Read the tests

The tests show the expected behavior. `tests/models/` holds small models with known verdicts, and `tests/conftest.py` builds the shared fixtures one time for each session.

Run only the files that apply:

```bash
pytest -q tests/test_export.py tests/test_engine.py
```

Read these files in this order:

| File | Shows |
|---|---|
| `tests/test_export.py` | The verdict for each model in `tests/models/`. |
| `tests/test_engine.py` | `choose_backend` and the boot builders. |
| `tests/test_serve.py` | The routes and their status codes. |
| `tests/test_export_cache.py` | The memo and the disk cache. |

Run the full suite one time, at the end of a change. CI also needs 95% line coverage. Refer to [`../AGENTS.md`](../AGENTS.md) for the exact commands.

## Step 9. Make a small change

To see how a change goes through the project, do this exercise on a branch:

1. Pick a `serve` option, for example `--warmup`.
2. Find it in `settings.py`, `cli/options.py`, `serve_cmd` in `cli/main.py`, `serve/options.py`, `docs/cli.md` and `docs/production.md`. These are the places that a new option touches.
3. Read "Add a `serve` option" in [`cli.md`](cli.md).
4. Do not commit the exercise.

## What to read next

- [`architecture.md`](architecture.md) for the model of the whole project.
- [`serving.md`](serving.md) for the request path in detail.
- [`adapters.md`](adapters.md) to add a model family.
- [`production.md`](production.md) to deploy.
