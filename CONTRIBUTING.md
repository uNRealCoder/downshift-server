# Contributing

Thank you for your help. This page tells you how to make a change that the project accepts. To learn the code first, read [`docs/architecture.md`](docs/architecture.md) and [`docs/walkthrough.md`](docs/walkthrough.md). A coding agent must also read [`AGENTS.md`](AGENTS.md).

## Set up

1. Install Python 3.12 to 3.14.
2. Install a CPU build of torch:

   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   ```

3. Install the package in editable mode:

   ```bash
   pip install -e ".[dev,gnn,hf,fast]"
   ```

4. If a non-editable `downshift-server` is also installed, set `PYTHONPATH=src`. If you do not, the tests import the installed copy and not your changes.

The tests run on the CPU. You do not need a GPU. The HTTPS tests need the `openssl` command on `PATH`. Without it, they skip. In CI, they fail.

## Make a change

1. Make a branch from the branch that the maintainers tell you to use.
2. Make the change. Keep it small and keep it to one subject.
3. Add a test for each new code path.
4. Run the tests that apply to the change, while you work:

   ```bash
   python -m pytest -q tests/test_export.py
   ```

5. Run the checks (next section) one time, at the end.
6. Record each change that a user can see in `CHANGELOG.md`.
7. Open a pull request. Say what you changed and why.

## Check a change

CI runs these four commands. Run them before you open a pull request:

```bash
ruff check .
ruff format --check .
mypy src
pytest -q --cov=src/downshift --cov-report=term-missing --cov-fail-under=95
```

CI needs 95% line coverage. Plain `pytest -q` does not measure coverage. If the coverage is too low, the report shows the lines that no test runs (`Missing`). Add a test for them.

To fix the format, run `ruff format .`. The line length is 100.

CI runs the tests on Python 3.12, 3.13 and 3.14. A second job pins the oldest torch, onnx and onnxruntime that downshift supports. If your change uses a new torch API, check that the oldest torch has it.

## Rules for the code

- **Keep the CLI modules free of torch at import time.** `downshift --help` and `--version` must stay fast. Import heavy modules (torch, onnx, onnxruntime, fastapi, uvicorn) inside the function bodies. Refer to [`docs/architecture.md`](docs/architecture.md#layers).
- **Do not add work to the request path without a measurement.** Per-request speed has priority over boot time. Put work in the boot if you can.
- **Keep the code simple.** Prefer a plain function or a small dataclass to a new class hierarchy, registry or wrapper.
- **Give each usage error a next step.** A `ValueError` from a command gives exit code `4`. Its message must say what to change.
- **Do not break a listed item.** [`docs/stability.md`](docs/stability.md) lists the routes, codes, metrics, options and names that stay the same in 1.x. If your change breaks one, it needs a new major version. Refer to the deprecation rule on that page.
- **Do not fetch anything.** Downshift serves only models that are already on the machine.

## Rules for the text

Write docs, comments, CLI help and error messages in ASD-STE100 Simplified Technical English:

- Use short sentences.
- Use one word for one meaning, and use the same word each time.
- Write steps as commands.
- Put the condition first: "If the model has no graph, fall back to torch."

## Recipes

### Add a `serve` option

Follow "Add a `serve` option" in [`docs/cli.md`](docs/cli.md#add-a-serve-option). A new option touches these places:

| Place | Change |
|---|---|
| `src/downshift/settings.py` | The default, and the `DOWNSHIFT_*` variable. |
| `src/downshift/cli/options.py` | The `Annotated` option type. |
| `src/downshift/cli/main.py` | A parameter of `serve_cmd` with the name of the field. |
| `src/downshift/entities/options.py` | The field of `ServeOptions`. |
| `src/downshift/cli/runtime.py` | `ServeArgs.from_json`, if the field is an enum. |
| `docs/cli.md`, `docs/production.md` | The table row and the description. |
| `tests/` and `CHANGELOG.md` | A test and a changelog entry. |

### Add a model family

Write an adapter. Refer to [`docs/adapters.md`](docs/adapters.md). An adapter has `matches`, `example_inputs` and `prepare`. `prepare` returns a `Prepared` object. A built-in adapter goes in `src/downshift/adapters/`. Add a small model for it in `tests/models/`, with a known verdict.

### Add a test model

1. Put the model in `tests/models/`. Keep it small. The export and verification are the slow part of the suite.
2. Say in the module docstring which hazard the model has, or that it is a control. Use a known verdict (CLEAN, DEGRADED or FAILED) in the test.
3. If more than one test file needs the exported model, build it in `tests/conftest.py`. The fixtures there run one time for each session.

### Change the docs

- Do not edit `docs/compatibility.md` by hand. `scripts/gen_matrix.py` generates it.
- If you change an option, a route or a default, change the page that describes it in the same pull request.
- Run each command that you put in a doc. Write only output that you saw.

## Report a problem

Open an issue. Include:

1. The output of `downshift --version`.
2. The versions of torch, onnx and onnxruntime.
3. The command that you ran.
4. The full output. For a verdict problem, add `--log-level debug`.

Do not post secrets, API keys or private model paths. Downshift hides local paths in its responses, but your own logs can show them.
