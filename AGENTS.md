# AGENTS.md

This file is for coding agents that work in this repository. To use downshift as a package, read [llms.txt](llms.txt).

## Set up

1. Install a CPU build of torch: `pip install torch --index-url https://download.pytorch.org/whl/cpu`.
2. Install the package in editable mode: `pip install -e ".[dev,gnn,hf,fast]"`.
3. If a non-editable `downshift-server` is also installed, set `PYTHONPATH=src`. If you do not, the tests import the installed copy and not your changes.

The project needs Python 3.12 to 3.14. The tests run on the CPU. A GPU is not necessary. The HTTPS tests need the `openssl` command on `PATH`. They make a throwaway certificate with it. Without `openssl`, they skip. In CI, they fail.

## Check a change

CI runs these commands. Run them before you finish:

```bash
ruff check .
ruff format --check .
mypy src
pytest -q
```

CI requires 95% line coverage (`--cov-fail-under=95`). Add a test for each new code path.

While you work, run only the test files that apply to the change. Run the full suite one time, at the end.

## Layout

| Path | Contents |
|---|---|
| `src/downshift/core/` | Export, verification, the verdict, the export cache. |
| `src/downshift/adapters/` | The model families: generic, pyg, hf, onnx. |
| `src/downshift/serve/` | The FastAPI app, the backends, the request path. |
| `src/downshift/cli/` | The `check`, `export` and `serve` commands. |
| `src/downshift/settings.py` | The defaults and the `DOWNSHIFT_*` environment variables. |
| `tests/` | pytest. `tests/models/` has small models with known verdicts. |
| `docs/` | The user documentation. |
| `examples/` | The models of the README commands. |

## Rules

- Keep the CLI modules free of torch at import time. `downshift --help` and `--version` must stay fast. Import heavy modules inside the function bodies.
- Per-request speed has priority over boot time. Do not add work to the request path without a measurement.
- Keep the code simple. Prefer a plain function or a small dataclass to a new class hierarchy, registry or wrapper.
- Give each usage error a next step. A `ValueError` from a command is exit code `4`. Its message must say what to change.
- Write docs, comments, CLI help and error messages in ASD-STE100 Simplified Technical English. Use short sentences and one meaning for each word. Write steps as commands. Put conditions first.
- Add a new `serve` option in these places: `settings.py`, `cli/options.py`, `serve_cmd`, `docs/code-docs/cli.md` and `docs/production.md`. Refer to "Add a `serve` option" in `docs/code-docs/cli.md`.
- Record each change that a user can see in `CHANGELOG.md`.
