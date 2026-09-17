# %% [markdown]
# # Lesson 7: the command line
#
# Everything in lessons 1-6 has a CLI equivalent. This lesson runs the real
# `downshift` CLI as a subprocess against the demo models shipped inside the `downshift`
# package, and prints exactly what it prints. Nothing here is simulated.

# %%
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# The CLI's tables use box-drawing characters. On Windows a piped stdout defaults to the
# system codepage, which can't encode them; force UTF-8 for this script's own output too.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def run(*args: str) -> subprocess.CompletedProcess:
    """Run `downshift <args>` the way a user would from a shell, and show it."""
    print("$ downshift", " ".join(args))
    env = os.environ | {"PYTHONPATH": str(REPO_ROOT / "src"), "PYTHONIOENCODING": "utf-8"}
    result = subprocess.run(
        [sys.executable, "-m", "downshift.cli.main", *args],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        encoding="utf-8",  # matches PYTHONIOENCODING above; the boot banner uses box-drawing characters
    )
    print(result.stdout)
    if result.returncode not in (0, 1, 2, 3):  # those three are verdict exit codes, not errors
        print(result.stderr, file=sys.stderr)
    print(f"[exit code {result.returncode}]\n")
    return result


# %% [markdown]
# ## `check`
#
# Same verdict as `downshift.check()` in Python, as a table. `downshift.demo.clean_mlp`
# is one of the package's own demo modules — an import spec (`pkg.module:attr`) is one
# of the four model forms the CLI accepts, alongside a `.onnx` path, a `weights.pt`
# state dict with `--model-class`, and a Hugging Face repo id.

# %%
run("check", "downshift.demo.clean_mlp:make_model")

# %% [markdown]
# `--json` for machine-readable output and a script-friendly exit code — this is what a
# CI job greps.

# %%
run("check", "downshift.demo.scatter_include_self_false:make_model", "--json")

# %% [markdown]
# ## `export`
#
# Writes the `.onnx` and its manifest to `-o`/`--output`.

# %%
export_dir = Path(tempfile.mkdtemp(prefix="downshift-cli-tutorial-"))
run("export", "downshift.demo.clean_mlp:make_model", "-o", str(export_dir))

# %% [markdown]
# ## `serve`
#
# Not run here — it starts a long-lived server, which doesn't fit a script that finishes.
# From a real shell:
#
# ```bash
# downshift serve downshift.demo.scatter_include_self_false:make_model --port 8000
# ```
#
# prints the boot banner (lesson 1's DEGRADED verdict, formatted for a terminal) and
# starts listening. In another terminal:
#
# ```bash
# curl http://localhost:8000/metadata
# curl -X POST http://localhost:8000/predict \
#      -H "Content-Type: application/json" \
#      -d '{"inputs": {"x": [[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]], "segment_ids": [0]}}'
# ```
#
# Useful flags: `--force-onnx` serves a DEGRADED graph anyway (loudly, with a warning);
# `--reference model.pt` verifies a pre-built `.onnx` you pass instead of a PyTorch
# model; `--middleware pkg.mod:Attr` attaches a Starlette middleware or async function,
# repeatable.

# %% [markdown]
# ## `version`

# %%
run("version")

# %% [markdown]
# That's the whole surface: `check`, `export`, `serve`, `version`. Every command loads a
# model, calls a library function, and hands the result to the same rendering code the
# tables and banners above came from — nothing the CLI can do is unavailable from Python,
# and nothing in these seven lessons is unavailable from the CLI.
