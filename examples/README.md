# Tutorials

Seven short, runnable lessons on `downshift`: checking whether a model survives ONNX
export, exporting it, serving it, and extending it to a new model family.

Each lesson exists twice: as a plain script (`NN_name.py`) and as a Jupyter notebook
(`NN_name.ipynb`) with the same content in cells. Run the script, or open the notebook
and step through it — same code either way.

## Setup

From a checkout that isn't pip-installed, run scripts with `src/` on the path:

```bash
cd downshift-server
PYTHONPATH=src python examples/01_check_a_model.py
```

Each script also inserts `../src` onto `sys.path` itself, so `python examples/01_...py`
works directly from the repo root without setting `PYTHONPATH`. If you `pip install -e .`
first, neither is necessary.

Lessons 4 and 5 need extras:

```bash
pip install -e ".[gnn]"   # lesson 4, PyTorch Geometric
pip install -e ".[hf]"    # lesson 5, Hugging Face transformers
```

## Order

1. **`01_check_a_model.py`** — the four verdicts (CLEAN / DEGRADED / FAILED / UNVERIFIED),
   read against a clean model, a model that exports but lies, and one that doesn't export.
2. **`02_export_and_manifest.py`** — write the `.onnx` artifact and read its manifest.
3. **`03_serve_and_query.py`** — build a serving state and call it over HTTP in-process.
4. **`04_pyg_graph_neural_networks.py`** — a GCN, and why node/edge counts need
   independent dynamic dimensions.
5. **`05_huggingface_encoder.py`** — a Hugging Face encoder, no download required.
6. **`06_custom_adapter.py`** — teach `downshift` a model family it doesn't know.
7. **`07_cli_walkthrough.py`** — the same things, from the command line.

`_build_notebooks.py` regenerates the `.ipynb` files from the `.py` sources; edit the
`.py` files and rerun it if you change a lesson.
