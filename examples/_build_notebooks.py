"""Regenerate the tutorial .ipynb files from the .py sources.

Each tutorial script is plain Python using the "percent" cell convention:

    # %% [markdown]
    # # A heading
    # More markdown.

    # %%
    code_here()

No dependency on jupytext or nbformat — this is a small, direct writer for the notebook
JSON format, run with `python examples/_build_notebooks.py`.
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
CODE_MARKER = "# %%"
MARKDOWN_MARKER = "# %% [markdown]"


def parse_cells(source: str) -> list[dict]:
    cells: list[dict] = []
    kind: str | None = None
    lines: list[str] = []

    def flush() -> None:
        if kind is None:
            return
        text = "\n".join(lines).strip("\n")
        if kind == "markdown":
            body = "\n".join(line.removeprefix("# ").removeprefix("#") for line in lines)
            cells.append({"cell_type": "markdown", "metadata": {}, "source": body.strip("\n")})
        else:
            cells.append(
                {
                    "cell_type": "code",
                    "metadata": {},
                    "execution_count": None,
                    "outputs": [],
                    "source": text,
                }
            )

    for raw_line in source.splitlines():
        line = raw_line.rstrip("\n")
        if line == MARKDOWN_MARKER:
            flush()
            kind, lines = "markdown", []
        elif line == CODE_MARKER:
            flush()
            kind, lines = "code", []
        else:
            lines.append(line)
    flush()
    return cells


def to_notebook(cells: list[dict]) -> dict:
    for cell in cells:
        cell["source"] = [line + "\n" for line in cell["source"].split("\n")]
        if cell["source"]:
            cell["source"][-1] = cell["source"][-1].removesuffix("\n")
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def build(py_path: Path) -> Path:
    cells = parse_cells(py_path.read_text(encoding="utf-8"))
    notebook = to_notebook(cells)
    out_path = py_path.with_suffix(".ipynb")
    out_path.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
    return out_path


def main() -> None:
    scripts = sorted(HERE.glob("[0-9][0-9]_*.py"))
    if not scripts:
        print("no numbered tutorial scripts found", file=sys.stderr)
        sys.exit(1)
    for script in scripts:
        out_path = build(script)
        print(f"wrote {out_path.relative_to(HERE.parent)}")


if __name__ == "__main__":
    main()
