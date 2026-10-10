"""Run downshift.check on each fixture in tests/models and write docs/compatibility.md.

Usage: python scripts/gen_matrix.py   (from a checkout. It works with or without pip install)
The exit code is always 0. This is a report and not a gate.
"""

from __future__ import annotations

import ast
import importlib
import pkgutil
import re
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))  # the imports of `tests.models.*` then work from all places
sys.path.insert(1, str(REPO_ROOT / "src"))  # a checkout in editable mode, without pip install

import downshift  # noqa: E402

FIXTURES_DIR = REPO_ROOT / "tests" / "models"
OUTPUT = REPO_ROOT / "docs" / "compatibility.md"
README = REPO_ROOT / "README.md"
MATRIX_START = "<!-- matrix:start -->"
MATRIX_END = "<!-- matrix:end -->"
K = 8
VERSIONED = ("torch", "onnx", "onnxruntime", "torch_geometric", "transformers")
COLUMNS = ("Model", "Hazard", "Family", "Export", "Capture", "Numerics", "Shape-general", "Backend")
DASH = "—"

LEGEND = (
    "**CLEAN**: The model exports, matches PyTorch on every sample, and works on shapes that "
    "the exporter did not see. Served by ONNX Runtime. **DEGRADED**: The model exports "
    "without error, but the numbers differ from PyTorch by more than the tolerance on at "
    "least one sample. Served by eager PyTorch, unless you set `--force-onnx`. **FAILED**: "
    "The model does not export, or it exports but ONNX Runtime cannot load or run the graph. "
    "Served by eager PyTorch in both cases. **UNVERIFIED**: A `.onnx` file with no reference "
    "model. Downshift never checked the numbers. Served by ONNX Runtime and labelled as "
    "unverified."
)
HOW_TO_READ = (
    "CLEAN means that the ONNX graph agrees with PyTorch. It does not mean that the graph is "
    "fast. DEGRADED means that the graph runs, but the numbers are wrong on at least one of "
    "the K samples. The Numerics column shows the largest absolute error. Downshift still "
    "serves FAILED models, through eager PyTorch behind the same endpoint. UNVERIFIED does not "
    "appear here, because every model in the corpus has a PyTorch reference."
)


@dataclass
class Row:
    model: str
    hazard: str
    cells: tuple[str, ...]  # Family .. Backend, in the order of COLUMNS


def _hazard(path: Path) -> str:
    """First sentence of the module docstring, which names the hazard the fixture isolates."""
    doc = ast.get_docstring(ast.parse(path.read_text(encoding="utf-8"))) or ""
    line = doc.splitlines()[0] if doc else ""
    line = re.split(r"\.\s|\s—\s", line)[0]
    return re.sub(r"^Hazard:\s*", "", line).rstrip(".").strip() or DASH


def _dist_version(name: str) -> str:
    try:
        return version(name.replace("_", "-"))
    except PackageNotFoundError:
        return "not installed"


def _run_fixture(name: str) -> Row:
    hazard = _hazard(FIXTURES_DIR / f"{name}.py")
    try:
        module = importlib.import_module(f"tests.models.{name}")
    except ImportError as exc:
        missing = exc.name or "dependency"
        return Row(name, hazard, (DASH, f"skipped (missing {missing})", DASH, DASH, DASH, DASH))
    try:
        inputs = module.make_inputs() if hasattr(module, "make_inputs") else None
        verdict = downshift.check(module.make_model(), inputs, k=K)
    except Exception as exc:  # noqa: BLE001 - a fixture that is not a hazard (for example a
        # factory that raises by design for CLI tests) is a skip and not a crash of the script.
        return Row(name, hazard, (DASH, f"skipped ({type(exc).__name__})", DASH, DASH, DASH, DASH))
    numerics = verdict.numerics
    shape = {True: "✓", False: "✗", None: DASH}[verdict.shape_generalization]
    return Row(
        name,
        hazard,
        (
            verdict.model_family,
            verdict.status,
            verdict.capture_strategy or DASH,
            f"{numerics.max_abs_err:.1e}" if numerics else DASH,
            shape,
            verdict.recommended_backend,
        ),
    )


def _discover() -> list[str]:
    return sorted(
        info.name
        for info in pkgutil.iter_modules([str(FIXTURES_DIR)])
        if "def make_model" in (FIXTURES_DIR / f"{info.name}.py").read_text(encoding="utf-8")
    )


def render(rows: list[Row]) -> str:
    versions = ", ".join(f"{name} {_dist_version(name)}" for name in VERSIONED)
    lines = [
        "# Compatibility matrix",
        "",
        f"Generated {datetime.now(UTC):%Y-%m-%d} by `scripts/gen_matrix.py` with {versions}.",
        f"Downshift checked each model with `downshift.check(..., k={K})`.",
        "",
        LEGEND,
        "",
        "| " + " | ".join(COLUMNS) + " |",
        "|" + "---|" * len(COLUMNS),
    ]
    for row in rows:
        lines.append("| " + " | ".join((f"`{row.model}`", row.hazard, *row.cells)) + " |")
    lines += ["", "## How to read this", "", HOW_TO_READ, ""]
    return "\n".join(lines)


def render_table(rows: list[Row]) -> str:
    lines = ["| " + " | ".join(COLUMNS) + " |", "|" + "---|" * len(COLUMNS)]
    for row in rows:
        lines.append("| " + " | ".join((f"`{row.model}`", row.hazard, *row.cells)) + " |")
    return "\n".join(lines)


def update_readme(rows: list[Row]) -> None:
    text = README.read_text(encoding="utf-8")
    if MATRIX_START not in text or MATRIX_END not in text:
        print(f"no {MATRIX_START}/{MATRIX_END} markers in {README}, skipping", file=sys.stderr)
        return
    before, rest = text.split(MATRIX_START, 1)
    _, after = rest.split(MATRIX_END, 1)
    new_text = f"{before}{MATRIX_START}\n{render_table(rows)}\n{MATRIX_END}{after}"
    README.write_text(new_text, encoding="utf-8")
    print(f"wrote {README}", file=sys.stderr)


def main() -> int:
    rows = []
    for name in _discover():
        print(f"checking {name} ...", file=sys.stderr)
        rows.append(_run_fixture(name))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(render(rows), encoding="utf-8")
    print(f"wrote {OUTPUT}", file=sys.stderr)
    update_readme(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
