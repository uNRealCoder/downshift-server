"""Check real public models with `downshift check` and publish the results.

Run from the repository root:

    python scripts/check_real_models.py [--only ID] [--large] [--out FILE]

Each model runs in its own subprocess, one at a time, so the memory is free between models.
A full run replaces the "Tested real models" section of docs/compatibility.md and writes the
raw `check --json` results to --out. Exit code: 0 if no model FAILED or crashed, else 1.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS = REPO_ROOT / "docs" / "compatibility.md"
DEFAULT_OUT = REPO_ROOT / "bench" / "results" / "v1.0.0" / "real_models.json"
LOCAL = REPO_ROOT / "bench" / "models"
START = "<!-- real-models:start -->"
END = "<!-- real-models:end -->"
TIMEOUT = 15 * 60
VERSIONED = ("torch", "onnx", "onnxruntime", "transformers", "torch_geometric", "torchvision")

# kind: "hf" is a hub id (or a local dir that is used first); "local" is a dir that must exist;
# "code" is a pkg.module:attr with example inputs. `large` models run only with --large.
MODELS: list[dict] = [
    {"id": "sentence-transformers/all-MiniLM-L6-v2", "kind": "hf", "local": "all-MiniLM-L6-v2"},
    {"id": "BAAI/bge-small-en-v1.5", "kind": "hf"},
    {"id": "intfloat/e5-small-v2", "kind": "hf"},
    {"id": "distilbert/distilbert-base-uncased-finetuned-sst-2-english", "kind": "hf"},
    {"id": "cross-encoder/ms-marco-MiniLM-L-6-v2", "kind": "hf"},
    {"id": "meta-llama/Prompt-Guard-86M", "kind": "local", "local": "Prompt-Guard-86M"},
    {
        "id": "torchvision/resnet18",
        "kind": "code",
        "module": "torchvision",
        "target": "torchvision.models:resnet18",
        "inputs": "scripts.check_real_models:resnet_inputs",
    },
    {
        "id": "pyg/GCN (tests/models/gnn_gcn)",
        "kind": "code",
        "module": "torch_geometric",
        "target": "tests.models.gnn_gcn:make_model",
        "inputs": "tests.models.gnn_gcn:make_inputs",
    },
    {
        "id": "Qwen/Qwen3-Embedding-0.6B",
        "kind": "local",
        "local": "Qwen3-Embedding-0.6B",
        "large": True,
    },
]


def resnet_inputs():
    import torch

    return (torch.randn(1, 3, 224, 224),)


def _installed(module: str) -> bool:
    try:
        version(module)
    except PackageNotFoundError:
        return False
    return True


def resolve(spec: dict) -> tuple[str | None, str]:
    """Return (the argument for `check`, or None) and a note for a row with no result."""
    kind = spec["kind"]
    if kind == "code":
        if not _installed(spec["module"]):
            return None, "not installed"
        return spec["target"], ""
    local = LOCAL / spec.get("local", "")
    if spec.get("local") and local.is_dir():
        return str(local), ""
    if kind == "local":
        return None, "not downloaded"
    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(spec["id"]), ""
    except Exception as error:  # a network or hub error must not stop the run
        print(f"download of {spec['id']} failed: {error}", file=sys.stderr)
        return None, "not downloaded"


def run_check(spec: dict, target: str) -> dict:
    command = [sys.executable, "-m", "downshift", "check", target, "--json"]
    if spec["kind"] == "code":
        command += ["--inputs", spec["inputs"]]
    start = time.perf_counter()
    try:
        done = subprocess.run(
            command, capture_output=True, text=True, timeout=TIMEOUT, cwd=REPO_ROOT
        )
    except subprocess.TimeoutExpired:
        return {"outcome": "crashed", "error": f"timeout after {TIMEOUT} s"}
    seconds = round(time.perf_counter() - start, 1)
    try:
        verdict = json.loads(done.stdout)
    except json.JSONDecodeError:
        tail = (done.stderr or "").strip().splitlines()[-3:]
        return {
            "outcome": "crashed",
            "error": f"exit {done.returncode}: " + " | ".join(tail),
            "seconds": seconds,
        }
    return {"outcome": "ok", "exit_code": done.returncode, "seconds": seconds, "verdict": verdict}


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.1e}"


def _cosine(value: float | None) -> str:
    return "—" if value is None else f"{value:.6f}"


def table_row(spec: dict, result: dict) -> str:
    model = f"`{spec['id']}`"
    if "note" in result:
        return f"| {model} | — | {result['note']} | — | — | — | — | — |"
    if result["outcome"] == "crashed":
        return f"| {model} | — | crashed | — | — | — | — | {result.get('seconds', '—')} |"
    verdict = result["verdict"]
    numerics = verdict.get("numerics") or {}
    return (
        f"| {model} | {verdict.get('model_family', '—')} | {verdict['status']} "
        f"| {numerics.get('samples_tested', '—')} | {_fmt(numerics.get('max_abs_err'))} "
        f"| {_cosine(numerics.get('min_cosine'))} | {verdict.get('recommended_backend', '—')} "
        f"| {result['seconds']} |"
    )


def render_section(specs: list[dict], results: list[dict]) -> str:
    versions = ", ".join(f"{name} {_dist_version(name)}" for name in VERSIONED)
    lines = [
        START,
        "## Tested real models",
        "",
        f"Generated {datetime.now(UTC):%Y-%m-%d} by `scripts/check_real_models.py` with {versions}.",
        "Downshift ran `downshift check --json` on each model, on the CPU.",
        "The Rows column is the number of verification samples. The script is in the repository: "
        "to run it again, use `python scripts/check_real_models.py`.",
        "",
        "| Model | Family | Verdict | Rows | Max abs err | Min cosine | Backend | Seconds |",
        "|---|---|---|---|---|---|---|---|",
    ]
    lines += [table_row(s, r) for s, r in zip(specs, results, strict=True)]
    notes = [
        f"- `{s['id']}`: {r['verdict'].get('reason')}"
        for s, r in zip(specs, results, strict=True)
        if r.get("outcome") == "ok" and r["verdict"]["status"] != "CLEAN"
    ]
    if notes:
        lines += ["", "Reason for each verdict that is not CLEAN:", "", *notes]
    lines.append(END)
    return "\n".join(lines)


def _dist_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not installed"


def write_docs(section: str) -> None:
    text = DOCS.read_text(encoding="utf-8")
    if START in text and END in text:
        before, rest = text.split(START, 1)
        _, after = rest.split(END, 1)
        text = before + section + after
    else:
        text = text.rstrip("\n") + "\n\n" + section + "\n"
    DOCS.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", help="Run only the model with this id.")
    parser.add_argument("--large", action="store_true", help="Include the large models.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="The JSON results file.")
    args = parser.parse_args()

    specs = [s for s in MODELS if args.large or not s.get("large")]
    if args.only:
        specs = [s for s in MODELS if s["id"] == args.only]
        if not specs:
            ids = ", ".join(s["id"] for s in MODELS)
            parser.error(f"unknown id {args.only!r}. Use one of: {ids}")
    results = []
    for spec in specs:
        print(f"checking {spec['id']} ...", file=sys.stderr)
        target, note = resolve(spec)
        if target is None:
            results.append({"outcome": "skipped", "note": note})
        else:
            results.append(run_check(spec, target))
        print(f"  {table_row(spec, results[-1])}", file=sys.stderr)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "date": f"{datetime.now(UTC):%Y-%m-%d}",
        "versions": {name: _dist_version(name) for name in VERSIONED},
        "models": [{"id": s["id"], **r} for s, r in zip(specs, results, strict=True)],
    }
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if not args.only:
        write_docs(render_section(specs, results))

    failed = [
        s["id"]
        for s, r in zip(specs, results, strict=True)
        if r["outcome"] == "crashed"
        or (r["outcome"] == "ok" and r["verdict"]["status"] == "FAILED")
    ]
    if failed:
        print("FAILED or crashed: " + ", ".join(failed), file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
