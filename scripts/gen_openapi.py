"""Write docs/openapi.json, the OpenAPI document of the HTTP API.

Usage: python scripts/gen_openapi.py   (from a checkout. It works with or without pip install)
The app has a loader that never runs, so the script needs no model. A test fails when the file
is out of date. Run the script again to fix it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))  # a checkout in editable mode, without pip install

OUTPUT = REPO_ROOT / "docs" / "openapi.json"


def build_spec() -> dict:
    from downshift.serve.app import build_app

    # metrics=True lists GET /metrics, which is opt-in on a real server.
    app = build_app(loader=lambda: None, api_key=None, metrics=True)  # type: ignore[arg-type,return-value]
    return app.openapi()


def main() -> None:
    OUTPUT.write_text(json.dumps(build_spec(), indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
