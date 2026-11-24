"""docs/openapi.json must match the app. Run scripts/gen_openapi.py to update it."""

from __future__ import annotations

import json

from scripts.gen_openapi import OUTPUT, build_spec


def test_committed_openapi_matches_the_app():
    assert json.loads(OUTPUT.read_text(encoding="utf-8")) == build_spec(), (
        "docs/openapi.json is out of date. Run: python scripts/gen_openapi.py"
    )
