"""docs/openapi.json must match the app. Run scripts/gen_openapi.py to update it."""

from __future__ import annotations

import json

from scripts.gen_openapi import OUTPUT, build_spec


def _comparable(node):
    """Drop the details that change between pydantic and FastAPI versions.

    The floor job runs the oldest pydantic. It does not write `additionalProperties: true`
    for a plain `dict`. It also leaves `ctx` and `input` out of `ValidationError`.
    """
    if isinstance(node, dict):
        return {
            key: _comparable(value)
            for key, value in node.items()
            if not (key == "additionalProperties" and value is True)
        }
    if isinstance(node, list):
        return [_comparable(item) for item in node]
    return node


def test_committed_openapi_matches_the_app():
    committed = json.loads(OUTPUT.read_text(encoding="utf-8"))
    current = build_spec()
    for spec in (committed, current):
        props = spec["components"]["schemas"]["ValidationError"]["properties"]
        props.pop("ctx", None)
        props.pop("input", None)
    assert _comparable(committed) == _comparable(current), (
        "docs/openapi.json is out of date. Run: python scripts/gen_openapi.py"
    )
