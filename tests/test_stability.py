"""docs/stability.md names each public item that 1.0 freezes. A new item needs a line there."""

import re
from pathlib import Path

import downshift
from downshift import settings
from downshift.serve.errors import ERROR_CODES

ROOT = Path(__file__).resolve().parent.parent
PAGE = (ROOT / "docs" / "stability.md").read_text(encoding="utf-8")


def source(relative: str) -> str:
    return (ROOT / "src" / "downshift" / relative).read_text(encoding="utf-8")


def test_each_environment_variable_is_on_the_page():
    names = set(re.findall(r'"(DOWNSHIFT_[A-Z0-9_]+)"', source("settings.py")))
    names |= {
        f"DOWNSHIFT_TOL_{dtype.upper()}_{kind}"
        for dtype in settings.TOLERANCES
        for kind in ("ATOL", "RTOL")
    }
    assert len(names) > 25
    assert [name for name in sorted(names) if name not in PAGE] == []


def test_each_public_name_is_on_the_page():
    assert [name for name in downshift.__all__ if f"`{name}`" not in PAGE] == []


def test_each_error_code_is_on_the_page():
    assert [code for code in ERROR_CODES if f"`{code}`" not in PAGE] == []


def test_each_metric_is_on_the_page():
    names = set(re.findall(r'"(downshift_[a-z_]+)"', source("serve/metrics.py")))
    assert len(names) > 10
    assert [name for name in sorted(names) if f"`{name}`" not in PAGE] == []


def test_the_page_is_linked():
    assert "stability.md" in (ROOT / "docs" / "README.md").read_text(encoding="utf-8")
    assert "stability.md" in (ROOT / "llms.txt").read_text(encoding="utf-8")
