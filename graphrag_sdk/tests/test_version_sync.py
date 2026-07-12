"""Guard against src/graphrag_sdk/__init__.py __version__ drifting from pyproject.toml."""
from __future__ import annotations

import re
from pathlib import Path

import graphrag_sdk


def test_version_matches_pyproject() -> None:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    m = re.search(r'^version\s*=\s*"([^"]+)"', pyproject.read_text(encoding="utf-8"), re.M)
    assert m, "version not found in pyproject.toml"
    assert graphrag_sdk.__version__ == m.group(1)
