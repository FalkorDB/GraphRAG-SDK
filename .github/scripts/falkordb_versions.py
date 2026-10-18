"""Print the CI matrix of FalkorDB images: the newest C engine and the newest Rust engine.

FalkorDB ships two engines: 4.x is the C engine, 6.x the Rust engine. They
differ in index DDL, procedures, error text and row order, so the integration
suite runs against the newest stable release of each. Release tags are read
from Docker Hub; only plain release tags count (``v4.22.1``, ``6.0.2``), not
``latest``, ``edge*`` or per-platform variants such as ``v4.22.1-alpine``.

If Docker Hub cannot be reached the pinned fallbacks below are used, with a
warning, so a Docker Hub outage cannot fail CI by itself.

Usage: ``python falkordb_versions.py`` prints ``matrix=<json>`` for
``$GITHUB_OUTPUT``; ``--json`` prints only the JSON.
"""

from __future__ import annotations

import json
import re
import sys
import urllib.request

TAGS_URL = "https://hub.docker.com/v2/repositories/falkordb/falkordb/tags?page_size=100"
MAX_PAGES = 30
RELEASE = re.compile(r"^v?(\d+)\.(\d+)\.(\d+)$")

# Used only when Docker Hub cannot be read.
FALLBACK = {"C": "v4.22.1", "Rust": "6.0.2"}


def engine_of(major: int) -> str | None:
    if major == 4:
        return "C"
    if major >= 6:
        return "Rust"
    return None


def fetch_tags() -> list[str]:
    tags: list[str] = []
    url: str | None = TAGS_URL
    for _ in range(MAX_PAGES):
        if not url:
            break
        with urllib.request.urlopen(url, timeout=20) as response:
            page = json.load(response)
        tags.extend(result["name"] for result in page.get("results", []))
        url = page.get("next")
    return tags


def newest_per_engine(tags: list[str]) -> dict[str, str]:
    newest: dict[str, tuple[tuple[int, int, int], str]] = {}
    for tag in tags:
        match = RELEASE.match(tag)
        if not match:
            continue
        version = tuple(int(part) for part in match.groups())
        engine = engine_of(version[0])
        if engine and (engine not in newest or version > newest[engine][0]):
            newest[engine] = (version, tag)
    return {engine: tag for engine, (_, tag) in newest.items()}


def build_matrix() -> list[dict[str, str]]:
    try:
        found = newest_per_engine(fetch_tags())
    except Exception as exc:  # network, rate limit, malformed reply
        print(
            f"::warning::Could not read FalkorDB tags from Docker Hub ({exc}); "
            f"using pinned versions {FALLBACK}",
            file=sys.stderr,
        )
        found = dict(FALLBACK)
    matrix = []
    for engine in ("C", "Rust"):
        tag = found.get(engine)
        if tag is None:
            print(
                f"::warning::No {engine} engine release found on Docker Hub; "
                f"using {FALLBACK[engine]}",
                file=sys.stderr,
            )
            tag = FALLBACK[engine]
        matrix.append({"engine": engine, "version": tag})
    return matrix


def main() -> None:
    matrix = json.dumps(build_matrix())
    print(matrix if "--json" in sys.argv else f"matrix={matrix}")


if __name__ == "__main__":
    main()
