"""Print the CI matrix of FalkorDB images: one C engine and one Rust engine.

FalkorDB ships two engines: 4.x is the C engine, 6.x the Rust engine. They
differ in index DDL, procedures, error text and row order, so the integration
suite runs against both.

Pushes and pull requests test the versions pinned in ``PINNED``, so the same
commit always tests the same servers and a FalkorDB release cannot turn CI red
for code that did not change. ``--newest`` (the weekly run) tests the newest
release of each engine instead, read from Docker Hub, and warns when it is
newer than the pin: that is the cue to bump ``PINNED`` in a PR. Only plain
release tags count (``v4.22.1``, ``6.0.2``), not ``latest``, ``edge*`` or
per-platform variants such as ``v4.22.1-alpine``. If Docker Hub cannot be
reached the pins are used, with a warning.

Usage: ``python falkordb_versions.py [--newest]`` prints ``matrix=<json>`` for
``$GITHUB_OUTPUT``; ``--json`` prints only the JSON.
"""

from __future__ import annotations

import json
import re
import sys
import urllib.request

TAGS_URL = "https://hub.docker.com/v2/repositories/falkordb/falkordb/tags?page_size=100"
# Every page is read: the newest release can be on any page. This only stops a
# runaway loop (about 600 tags exist today, 7 pages); reaching it is reported.
MAX_PAGES = 500
RELEASE = re.compile(r"^v?(\d+)\.(\d+)\.(\d+)$")

# The versions pushes and pull requests test. Bump deliberately, in a PR.
PINNED = {"C": "v4.22.2", "Rust": "6.0.2"}


def release(tag: str) -> tuple[int, int, int]:
    """The release number of a tag: ``v4.22.2`` and ``4.22.2`` are one release."""
    match = RELEASE.match(tag)
    if not match:
        raise ValueError(f"not a release tag: {tag!r}")
    major, minor, patch = (int(part) for part in match.groups())
    return major, minor, patch


def engine_of(major: int) -> str | None:
    # Exactly these release lines. A new major (7.x, ...) is a deliberate
    # change to make here, not something CI should start testing on its own.
    if major == 4:
        return "C"
    if major == 6:
        return "Rust"
    return None


def fetch_tags() -> list[str]:
    tags: list[str] = []
    url: str | None = TAGS_URL
    for _ in range(MAX_PAGES):
        if not url:
            return tags
        with urllib.request.urlopen(url, timeout=20) as response:
            page = json.load(response)
        tags.extend(result["name"] for result in page.get("results", []))
        url = page.get("next")
    if url:
        print(
            f"::warning::Stopped reading Docker Hub tags after {MAX_PAGES} pages; "
            "the newest release may have been missed",
            file=sys.stderr,
        )
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


def build_matrix(newest: bool) -> list[dict[str, str]]:
    if not newest:
        return [
            {"engine": engine, "version": PINNED[engine]} for engine in ("C", "Rust")
        ]
    try:
        found = newest_per_engine(fetch_tags())
    except Exception as exc:  # network, rate limit, malformed reply
        print(
            f"::warning::Could not read FalkorDB tags from Docker Hub ({exc}); "
            f"using pinned versions {PINNED}",
            file=sys.stderr,
        )
        found = dict(PINNED)
    matrix = []
    for engine in ("C", "Rust"):
        tag = found.get(engine)
        if tag is None:
            print(
                f"::warning::No {engine} engine release found on Docker Hub; "
                f"using {PINNED[engine]}",
                file=sys.stderr,
            )
            tag = PINNED[engine]
        elif release(tag) > release(PINNED[engine]):
            print(
                f"::warning::FalkorDB {engine} {tag} is newer than the pinned "
                f"{PINNED[engine]}; if this run passes, bump PINNED in "
                ".github/scripts/falkordb_versions.py",
                file=sys.stderr,
            )
        elif release(tag) < release(PINNED[engine]):
            print(
                f"::warning::The newest FalkorDB {engine} release on Docker Hub, {tag}, "
                f"is older than the pinned {PINNED[engine]}; check PINNED",
                file=sys.stderr,
            )
        matrix.append({"engine": engine, "version": tag})
    return matrix


def main() -> None:
    matrix = json.dumps(build_matrix(newest="--newest" in sys.argv))
    # Also to the log: $GITHUB_OUTPUT alone does not show what was chosen.
    print(f"FalkorDB matrix: {matrix}", file=sys.stderr)
    print(matrix if "--json" in sys.argv else f"matrix={matrix}")


if __name__ == "__main__":
    main()
