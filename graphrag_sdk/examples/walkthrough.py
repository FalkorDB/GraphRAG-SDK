"""Printing helpers for 12_hybrid_walkthrough.ipynb.

Nothing here is part of the SDK. It exists so the notebook can show a query,
what to expect from it, and what it actually returned, without 30 lines of
formatting in the middle of the narrative. Paste any query it prints into the
FalkorDB browser to see the same thing as a picture.
"""

from __future__ import annotations

import textwrap

MAX_ROWS = 12


def _cell(value: object) -> object:
    """Render a node as ``Label(name)`` and an edge as ``-[TYPE]->``.

    The queries return whole nodes and edges on purpose — that is what draws a
    picture in the browser — but printed raw they are ``<Node object at 0x...>``.
    Scalars pass through untouched.
    """
    properties = getattr(value, "properties", None) or {}
    labels = getattr(value, "labels", None)
    relation = getattr(value, "relation", None)

    if relation is not None:
        # Every data edge is RELATES; the meaning is in rel_type.
        return f"-[{properties.get('rel_type') or relation}]->"
    if labels is None:
        return value
    label = next((each for each in labels if each != "__Entity__"), "Node")
    name = properties.get("name") or properties.get("record_key") or properties.get("id")
    return f"{label}({name})" if name else label


async def look(rag, what: str, cypher: str, expect: str = "") -> list:
    """Run one query, print it with its result, and return the rows."""
    rows = await rag.query(cypher)
    rule = "-" * 78
    print(f"\n{rule}\n{what}\n{rule}")
    print(textwrap.indent(cypher.strip(), "    "))
    if expect:
        print(f"\n  expect: {expect}")
    print(f"\n  {len(rows)} row(s):")
    for row in rows[:MAX_ROWS]:
        print("    ", [_cell(value) for value in row])
    if len(rows) > MAX_ROWS:
        print(f"     ... and {len(rows) - MAX_ROWS} more")
    return rows


def paragraph(text: object, width: int = 74) -> str:
    """Wrap and indent a long message — used for the SDK's refusal text."""
    return textwrap.indent(textwrap.fill(str(text), width), "  ")


async def counts(rag, note: str = "") -> None:
    """Print one line: total nodes, and the count under each label."""
    rows = await rag.query(
        "MATCH (n) RETURN head([l IN labels(n) WHERE l <> '__Entity__']) AS label, "
        "count(n) AS n ORDER BY n DESC"
    )
    total = sum(count for _, count in rows)
    breakdown = ", ".join(f"{label}={count}" for label, count in rows if label)
    print(f"  {note + '  ' if note else ''}{total} nodes: {breakdown}")
