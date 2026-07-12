"""Rendering and shape tests for graphrag_sdk.tools result models."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from graphrag_sdk.core.exceptions import GraphRAGError, ReadOnlyViolation
from graphrag_sdk.tools import (
    AnswerResult,
    ChunkRef,
    Citation,
    CypherResult,
    DocumentRef,
    EntityCard,
    EntityResult,
    EntityTypeInfo,
    RelationTriple,
    RelationTypeInfo,
    RememberResult,
    SchemaResult,
    SearchResult,
)

GOLDEN = Path(__file__).parent / "golden" / "tools"


def check_golden(name: str, actual: str) -> None:
    path = GOLDEN / name
    if os.getenv("UPDATE_GOLDEN") == "1":
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(actual + "\n", encoding="utf-8")
        return
    assert actual + "\n" == path.read_text(encoding="utf-8"), f"golden mismatch: {name}"


def _search_result() -> SearchResult:
    return SearchResult(
        query="Who works at Acme?",
        entities=[
            EntityCard(
                name="Alice",
                label="Person",
                description="Engineer at Acme",
                properties={"seniority": "senior"},
            ),
            EntityCard(
                name="Acme Corp",
                label="Organization",
                description="A tech company",
                properties={},
            ),
        ],
        relations=[
            RelationTriple(
                source="Alice",
                type="WORKS_AT",
                target="Acme Corp",
                fact="Alice is employed at Acme Corp",
            )
        ],
        facts=["Alice —[WORKS_AT]→ Acme Corp: employment"],
        chunks=[
            ChunkRef(
                chunk_id="c1",
                document_id="doc-a",
                document_path="docs/a.md",
                text="Alice works at Acme Corp.",
            )
        ],
        documents=[DocumentRef(document_id="doc-a", document_path="docs/a.md")],
    )


def test_readonly_violation_hierarchy():
    err = ReadOnlyViolation("no", offending_token="MERGE")
    assert isinstance(err, GraphRAGError)
    assert err.offending_token == "MERGE"
    assert ReadOnlyViolation("no").offending_token is None


def test_model_dump_json_round_trip():
    sr = _search_result()
    assert SearchResult.model_validate(json.loads(sr.model_dump_json())) == sr


def test_to_llm_text_deterministic_and_bounded():
    sr = _search_result()
    assert sr.to_llm_text() == sr.to_llm_text()
    for max_chars in (10, 50, 120, 4000):
        out = sr.to_llm_text(max_chars=max_chars)
        assert len(out) <= max_chars


def test_truncation_marker_at_item_boundary():
    sr = SearchResult(
        query="q",
        entities=[
            EntityCard(name=f"E{i:02d}", label="Person", description="d" * 30, properties={})
            for i in range(20)
        ],
        relations=[],
        facts=[],
        chunks=[],
        documents=[],
    )
    out = sr.to_llm_text(max_chars=300)
    assert "…(" in out and "more)" in out
    assert len(out) <= 300
    # never truncates mid-entity: every emitted entity line is complete
    for line in out.splitlines():
        if line.startswith("- E"):
            assert line.endswith("d" * 30)


def test_no_dangling_section_header():
    """Every emitted section header is followed by an item or a drop-marker."""
    sr = _search_result()
    for max_chars in range(40, 400, 7):
        out = sr.to_llm_text(max_chars=max_chars)
        lines = out.splitlines()
        for i, line in enumerate(lines):
            if re.match(r"^[A-Z][A-Za-z ]+ \(\d+\):$", line):
                assert i + 1 < len(lines), f"dangling header at max_chars={max_chars}: {out!r}"
                nxt = lines[i + 1]
                assert nxt.startswith("- ") or nxt.startswith(f"  {chr(0x2026)}"), (
                    f"header not followed by content at max_chars={max_chars}: {out!r}"
                )


def test_control_characters_stripped():
    sr = SearchResult(
        query="bad\x00query\x07", entities=[], relations=[], facts=[], chunks=[], documents=[]
    )
    out = sr.to_llm_text()
    assert "\x00" not in out and "\x07" not in out and "badquery" in out


def test_golden_search():
    check_golden("search_result.txt", _search_result().to_llm_text())


def test_golden_search_truncated():
    check_golden("search_result_truncated.txt", _search_result().to_llm_text(max_chars=160))


def test_golden_answer():
    ar = AnswerResult(
        answer="Alice works at Acme Corp.\nShe is an engineer.",
        citations=[
            Citation(
                document_id="doc-a",
                document_path="docs/a.md",
                chunk_id="c1",
                snippet="Alice works at Acme Corp.",
            )
        ],
        entities_touched=["Alice", "Acme Corp"],
        cypher_used=None,
    )
    check_golden("answer_result.txt", ar.to_llm_text())


def test_golden_schema():
    sr = SchemaResult(
        entities=[
            EntityTypeInfo(
                label="Person", description="A human", count=2, properties=["seniority"]
            ),
            EntityTypeInfo(label="Organization", description=None, count=1, properties=[]),
        ],
        relations=[
            RelationTypeInfo(
                label="WORKS_AT",
                description=None,
                patterns=[("Person", "Organization")],
                count=1,
            )
        ],
        node_count=7,
        edge_count=9,
    )
    check_golden("schema_result.txt", sr.to_llm_text())


def test_golden_cypher():
    cr = CypherResult(
        columns=["name", "n"], rows=[["Alice", 3], ["Bob", 1]], row_count=2, truncated=True
    )
    check_golden("cypher_result.txt", cr.to_llm_text())


def test_golden_entity_found_and_not_found():
    er = EntityResult(
        query="Alice",
        found=True,
        entity=EntityCard(
            name="Alice",
            label="Person",
            description="Engineer",
            properties={"seniority": "senior"},
        ),
        neighbors=[RelationTriple(source="Alice", type="WORKS_AT", target="Acme Corp", fact=None)],
        nearby=["Alice Smith"],
        documents=[DocumentRef(document_id="doc-a", document_path="docs/a.md")],
    )
    check_golden("entity_result.txt", er.to_llm_text())
    missing = EntityResult(
        query="Zorp", found=False, entity=None, neighbors=[], nearby=[], documents=[]
    )
    assert "No entity" in missing.to_llm_text() and "Zorp" in missing.to_llm_text()


def test_golden_remember():
    rr = RememberResult(
        document_id="text-abc123",
        chunks_indexed=1,
        nodes_created=2,
        relationships_created=1,
        finalized=False,
    )
    check_golden("remember_result.txt", rr.to_llm_text())
    assert (
        "Finalized."
        in RememberResult(
            document_id="d",
            chunks_indexed=0,
            nodes_created=0,
            relationships_created=0,
            finalized=True,
        ).to_llm_text()
    )
