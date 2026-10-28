"""Tests for graphrag_sdk.tools.models: typed results and their LLM text."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from graphrag_sdk.tools.models import (
    ChunkRef,
    CitationCheckResult,
    CypherResult,
    CypherRow,
    DocumentRef,
    EntityCard,
    EntityMatch,
    EntityTypeInfo,
    EvidenceRef,
    Fact,
    FinalizeReport,
    IngestResult,
    LookupResult,
    OntologyResult,
    RelationTriple,
    RelationTypeInfo,
    RetrieveResult,
    SkillRunResult,
    StatusResult,
    ToolResult,
    TraverseNode,
    TraverseResult,
    WalkPath,
    WalkResult,
)


def retrieve_result(n_entities: int = 3) -> RetrieveResult:
    return RetrieveResult(
        question="Where does Alice work?",
        entities=[
            EntityCard(n=i + 1, id=f"e{i}", name=f"Entity {i}", label="Person", description="d")
            for i in range(n_entities)
        ],
        relations=[
            RelationTriple(n=10, source="Alice", type="WORKS_AT", target="Acme", fact="since 2020")
        ],
        facts=[Fact(n=11, text="Alice joined Acme in 2020.")],
        chunks=[
            ChunkRef(n=12, chunk_id="c1", document_path="hr.pdf", text="Alice works at Acme."),
            ChunkRef(n=13, chunk_id="c2", document_id="doc-2", text="Second passage."),
        ],
        documents=[DocumentRef(document_id="doc-1", document_path="hr.pdf")],
    )


def big_cypher_result(rows: int = 50) -> CypherResult:
    return CypherResult(
        columns=["name", "n"],
        rows=[CypherRow(n=i + 1, values=[f"name-{i}", i]) for i in range(rows)],
        row_count=rows,
    )


ALL_RESULTS: list[ToolResult] = [
    StatusResult(
        entities=3,
        relations=2,
        documents=1,
        chunks=4,
        entity_types=[EntityTypeInfo(label="Person", count=2)],
        relation_types=[RelationTypeInfo(label="WORKS_AT", count=2)],
    ),
    OntologyResult(
        entity_types=[EntityTypeInfo(label="Person", description="A human", count=2)],
        relation_types=[RelationTypeInfo(label="WORKS_AT", patterns=[("Person", "Company")])],
    ),
    LookupResult(query="acme", matches=[EntityMatch(id="acme", name="Acme", label="Company")]),
    TraverseResult(
        start="Alice",
        start_id="alice",
        nodes=[TraverseNode(n=1, id="acme", name="Acme", depth=1)],
        edges=[RelationTriple(n=2, source="Alice", type="WORKS_AT", target="Acme")],
    ),
    WalkResult(start="Alice", paths=[WalkPath(n=1, nodes=["Alice", "Acme"], edges=["WORKS_AT"])]),
    big_cypher_result(5),
    retrieve_result(),
    SkillRunResult(skill="impact_analysis", n=1, summary="2 impacted", data={"k": [1, 2]}),
    CitationCheckResult(grounded=True, reason="cited", cited=[1], answer="x [1]"),
    IngestResult(document_id="doc-1", chunks_indexed=2),
    FinalizeReport(),
]


# ── Bounded, deterministic rendering ─────────────────────────────


@pytest.mark.parametrize("model", ALL_RESULTS, ids=lambda m: type(m).__name__)
@pytest.mark.parametrize("max_chars", [1, 30, 80, 200, 4000])
def test_rendering_is_bounded_and_deterministic(model, max_chars):
    text = model.to_llm_text(max_chars=max_chars)
    assert len(text) <= max_chars
    if max_chars >= 30:
        assert text
    assert text == model.to_llm_text(max_chars=max_chars)


@pytest.mark.parametrize("model", ALL_RESULTS, ids=lambda m: type(m).__name__)
def test_json_round_trip(model):
    again = type(model).model_validate_json(model.model_dump_json())
    assert again == model
    assert again.to_llm_text() == model.to_llm_text()


@pytest.mark.parametrize("model", ALL_RESULTS, ids=lambda m: type(m).__name__)
def test_results_reject_unknown_fields(model):
    with pytest.raises(ValidationError):
        type(model).model_validate({**model.model_dump(), "surprise": 1})


@pytest.mark.parametrize(
    ("cls", "kwargs"),
    [
        (EntityMatch, {"id": "a", "name": "A"}),
        (TraverseNode, {"id": "a", "name": "A", "depth": 1}),
        (WalkPath, {}),
        (CypherRow, {}),
        (EvidenceRef, {"n": 1, "tool": "t", "kind": "k", "content": "c"}),
    ],
)
def test_strict_item_models_reject_unknown_fields(cls, kwargs):
    with pytest.raises(ValidationError):
        cls(**kwargs, surprise=1)


def test_zero_budget_renders_nothing():
    assert big_cypher_result().to_llm_text(max_chars=0) == ""


def test_truncation_happens_at_item_boundaries():
    model = big_cypher_result(50)
    full = model.to_llm_text(max_chars=100_000)
    text = model.to_llm_text(max_chars=400)
    lines = text.splitlines()
    assert lines[-1].startswith("  …(") and lines[-1].endswith(" more)")
    dropped = int(lines[-1].split("(")[1].split()[0])
    shown = [ln for ln in lines if ln.startswith("- [")]
    assert len(shown) + dropped == 50
    full_lines = set(full.splitlines())
    assert all(ln in full_lines for ln in shown)  # whole items only


def test_no_header_dangles_without_items():
    model = retrieve_result(n_entities=3)
    for max_chars in range(20, 400, 7):
        lines = model.to_llm_text(max_chars=max_chars).splitlines()
        for i, line in enumerate(lines):
            if line.endswith("):") and " (" in line:  # a "Header (N):" line
                assert i + 1 < len(lines), (max_chars, lines)
                assert lines[i + 1].startswith(("- ", "  …(")), (max_chars, lines)


def test_section_that_cannot_fit_is_summarised():
    model = retrieve_result(n_entities=3)
    # Room for the one-line summary but not for a header plus its drop marker.
    text = model.to_llm_text(max_chars=55)
    assert text == "Question: Where does Alice work?\nEntities: …(3 items)"
    # With a little more room the header is kept and followed by a marker.
    text = model.to_llm_text(max_chars=60)
    assert text == "Question: Where does Alice work?\nEntities (3):\n  …(3 more)"


def test_preamble_that_cannot_fit_is_cut():
    model = LookupResult(query="x" * 100)
    text = model.to_llm_text(max_chars=20)
    assert len(text) == 20 and text.endswith("…")


def test_control_characters_and_whitespace_are_cleaned():
    model = LookupResult(
        query="ac\x00me\x1b",
        matches=[EntityMatch(id="a\x07", name="Ac\x00me\n  Corp", description="line1\r\nline2")],
    )
    text = model.to_llm_text()
    assert "\x00" not in text and "\x1b" not in text and "\x07" not in text
    assert 'Lookup: acme\nMatches (1):\n- "Acme Corp" (id a): line1 line2' == text


def test_long_snippets_are_capped():
    text = RetrieveResult(question="q", facts=[Fact(n=1, text="y" * 1000)]).to_llm_text()
    fact_line = text.splitlines()[-1]
    assert fact_line.startswith("- [1] y") and fact_line.endswith("…")
    assert len(fact_line) == len("- [1] ") + 240


# ── Per-model rendering ──────────────────────────────────────────


def test_status_rendering():
    text = ALL_RESULTS[0].to_llm_text()
    assert text == (
        "Entities: 3 | Relations: 2 | Documents: 1 | Chunks: 4\n"
        "Entity types (1):\n- Person: 2\n"
        "Relation types (1):\n- WORKS_AT: 2"
    )
    assert "Hint: Ingest first" in StatusResult(hint="Ingest first").to_llm_text()


def test_ontology_rendering():
    model = OntologyResult(
        entity_types=[
            EntityTypeInfo(label="Person", description="A human", count=2, properties=["age"])
        ],
        relation_types=[
            RelationTypeInfo(
                label="WORKS_AT", description="Employment", patterns=[("Person", "Company")]
            )
        ],
    )
    assert model.to_llm_text() == (
        "Entity types (1):\n- Person: 2 — A human (props: age)\n"
        "Relation types (1):\n- WORKS_AT: 0 [Person->Company] — Employment"
    )


def test_lookup_rendering():
    model = LookupResult(
        query="acme",
        matches=[
            EntityMatch(id="acme", name="Acme", label="Company", description="Maker"),
            EntityMatch(id="acme2", name="Acme Labs"),
        ],
        hint="Pass an id to traverse.",
    )
    assert model.to_llm_text() == (
        "Lookup: acme\nHint: Pass an id to traverse.\nMatches (2):\n"
        '- "Acme" (Company, id acme): Maker\n- "Acme Labs" (id acme2)'
    )


def test_traverse_rendering():
    model = TraverseResult(
        start="Alice",
        start_id="alice",
        via=["WORKS_AT", "LOCATED_IN"],
        nodes=[TraverseNode(n=3, id="berlin", name="Berlin", depth=2)],
        edges=[RelationTriple(n=2, source="Alice", type="WORKS_AT", target="Acme", fact="f")],
        stopped="depth_budget",
    )
    assert model.to_llm_text() == (
        "From Alice via WORKS_AT -> LOCATED_IN: stopped=depth_budget\n"
        "Edges (1):\n- [2] Alice -[WORKS_AT]-> Acme: f\n"
        "Reached (1):\n- [3] Berlin (depth 2)"
    )


def test_walk_rendering():
    model = WalkResult(
        start="Alice",
        goal="Berlin",
        paths=[
            WalkPath(
                n=4,
                nodes=["Alice", "Acme", "Berlin"],
                edges=["WORKS_AT", "LOCATED_IN"],
                score=0.5,
            ),
            WalkPath(nodes=["A", "B"], score=0.25),  # missing edge name
        ],
    )
    assert model.to_llm_text() == (
        "Paths from Alice to Berlin\nPaths (2):\n"
        "- [4] Alice -[WORKS_AT]-> Acme -[LOCATED_IN]-> Berlin (score 0.500)\n"
        "- A -[?]-> B (score 0.250)"
    )


def test_cypher_rendering():
    model = CypherResult(
        columns=["name", "n"],
        rows=[CypherRow(n=1, values=["Älice", 2]), CypherRow(values=[None, {"k": 1}])],
        row_count=2,
        truncated=True,
        hint="Raise max_rows.",
    )
    assert model.to_llm_text() == (
        "Columns: name, n\nRows: 2 (truncated)\nHint: Raise max_rows.\n"
        'Rows (2):\n- [1] ["Älice", 2]\n- [null, {"k": 1}]'
    )


def test_retrieve_rendering():
    text = retrieve_result(n_entities=1).to_llm_text()
    assert text == (
        "Question: Where does Alice work?\n"
        "Entities (1):\n- [1] Entity 0 [Person]: d\n"
        "Relations (1):\n- [10] Alice -[WORKS_AT]-> Acme: since 2020\n"
        "Facts (1):\n- [11] Alice joined Acme in 2020.\n"
        "Passages (2):\n- [12] [hr.pdf] Alice works at Acme.\n- [13] [doc-2] Second passage."
    )


def test_skill_rendering():
    model = SkillRunResult(skill="impact", n=5, summary="Two hits.", data={"hits": ["a", "b"]})
    assert model.to_llm_text() == '[5] Skill impact\nTwo hits.\nData (1):\n- hits: ["a", "b"]'
    assert SkillRunResult(skill="s").to_llm_text() == "Skill s"


def test_long_skill_summary_is_cut_not_dropped():
    model = SkillRunResult(skill="impact", n=5, summary="word " * 400, data={"k": 1})
    text = model.to_llm_text(max_chars=300)
    head, summary = text.split("\n")
    assert head == "[5] Skill impact"
    assert summary.startswith("word word") and summary.endswith("…")
    assert len(text) == 300


def test_citation_check_rendering():
    model = CitationCheckResult(
        grounded=True,
        reason="cited",
        cited=[1],
        invalid=[9],
        answer="Acme [1].",
        sources=[EvidenceRef(n=1, tool="retrieve", kind="passage", content="Acme text")],
    )
    assert model.to_llm_text() == (
        "Grounded: True (cited)\nInvalid citations removed: [9]\n"
        "Sources (1):\n- [1] (retrieve) Acme text"
    )
    plain = CitationCheckResult(grounded=False, reason="no_citations").to_llm_text()
    assert plain == "Grounded: False (no_citations)"


def test_ingest_rendering():
    model = IngestResult(document_id="doc-1", chunks_indexed=2, nodes_created=3)
    assert model.to_llm_text() == (
        "Stored document 'doc-1' (2 chunks, 3 nodes, 0 relations).\nNot finalized yet."
    )
    assert IngestResult(document_id="d", finalized=True).to_llm_text().endswith("Finalized.")


def test_finalize_rendering():
    assert FinalizeReport(entities_deduplicated=2, entities_embedded=5).to_llm_text() == (
        "Finalized: 2 duplicates merged, 5 entities and 0 relations embedded, "
        "0 empty stubs removed."
    )
    assert FinalizeReport(finalized=False, hint="Busy").to_llm_text() == (
        "Not finalized.\nHint: Busy"
    )
