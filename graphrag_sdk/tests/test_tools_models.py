"""Tests for graphrag_sdk.tools.models: typed results and their LLM text."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from graphrag_sdk.tools.models import (
    ChunkRef,
    CitationCheckResult,
    DocumentRef,
    EntityCard,
    EntityTypeInfo,
    EvidenceRef,
    Fact,
    FinalizeReport,
    IngestResult,
    OntologyResult,
    RelationTriple,
    RelationTypeInfo,
    RetrieveResult,
    StatusResult,
    ToolResult,
    _render,
    properties_text,
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


def big_retrieve_result(rows: int = 50) -> RetrieveResult:
    return RetrieveResult(
        question="q",
        entities=[EntityCard(n=i + 1, name=f"name-{i}", label="Person") for i in range(rows)],
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
    retrieve_result(),
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
        (EntityCard, {"name": "A"}),
        (Fact, {"text": "t"}),
        (EvidenceRef, {"n": 1, "tool": "t", "kind": "k", "content": "c"}),
    ],
)
def test_strict_item_models_reject_unknown_fields(cls, kwargs):
    with pytest.raises(ValidationError):
        cls(**kwargs, surprise=1)


def test_zero_budget_renders_nothing():
    assert big_retrieve_result().to_llm_text(max_chars=0) == ""


def test_truncation_happens_at_item_boundaries():
    model = big_retrieve_result(50)
    full = model.to_llm_text(max_chars=100_000)
    text = model.to_llm_text(max_chars=400)
    lines = text.splitlines()
    assert lines[-1].startswith("  …(") and lines[-1].endswith(" more omitted)")
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
    assert text == "Question: Where does Alice work?\nEntities: …(3 omitted)"
    # With a little more room the header is kept and followed by a marker.
    text = model.to_llm_text(max_chars=70)
    assert text == "Question: Where does Alice work?\nEntities (3):\n  …(3 more omitted)"


def test_a_long_section_does_not_push_out_the_passages():
    model = big_retrieve_result(200)
    model.facts = [Fact(n=300 + i, text=f"Fact number {i}.") for i in range(5)]
    model.chunks = [
        ChunkRef(n=400 + i, chunk_id=f"c{i}", document_id="d", text=f"Passage {i}. " * 9)
        for i in range(30)
    ]
    text = model.to_llm_text(max_chars=2000)
    assert len(text) <= 2000
    assert "Facts (5):" in text and "- [304] Fact number 4." in text  # small: kept whole
    assert "Passages (30):" in text and "- [400] [d] Passage 0." in text
    assert "more omitted)" in text.split("Passages (30):")[1]
    assert "Entities (200):" in text and "more omitted)" in text.split("Facts")[0]


def test_default_budget_keeps_a_full_retrieve_result():
    model = big_retrieve_result(20)
    model.chunks = [
        ChunkRef(n=100 + i, chunk_id=f"c{i}", text="Passage text. " * 20) for i in range(10)
    ]
    assert "omitted" not in model.to_llm_text()


def test_preamble_that_cannot_fit_is_cut():
    model = RetrieveResult(question="x" * 100)
    text = model.to_llm_text(max_chars=20)
    assert len(text) == 20 and text.endswith("…")


def test_control_characters_and_whitespace_are_cleaned():
    model = RetrieveResult(
        question="ac\x00me\x1b",
        entities=[
            EntityCard(label="Co\x07", name="Ac\x00me\n  Corp", description="line1\r\nline2")
        ],
    )
    text = model.to_llm_text()
    assert "\x00" not in text and "\x1b" not in text and "\x07" not in text
    assert "Question: acme\nEntities (1):\n- Acme Corp [Co]: line1 line2" == text


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


def test_whole_floats_are_shown_as_integers():
    props = {"age": 40.0, "rate": 2.5, "ids": [1.0, 2.25], "big": 1e20, "flag": True}
    assert properties_text(props) == "age: 40; rate: 2.5; ids: 1, 2.25; big: 1e+20; flag: True"


def test_status_and_ontology_carry_their_evidence_number():
    status = StatusResult(entities=3, n=4).to_llm_text()
    assert status.startswith("[4] Entities: 3 |")
    ontology = OntologyResult(entity_types=[EntityTypeInfo(label="Person", count=2)], n=5)
    assert ontology.to_llm_text().startswith("[5] Ontology with live counts:\nEntity types (1):")


def test_a_section_that_does_not_fit_leaves_a_marker():
    text = _render(["x" * 30], [("Edges", ["- a"] * 3)], max_chars=45)
    assert text == "x" * 30 + "\n" + "…"


def test_retrieve_rendering():
    text = retrieve_result(n_entities=1).to_llm_text()
    assert text == (
        "Question: Where does Alice work?\n"
        "Entities (1):\n- [1] Entity 0 [Person]: d\n"
        "Relations (1):\n- [10] Alice -[WORKS_AT]-> Acme: since 2020\n"
        "Facts (1):\n- [11] Alice joined Acme in 2020.\n"
        "Passages (2):\n- [12] [hr.pdf] Alice works at Acme.\n- [13] [doc-2] Second passage."
    )


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
