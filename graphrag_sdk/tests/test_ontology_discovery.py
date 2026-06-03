"""Unit tests for ``Ontology.from_sources`` and its discovery internals.

Three layers covered:
  - ``extract_with_retry`` (instructor pattern wrapper)
  - ``_validate_proposal`` (semantic validation)
  - ``discover_ontology`` + the public ``Ontology.from_sources`` classmethod
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from graphrag_sdk import Ontology, OntologyDiscoveryError, SchemaExtensionProposal
from graphrag_sdk.core.context import Context
from graphrag_sdk.core.models import (
    Attribute,
    Entity,
    Relation,
    TextChunk,
    TextChunks,
)
from graphrag_sdk.discovery.instructor import extract_with_retry
from graphrag_sdk.discovery.pipeline import (
    _diff_ontologies,
    _ensure_sdk_managed_attributes,
    _validate_proposal,
    discover_ontology,
    suggest_extensions,
)
from graphrag_sdk.discovery.proposal import (
    ChunkProposal,
    DocSummary,
    _ProposedAttribute,
    _ProposedEntity,
    _ProposedRelation,
)
from graphrag_sdk.ingestion.chunking_strategies.base import ChunkingStrategy

from .conftest import MockLLM


# ── Test helpers ───────────────────────────────────────────────────


class RecordingMockLLM(MockLLM):
    """MockLLM that captures every call's full message history.

    The base ``MockLLM`` only stores the latest. For instructor-pattern
    tests we want to verify retry feedback is appended to the
    conversation, which means inspecting an arbitrary attempt's
    messages.
    """

    def __init__(self, responses: list[str], **kwargs: Any) -> None:
        super().__init__(responses=responses, strict=True, **kwargs)
        self.all_calls: list[list] = []

    async def ainvoke_messages(self, messages, *, max_retries=3, **kwargs):
        # Snapshot the message list (caller mutates it after the call).
        self.all_calls.append([m.model_copy() for m in messages])
        self.last_messages = messages
        return self.invoke("")


class FixedChunker(ChunkingStrategy):
    """Deterministic chunker that yields a pre-specified chunk list.

    Lets pipeline tests pin the exact number of chunk-level LLM calls.
    """

    def __init__(self, chunks: list[str]) -> None:
        self._chunks = chunks

    async def chunk(self, text: str, ctx: Context) -> TextChunks:
        return TextChunks(
            chunks=[
                TextChunk(text=c, index=i, uid=f"chunk-{i}")
                for i, c in enumerate(self._chunks)
            ]
        )


def _valid_chunk_proposal_json() -> str:
    return json.dumps(
        {
            "entities": [
                {"label": "Person", "description": "A human", "properties": []},
                {"label": "Company", "description": "A business", "properties": []},
            ],
            "relations": [
                {
                    "label": "WORKS_AT",
                    "description": "Employment",
                    "patterns": [["Person", "Company"]],
                    "properties": [],
                },
            ],
        }
    )


def _valid_doc_summary_json() -> str:
    return json.dumps(
        {"main_entities": ["Alice", "Acme Corp"], "aboutness": "Test doc"}
    )


def _valid_normalized_json() -> str:
    # Same shape as ChunkProposal but interpreted as a NormalizedDraft.
    return _valid_chunk_proposal_json()


# ── extract_with_retry ─────────────────────────────────────────────


class TestExtractWithRetry:
    """The single piece lifted from ``barakb/text-to-rdf``."""

    @pytest.mark.asyncio
    async def test_succeeds_on_first_try(self) -> None:
        llm = RecordingMockLLM([_valid_doc_summary_json()])
        result = await extract_with_retry(
            llm,
            system_prompt="sys",
            user_prompt="usr",
            response_model=DocSummary,
            max_retries=2,
        )
        assert isinstance(result, DocSummary)
        assert result.main_entities == ["Alice", "Acme Corp"]
        assert len(llm.all_calls) == 1

    @pytest.mark.asyncio
    async def test_parse_fail_then_succeed_with_feedback(self) -> None:
        llm = RecordingMockLLM(["not valid json", _valid_doc_summary_json()])
        result = await extract_with_retry(
            llm,
            system_prompt="sys",
            user_prompt="usr",
            response_model=DocSummary,
            max_retries=2,
        )
        assert result.aboutness == "Test doc"
        # Two calls happened.
        assert len(llm.all_calls) == 2
        # Second call's history contains a user message with parse feedback.
        second_call = llm.all_calls[1]
        feedback_messages = [m for m in second_call if m.role == "user"]
        assert len(feedback_messages) == 2  # original + feedback
        assert "could not be parsed" in feedback_messages[1].content

    @pytest.mark.asyncio
    async def test_validate_fail_then_succeed_with_feedback(self) -> None:
        # First response: valid JSON but bad attribute type → validator rejects.
        bad = json.dumps(
            {
                "entities": [
                    {
                        "label": "Person",
                        "properties": [
                            {"name": "age", "type": "INT", "description": None}
                        ],
                    }
                ],
                "relations": [],
            }
        )
        good = _valid_chunk_proposal_json()
        llm = RecordingMockLLM([bad, good])
        result = await extract_with_retry(
            llm,
            system_prompt="sys",
            user_prompt="usr",
            response_model=ChunkProposal,
            extra_validate=_validate_proposal,
            max_retries=2,
        )
        assert isinstance(result, ChunkProposal)
        assert len(llm.all_calls) == 2
        second_call_user_msgs = [m for m in llm.all_calls[1] if m.role == "user"]
        # Second user message is the validation feedback citing the bad type.
        assert "did not pass validation" in second_call_user_msgs[1].content
        assert "INT" in second_call_user_msgs[1].content

    @pytest.mark.asyncio
    async def test_exhausts_retries_raises(self) -> None:
        llm = RecordingMockLLM(["bad1", "bad2", "bad3"])
        with pytest.raises(OntologyDiscoveryError) as exc_info:
            await extract_with_retry(
                llm,
                system_prompt="sys",
                user_prompt="usr",
                response_model=DocSummary,
                max_retries=2,  # 1 initial + 2 retries = 3 attempts
                chunk_id="chunk-X",
            )
        err = exc_info.value
        assert err.attempts == 3
        assert err.chunk_id == "chunk-X"
        assert err.last_error is not None

    @pytest.mark.asyncio
    async def test_strips_markdown_fences(self) -> None:
        # LLM sometimes wraps JSON in ```json … ``` fences.
        fenced = f"```json\n{_valid_doc_summary_json()}\n```"
        llm = RecordingMockLLM([fenced])
        result = await extract_with_retry(
            llm,
            system_prompt="sys",
            user_prompt="usr",
            response_model=DocSummary,
            max_retries=0,
        )
        assert result.aboutness == "Test doc"

    @pytest.mark.asyncio
    async def test_disables_provider_retries(self) -> None:
        """The wrapper owns retry policy — provider retries would multiply
        the budget and skew OntologyDiscoveryError.attempts."""

        # Build a MockLLM that records the kwargs each ainvoke_messages
        # call receives so we can assert max_retries=1 is forwarded.
        observed_max_retries: list[int] = []

        class _Spy(MockLLM):
            async def ainvoke_messages(self, messages, *, max_retries=3, **kwargs):
                observed_max_retries.append(max_retries)
                self.last_messages = messages
                return self.invoke("")

        llm = _Spy(responses=[_valid_doc_summary_json()], strict=True)
        await extract_with_retry(
            llm,
            system_prompt="sys",
            user_prompt="usr",
            response_model=DocSummary,
            max_retries=2,
        )
        assert observed_max_retries == [1], (
            "extract_with_retry must pass max_retries=1 so the provider does "
            "not silently multiply the retry budget."
        )

    @pytest.mark.asyncio
    async def test_extra_field_in_response_triggers_retry(self) -> None:
        """Unknown keys in the LLM JSON must be rejected (extra='forbid')
        so the retry loop can ask the model to remove them, instead of
        silently dropping fields that may reflect a misunderstood schema."""
        bad = json.dumps({"main_entities": [], "aboutness": "ok", "junk": 1})
        good = _valid_doc_summary_json()
        llm = RecordingMockLLM([bad, good])
        result = await extract_with_retry(
            llm,
            system_prompt="sys",
            user_prompt="usr",
            response_model=DocSummary,
            max_retries=1,
        )
        assert result.aboutness == "Test doc"
        assert len(llm.all_calls) == 2, "Expected parse-then-retry on extra='junk'"


# ── _validate_proposal ─────────────────────────────────────────────


class TestValidateProposal:
    def test_accepts_valid_proposal(self) -> None:
        proposal = ChunkProposal(
            entities=[
                _ProposedEntity(label="Person", properties=[]),
                _ProposedEntity(label="Company", properties=[]),
            ],
            relations=[
                _ProposedRelation(
                    label="WORKS_AT", patterns=[("Person", "Company")], properties=[]
                )
            ],
        )
        assert _validate_proposal(proposal) == []

    def test_rejects_bad_attribute_type(self) -> None:
        proposal = ChunkProposal(
            entities=[
                _ProposedEntity(
                    label="Person",
                    properties=[_ProposedAttribute(name="age", type="INT")],
                )
            ],
            relations=[],
        )
        errors = _validate_proposal(proposal)
        assert any("INT" in e for e in errors)

    def test_rejects_reserved_attribute_name_on_entity(self) -> None:
        """Truly-internal reserved names are rejected on entities."""
        # `description` is one of the names the SDK auto-fills on every node;
        # `name` is the entity identifier and is allowed in the schema as
        # documentation (filtered out of extraction). This test uses
        # `description` to ensure non-managed reserved names still raise.
        proposal = ChunkProposal(
            entities=[
                _ProposedEntity(
                    label="Person",
                    properties=[_ProposedAttribute(name="description")],
                )
            ],
            relations=[],
        )
        errors = _validate_proposal(proposal)
        assert any("reserved" in e for e in errors)

    def test_accepts_name_as_sdk_managed_attribute_on_entity(self) -> None:
        """`name` is in _RESERVED but in _SDK_MANAGED — allowed on entities."""
        proposal = ChunkProposal(
            entities=[
                _ProposedEntity(
                    label="Person",
                    properties=[_ProposedAttribute(name="name", type="STRING")],
                )
            ],
            relations=[],
        )
        assert _validate_proposal(proposal) == []

    def test_rejects_name_on_relation_attribute(self) -> None:
        """`name` is SDK-managed on entities but relations don't have a
        name field, so it must still be rejected there."""
        proposal = ChunkProposal(
            entities=[_ProposedEntity(label="Person")],
            relations=[
                _ProposedRelation(
                    label="MENTIONS",
                    properties=[_ProposedAttribute(name="name")],
                )
            ],
        )
        errors = _validate_proposal(proposal)
        assert any("reserved" in e for e in errors)

    def test_rejects_reserved_attribute_name_on_relation(self) -> None:
        proposal = ChunkProposal(
            entities=[_ProposedEntity(label="Person")],
            relations=[
                _ProposedRelation(
                    label="MENTIONS",
                    properties=[_ProposedAttribute(name="source_chunk_ids")],
                )
            ],
        )
        errors = _validate_proposal(proposal)
        assert any("reserved" in e for e in errors)

    def test_rejects_pattern_with_missing_entity(self) -> None:
        proposal = ChunkProposal(
            entities=[_ProposedEntity(label="Person")],
            relations=[
                _ProposedRelation(
                    label="WORKS_AT",
                    patterns=[("Person", "Company")],  # Company not declared
                )
            ],
        )
        errors = _validate_proposal(proposal)
        assert any("Company" in e for e in errors)


# ── discover_ontology pipeline ─────────────────────────────────────


class TestPipeline:
    @pytest.mark.asyncio
    async def test_one_doc_one_chunk_end_to_end(self, tmp_path: Path) -> None:
        # Source file with one chunk's worth of text.
        src = tmp_path / "doc.txt"
        src.write_text("Alice works at Acme Corp.")

        # Sequence: 1 doc summary + 1 chunk proposal + 1 normalize.
        llm = RecordingMockLLM(
            [
                _valid_doc_summary_json(),
                _valid_chunk_proposal_json(),
                _valid_normalized_json(),
            ]
        )

        ontology = await discover_ontology(
            str(src),
            llm,
            sample_chunks_per_doc=1,
            max_retries=1,
            concurrency=1,
            chunker=FixedChunker(["Alice works at Acme Corp."]),
            seed=42,
        )

        assert len(llm.all_calls) == 3, "Expected 1 summary + 1 chunk + 1 normalize"
        labels = {e.label for e in ontology.entities}
        assert {"Person", "Company"} <= labels
        rel_labels = {r.label for r in ontology.relations}
        assert "WORKS_AT" in rel_labels

    @pytest.mark.asyncio
    async def test_existing_prior_is_merged_into_result(self, tmp_path: Path) -> None:
        src = tmp_path / "doc.txt"
        src.write_text("Alice works at Acme Corp.")
        existing = Ontology(
            entities=[Entity(label="Document", description="A file")],
            relations=[],
        )
        llm = RecordingMockLLM(
            [
                _valid_doc_summary_json(),
                _valid_chunk_proposal_json(),
                _valid_normalized_json(),
            ]
        )
        result = await discover_ontology(
            str(src),
            llm,
            existing=existing,
            sample_chunks_per_doc=1,
            max_retries=1,
            concurrency=1,
            chunker=FixedChunker(["Alice works at Acme Corp."]),
            seed=42,
        )
        labels = {e.label for e in result.entities}
        assert "Document" in labels  # from `existing`
        assert "Person" in labels  # from discovery

    @pytest.mark.asyncio
    async def test_soft_fails_on_unrecoverable_chunk(self, tmp_path: Path) -> None:
        """A chunk that exhausts its retries is skipped, not fatal."""
        src = tmp_path / "doc.txt"
        src.write_text("Some content.")
        # Sequence: summary OK, chunk1 fails twice (bad+bad), chunk2 succeeds,
        # then normalize succeeds.
        llm = RecordingMockLLM(
            [
                _valid_doc_summary_json(),  # 1: summary
                "garbage",  # 2: chunk1 attempt 1
                "still bad",  # 3: chunk1 attempt 2 → exhausted
                _valid_chunk_proposal_json(),  # 4: chunk2 attempt 1 OK
                _valid_normalized_json(),  # 5: normalize
            ]
        )
        result = await discover_ontology(
            str(src),
            llm,
            sample_chunks_per_doc=2,
            max_retries=1,  # 1 initial + 1 retry = 2 attempts per call
            concurrency=1,  # sequential so the script order is deterministic
            chunker=FixedChunker(["chunk1 text", "chunk2 text"]),
            seed=42,
        )
        # Despite chunk1 failing, the final ontology came through.
        assert len(result.entities) >= 1
        assert {"Person", "Company"} <= {e.label for e in result.entities}

    @pytest.mark.asyncio
    async def test_classmethod_dispatches(self, tmp_path: Path) -> None:
        src = tmp_path / "doc.txt"
        src.write_text("Alice works at Acme Corp.")
        llm = RecordingMockLLM(
            [
                _valid_doc_summary_json(),
                _valid_chunk_proposal_json(),
                _valid_normalized_json(),
            ]
        )
        ontology = await Ontology.from_sources(
            str(src),
            llm,
            sample_chunks_per_doc=1,
            max_retries=1,
            concurrency=1,
            chunker=FixedChunker(["text"]),
            seed=42,
        )
        assert isinstance(ontology, Ontology)
        assert {"Person", "Company"} <= {e.label for e in ontology.entities}

    @pytest.mark.asyncio
    async def test_normalization_failure_falls_back_to_unnormalized(
        self, tmp_path: Path
    ) -> None:
        """If the final normalize LLM call fails, return the merged draft."""
        src = tmp_path / "doc.txt"
        src.write_text("Alice works at Acme.")
        # Summary OK, chunk OK, normalize permanently bad.
        llm = RecordingMockLLM(
            [
                _valid_doc_summary_json(),
                _valid_chunk_proposal_json(),
                "still not json",
                "still not json",
            ]
        )
        result = await discover_ontology(
            str(src),
            llm,
            sample_chunks_per_doc=1,
            max_retries=1,
            concurrency=1,
            chunker=FixedChunker(["text"]),
            seed=42,
        )
        # Fell back to the un-normalized merged draft, which still has the entities.
        assert {"Person", "Company"} <= {e.label for e in result.entities}


# ── _ensure_sdk_managed_attributes ────────────────────────────────


class TestEnsureSDKManagedAttributes:
    def test_adds_name_to_entity_that_lacks_it(self) -> None:
        ont = Ontology(
            entities=[Entity(label="Person", properties=[Attribute(name="role")])]
        )
        result = _ensure_sdk_managed_attributes(ont)
        person = next(e for e in result.entities if e.label == "Person")
        attr_names = [a.name for a in person.properties]
        assert "name" in attr_names
        assert attr_names[0] == "name", "name should appear first"
        assert "role" in attr_names

    def test_preserves_existing_name_attribute(self) -> None:
        original_attr = Attribute(name="name", type="STRING", description="custom")
        ont = Ontology(entities=[Entity(label="Person", properties=[original_attr])])
        result = _ensure_sdk_managed_attributes(ont)
        person = next(e for e in result.entities if e.label == "Person")
        # The original (custom-described) attribute is kept, not overwritten.
        name_attr = next(a for a in person.properties if a.name == "name")
        assert name_attr.description == "custom"

    def test_no_op_on_empty_ontology(self) -> None:
        empty = Ontology()
        assert _ensure_sdk_managed_attributes(empty).entities == []

    def test_returns_same_instance_when_nothing_changes(self) -> None:
        """Optimisation check — don't rebuild when every entity already has name."""
        ont = Ontology(
            entities=[
                Entity(label="Person", properties=[Attribute(name="name")]),
                Entity(label="Company", properties=[Attribute(name="name")]),
            ]
        )
        assert _ensure_sdk_managed_attributes(ont) is ont

    def test_does_not_touch_relations(self) -> None:
        """Relations are not entity-shaped; no name attribute is added there."""
        rel = Relation(label="WORKS_AT", patterns=[("Person", "Company")])
        ont = Ontology(entities=[Entity(label="Person")], relations=[rel])
        result = _ensure_sdk_managed_attributes(ont)
        result_rel = next(r for r in result.relations if r.label == "WORKS_AT")
        assert result_rel.properties == []


# ── _diff_ontologies (Layer B internal) ────────────────────────────


class TestDiffOntologies:
    def test_empty_diff_when_identical(self) -> None:
        ont = Ontology(
            entities=[Entity(label="Person")],
            relations=[Relation(label="KNOWS", patterns=[("Person", "Person")])],
        )
        proposal = _diff_ontologies(ont, ont)
        assert proposal.is_empty

    def test_detects_new_entity(self) -> None:
        existing = Ontology(entities=[Entity(label="Person")])
        discovered = Ontology(
            entities=[Entity(label="Person"), Entity(label="Company")]
        )
        proposal = _diff_ontologies(existing, discovered)
        assert [e.label for e in proposal.new_entities] == ["Company"]
        assert proposal.new_attributes == []

    def test_detects_new_attribute_on_existing_entity(self) -> None:
        existing = Ontology(entities=[Entity(label="Person")])
        discovered = Ontology(
            entities=[
                Entity(
                    label="Person",
                    properties=[Attribute(name="role", type="STRING")],
                )
            ]
        )
        proposal = _diff_ontologies(existing, discovered)
        assert proposal.new_entities == []
        assert len(proposal.new_attributes) == 1
        owner, attr = proposal.new_attributes[0]
        assert owner == "Person"
        assert attr.name == "role"

    def test_detects_new_relation_and_pattern(self) -> None:
        existing = Ontology(
            entities=[Entity(label="Person"), Entity(label="Company")],
            relations=[Relation(label="WORKS_AT", patterns=[("Person", "Company")])],
        )
        discovered = Ontology(
            entities=[Entity(label="Person"), Entity(label="Company")],
            relations=[
                Relation(
                    label="WORKS_AT",
                    patterns=[("Person", "Company"), ("Person", "NonProfit")],
                ),
                Relation(label="FOUNDED", patterns=[("Person", "Company")]),
            ],
        )
        proposal = _diff_ontologies(existing, discovered)
        assert [r.label for r in proposal.new_relations] == ["FOUNDED"]
        assert proposal.new_patterns == [("WORKS_AT", "Person", "NonProfit")]

    def test_detects_new_attribute_on_existing_relation(self) -> None:
        """A discovered relation that adds a new property to an existing
        relation label must surface in proposal.new_attributes — without
        this we silently drop relation-property additions."""
        existing = Ontology(
            entities=[Entity(label="Person"), Entity(label="Company")],
            relations=[Relation(label="WORKS_AT", patterns=[("Person", "Company")])],
        )
        discovered = Ontology(
            entities=[Entity(label="Person"), Entity(label="Company")],
            relations=[
                Relation(
                    label="WORKS_AT",
                    patterns=[("Person", "Company")],
                    properties=[Attribute(name="since", type="DATE")],
                )
            ],
        )
        proposal = _diff_ontologies(existing, discovered)
        assert any(
            owner == "WORKS_AT" and attr.name == "since"
            for owner, attr in proposal.new_attributes
        )

    def test_does_not_propose_deletions(self) -> None:
        """Schema-shrinking is never proposed — absence from the new corpus
        is not evidence of irrelevance."""
        existing = Ontology(
            entities=[Entity(label="Person"), Entity(label="Country")]
        )
        discovered = Ontology(entities=[Entity(label="Person")])
        proposal = _diff_ontologies(existing, discovered)
        assert proposal.is_empty


# ── suggest_extensions (Layer B end-to-end) ────────────────────────


class TestSuggestExtensions:
    @pytest.mark.asyncio
    async def test_returns_proposal_with_additions_only(self, tmp_path: Path) -> None:
        src = tmp_path / "doc.txt"
        src.write_text("Alice works at Acme Corp.")
        existing = Ontology(entities=[Entity(label="Person")])
        # Discovery emits Person + Company + WORKS_AT(Person, Company).
        llm = RecordingMockLLM(
            [
                _valid_doc_summary_json(),
                _valid_chunk_proposal_json(),
                _valid_normalized_json(),
            ]
        )
        proposal = await suggest_extensions(
            existing,
            str(src),
            llm,
            sample_chunks_per_doc=1,
            max_retries=1,
            concurrency=1,
            chunker=FixedChunker(["text"]),
            seed=42,
        )
        assert isinstance(proposal, SchemaExtensionProposal)
        new_labels = {e.label for e in proposal.new_entities}
        assert "Company" in new_labels
        assert "Person" not in new_labels  # already in `existing`
        assert any(r.label == "WORKS_AT" for r in proposal.new_relations)
        assert proposal.sources_scanned == [str(src)]

    @pytest.mark.asyncio
    async def test_proposal_is_empty_when_discovery_adds_nothing(
        self, tmp_path: Path
    ) -> None:
        """If the discovery output is already covered by ``existing``,
        the proposal returns empty."""
        src = tmp_path / "doc.txt"
        src.write_text("Alice works at Acme Corp.")
        existing = Ontology(
            entities=[Entity(label="Person"), Entity(label="Company")],
            relations=[
                Relation(label="WORKS_AT", patterns=[("Person", "Company")])
            ],
        )
        llm = RecordingMockLLM(
            [
                _valid_doc_summary_json(),
                _valid_chunk_proposal_json(),
                _valid_normalized_json(),
            ]
        )
        proposal = await suggest_extensions(
            existing,
            str(src),
            llm,
            sample_chunks_per_doc=1,
            max_retries=1,
            concurrency=1,
            chunker=FixedChunker(["text"]),
            seed=42,
        )
        assert proposal.is_empty
        # Coarse evidence still populated.
        assert proposal.sources_scanned == [str(src)]


# ── Catalog + grounded discovery ───────────────────────────────────


# Minimal Schema.org JSON-LD subset used to exercise SchemaOrgCatalog
# without touching the network. Realistic shape — same ``@graph`` /
# ``@type`` / ``schema:domainIncludes`` / ``rdfs:subClassOf`` keys the
# live endpoint emits. Just enough to cover all parsing branches.
_SCHEMA_ORG_FIXTURE = json.dumps(
    {
        "@context": {"schema": "https://schema.org/", "rdf": "...", "rdfs": "..."},
        "@graph": [
            {
                "@id": "schema:Person",
                "@type": "rdfs:Class",
                "rdfs:label": "Person",
                "rdfs:comment": "A human individual.",
            },
            {
                "@id": "schema:Organization",
                "@type": "rdfs:Class",
                "rdfs:label": "Organization",
                "rdfs:comment": "A business or institution.",
            },
            {
                "@id": "schema:Place",
                "@type": "rdfs:Class",
                "rdfs:label": "Place",
                "rdfs:comment": "A geographic place.",
            },
            {
                "@id": "schema:CreativeWork",
                "@type": "rdfs:Class",
                "rdfs:label": "CreativeWork",
                "rdfs:comment": "An authored work.",
            },
            {
                "@id": "schema:Article",
                "@type": "rdfs:Class",
                "rdfs:label": "Article",
                "rdfs:comment": "An article.",
                "rdfs:subClassOf": {"@id": "schema:CreativeWork"},
            },
            # Attribute on Person: birthDate → Date (primitive range).
            {
                "@id": "schema:birthDate",
                "@type": "rdf:Property",
                "rdfs:comment": "Date of birth.",
                "schema:domainIncludes": {"@id": "schema:Person"},
                "schema:rangeIncludes": {"@id": "schema:Date"},
            },
            # Relation on Person: worksFor → Organization (entity range).
            {
                "@id": "schema:worksFor",
                "@type": "rdf:Property",
                "rdfs:comment": "Organization the Person works for.",
                "schema:domainIncludes": {"@id": "schema:Person"},
                "schema:rangeIncludes": {"@id": "schema:Organization"},
            },
            # Attribute on CreativeWork: datePublished → Date. Article
            # should inherit this via subClassOf.
            {
                "@id": "schema:datePublished",
                "@type": "rdf:Property",
                "rdfs:comment": "Date the work was published.",
                "schema:domainIncludes": {"@id": "schema:CreativeWork"},
                "schema:rangeIncludes": {"@id": "schema:Date"},
            },
            # Relation on CreativeWork: author → Person. Article should
            # inherit this too.
            {
                "@id": "schema:author",
                "@type": "rdf:Property",
                "rdfs:comment": "The author of the work.",
                "schema:domainIncludes": {"@id": "schema:CreativeWork"},
                "schema:rangeIncludes": {"@id": "schema:Person"},
            },
        ],
    }
).encode("utf-8")


def _mock_urlopen_fixture(*args, **kwargs):
    """Drop-in replacement for ``urllib.request.urlopen`` that yields the
    fixture bytes as a context-manager-compatible BytesIO. Accepts and
    discards any positional/keyword args (``timeout``, etc.) the catalog
    might pass."""
    import io

    return io.BytesIO(_SCHEMA_ORG_FIXTURE)


class TestSchemaOrgCatalog:
    """Live-fetch SchemaOrgCatalog with the network mocked.

    The catalog now downloads Schema.org's JSON-LD on first use and
    caches the processed result under
    ``$XDG_CACHE_HOME/graphrag-sdk/schema_org.json``. Tests redirect the
    cache to ``tmp_path`` to keep them hermetic and patch
    ``urllib.request.urlopen`` so no real network traffic happens.
    """

    @pytest.fixture
    def schema_org_catalog(self, tmp_path):
        """A fresh catalog with the network mocked and cache under tmp_path."""
        from unittest.mock import patch
        from graphrag_sdk.discovery.catalog import SchemaOrgCatalog

        cache_path = tmp_path / "schema_org.json"
        with patch(
            "graphrag_sdk.discovery.catalog.urllib.request.urlopen",
            new=_mock_urlopen_fixture,
        ):
            yield SchemaOrgCatalog(
                cache_path=cache_path,
                curated_types={"Person", "Organization", "Place", "CreativeWork", "Article"},
            )

    def test_known_types_returns_curated(self, schema_org_catalog) -> None:
        assert schema_org_catalog.known_types() == {
            "Person",
            "Organization",
            "Place",
            "CreativeWork",
            "Article",
        }

    def test_lookup_returns_schema_org_provenance(self, schema_org_catalog) -> None:
        person = schema_org_catalog.lookup("Person")
        assert person is not None
        assert person.label == "Person"
        # The catalog records the canonical URI in the description as a
        # provenance marker in the exact form
        # ``"Schema.org: https://schema.org/<Type>"``.
        assert person.description is not None
        assert "Schema.org: https://schema.org/Person" in person.description
        # ``birthDate`` is on Person directly.
        birth = next(a for a in person.properties if a.name == "birth_date")
        assert birth.type == "DATE"
        assert birth.description is not None
        assert birth.description.startswith("Schema.org birthDate")

    def test_lookup_returns_none_for_unknown(self, schema_org_catalog) -> None:
        assert schema_org_catalog.lookup("Penguin") is None

    def test_relations_among_filters_to_input_set(self, schema_org_catalog) -> None:
        relations = schema_org_catalog.relations_among({"Person", "Organization"})
        for r in relations:
            for src, tgt in r.patterns:
                assert src in {"Person", "Organization"}
                assert tgt in {"Person", "Organization"}
        # The fixture has worksFor (Person → Organization) → WORKS_FOR.
        assert any(
            ("Person", "Organization") in r.patterns for r in relations
        )

    def test_subclass_inherits_base_class_attributes(self, schema_org_catalog) -> None:
        """``Article`` is a subClassOf ``CreativeWork`` in the fixture.
        Walking ``rdfs:subClassOf`` should propagate CreativeWork's
        ``datePublished`` attribute and ``author`` relation onto Article."""
        article = schema_org_catalog.lookup("Article")
        creative_work = schema_org_catalog.lookup("CreativeWork")
        assert article is not None and creative_work is not None
        article_attrs = {a.name for a in article.properties}
        cw_attrs = {a.name for a in creative_work.properties}
        assert "date_published" in article_attrs
        assert cw_attrs <= article_attrs

        # The author relation should also inherit: Article → Person.
        rels = schema_org_catalog.relations_among({"Article", "Person"})
        rel_labels = {r.label for r in rels}
        assert "AUTHOR" in rel_labels

    def test_cache_is_used_on_second_construction(self, tmp_path) -> None:
        """Once written, the cache file is read on a fresh construction
        and the network is not hit again. The mocked urlopen would
        raise on a second call, proving it wasn't called."""
        from unittest.mock import patch, MagicMock
        from graphrag_sdk.discovery.catalog import SchemaOrgCatalog

        cache_path = tmp_path / "cached.json"

        # First instantiation: network returns the fixture, cache is written.
        with patch(
            "graphrag_sdk.discovery.catalog.urllib.request.urlopen",
            new=_mock_urlopen_fixture,
        ):
            first = SchemaOrgCatalog(cache_path=cache_path)
            first.known_types()
        assert cache_path.exists()

        # Second instantiation: any network call would raise. Cache hit.
        sentinel = MagicMock(side_effect=AssertionError("network should not be hit"))
        with patch(
            "graphrag_sdk.discovery.catalog.urllib.request.urlopen",
            new=sentinel,
        ):
            second = SchemaOrgCatalog(cache_path=cache_path)
            assert "Person" in second.known_types()

    def test_network_failure_raises_when_cache_missing(self, tmp_path) -> None:
        from unittest.mock import patch, MagicMock
        from graphrag_sdk.discovery.catalog import (
            SchemaOrgCatalog,
            SchemaOrgFetchError,
        )

        cache_path = tmp_path / "missing.json"
        with patch(
            "graphrag_sdk.discovery.catalog.urllib.request.urlopen",
            new=MagicMock(side_effect=OSError("connection refused")),
        ):
            catalog = SchemaOrgCatalog(cache_path=cache_path)
            with pytest.raises(SchemaOrgFetchError) as exc_info:
                catalog.known_types()
            # Error message should point at the URL and the fact that
            # there is no offline fallback so the user knows what's
            # going on.
            msg = str(exc_info.value).lower()
            assert "schema.org" in msg
            assert "no offline fallback" in msg

    def test_ttl_expiry_triggers_refetch(self, tmp_path) -> None:
        """When ``cache_ttl_days=0``, every construction re-fetches."""
        from unittest.mock import patch, MagicMock
        from graphrag_sdk.discovery.catalog import SchemaOrgCatalog

        cache_path = tmp_path / "ttl.json"
        urlopen_spy = MagicMock(side_effect=_mock_urlopen_fixture)
        with patch(
            "graphrag_sdk.discovery.catalog.urllib.request.urlopen",
            new=urlopen_spy,
        ):
            first = SchemaOrgCatalog(cache_path=cache_path, cache_ttl_days=0)
            first.known_types()
            second = SchemaOrgCatalog(cache_path=cache_path, cache_ttl_days=0)
            second.known_types()
        assert urlopen_spy.call_count == 2


class TestDiscoverGrounded:
    """End-to-end grounded discovery with a mock catalog + mock extractor."""

    @pytest.fixture
    def tiny_catalog(self):
        """A toy in-process catalog: 3 types, 2 relations among them."""
        from graphrag_sdk.discovery.catalog import Catalog

        class _ToyCatalog(Catalog):
            def known_types(self) -> set[str]:
                return {"Person", "Organization", "Place"}

            def lookup(self, name: str):
                if name == "Person":
                    return Entity(
                        label="Person",
                        properties=[Attribute(name="role", type="STRING")],
                    )
                if name == "Organization":
                    return Entity(
                        label="Organization",
                        properties=[Attribute(name="founded", type="DATE")],
                    )
                if name == "Place":
                    return Entity(label="Place", properties=[])
                return None

            def relations_among(self, names):
                wanted = set(names)
                result = []
                if {"Person", "Organization"} <= wanted:
                    result.append(
                        Relation(
                            label="WORKS_AT", patterns=[("Person", "Organization")]
                        )
                    )
                if {"Person", "Place"} <= wanted:
                    result.append(
                        Relation(label="BORN_IN", patterns=[("Person", "Place")])
                    )
                return result

        return _ToyCatalog()

    class _FakeExtractor:
        """Returns scripted ExtractedEntities by type for each chunk."""

        def __init__(self, types_per_chunk: list[list[str]]) -> None:
            self._scripts = list(types_per_chunk)
            self._calls = 0

        async def extract_entities(self, text, entity_types, source_chunk_id):
            from graphrag_sdk.core.models import ExtractedEntity

            types = self._scripts[min(self._calls, len(self._scripts) - 1)]
            self._calls += 1
            return [
                ExtractedEntity(name=f"E{i}", type=t)
                for i, t in enumerate(types)
            ]

    @pytest.mark.asyncio
    async def test_returns_only_types_detected_in_corpus(
        self, tmp_path: Path, tiny_catalog
    ) -> None:
        from graphrag_sdk.discovery.pipeline import discover_grounded

        src = tmp_path / "doc.txt"
        src.write_text("Alice works at Acme")
        # Detect only Person + Organization — Place is in the catalog but
        # not in the corpus, so it must not appear in the result.
        extractor = self._FakeExtractor([["Person", "Organization"]])
        ontology = await discover_grounded(
            str(src),
            catalog=tiny_catalog,
            entity_extractor=extractor,
            sample_chunks_per_doc=1,
            concurrency=1,
            chunker=FixedChunker(["Alice works at Acme"]),
            seed=42,
        )
        labels = {e.label for e in ontology.entities}
        assert labels == {"Person", "Organization"}
        rel_labels = {r.label for r in ontology.relations}
        assert "WORKS_AT" in rel_labels
        assert "BORN_IN" not in rel_labels

    @pytest.mark.asyncio
    async def test_bridge_relations_to_existing_labels_are_surfaced(
        self, tmp_path: Path, tiny_catalog
    ) -> None:
        """If ``existing`` carries Person and the corpus surfaces only
        Organization, the catalog must still be queried with the union
        so the Person↔Organization bridge relation (``WORKS_AT``)
        appears in the result. Regression test for the gap where
        relations_among was called with detected_types only."""
        from graphrag_sdk.discovery.pipeline import discover_grounded

        src = tmp_path / "doc.txt"
        src.write_text("Acme Corp")
        existing = Ontology(entities=[Entity(label="Person")])
        # NER only sees Organization in the corpus.
        extractor = self._FakeExtractor([["Organization"]])
        ontology = await discover_grounded(
            str(src),
            catalog=tiny_catalog,
            entity_extractor=extractor,
            existing=existing,
            sample_chunks_per_doc=1,
            concurrency=1,
            chunker=FixedChunker(["Acme Corp"]),
            seed=42,
        )
        labels = {e.label for e in ontology.entities}
        assert {"Person", "Organization"} <= labels
        rel_labels = {r.label for r in ontology.relations}
        assert "WORKS_AT" in rel_labels, (
            "Bridge relation Person->Organization should appear because "
            "Person is in `existing` and Organization is newly detected — "
            "even though Person was not part of detected_types."
        )

    @pytest.mark.asyncio
    async def test_existing_prior_is_merged(
        self, tmp_path: Path, tiny_catalog
    ) -> None:
        from graphrag_sdk.discovery.pipeline import discover_grounded

        src = tmp_path / "doc.txt"
        src.write_text("Alice")
        existing = Ontology(entities=[Entity(label="Document")])
        extractor = self._FakeExtractor([["Person"]])
        result = await discover_grounded(
            str(src),
            catalog=tiny_catalog,
            entity_extractor=extractor,
            existing=existing,
            sample_chunks_per_doc=1,
            concurrency=1,
            chunker=FixedChunker(["Alice"]),
            seed=42,
        )
        labels = {e.label for e in result.entities}
        assert "Document" in labels
        assert "Person" in labels

    @pytest.mark.asyncio
    async def test_from_sources_dispatches_to_grounded(
        self, tmp_path: Path, tiny_catalog
    ) -> None:
        """Public API: from_sources(method='grounded') routes to grounded path."""
        src = tmp_path / "doc.txt"
        src.write_text("Alice works at Acme")
        extractor = self._FakeExtractor([["Person", "Organization"]])
        ontology = await Ontology.from_sources(
            str(src),
            method="grounded",
            catalog=tiny_catalog,
            entity_extractor=extractor,
            sample_chunks_per_doc=1,
            concurrency=1,
            chunker=FixedChunker(["Alice works at Acme"]),
            seed=42,
        )
        assert {"Person", "Organization"} <= {e.label for e in ontology.entities}

    @pytest.mark.asyncio
    async def test_from_sources_grounded_requires_catalog(
        self, tmp_path: Path
    ) -> None:
        src = tmp_path / "doc.txt"
        src.write_text("text")
        with pytest.raises(ValueError, match="catalog"):
            await Ontology.from_sources(
                str(src),
                method="grounded",
                sample_chunks_per_doc=1,
            )

    @pytest.mark.asyncio
    async def test_from_sources_llm_requires_llm(self, tmp_path: Path) -> None:
        src = tmp_path / "doc.txt"
        src.write_text("text")
        with pytest.raises(ValueError, match="llm"):
            await Ontology.from_sources(
                str(src),
                method="llm",
                sample_chunks_per_doc=1,
            )

    @pytest.mark.asyncio
    async def test_from_sources_unknown_method_raises(
        self, tmp_path: Path
    ) -> None:
        src = tmp_path / "doc.txt"
        src.write_text("text")
        with pytest.raises(ValueError, match="unknown method"):
            await Ontology.from_sources(
                str(src),
                method="bogus",
                sample_chunks_per_doc=1,
            )


class TestGroundedTrim:
    """Per-type LLM trim pass on grounded discovery — passing llm to
    ``method="grounded"`` filters each catalog type's property list down
    to what the corpus actually mentions."""

    @pytest.fixture
    def trim_catalog(self):
        """Catalog whose Person carries many properties — the trim's job
        is to shrink this list."""
        from graphrag_sdk.discovery.catalog import Catalog

        class _RichCatalog(Catalog):
            def known_types(self) -> set[str]:
                return {"Person"}

            def lookup(self, name: str):
                if name == "Person":
                    return Entity(
                        label="Person",
                        properties=[
                            Attribute(name="name", type="STRING"),
                            Attribute(name="birth_date", type="DATE"),
                            Attribute(name="job_title", type="STRING"),
                            Attribute(name="email", type="STRING"),
                            Attribute(name="duns", type="STRING"),
                            Attribute(name="call_sign", type="STRING"),
                        ],
                    )
                return None

            def relations_among(self, names):
                return []

        return _RichCatalog()

    class _FakeExtractor:
        async def extract_entities(self, text, entity_types, source_chunk_id):
            from graphrag_sdk.core.models import ExtractedEntity

            return [ExtractedEntity(name="Alice", type="Person")]

    @pytest.mark.asyncio
    async def test_trim_keeps_only_llm_selected_properties(
        self, tmp_path: Path, trim_catalog
    ) -> None:
        from graphrag_sdk.discovery.pipeline import discover_grounded

        src = tmp_path / "doc.txt"
        src.write_text("Alice is a software engineer born in 1990")
        # LLM keeps name + birth_date + job_title, drops the rest.
        llm = RecordingMockLLM(
            [json.dumps({"keep": ["name", "birth_date", "job_title"]})]
        )
        ontology = await discover_grounded(
            str(src),
            catalog=trim_catalog,
            llm=llm,
            entity_extractor=self._FakeExtractor(),
            sample_chunks_per_doc=1,
            concurrency=1,
            chunker=FixedChunker(["Alice is a software engineer born in 1990"]),
            seed=42,
        )
        person = next(e for e in ontology.entities if e.label == "Person")
        kept = {a.name for a in person.properties}
        assert kept == {"name", "birth_date", "job_title"}
        assert "duns" not in kept
        assert "call_sign" not in kept

    @pytest.mark.asyncio
    async def test_trim_always_keeps_name(
        self, tmp_path: Path, trim_catalog
    ) -> None:
        """Even if the LLM forgets ``name``, the trim adds it back —
        the SDK-managed identifier filter relies on it being declared."""
        from graphrag_sdk.discovery.pipeline import discover_grounded

        src = tmp_path / "doc.txt"
        src.write_text("Alice was born in 1990")
        llm = RecordingMockLLM([json.dumps({"keep": ["birth_date"]})])
        ontology = await discover_grounded(
            str(src),
            catalog=trim_catalog,
            llm=llm,
            entity_extractor=self._FakeExtractor(),
            sample_chunks_per_doc=1,
            concurrency=1,
            chunker=FixedChunker(["Alice was born in 1990"]),
            seed=42,
        )
        person = next(e for e in ontology.entities if e.label == "Person")
        kept = {a.name for a in person.properties}
        assert "name" in kept
        assert "birth_date" in kept

    @pytest.mark.asyncio
    async def test_trim_soft_fails_to_full_list(
        self, tmp_path: Path, trim_catalog
    ) -> None:
        """If the LLM call exhausts retries, the pipeline falls back to
        the catalog's full property list — flaky LLM does not silently
        lose schema information."""
        from graphrag_sdk.discovery.pipeline import discover_grounded

        src = tmp_path / "doc.txt"
        src.write_text("Alice")
        # All responses bad — the wrapper exhausts retries and raises
        # OntologyDiscoveryError, which discover_grounded catches.
        llm = RecordingMockLLM(["garbage", "still garbage"])
        ontology = await discover_grounded(
            str(src),
            catalog=trim_catalog,
            llm=llm,
            entity_extractor=self._FakeExtractor(),
            sample_chunks_per_doc=1,
            max_retries=1,
            concurrency=1,
            chunker=FixedChunker(["Alice"]),
            seed=42,
        )
        person = next(e for e in ontology.entities if e.label == "Person")
        # Full catalog list preserved.
        assert {a.name for a in person.properties} == {
            "name",
            "birth_date",
            "job_title",
            "email",
            "duns",
            "call_sign",
        }

    @pytest.mark.asyncio
    async def test_no_trim_when_llm_not_provided(
        self, tmp_path: Path, trim_catalog
    ) -> None:
        """Without llm=, the catalog's full property list is returned —
        unchanged from the prior behavior."""
        from graphrag_sdk.discovery.pipeline import discover_grounded

        src = tmp_path / "doc.txt"
        src.write_text("Alice")
        ontology = await discover_grounded(
            str(src),
            catalog=trim_catalog,
            entity_extractor=self._FakeExtractor(),
            sample_chunks_per_doc=1,
            concurrency=1,
            chunker=FixedChunker(["Alice"]),
            seed=42,
        )
        person = next(e for e in ontology.entities if e.label == "Person")
        assert len(person.properties) == 6

    @pytest.mark.asyncio
    async def test_from_sources_grounded_with_llm_runs_trim(
        self, tmp_path: Path, trim_catalog
    ) -> None:
        """Public API: from_sources(method='grounded', llm=...) triggers trim."""
        src = tmp_path / "doc.txt"
        src.write_text("Alice")
        llm = RecordingMockLLM([json.dumps({"keep": ["name", "email"]})])
        ontology = await Ontology.from_sources(
            str(src),
            llm=llm,
            method="grounded",
            catalog=trim_catalog,
            entity_extractor=self._FakeExtractor(),
            sample_chunks_per_doc=1,
            concurrency=1,
            chunker=FixedChunker(["Alice"]),
            seed=42,
        )
        person = next(e for e in ontology.entities if e.label == "Person")
        assert {a.name for a in person.properties} == {"name", "email"}
