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
