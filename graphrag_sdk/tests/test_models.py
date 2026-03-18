"""Tests for core/models.py — all Pydantic v2 data models."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from graphrag_sdk.core.models import (
    DataModel,
    DocumentInfo,
    DocumentOutput,
    EntityType,
    GraphData,
    GraphNode,
    GraphRelationship,
    GraphSchema,
    IngestionResult,
    LLMMessage,
    LLMResponse,
    PropertyType,
    RagResult,
    RawSearchResult,
    RelationType,
    ResolutionResult,
    RetrieverResult,
    RetrieverResultItem,
    SchemaPattern,
    SearchType,
    TextChunk,
    TextChunks,
)


class TestDataModel:
    def test_base_model_allows_extra(self):
        model = DataModel(extra_field="hello")
        assert model.extra_field == "hello"  # type: ignore[attr-defined]

    def test_base_model_is_mutable(self):
        model = DataModel()
        model.new_attr = "value"  # type: ignore[attr-defined]


class TestGraphNode:
    def test_creation(self):
        node = GraphNode(id="n1", label="Person", properties={"name": "Alice"})
        assert node.id == "n1"
        assert node.label == "Person"
        assert node.properties["name"] == "Alice"

    def test_defaults(self):
        node = GraphNode(id="n1", label="Test")
        assert node.properties == {}
        assert node.embedding_properties is None

    def test_hash(self):
        n1 = GraphNode(id="n1", label="Person")
        n2 = GraphNode(id="n1", label="Company")
        assert hash(n1) == hash(n2)  # hash based on id

    def test_different_ids_hash_differently(self):
        n1 = GraphNode(id="n1", label="Person")
        n2 = GraphNode(id="n2", label="Person")
        assert hash(n1) != hash(n2)

    def test_embedding_properties(self):
        node = GraphNode(
            id="n1",
            label="Test",
            embedding_properties={"vec": [0.1, 0.2, 0.3]},
        )
        assert node.embedding_properties["vec"] == [0.1, 0.2, 0.3]


class TestGraphRelationship:
    def test_creation(self):
        rel = GraphRelationship(
            start_node_id="a", end_node_id="b", type="KNOWS", properties={"since": 2020}
        )
        assert rel.start_node_id == "a"
        assert rel.end_node_id == "b"
        assert rel.type == "KNOWS"
        assert rel.properties["since"] == 2020

    def test_defaults(self):
        rel = GraphRelationship(start_node_id="a", end_node_id="b", type="REL")
        assert rel.properties == {}
        assert rel.embedding_properties is None


class TestTextChunk:
    def test_creation(self):
        chunk = TextChunk(text="Hello world", index=0)
        assert chunk.text == "Hello world"
        assert chunk.index == 0
        assert chunk.uid  # auto-generated UUID

    def test_custom_uid(self):
        chunk = TextChunk(text="Hello", index=0, uid="custom-id")
        assert chunk.uid == "custom-id"

    def test_metadata(self):
        chunk = TextChunk(text="Hi", index=0, metadata={"start_char": 0, "end_char": 2})
        assert chunk.metadata["start_char"] == 0


class TestTextChunks:
    def test_empty(self):
        chunks = TextChunks()
        assert chunks.chunks == []

    def test_with_chunks(self):
        chunks = TextChunks(
            chunks=[TextChunk(text="a", index=0), TextChunk(text="b", index=1)]
        )
        assert len(chunks.chunks) == 2


class TestDocumentModels:
    def test_document_info_defaults(self):
        info = DocumentInfo()
        assert info.path is None
        assert info.uid  # auto-generated
        assert info.metadata == {}

    def test_document_info_with_path(self):
        info = DocumentInfo(path="/test/file.txt", metadata={"loader": "text"})
        assert info.path == "/test/file.txt"
        assert info.metadata["loader"] == "text"

    def test_document_output(self):
        doc = DocumentOutput(text="Hello world")
        assert doc.text == "Hello world"
        assert doc.document_info.uid  # auto-created


class TestSchemaTypes:
    def test_property_type(self):
        pt = PropertyType(name="age", type="INT", required=True)
        assert pt.name == "age"
        assert pt.type == "INT"
        assert pt.required is True

    def test_entity_type(self):
        et = EntityType(label="Person", description="A human")
        assert et.label == "Person"
        assert et.properties == []

    def test_entity_type_hash(self):
        e1 = EntityType(label="Person")
        e2 = EntityType(label="Person", description="Different desc")
        assert hash(e1) == hash(e2)

    def test_relation_type(self):
        rt = RelationType(label="KNOWS", description="Social link")
        assert rt.label == "KNOWS"
        assert hash(rt) == hash(RelationType(label="KNOWS"))

    def test_schema_pattern(self):
        sp = SchemaPattern(source="Person", relationship="WORKS_AT", target="Company")
        assert sp.source == "Person"

    def test_graph_schema(self, sample_schema):
        assert len(sample_schema.entities) == 2
        assert len(sample_schema.relations) == 2
        assert len(sample_schema.patterns) == 2

    def test_empty_schema(self):
        schema = GraphSchema()
        assert schema.entities == []
        assert schema.relations == []
        assert schema.patterns == []


class TestGraphData:
    def test_creation(self, sample_graph_data):
        assert len(sample_graph_data.nodes) == 3
        assert len(sample_graph_data.relationships) == 2

    def test_empty(self):
        gd = GraphData()
        assert gd.nodes == []
        assert gd.relationships == []


class TestResolutionResult:
    def test_creation(self):
        rr = ResolutionResult(
            nodes=[GraphNode(id="a", label="X")],
            relationships=[],
            merged_count=1,
        )
        assert rr.merged_count == 1
        assert len(rr.nodes) == 1


class TestRetrieverModels:
    def test_result_item(self):
        item = RetrieverResultItem(content="chunk text", score=0.95)
        assert item.content == "chunk text"
        assert item.score == 0.95

    def test_result(self):
        result = RetrieverResult(
            items=[RetrieverResultItem(content="a"), RetrieverResultItem(content="b")]
        )
        assert len(result.items) == 2

    def test_raw_search_result(self):
        raw = RawSearchResult(records=[{"text": "hello"}], metadata={"strategy": "local"})
        assert len(raw.records) == 1


class TestLLMModels:
    def test_llm_message(self):
        msg = LLMMessage(role="user", content="Hello")
        assert msg.role == "user"

    def test_llm_response(self):
        resp = LLMResponse(content="Answer")
        assert resp.content == "Answer"
        assert resp.tool_calls is None


class TestRagResult:
    def test_creation(self):
        result = RagResult(answer="42", metadata={"model": "gpt-4"})
        assert result.answer == "42"
        assert result.retriever_result is None


class TestIngestionResult:
    def test_defaults(self):
        result = IngestionResult()
        assert result.nodes_created == 0
        assert result.chunks_indexed == 0

    def test_with_counts(self):
        result = IngestionResult(nodes_created=10, relationships_created=5, chunks_indexed=3)
        assert result.nodes_created == 10
        assert result.relationships_created == 5


class TestSearchType:
    def test_enum_values(self):
        assert SearchType.VECTOR == "vector"
        assert SearchType.FULLTEXT == "fulltext"
        assert SearchType.HYBRID == "hybrid"

    def test_from_string(self):
        assert SearchType("vector") == SearchType.VECTOR
