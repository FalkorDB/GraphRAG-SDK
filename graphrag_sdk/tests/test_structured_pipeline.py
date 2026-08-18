"""Writing a structured source into the graph.

The whole point of this path is that it is deterministic: the same table always
produces the same graph, because identity comes from a declared key and every
type is declared rather than inferred. These tests pin that down, and pin the
shape that lets a row and a sentence end up on the same node.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from graphrag_sdk.core.context import Context
from graphrag_sdk.ingestion.extraction_strategies.entity_extractors import compute_entity_id
from graphrag_sdk.ingestion.loaders.record_loader import CsvRecordLoader
from graphrag_sdk.ingestion.mapping import (
    Column,
    EdgeMapping,
    MappingError,
    NodeMapping,
    RecordMapping,
    Table,
)
from graphrag_sdk.ingestion.structured_pipeline import (
    StructuredIngestionPipeline,
    record_cells,
    record_chunk_uid,
    render_record,
)


class RecordingGraphStore:
    """Captures what the pipeline would write, without a database."""

    def __init__(self) -> None:
        self.nodes: list[Any] = []
        self.references: list[tuple[str, str, str]] = []
        self.relationships: list[Any] = []
        self.queries: list[tuple[str, dict]] = []

    async def upsert_nodes(self, nodes):
        self.nodes.extend(nodes)
        return len(nodes)

    async def upsert_reference_nodes(self, references):
        self.references.extend(references)
        return len(references)

    async def upsert_relationships(self, relationships):
        self.relationships.extend(relationships)
        return len(relationships)

    async def query_raw(self, cypher, params=None):
        self.queries.append((cypher, params or {}))
        return AsyncMock(result_set=[])

    def __getattr__(self, name):
        # The lexical writer touches more of the store than this test needs.
        return AsyncMock(return_value=0)

    def entity_nodes(self):
        """Only the mapped entities. The lexical writer puts the Document and
        one Chunk per record through the same upsert."""
        return [n for n in self.nodes if n.label not in ("Document", "Chunk")]

    def node(self, node_id):
        for node in self.nodes:
            if node.id == node_id:
                return node
        raise AssertionError(f"{node_id} was never written; wrote {[n.id for n in self.nodes]}")

    def rel_types(self):
        return sorted(r.properties.get("rel_type", r.type) for r in self.relationships)


EMPLOYEES = RecordMapping(
    nodes=[
        NodeMapping(
            label="Person",
            key="employee_id",
            name="full_name",
            properties={"age": Column("age", "INTEGER"), "title": Column("job_title")},
        ),
        NodeMapping(label="Organization", key="org_id", reference=True),
    ],
    edges=[EdgeMapping(type="WORKS_AT", source="Person", target="Organization")],
)


@pytest.fixture
def employees_csv(tmp_path):
    path = tmp_path / "employees.csv"
    path.write_text(
        "employee_id,full_name,age,job_title,org_id\n"
        "E-1,Alice Smith,34,Engineer,ORG-42\n"
        "E-2,Bob Jones,45,CFO,ORG-42\n",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def pipeline():
    store = RecordingGraphStore()
    return StructuredIngestionPipeline(loader=CsvRecordLoader(), graph_store=store), store


class TestHelpers:
    def test_chunk_uid_is_deterministic(self):
        assert record_chunk_uid("employees.csv", "E-1") == record_chunk_uid("employees.csv", "E-1")

    def test_chunk_uid_is_scoped_to_the_document(self):
        """Two sources may legitimately share a key. Scoping keeps their records
        apart, and keying on the effective uid is what stops a pending update
        from merging onto the live document's chunks."""
        assert record_chunk_uid("a.csv", "E-1") != record_chunk_uid("b.csv", "E-1")

    def test_render_record_reads_as_language(self):
        """text is what chunk embedding and full-text search read, and embedders
        are trained on language, not key-value soup."""
        text = render_record({"full_name": "Alice Smith", "job_title": "Engineer"})
        assert text == "full name Alice Smith, job title Engineer."

    def test_render_record_omits_empty_cells(self):
        assert render_record({"a": "1", "b": "", "c": None}) == "a 1."

    def test_record_cells_keeps_the_row_verbatim(self):
        assert record_cells({"age": "34", "b": ""}) == {"age": "34"}

    def test_record_cells_renames_a_column_that_would_shadow_the_chunk(self):
        """A column called "text" would overwrite the chunk's own text."""
        cells = record_cells({"text": "hello", "age": "34"})
        assert cells == {"col_text": "hello", "age": "34"}


class TestStructuredIngest:
    async def test_each_record_becomes_one_chunk(self, pipeline, employees_csv, ctx: Context):
        pipe, _ = pipeline
        result = await pipe.run(str(employees_csv), EMPLOYEES, ctx)
        assert (result.records, result.chunks) == (2, 2)

    async def test_declared_columns_become_typed_properties(
        self, pipeline, employees_csv, ctx: Context
    ):
        """The reason to declare a mapping at all: age is a number that can be
        averaged, not the string "34"."""
        pipe, store = pipeline
        await pipe.run(str(employees_csv), EMPLOYEES, ctx)
        alice = store.node(compute_entity_id("E-1", "Person"))
        assert alice.properties["age"] == 34
        assert isinstance(alice.properties["age"], int)
        assert alice.properties["title"] == "Engineer"
        assert alice.properties["name"] == "Alice Smith"
        assert alice.properties["employee_id"] == "E-1"

    async def test_identity_comes_from_the_declared_key(
        self, pipeline, employees_csv, ctx: Context
    ):
        pipe, store = pipeline
        await pipe.run(str(employees_csv), EMPLOYEES, ctx)
        assert {n.id for n in store.entity_nodes()} == {
            compute_entity_id("E-1", "Person"),
            compute_entity_id("E-2", "Person"),
        }

    async def test_a_keyed_and_named_node_publishes_the_extracted_id_as_an_alias(
        self, pipeline, employees_csv, ctx: Context
    ):
        """This is the bridge between the two halves of the graph. The node is
        keyed E-1, but prose about "Alice Smith" resolves to a different id, so
        the source that holds both publishes the one an extractor would compute.
        """
        pipe, store = pipeline
        await pipe.run(str(employees_csv), EMPLOYEES, ctx)
        alice = store.node(compute_entity_id("E-1", "Person"))
        assert alice.properties["alias_ids"] == [compute_entity_id("Alice Smith", "Person")]

    async def test_a_foreign_key_is_written_as_a_reference_not_an_entity(
        self, pipeline, employees_csv, ctx: Context
    ):
        """The org column says the organization exists, not what it looks like.
        Writing it as a full entity would let a key overwrite a real name."""
        pipe, store = pipeline
        result = await pipe.run(str(employees_csv), EMPLOYEES, ctx)
        assert result.references == 2
        assert {r.id for r in store.references} == {compute_entity_id("ORG-42", "Organization")}
        # Keyed, so a later source and a generated query can both join on it.
        assert store.references[0].properties["org_id"] == "ORG-42"
        assert all(n.label != "Organization" for n in store.entity_nodes())

    async def test_a_reference_with_a_name_column_labels_the_stub(self, tmp_path, ctx: Context):
        """A denormalised name makes the stub "Acme Corp" instead of "ORG-42"."""
        path = tmp_path / "e.csv"
        path.write_text("employee_id,full_name,org_id,org_name\nE-1,Alice,ORG-42,Acme Corp\n")
        mapping = RecordMapping(
            nodes=[
                NodeMapping(label="Person", key="employee_id", name="full_name"),
                NodeMapping(label="Organization", key="org_id", name="org_name", reference=True),
            ],
            edges=[EdgeMapping(type="WORKS_AT", source="Person", target="Organization")],
        )
        store = RecordingGraphStore()
        pipe = StructuredIngestionPipeline(loader=CsvRecordLoader(), graph_store=store)
        await pipe.run(str(path), mapping, ctx)
        assert store.references[0].name == "Acme Corp"

    async def test_declared_edges_carry_their_semantic_type(
        self, pipeline, employees_csv, ctx: Context
    ):
        """Every data edge is RELATES; the meaning lives in rel_type."""
        pipe, store = pipeline
        await pipe.run(str(employees_csv), EMPLOYEES, ctx)
        works_at = [r for r in store.relationships if r.properties.get("rel_type") == "WORKS_AT"]
        assert len(works_at) == 2
        assert all(r.type == "RELATES" for r in works_at)

    async def test_every_entity_is_linked_to_its_record(
        self, pipeline, employees_csv, ctx: Context
    ):
        """MENTIONED_IN is what makes a row traceable back to its source."""
        pipe, store = pipeline
        await pipe.run(str(employees_csv), EMPLOYEES, ctx)
        mentions = [r for r in store.relationships if r.type == "MENTIONED_IN"]
        # Two records, each with a person and an organization.
        assert len(mentions) == 4

    async def test_records_are_not_chained_as_a_sequence(
        self, pipeline, employees_csv, ctx: Context
    ):
        """Rows have no reading order. NEXT_CHUNK means "the next sequential
        chunk" to text-to-Cypher, so chaining rows asserts a sequence that does
        not exist."""
        pipe, store = pipeline
        await pipe.run(str(employees_csv), EMPLOYEES, ctx)
        assert not [r for r in store.relationships if r.type == "NEXT_CHUNK"]

    async def test_the_same_source_twice_produces_the_same_ids(self, employees_csv, ctx: Context):
        """Deterministic, so re-ingesting updates in place instead of doubling."""
        ids = []
        for _ in range(2):
            store = RecordingGraphStore()
            pipe = StructuredIngestionPipeline(loader=CsvRecordLoader(), graph_store=store)
            await pipe.run(str(employees_csv), EMPLOYEES, ctx)
            ids.append(sorted(n.id for n in store.entity_nodes()))
        assert ids[0] == ids[1]

    async def test_a_row_missing_its_key_is_skipped_not_guessed(self, tmp_path, ctx: Context):
        """Without a key a record has no stable identity, so it could never be
        updated or deleted. Inventing one would be worse than skipping."""
        path = tmp_path / "gap.csv"
        path.write_text(
            "employee_id,full_name,age,job_title,org_id\n,Nameless,20,X,ORG-1\n"
            "E-9,Real Person,30,Y,ORG-1\n"
        )
        store = RecordingGraphStore()
        pipe = StructuredIngestionPipeline(loader=CsvRecordLoader(), graph_store=store)
        result = await pipe.run(str(path), EMPLOYEES, ctx)
        assert result.records == 1
        assert [n.properties["name"] for n in store.entity_nodes()] == ["Real Person"]

    async def test_a_bad_cell_fails_the_ingest_rather_than_writing_a_wrong_type(
        self, tmp_path, ctx: Context
    ):
        path = tmp_path / "bad.csv"
        path.write_text(
            "employee_id,full_name,age,job_title,org_id\nE-1,Alice,thirty-four,Engineer,ORG-42\n"
        )
        store = RecordingGraphStore()
        pipe = StructuredIngestionPipeline(loader=CsvRecordLoader(), graph_store=store)
        with pytest.raises(MappingError, match="'age'"):
            await pipe.run(str(path), EMPLOYEES, ctx)

    async def test_a_mapping_that_does_not_fit_writes_nothing(self, tmp_path, ctx: Context):
        path = tmp_path / "other.csv"
        path.write_text("a,b\n1,2\n")
        store = RecordingGraphStore()
        pipe = StructuredIngestionPipeline(loader=CsvRecordLoader(), graph_store=store)
        with pytest.raises(MappingError, match="does not fit"):
            await pipe.run(str(path), EMPLOYEES, ctx)
        assert store.entity_nodes() == []
        assert store.relationships == []

    async def test_strict_mode_reports_an_unmapped_column(self, tmp_path, ctx: Context):
        path = tmp_path / "extra.csv"
        path.write_text("org_id,org_name,salary\nORG-1,Acme,100\n")
        store = RecordingGraphStore()
        pipe = StructuredIngestionPipeline(loader=CsvRecordLoader(), graph_store=store)
        mapping = Table("Organization", key="org_id", name="org_name")
        with pytest.raises(MappingError, match="salary"):
            await pipe.run(str(path), mapping, ctx, strict=True)
        await pipe.run(str(path), mapping, ctx)  # permissive by default

    async def test_an_empty_source_writes_nothing(self, tmp_path, ctx: Context):
        path = tmp_path / "headers_only.csv"
        path.write_text("org_id,org_name\n")
        store = RecordingGraphStore()
        pipe = StructuredIngestionPipeline(loader=CsvRecordLoader(), graph_store=store)
        mapping = Table("Organization", key="org_id", name="org_name")
        result = await pipe.run(str(path), mapping, ctx)
        assert result.records == 0
        assert store.entity_nodes() == []

    async def test_document_id_override_reaches_the_chunk_ids(
        self, pipeline, employees_csv, ctx: Context
    ):
        pipe, _ = pipeline
        result = await pipe.run(str(employees_csv), EMPLOYEES, ctx, document_id="hr-export")
        assert result.document_id == "hr-export"

    async def test_two_aliases_resolving_to_one_entity_write_no_self_loop(
        self, tmp_path, ctx: Context
    ):
        path = tmp_path / "same.csv"
        path.write_text("a_id,b_id\nX-1,X-1\n")
        mapping = RecordMapping(
            nodes=[
                NodeMapping(alias="a", label="Thing", key="a_id"),
                NodeMapping(alias="b", label="Thing", key="b_id"),
            ],
            edges=[EdgeMapping(type="LINKS", source="a", target="b")],
        )
        store = RecordingGraphStore()
        pipe = StructuredIngestionPipeline(loader=CsvRecordLoader(), graph_store=store)
        result = await pipe.run(str(path), mapping, ctx)
        assert result.edges == 0
        assert "LINKS" not in store.rel_types()
