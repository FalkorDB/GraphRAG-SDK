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
    Link,
    MappingError,
    TableMapping,
    record_mapping_for,
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

    # Identity reconciliation reads the graph; with nothing stored there is
    # nothing to resolve or rename, which is what a fresh write sees.
    async def resolve_by_entity_key(self, label, keys):
        return {}

    async def reconcile_keyed_identity(self, label, signed_key, rows):
        return {"renamed": 0, "merged": 0}

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


EMPLOYEES_TABLE = TableMapping(
    source="employees.csv",
    label="Person",
    key="employee_id",
    name="full_name",
    properties={"age": Column("age", "INTEGER"), "title": Column("job_title")},
    links=[Link("WORKS_AT", to="Organization", by="org_id")],
)
# The write path wants the declaration flattened into nodes and edges, and that
# translation is where the source's signature is stamped on: every property this
# table writes lands as ``employees__<property>``.
EMPLOYEES = record_mapping_for(EMPLOYEES_TABLE)


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
        alice = store.node(compute_entity_id("Alice Smith", "Person"))
        assert alice.properties["employees__age"] == 34
        assert isinstance(alice.properties["employees__age"], int)
        assert alice.properties["employees__title"] == "Engineer"
        assert alice.properties["name"] == "Alice Smith"
        assert alice.properties["employees__employee_id"] == "E-1"

    async def test_an_empty_cell_takes_the_column_off_the_row(
        self, pipeline, tmp_path, ctx: Context
    ):
        """A blank cell is carried as ``None`` — which the store's write skips — and
        then removed by column, so a value from the previous export cannot stay."""
        path = tmp_path / "employees.csv"
        path.write_text(
            "employee_id,full_name,age,job_title,org_id\n"
            "E-1,Alice Smith,34,,ORG-42\n"
            "E-2,Bob Jones,,CFO,ORG-42\n",
            encoding="utf-8",
        )
        pipe, store = pipeline
        removed = AsyncMock(return_value=1)
        store.drop_node_property = removed
        await pipe.run(str(path), EMPLOYEES, ctx)

        alice = store.node(compute_entity_id("Alice Smith", "Person"))
        assert alice.properties["employees__title"] is None
        assert alice.properties["employees__age"] == 34
        calls = {(c.args[0], c.args[1], tuple(c.kwargs["ids"])) for c in removed.call_args_list}
        assert calls == {
            ("Person", "employees__title", (compute_entity_id("Alice Smith", "Person"),)),
            ("Person", "employees__age", (compute_entity_id("Bob Jones", "Person"),)),
        }

    async def test_identity_comes_from_the_name(self, pipeline, employees_csv, ctx: Context):
        """One identity scheme for both halves. A row's id is derived from its
        name exactly as a prose mention's is, so "Alice Smith" in a PDF and
        ``E-1, Alice Smith`` in a CSV are the same node with no merge step.
        The key is carried beside it as ``entity_key``, for links and re-sync.
        """
        pipe, store = pipeline
        await pipe.run(str(employees_csv), EMPLOYEES, ctx)
        assert {n.id for n in store.entity_nodes()} == {
            compute_entity_id("Alice Smith", "Person"),
            compute_entity_id("Bob Jones", "Person"),
        }
        alice = store.node(compute_entity_id("Alice Smith", "Person"))
        assert alice.properties["entity_key"] == "E-1"

    async def test_a_named_node_needs_no_alias_to_meet_its_prose_mention(
        self, pipeline, employees_csv, ctx: Context
    ):
        """There used to be an ``alias_ids`` bridge: the node was keyed E-1, prose
        about "Alice Smith" resolved to a different id, so the row published the
        id an extractor would compute. With one identity scheme the two ids are
        the same string, and the bridge has nothing to bridge.
        """
        pipe, store = pipeline
        await pipe.run(str(employees_csv), EMPLOYEES, ctx)
        alice = store.node(compute_entity_id("Alice Smith", "Person"))
        assert alice.id == compute_entity_id("Alice Smith", "Person")
        assert "alias_ids" not in alice.properties

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
        assert store.references[0].properties["employees__org_id"] == "ORG-42"
        assert all(n.label != "Organization" for n in store.entity_nodes())

    async def test_a_reference_with_a_name_column_labels_the_stub(self, tmp_path, ctx: Context):
        """A denormalised name makes the stub "Acme Corp" instead of "ORG-42"."""
        path = tmp_path / "e.csv"
        path.write_text("employee_id,full_name,org_id,org_name\nE-1,Alice,ORG-42,Acme Corp\n")
        mapping = record_mapping_for(
            TableMapping(
                source="e.csv",
                label="Person",
                key="employee_id",
                name="full_name",
                links=[Link("WORKS_AT", to="Organization", by="org_id", name="org_name")],
            )
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
        mapping = record_mapping_for(
            TableMapping(source="extra.csv", label="Organization", key="org_id", name="org_name")
        )
        with pytest.raises(MappingError, match="salary"):
            await pipe.run(str(path), mapping, ctx, strict=True)
        await pipe.run(str(path), mapping, ctx)  # permissive by default

    async def test_an_empty_source_writes_nothing(self, tmp_path, ctx: Context):
        path = tmp_path / "headers_only.csv"
        path.write_text("org_id,org_name\n")
        store = RecordingGraphStore()
        pipe = StructuredIngestionPipeline(loader=CsvRecordLoader(), graph_store=store)
        mapping = record_mapping_for(
            TableMapping(
                source="headers_only.csv", label="Organization", key="org_id", name="org_name"
            )
        )
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
        mapping = record_mapping_for(
            TableMapping(
                source="same.csv",
                label="Thing",
                key="a_id",
                links=[Link("LINKS", to="Thing", by="b_id")],
            )
        )
        store = RecordingGraphStore()
        pipe = StructuredIngestionPipeline(loader=CsvRecordLoader(), graph_store=store)
        result = await pipe.run(str(path), mapping, ctx)
        assert result.edges == 0
        assert "LINKS" not in store.rel_types()


class TestRowsSharingAKey:
    """A chunk identifies a row; only the entity identifies by key.

    Measured before this: two rows keyed ``K1`` produced one chunk holding the
    first row's cells while the ingest reported two records. A source whose key
    column turns out not to be unique lost rows with nothing raised or logged.
    """

    MAPPING = record_mapping_for(
        TableMapping(
            source="dup.csv",
            label="Org",
            key="k",
            name="n",
            properties={"v": Column("v", "INTEGER")},
        )
    )

    @pytest.fixture
    def duplicate_csv(self, tmp_path):
        path = tmp_path / "dup.csv"
        path.write_text("k,n,v\nK1,First,1\nK1,Second,2\nK2,Other,3\n", encoding="utf-8")
        return str(path)

    async def test_every_row_keeps_its_own_chunk(self, pipeline, duplicate_csv, ctx: Context):
        pipe, store = pipeline
        result = await pipe.run(duplicate_csv, self.MAPPING, ctx)
        assert result.records == 3
        chunks = [n for n in store.nodes if n.label == "Chunk"]
        assert len(chunks) == 3, "no row may be dropped because another shares its key"
        assert len({c.id for c in chunks}) == 3, "and their ids must differ"

    async def test_rows_sharing_a_key_but_not_a_name_are_separate_entities(
        self, pipeline, duplicate_csv, ctx: Context
    ):
        """``K1, First`` and ``K1, Second`` disagree about what K1 is. Identity is
        the name, so they are two entities both carrying ``entity_key = K1`` —
        and the repeated-key warning (next test) is what says the data is
        contradictory. Collapsing them on the key silently kept one name and
        lost the other; this keeps both and says so.
        """
        pipe, store = pipeline
        await pipe.run(duplicate_csv, self.MAPPING, ctx)
        assert {n.id for n in store.entity_nodes()} == {
            compute_entity_id("First", "Org"),
            compute_entity_id("Second", "Org"),
            compute_entity_id("Other", "Org"),
        }
        assert {n.properties["entity_key"] for n in store.entity_nodes()} == {"K1", "K2"}

    async def test_a_repeated_key_is_reported(self, pipeline, duplicate_csv, ctx, caplog):
        """Silently collapsing rows onto one entity is the kind of thing that is
        only noticed months later, so it is said out loud."""
        pipe, _ = pipeline
        with caplog.at_level("WARNING"):
            await pipe.run(duplicate_csv, self.MAPPING, ctx)
        assert any("not unique" in record.getMessage() for record in caplog.records)

    async def test_a_unique_key_reports_nothing(self, pipeline, employees_csv, ctx, caplog):
        pipe, _ = pipeline
        with caplog.at_level("WARNING"):
            await pipe.run(str(employees_csv), EMPLOYEES, ctx)
        assert not [r for r in caplog.records if "not unique" in r.getMessage()]

    async def test_the_first_row_of_a_key_keeps_its_chunk_id(self):
        """The occurrence suffix must not move ids that already exist in a graph,
        so occurrence zero reproduces the original digest."""
        from graphrag_sdk.ingestion.structured_pipeline import record_chunk_uid

        assert record_chunk_uid("d", "K1") == record_chunk_uid("d", "K1", 0)
        assert record_chunk_uid("d", "K1", 1) != record_chunk_uid("d", "K1", 0)


class TestOneNodePerKeyWithinABatch:
    """A reference's id comes from the name a row supplies, else from the key.

    Two columns of one file referencing the same entity — one bare, one with a
    name — derived two ids for one key, and the lookup against the graph could
    not catch it because nothing had been written yet. Measured on a citations
    table: two ``Paper`` placeholders both carrying ``entity_key = 2603.20674``,
    one named by its title and one named by its id.
    """

    CITATIONS = record_mapping_for(
        TableMapping(
            source="citations.csv",
            label="Citation",
            key="citation_id",
            links=[
                Link("CITING", to="Paper", by="citing"),
                Link("CITES", to="Paper", by="cited", name="cited_title"),
            ],
        )
    )

    @pytest.fixture
    def citations_csv(self, tmp_path):
        path = tmp_path / "citations.csv"
        # Row 2 cites the paper row 1 was citing from, and supplies its title.
        path.write_text(
            "citation_id,citing,cited,cited_title\n"
            "C-1,1801.02681,1712.01815,Mastering chess\n"
            "C-2,2603.20674,1801.02681,Diversification and exports\n",
            encoding="utf-8",
        )
        return str(path)

    async def test_a_key_referenced_bare_and_by_name_is_one_placeholder(
        self, pipeline, citations_csv, ctx: Context
    ):
        pipe, store = pipeline
        await pipe.run(citations_csv, self.CITATIONS, ctx)
        papers = [r for r in store.references if r.label == "Paper"]
        by_key: dict[str, list[str]] = {}
        for ref in papers:
            by_key.setdefault(ref.properties["entity_key"], []).append(ref.id)
        assert all(len(ids) == 1 for ids in by_key.values()), by_key
        # The named reference is the survivor: a placeholder called by its title
        # is findable, one called "1801.02681" is not.
        assert by_key["1801.02681"] == [compute_entity_id("Diversification and exports", "Paper")]

    async def test_edges_follow_the_surviving_placeholder(
        self, pipeline, citations_csv, ctx: Context
    ):
        pipe, store = pipeline
        await pipe.run(citations_csv, self.CITATIONS, ctx)
        bare = compute_entity_id("1801.02681", "Paper")
        touched = {r.start_node_id for r in store.relationships} | {
            r.end_node_id for r in store.relationships
        }
        assert bare not in touched, "no edge may point at a placeholder that was never written"
        citing = [r for r in store.relationships if r.properties.get("rel_type") == "CITING"]
        assert compute_entity_id("Diversification and exports", "Paper") in {
            r.end_node_id for r in citing
        }

    async def test_a_self_reference_lands_on_the_row(self, pipeline, tmp_path, ctx: Context):
        """``manager_id`` points at rows of the same file. The row derives its id
        from its name, the reference from the key; without the collapse a
        placeholder "E-1" was raised beside Alice."""
        path = tmp_path / "staff.csv"
        path.write_text(
            "employee_id,full_name,manager_id\nE-1,Alice Smith,\nE-2,Bob Jones,E-1\n",
            encoding="utf-8",
        )
        mapping = record_mapping_for(
            TableMapping(
                source="staff.csv",
                label="Person",
                key="employee_id",
                name="full_name",
                links=[Link("REPORTS_TO", to="Person", by="manager_id")],
            )
        )
        pipe, store = pipeline
        await pipe.run(str(path), mapping, ctx)
        assert store.references == [], "every referenced key is a row of this file"
        reports_to = [
            r for r in store.relationships if r.properties.get("rel_type") == "REPORTS_TO"
        ]
        assert [(r.start_node_id, r.end_node_id) for r in reports_to] == [
            (compute_entity_id("Bob Jones", "Person"), compute_entity_id("Alice Smith", "Person"))
        ]


class TestKeyWhitespace:
    async def test_a_key_with_surrounding_whitespace_is_stored_stripped(
        self, pipeline, tmp_path, ctx: Context
    ):
        """The id derivation strips, so ``ORG-42 `` already landed on the ORG-42
        node's id; the stored ``entity_key`` did not, so the owning source could
        never find the placeholder by key. Both now agree."""
        path = tmp_path / "employees.csv"
        path.write_text(
            "employee_id,full_name,age,job_title,org_id\nE-1 ,Alice Smith,34,Engineer, ORG-42 \n",
            encoding="utf-8",
        )
        pipe, store = pipeline
        await pipe.run(str(path), EMPLOYEES, ctx)
        alice = store.node(compute_entity_id("Alice Smith", "Person"))
        assert alice.properties["entity_key"] == "E-1"
        assert [r.properties["entity_key"] for r in store.references] == ["ORG-42"]
