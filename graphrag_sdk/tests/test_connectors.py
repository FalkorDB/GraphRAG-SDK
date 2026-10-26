"""Tests for graphrag_sdk.connectors: syncing external sources into the graph.

The unit tests run against a fake graph that records the calls a sync makes.
The integration tests at the end run a real sync against FalkorDB and are
skipped unless RUN_INTEGRATION=1.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from graphrag_sdk.api.main import GraphRAG
from graphrag_sdk.connectors import (
    SourceDocument,
    document_id_for,
    sync,
    sync_documents,
    sync_table,
)
from graphrag_sdk.connectors.sync import graph_safe_metadata
from graphrag_sdk.core.context import Context
from graphrag_sdk.core.models import Entity, Ontology, UpdateResult
from graphrag_sdk.core.tables import TableMapping
from graphrag_sdk.ingestion.loaders.record_loader import CsvRecordLoader


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class FakeGraph:
    """Stands in for GraphRAG: documents are ``{id: (hash, metadata)}``."""

    def __init__(self, documents: dict[str, str] | None = None, tables: list | None = None):
        self.documents: dict[str, str] = dict(documents or {})
        self.metadata: dict[str, dict[str, Any]] = {}
        self.updates: list[str] = []
        self.deletes: list[str] = []
        self.ingests: list[tuple[str, list[dict[str, Any]]]] = []
        self.fail_on: set[str] = set()
        self.incomplete_on: set[str] = set()
        self.metadata_refreshes: list[tuple[str, dict[str, Any]]] = []
        self.ontology = Ontology(entities=[Entity(label="Person")], tables=tables or [])

    async def query(self, cypher: str, params: dict[str, Any]) -> list[list[Any]]:
        if "SET d += $props" in cypher:
            self.metadata_refreshes.append((params["id"], params["props"]))
            return []
        assert "STARTS WITH $prefix" in cypher
        return [[i, h] for i, h in self.documents.items() if i.startswith(params["prefix"])]

    async def update(self, source: str, *, document_id: str, loader: Any, **kwargs: Any):
        assert kwargs["if_missing"] == "ingest"
        self.update_kwargs = kwargs
        if document_id in self.fail_on:
            raise RuntimeError("provider down")
        self.updates.append(document_id)
        loaded = await loader.load(source, Context())
        self.metadata[document_id] = loaded.document_info.metadata
        new_hash = _hash(loaded.text)
        old_hash = self.documents.get(document_id)
        if document_id in self.incomplete_on:
            # As the pipeline does: written, but no hash recorded.
            self.documents[document_id] = None
            return UpdateResult(
                metadata={"incomplete_writes": ["extraction failed for 1 chunk(s)"]}
            )
        self.documents[document_id] = new_hash
        return UpdateResult(
            no_op=old_hash == new_hash,
            replaced_existing=old_hash is not None and old_hash != new_hash,
        )

    async def delete_document(self, document_id: str, *, if_missing: str = "error"):
        self.deletes.append(document_id)
        self.documents.pop(document_id, None)

    async def get_ontology(self) -> Ontology:
        return self.ontology

    async def ingest(self, source: str, *, record_loader: Any, ctx: Any = None):
        from graphrag_sdk.ingestion.structured_pipeline import StructuredIngestionResult

        self.last_loader = record_loader
        batch = await record_loader.load_records(source, Context())
        rows = list(batch)
        assert list(batch) == rows, "records must be re-openable"
        self.ingests.append((source, rows))
        result = StructuredIngestionResult(source)
        result.records = len(rows)
        return result


# ── sync_documents ──────────────────────────────────────────────


async def test_first_sync_adds_every_document_under_the_source_prefix():
    graph = FakeGraph()
    docs = [SourceDocument("p1", "Alice works at Acme."), SourceDocument("p2", "Bob.")]

    report = await sync_documents(graph, docs, source="notion")

    assert report.added == ["notion:p1", "notion:p2"]
    assert graph.updates == ["notion:p1", "notion:p2"]
    assert not report.updated and not report.unchanged and not report.deleted


async def test_resync_sorts_unchanged_changed_and_removed_documents():
    graph = FakeGraph(
        {
            "notion:same": _hash("same text"),
            "notion:edited": _hash("old text"),
            "notion:gone": _hash("removed text"),
            "drive:other": _hash("another source"),
        }
    )
    docs = [
        SourceDocument("same", "same text"),
        SourceDocument("edited", "new text"),
        SourceDocument("new", "brand new"),
    ]

    report = await sync_documents(graph, docs, source="notion")

    assert report.unchanged == ["notion:same"]
    assert report.updated == ["notion:edited"]
    assert report.added == ["notion:new"]
    assert report.deleted == ["notion:gone"]
    # Another source's document is outside the prefix, so it is never pruned.
    assert "drive:other" in graph.documents


async def test_prune_false_keeps_documents_the_source_left_out():
    graph = FakeGraph({"notion:gone": _hash("x")})

    report = await sync_documents(
        graph, [SourceDocument("p1", "text")], source="notion", prune=False
    )

    assert report.deleted == [] and graph.deletes == []


async def test_an_empty_source_does_not_wipe_its_documents():
    graph = FakeGraph({"notion:a": _hash("a"), "notion:b": _hash("b")})

    report = await sync_documents(graph, [], source="notion")

    assert graph.deletes == []
    assert report.prune_skipped and "returned no documents" in report.prune_skipped


async def test_dry_run_predicts_without_writing():
    graph = FakeGraph({"notion:same": _hash("same"), "notion:gone": _hash("x")})
    docs = [SourceDocument("same", "same"), SourceDocument("new", "new")]

    report = await sync_documents(graph, docs, source="notion", dry_run=True)

    assert report.dry_run
    assert (report.added, report.unchanged, report.deleted) == (
        ["notion:new"],
        ["notion:same"],
        ["notion:gone"],
    )
    assert graph.updates == [] and graph.deletes == []


async def test_a_failed_document_is_reported_and_not_pruned():
    graph = FakeGraph({"notion:p1": _hash("old")})
    graph.fail_on = {"notion:p1"}

    report = await sync_documents(
        graph, [SourceDocument("p1", "new"), SourceDocument("p2", "fine")], source="notion"
    )

    assert "provider down" in report.failed["notion:p1"]
    assert report.added == ["notion:p2"]
    assert "notion:p1" in graph.documents  # seen, so not pruned


async def test_a_repeated_id_is_written_once():
    graph = FakeGraph()

    report = await sync_documents(
        graph, [SourceDocument("p1", "a"), SourceDocument("p1", "b")], source="notion"
    )

    assert graph.updates == ["notion:p1"]
    assert "twice" in report.failed["notion:p1"]


async def test_metadata_reaches_the_document_without_reserved_or_unstorable_values():
    graph = FakeGraph()
    doc = SourceDocument(
        "p1",
        "text",
        metadata={
            "url": "https://notion.so/p1",
            "modified": datetime(2026, 10, 1, tzinfo=timezone.utc),
            "content_hash": "spoofed",
            "client": object(),
        },
    )

    await sync_documents(graph, [doc], source="notion")

    assert graph.metadata["notion:p1"] == {
        "url": "https://notion.so/p1",
        "modified": "2026-10-01T00:00:00+00:00",
    }


async def test_chunker_and_extractor_reach_update():
    graph = FakeGraph()
    chunker, extractor = object(), object()

    await sync_documents(
        graph, [SourceDocument("p1", "a")], source="notion", chunker=chunker, extractor=extractor
    )

    assert graph.update_kwargs["chunker"] is chunker
    assert graph.update_kwargs["extractor"] is extractor


async def test_a_plain_generator_is_read():
    graph = FakeGraph()

    def pages():
        yield SourceDocument("p1", "a")
        yield SourceDocument("p2", "b")

    report = await sync_documents(graph, pages(), source="notion")

    assert report.added == ["notion:p1", "notion:p2"]


async def test_items_that_are_not_documents_are_refused():
    with pytest.raises(TypeError, match="SourceDocument"):
        await sync_documents(FakeGraph(), [{"id": "p1"}], source="notion")


@pytest.mark.parametrize("name", ["", "  ", "x__pending__", "wiki:team"])
async def test_a_bad_source_name_is_refused(name):
    with pytest.raises(ValueError):
        await sync_documents(FakeGraph(), [], source=name)


async def test_incomplete_writes_are_reported_as_failed_and_not_pruned():
    graph = FakeGraph({"notion:p1": _hash("old")})
    graph.incomplete_on = {"notion:p1"}

    report = await sync_documents(graph, [SourceDocument("p1", "new")], source="notion")

    assert "extraction failed" in report.failed["notion:p1"]
    assert not (report.added or report.updated or report.unchanged)
    assert "notion:p1" in graph.documents


async def test_unchanged_text_still_refreshes_metadata():
    graph = FakeGraph({"notion:p1": _hash("same")})
    doc = SourceDocument("p1", "same", {"url": "https://notion.so/moved", "pending_delete": True})

    report = await sync_documents(graph, [doc], source="notion")

    assert report.unchanged == ["notion:p1"]
    assert graph.metadata_refreshes == [("notion:p1", {"url": "https://notion.so/moved"})]


def test_document_id_is_source_and_id():
    assert document_id_for("drive", "abc") == "drive:abc"


def test_graph_safe_metadata_keeps_scalars_and_lists():
    assert graph_safe_metadata(
        {
            "n": 3,
            "f": float("nan"),
            "tags": ["a", 1, None, object()],
            "none": None,
            "path": "x",
            "pending_delete": True,
            "cleanup_candidates": ["e1"],
        }
    ) == {"n": 3, "tags": ["a", 1]}


# ── sync_table ──────────────────────────────────────────────────


async def test_a_table_is_written_through_its_mapping():
    mapping = TableMapping(source="users", label="Person", key="id", name="name")
    graph = FakeGraph(tables=[mapping])
    rows = [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob", "city": "Rome"}]

    result = await sync_table(graph, rows, table="users")

    assert result.records == 2
    assert graph.ingests == [("users", rows)]


async def test_a_mapping_declared_with_a_file_suffix_matches_the_table_name():
    graph = FakeGraph(tables=[TableMapping(source="users.csv", label="Person", name="name")])

    await sync_table(graph, [{"name": "Alice"}], table="users")

    assert graph.ingests[0][0] == "users.csv"


async def test_an_empty_table_is_an_empty_snapshot_with_the_mapping_header():
    mapping = TableMapping(source="users", label="Person", key="id", name="name")
    graph = FakeGraph(tables=[mapping])

    await sync_table(graph, [], table="users")

    batch = await graph.last_loader.load_records("users", Context())
    assert graph.ingests == [("users", [])]
    assert set(batch.columns) == {"id", "name"}


async def test_a_table_without_a_mapping_is_refused():
    with pytest.raises(ValueError, match="No TableMapping"):
        await sync_table(FakeGraph(), [{"id": 1}], table="users")


async def test_a_list_with_a_record_loader_is_refused_as_tables():
    rag = GraphRAG.__new__(GraphRAG)

    with pytest.raises(ValueError, match="look like tables"):
        await rag.ingest(["users", "orders"], record_loader=CsvRecordLoader())


def test_a_record_loader_marks_a_suffixless_source_as_a_table():
    assert GraphRAG._is_tabular("users", record_loader=CsvRecordLoader())
    assert not GraphRAG._is_tabular("users")
    assert GraphRAG._is_tabular("users.csv")
    # A prose loader still wins, as it does for a .csv.
    assert not GraphRAG._is_tabular("users", loader=object(), record_loader=CsvRecordLoader())


# ── sync() over dlt resources ───────────────────────────────────


@pytest.fixture
def dlt():
    return pytest.importorskip("dlt")


async def test_sync_routes_documents_and_rows_by_what_a_resource_yields(dlt):
    graph = FakeGraph(tables=[TableMapping(source="users", label="Person", name="name")])

    @dlt.resource(name="notion")
    def notion():
        yield SourceDocument("p1", "Alice works at Acme.")

    @dlt.resource(name="users")
    def users():
        yield [{"name": "Alice"}, {"name": "Bob"}]

    result = await sync(graph, [notion(), users()])

    assert result.documents["notion"].added == ["notion:p1"]
    assert result.table_rows == {"users": 2}
    assert graph.ingests[0][0] == "users"


async def test_sync_dry_run_writes_no_table(dlt):
    graph = FakeGraph(tables=[TableMapping(source="users", label="Person", name="name")])

    @dlt.resource(name="users")
    def users():
        yield {"name": "Alice"}

    result = await sync(graph, users(), dry_run=True)

    assert result.table_rows == {"users": 1} and result.tables == {}
    assert graph.ingests == []


async def test_sync_reads_every_resource_of_a_source(dlt):
    graph = FakeGraph()

    @dlt.source
    def workspace():
        @dlt.resource(name="wiki")
        def wiki():
            yield SourceDocument("w1", "wiki page")

        @dlt.resource(name="tickets")
        def tickets():
            yield SourceDocument("t1", "ticket")

        return wiki, tickets

    result = await sync(graph, workspace())

    assert set(result.documents) == {"wiki", "tickets"}
    assert set(graph.documents) == {"wiki:w1", "tickets:t1"}


@pytest.mark.parametrize("prune", [True, False])
async def test_sync_refuses_a_cursor_before_writing_anything(dlt, prune):
    graph = FakeGraph(tables=[TableMapping(source="users", label="Person", name="name")])

    @dlt.resource(name="notion")
    def notion():
        yield SourceDocument("p1", "a")

    @dlt.resource(name="users")
    def users():
        yield {"name": "Alice"}

    table = users()
    table.apply_hints(incremental=dlt.sources.incremental("name"))

    # Refused with prune off too: a table is a snapshot, so a partial read would
    # remove the rows the cursor skipped.
    with pytest.raises(ValueError, match="incremental cursor"):
        await sync(graph, [notion(), table], prune=prune)
    assert graph.updates == [] and graph.ingests == []


async def test_sync_checks_table_mappings_before_writing_anything(dlt):
    graph = FakeGraph()

    @dlt.resource(name="notion")
    def notion():
        yield SourceDocument("p1", "a")

    @dlt.resource(name="users")
    def users():
        yield {"name": "Alice"}

    with pytest.raises(ValueError, match="No TableMapping"):
        await sync(graph, [notion(), users()])
    assert graph.updates == []


async def test_sync_checks_table_columns_before_writing_anything(dlt):
    graph = FakeGraph(tables=[TableMapping(source="users", label="Person", key="id", name="name")])

    @dlt.resource(name="notion")
    def notion():
        yield SourceDocument("p1", "a")

    @dlt.resource(name="users")
    def users():
        yield {"name": "Alice"}  # no id column

    with pytest.raises(ValueError, match="does not fit its mapping"):
        await sync(graph, [notion(), users()], dry_run=True)
    with pytest.raises(ValueError, match="does not fit its mapping"):
        await sync(graph, [notion(), users()])
    assert graph.updates == [] and graph.ingests == []


async def test_sync_sees_a_cursor_on_the_listing_a_transformer_reads(dlt, tmp_path):
    from dlt.sources.filesystem import filesystem

    from graphrag_sdk.connectors import file_documents

    (tmp_path / "a.md").write_text("a")
    listing = filesystem(bucket_url=tmp_path.as_uri(), file_glob="*")
    listing.apply_hints(incremental=dlt.sources.incremental("modification_date"))

    with pytest.raises(ValueError, match="incremental cursor"):
        await sync(FakeGraph(), file_documents(listing, name="files"))


async def test_a_failed_metadata_refresh_is_reported_and_the_sync_goes_on():
    graph = FakeGraph({"notion:p1": _hash("same")})

    async def broken(cypher, params):
        if "SET d += $props" in cypher:
            raise RuntimeError("connection reset")
        return await FakeGraph.query(graph, cypher, params)

    graph.query = broken

    report = await sync_documents(
        graph,
        [SourceDocument("p1", "same", {"url": "u"}), SourceDocument("p2", "new")],
        source="notion",
    )

    assert "connection reset" in report.failed["notion:p1"]
    assert report.added == ["notion:p2"]


async def test_sync_writes_an_empty_mapped_resource_as_an_empty_table(dlt):
    graph = FakeGraph(tables=[TableMapping(source="users", label="Person", name="name")])

    @dlt.resource(name="users")
    def users():
        yield from []

    result = await sync(graph, users())

    assert result.table_rows == {"users": 0}
    assert graph.ingests == [("users", [])]


async def test_sync_reports_an_empty_document_resource_without_pruning(dlt):
    graph = FakeGraph({"notion:p1": _hash("a")})

    @dlt.resource(name="notion")
    def notion():
        yield from []

    result = await sync(graph, notion())

    assert result.documents["notion"].prune_skipped
    assert graph.deletes == []


async def test_sync_refuses_two_resources_with_one_name(dlt):
    @dlt.resource(name="docs")
    def first():
        yield SourceDocument("a", "a")

    @dlt.resource(name="docs")
    def second():
        yield SourceDocument("b", "b")

    with pytest.raises(ValueError, match="named docs"):
        await sync(FakeGraph(), [first(), second()])


async def test_sync_refuses_a_resource_that_mixes_documents_and_rows(dlt):
    @dlt.resource(name="mixed")
    def mixed():
        yield SourceDocument("a", "a")
        yield {"id": 1}

    with pytest.raises(TypeError, match="not a mix"):
        await sync(FakeGraph(), mixed())


async def test_sync_refuses_a_plain_list(dlt):
    with pytest.raises(TypeError, match="dlt sources and resources"):
        await sync(FakeGraph(), [[SourceDocument("a", "a")]])


async def test_llamaindex_documents_takes_the_reader_id_and_metadata(dlt):
    from graphrag_sdk.connectors import llamaindex_documents

    class Doc:
        def __init__(self, id_, text, metadata):
            self.id_, self.text, self.metadata = id_, text, metadata

    class Reader:
        def load_data(self, **kwargs):
            assert kwargs == {"page_ids": ["p1"]}
            return [Doc("p1", "Alice.", {"url": "https://notion.so/p1"})]

    resource = llamaindex_documents(Reader(), name="notion", page_ids=["p1"])

    assert resource.name == "notion"
    assert list(resource) == [SourceDocument("p1", "Alice.", {"url": "https://notion.so/p1"})]


async def test_file_documents_reads_text_files_and_skips_the_rest(dlt, tmp_path):
    from dlt.sources.filesystem import filesystem

    from graphrag_sdk.connectors import file_documents

    (tmp_path / "a.md").write_text("# A\nAlice works at Acme.")
    (tmp_path / "b.bin").write_bytes(b"\x00\x01")

    resource = file_documents(filesystem(bucket_url=tmp_path.as_uri(), file_glob="*"), name="files")
    docs = list(resource)

    assert [d.text for d in docs] == ["# A\nAlice works at Acme."]
    assert docs[0].id == docs[0].metadata["source_url"]
    assert docs[0].id.endswith("/a.md")


async def test_file_documents_takes_one_file_at_a_time_too(dlt, tmp_path):
    from graphrag_sdk.connectors import file_documents

    path = tmp_path / "a.md"
    path.write_text("Alice.")

    class Item(dict):
        def read_bytes(self):
            return path.read_bytes()

    @dlt.resource(name="listing")
    def listing():
        yield Item(file_url=path.as_uri(), file_name="a.md")

    assert [d.text for d in file_documents(listing(), name="files")] == ["Alice."]


def test_a_pdf_is_read_from_a_closed_temporary_file(dlt, monkeypatch, tmp_path):
    from graphrag_sdk.connectors import sources
    from graphrag_sdk.core.models import DocumentOutput
    from graphrag_sdk.ingestion.loaders.pdf_loader import PdfLoader

    seen: list[str] = []
    paths: list[str] = []

    def load(self, path):
        paths.append(path)
        with open(path, "rb") as handle:  # reopening is what Windows refuses
            seen.append(handle.read().decode())
        return DocumentOutput(text="pdf text")

    monkeypatch.setattr(PdfLoader, "_load_sync", load)

    class Item(dict):
        def read_bytes(self):
            return b"%PDF-1.4"

    text = sources._file_text(Item(file_url="file:///a.pdf", file_name="a.pdf"))

    assert text == "pdf text" and seen == ["%PDF-1.4"]
    assert not Path(paths[0]).exists(), "the temporary file is removed"

    class Broken(dict):
        def read_bytes(self):
            raise OSError("network down")

    monkeypatch.setattr(sources.tempfile, "tempdir", str(tmp_path))
    with pytest.raises(OSError):
        sources._file_text(Broken(file_url="s3://b/a.pdf", file_name="a.pdf"))
    assert list(tmp_path.iterdir()) == [], "a failed fetch leaves no file behind"


# ── Real FalkorDB ───────────────────────────────────────────────


@pytest.mark.integration
async def test_resync_against_falkordb(real_falkordb_rag_factory, scripted_llm):
    """Add, skip, update and prune against a real graph.

    Step-1 NER is stubbed, as in test_cached_chunk_extraction: the default GLiNER
    extractor cannot load its tokenizer on every CI runner, and a chunk whose
    extraction failed withholds the content hash, so the unchanged re-sync would
    redo the documents for a reason unrelated to syncing.
    """
    from graphrag_sdk.ingestion.extraction_strategies.entity_extractors import (
        EntityExtractor,
    )
    from graphrag_sdk.ingestion.extraction_strategies.graph_extraction import (
        GraphExtraction,
    )

    class _NoLocalNER(EntityExtractor):
        async def extract_entities(self, text, entity_types, source_chunk_id):
            return []

    # Strict: one scripted reply per extraction call the test expects.
    llm = scripted_llm([], [], [])
    extractor = GraphExtraction(llm, entity_extractor=_NoLocalNER())
    rag = real_falkordb_rag_factory(
        llm=llm,
        resolver=None,
        ontology=Ontology(
            entities=[Entity(label="Person")],
            tables=[TableMapping(source="users", label="Person", key="id", name="name")],
        ),
    )
    pages = {"p1": "Alice works at Acme.", "p2": "Bob lives in Paris."}
    users = [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]

    def documents():
        return [SourceDocument(k, v, {"url": f"https://notion.so/{k}"}) for k, v in pages.items()]

    first = await sync_documents(rag, documents(), source="notion", extractor=extractor)
    await sync_table(rag, users, table="users")
    assert first.added == ["notion:p1", "notion:p2"]
    calls_after_first = llm._call_index

    # Nothing changed: no model call, nothing rewritten.
    again = await sync_documents(rag, documents(), source="notion", extractor=extractor)
    assert again.unchanged == ["notion:p1", "notion:p2"]
    assert llm._call_index == calls_after_first

    # One page edited, one removed, one table row removed.
    pages = {"p1": "Alice works at Acme in Berlin."}
    users = [{"id": 1, "name": "Alice"}]
    last = await sync_documents(rag, documents(), source="notion", extractor=extractor)
    table = await sync_table(rag, users, table="users")

    assert last.updated == ["notion:p1"] and last.deleted == ["notion:p2"]
    assert llm._call_index == calls_after_first + 1
    assert table.replaced_existing
    ids = {row[0] for row in await rag.query("MATCH (d:Document) RETURN d.id")}
    assert ids == {"notion:p1", "users"}
    url = await rag.query("MATCH (d:Document {id: 'notion:p1'}) RETURN d.url")
    assert url == [["https://notion.so/p1"]]
    keys = await rag.query("MATCH (p:Person) WHERE p.entity_key IS NOT NULL RETURN p.entity_key")
    assert keys == [["1"]]


@pytest.mark.integration
async def test_empty_table_and_metadata_refresh_against_falkordb(
    real_falkordb_rag_factory, scripted_llm
):
    """An emptied table loses its rows; a moved page keeps its text and gets its new URL."""
    from graphrag_sdk.ingestion.extraction_strategies.entity_extractors import (
        EntityExtractor,
    )
    from graphrag_sdk.ingestion.extraction_strategies.graph_extraction import (
        GraphExtraction,
    )

    class _NoLocalNER(EntityExtractor):
        async def extract_entities(self, text, entity_types, source_chunk_id):
            return []

    llm = scripted_llm([])
    extractor = GraphExtraction(llm, entity_extractor=_NoLocalNER())
    rag = real_falkordb_rag_factory(
        llm=llm,
        resolver=None,
        ontology=Ontology(
            entities=[Entity(label="Person")],
            tables=[TableMapping(source="users", label="Person", key="id", name="name")],
        ),
    )

    await sync_table(rag, [{"id": 1, "name": "Alice"}], table="users")
    emptied = await sync_table(rag, [], table="users")

    assert emptied.replaced_existing
    keys = await rag.query("MATCH (p:Person) WHERE p.entity_key IS NOT NULL RETURN p.entity_key")
    assert keys == []

    page = SourceDocument("p1", "Alice works at Acme.", {"url": "https://notion.so/old"})
    await sync_documents(rag, [page], source="notion", extractor=extractor)
    page.metadata = {"url": "https://notion.so/new"}
    report = await sync_documents(rag, [page], source="notion", extractor=extractor)

    assert report.unchanged == ["notion:p1"]
    url = await rag.query("MATCH (d:Document {id: 'notion:p1'}) RETURN d.url")
    assert url == [["https://notion.so/new"]]
