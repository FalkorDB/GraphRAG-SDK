from __future__ import annotations

from typing import Any

import pytest

from graphrag_sdk import (
    Attribute,
    Column,
    DocumentNotFoundError,
    EndpointMapping,
    Entity,
    GraphRAG,
    MappingError,
    Ontology,
    RelationshipMapping,
    RelationshipResolutionError,
)
from graphrag_sdk.core.context import Context
from graphrag_sdk.ingestion.loaders.record_loader import CsvRecordLoader
from graphrag_sdk.ingestion.mapping import ontology_for_relationship
from graphrag_sdk.ingestion.structured_pipeline import RelationshipIngestionPipeline


@pytest.mark.integration
async def test_relationship_snapshot_and_mapping_reload_on_real_falkordb(
    real_falkordb_rag_factory, llm, tmp_path
):
    from graphrag_sdk import TableMapping

    mapping = consumed_mapping()
    rag = real_falkordb_rag_factory(
        llm=llm,
        resolver=None,
        ontology=Ontology(
            entities=[Entity(label="Unit"), Entity(label="Batch")],
            tables=[
                TableMapping(source="unit.csv", label="Unit", key="unit_id", name="unit_id"),
                TableMapping(source="batch.csv", label="Batch", key="batch_id", name="batch_id"),
            ],
            relationship_tables=[mapping],
        ),
    )
    for filename, content in (
        ("unit.csv", "unit_id\nU-1\nU-2\n"),
        ("batch.csv", "batch_id\nB-1\nB-2\n"),
    ):
        path = tmp_path / filename
        path.write_text(content, encoding="utf-8")
        await rag.ingest(str(path))

    async def counts():
        result = await rag._graph_store.query_raw(
            "MATCH (n) RETURN count(n), count(CASE WHEN n:Document THEN 1 END), "
            "count(CASE WHEN n:Chunk THEN 1 END)"
        )
        return result.result_set

    before = await counts()
    path = write_csv(
        tmp_path, ["U-1,B-1,CONSUMED_BATCH,2026-09-20", "U-2,B-2,CONSUMED_BATCH,2026-09-21"]
    )
    added = await rag.apply_changes(added=[str(path)])
    assert added.added[0].is_success
    assert added.added[0].result.metadata["relationships_written"] == 2
    assert (await rag.ingest(str(path))).relationships_written == 2
    assert await counts() == before
    stored = await rag._ontology_store.load()
    assert stored.relationship_tables == [mapping]
    edges = await rag._graph_store.query_raw(
        "MATCH (u:Unit)-[r:RELATES {rel_type:'CONSUMED_BATCH'}]->(b:Batch) "
        "RETURN u.unit__unit_id, b.batch__batch_id, r.consumed_batch__consumed_on "
        "ORDER BY u.unit__unit_id"
    )
    assert edges.result_set == [["U-1", "B-1", "2026-09-20"], ["U-2", "B-2", "2026-09-21"]]

    write_csv(tmp_path, ["U-1,B-1,CONSUMED_BATCH,"])
    modified = await rag.apply_changes(modified=[str(path)])
    assert modified.modified[0].is_success
    assert modified.modified[0].result.metadata["relationships_deleted"] == 1
    edges = await rag._graph_store.query_raw(
        "MATCH ()-[r:RELATES {rel_type:'CONSUMED_BATCH'}]->() RETURN r.consumed_batch__consumed_on"
    )
    assert edges.result_set == [[None]]
    write_csv(tmp_path, ["U-1,B-404,CONSUMED_BATCH,"])
    with pytest.raises(RelationshipResolutionError):
        await rag.ingest(str(path))
    assert (
        await rag._graph_store.query_raw(
            "MATCH ()-[r:RELATES {rel_type:'CONSUMED_BATCH'}]->() RETURN count(r)"
        )
    ).result_set == [[1]]
    # A new instance registers a replacement mapping, then fails its first
    # ingest. Old property declarations must survive for the next retry.
    write_csv(tmp_path, ["U-1,B-1,CONSUMED_BATCH,2026-09-20"])
    await rag.ingest(str(path))
    replacement = real_falkordb_rag_factory(
        llm=llm,
        resolver=None,
        connection=rag._conn,
        ontology=Ontology(relationship_tables=[consumed_mapping(properties={})]),
    )
    await replacement.get_ontology()
    write_csv(tmp_path, ["U-1,B-404,CONSUMED_BATCH,"])
    with pytest.raises(RelationshipResolutionError):
        await replacement.ingest(str(path))
    assert (
        await rag._graph_store.query_raw(
            "MATCH ()-[r:RELATES {rel_type:'CONSUMED_BATCH'}]->() RETURN r.consumed_batch__consumed_on"
        )
    ).result_set == [["2026-09-20"]]
    retry = real_falkordb_rag_factory(llm=llm, resolver=None, connection=rag._conn)
    write_csv(tmp_path, ["U-1,B-1,CONSUMED_BATCH,"])
    await retry.ingest(str(path))
    assert (
        await rag._graph_store.query_raw(
            "MATCH ()-[r:RELATES {rel_type:'CONSUMED_BATCH'}]->() RETURN r.consumed_batch__consumed_on"
        )
    ).result_set == [[None]]
    declared = await retry.get_ontology()
    assert not any(
        prop.name == "consumed_batch__consumed_on"
        for relation in declared.relations
        for prop in relation.properties
    )

    async def edge_count():
        result = await rag._graph_store.query_raw(
            "MATCH ()-[r:RELATES {rel_type:'CONSUMED_BATCH'}]->() RETURN count(r)"
        )
        return result.result_set[0][0]

    # Deleting the file takes its edges and keeps its mapping, as a node table
    # keeps its own, so the file added back is read as the same table again.
    stored = (await rag._ontology_store.load()).relationship_tables
    assert [item.source for item in stored] == ["consumed_batch.csv"]
    deleted = await retry.apply_changes(deleted=[str(path)])
    assert deleted.deleted[0].is_success
    assert deleted.deleted[0].result.document_uid == str(path)
    assert await edge_count() == 0
    assert (await rag._ontology_store.load()).relationship_tables == stored
    assert await counts() == before
    readded = await retry.apply_changes(added=[str(path)])
    assert readded.added[0].is_success, readded.added[0].error
    assert readded.added[0].result.metadata["relationships_written"] == 1
    assert await edge_count() == 1
    # No proposed node table: no consumed_batch entity, Document or Chunk.
    assert await counts() == before
    await retry.drop_table("consumed_batch.csv")
    assert await edge_count() == 0
    assert (await rag._ontology_store.load()).relationship_tables == []
    assert await counts() == before


@pytest.mark.integration
@pytest.mark.parametrize(
    "relationship_source, doc_id",
    [
        ("consumed_batch.csv", "consumed_batch.csv"),
        ("exports/consumed_batch.csv", "notes/consumed_batch.csv"),
    ],
)
async def test_delete_prefers_a_document_named_like_a_relationship_table_on_real_falkordb(
    real_falkordb_rag_factory, llm, tmp_path, relationship_source, doc_id
):
    """``deleted`` removes a Document with that id and leaves the table alone."""
    from graphrag_sdk import TableMapping
    from graphrag_sdk.core.models import GraphData
    from graphrag_sdk.ingestion.extraction_strategies.base import ExtractionStrategy

    class NoEntities(ExtractionStrategy):
        async def extract(self, chunks, ontology, ctx):
            return GraphData(nodes=[], relationships=[])

    rag = real_falkordb_rag_factory(
        llm=llm,
        resolver=None,
        ontology=Ontology(
            entities=[Entity(label="Unit"), Entity(label="Batch")],
            tables=[
                TableMapping(source="unit.csv", label="Unit", key="unit_id", name="unit_id"),
                TableMapping(source="batch.csv", label="Batch", key="batch_id", name="batch_id"),
            ],
            relationship_tables=[consumed_mapping(source=relationship_source)],
        ),
    )
    for filename, content in (
        ("unit.csv", "unit_id\nU-1\nU-2\n"),
        ("batch.csv", "batch_id\nB-1\nB-2\n"),
    ):
        table = tmp_path / filename
        table.write_text(content, encoding="utf-8")
        await rag.ingest(str(table))
    path = write_csv(
        tmp_path, ["U-1,B-1,CONSUMED_BATCH,2026-09-20", "U-2,B-2,CONSUMED_BATCH,2026-09-21"]
    )
    assert (await rag.ingest(str(path))).relationships_written == 2
    # Prose stored under the table's name, or under another path with its basename.
    await rag.ingest(
        text="Unit U-1 consumed batch B-1 on 20 September.",
        document_id=doc_id,
        extractor=NoEntities(),
    )

    async def state():
        documents = await rag._graph_store.query_raw(
            "MATCH (d:Document {id: $id}) RETURN count(d)", {"id": doc_id}
        )
        edges = await rag._graph_store.query_raw(
            "MATCH ()-[r:RELATES {rel_type:'CONSUMED_BATCH'}]->() RETURN count(r)"
        )
        stored = await rag._ontology_store.load()
        return (
            documents.result_set[0][0],
            edges.result_set[0][0],
            [item.source for item in stored.relationship_tables],
        )

    assert await state() == (1, 2, [relationship_source])
    result = await rag.apply_changes(deleted=[doc_id])
    assert result.deleted[0].is_success, result.deleted[0].error
    assert result.deleted[0].result.chunks_deleted == 1
    assert await state() == (0, 2, [relationship_source])


async def _ingest_units_batches_and_edges(rag, tmp_path):
    for filename, content in (
        ("unit.csv", "unit_id\nU-1\nU-2\n"),
        ("batch.csv", "batch_id\nB-1\nB-2\n"),
    ):
        table = tmp_path / filename
        table.write_text(content, encoding="utf-8")
        await rag.ingest(str(table))
    path = write_csv(
        tmp_path, ["U-1,B-1,CONSUMED_BATCH,2026-09-20", "U-2,B-2,CONSUMED_BATCH,2026-09-21"]
    )
    assert (await rag.ingest(str(path))).relationships_written == 2
    return path


async def _consumed_edge_count(rag) -> int:
    result = await rag._graph_store.query_raw(
        "MATCH ()-[r:RELATES {rel_type:'CONSUMED_BATCH'}]->() RETURN count(r)"
    )
    return result.result_set[0][0]


def _unit_and_batch_tables():
    from graphrag_sdk import TableMapping

    return [
        TableMapping(source="unit.csv", label="Unit", key="unit_id", name="unit_id"),
        TableMapping(source="batch.csv", label="Batch", key="batch_id", name="batch_id"),
    ]


@pytest.mark.integration
async def test_endpoint_keys_cannot_be_removed_under_a_relationship_table_on_real_falkordb(
    real_falkordb_rag_factory, llm, tmp_path
):
    """drop_table, drop_attribute and rename_attribute refuse a key an edge table uses."""
    rag = real_falkordb_rag_factory(
        llm=llm,
        resolver=None,
        ontology=Ontology(
            entities=[Entity(label="Unit"), Entity(label="Batch")],
            tables=_unit_and_batch_tables(),
            relationship_tables=[consumed_mapping()],
        ),
    )
    await _ingest_units_batches_and_edges(rag, tmp_path)
    refused = r"'consumed_batch\.csv' uses {}.*drop_table\('consumed_batch\.csv'\) first"
    with pytest.raises(ValueError, match=refused.format(r"Unit\.unit__unit_id")):
        await rag.drop_table("unit.csv")
    with pytest.raises(ValueError, match=refused.format(r"Batch\.batch__batch_id")):
        await rag.drop_attribute("Batch", "batch__batch_id")
    with pytest.raises(ValueError, match=refused.format(r"Unit\.unit__unit_id")):
        await rag.rename_attribute("Unit", "unit__unit_id", "unit_code")
    # Nothing was removed, and a new instance on the graph still starts.
    stored = await rag._ontology_store.load()
    assert [m.source for m in stored.tables] == ["batch.csv", "unit.csv"]
    keys = {(e.label, p.name) for e in stored.entities for p in e.properties}
    assert {("Unit", "unit__unit_id"), ("Batch", "batch__batch_id")} <= keys
    held = await rag._graph_store.query_raw(
        "MATCH (u:Unit) WHERE u.unit__unit_id IS NOT NULL RETURN count(u)"
    )
    assert held.result_set == [[2]]
    assert await _consumed_edge_count(rag) == 2
    restarted = real_falkordb_rag_factory(llm=llm, resolver=None, connection=rag._conn)
    assert len((await restarted.get_ontology()).relationship_tables) == 1
    # The order the message asks for works.
    await restarted.drop_table("consumed_batch.csv")
    await restarted.drop_table("unit.csv")
    assert await _consumed_edge_count(rag) == 0
    again = real_falkordb_rag_factory(llm=llm, resolver=None, connection=rag._conn)
    ontology = await again.get_ontology()
    assert ontology.relationship_tables == []
    assert [m.source for m in ontology.tables] == ["batch.csv"]


@pytest.mark.integration
async def test_stored_mapping_with_a_missing_endpoint_key_does_not_stop_startup_on_real_falkordb(
    real_falkordb_rag_factory, llm, tmp_path, caplog
):
    """A graph whose relationship table lost its key still opens and can drop it."""
    import logging

    rag = real_falkordb_rag_factory(
        llm=llm,
        resolver=None,
        ontology=Ontology(
            entities=[Entity(label="Unit"), Entity(label="Batch")],
            tables=_unit_and_batch_tables(),
            relationship_tables=[consumed_mapping()],
        ),
    )
    path = await _ingest_units_batches_and_edges(rag, tmp_path)
    # The ontology an unguarded drop_table('unit.csv') left: the node table and
    # its key are gone, and the relationship mapping is still stored.
    await rag._ontology_store.drop_entity_property("Unit", "unit__unit_id")
    await rag._ontology_store.drop_table_mapping("unit.csv")
    restarted = real_falkordb_rag_factory(llm=llm, resolver=None, connection=rag._conn)
    with caplog.at_level(logging.WARNING, logger="graphrag_sdk.api.main"):
        ontology = await restarted.get_ontology()
    assert [m.source for m in ontology.relationship_tables] == ["consumed_batch.csv"]
    assert "start endpoint key Unit.unit__unit_id" in caplog.text
    assert "drop_table('consumed_batch.csv')" in caplog.text
    # Ingesting that file is still an error; the rest of the graph is usable.
    with pytest.raises(MappingError, match=r"start endpoint key Unit\.unit__unit_id"):
        await restarted.ingest(str(path))
    assert await _consumed_edge_count(rag) == 2
    batch = tmp_path / "batch.csv"
    batch.write_text("batch_id\nB-1\nB-2\nB-3\n", encoding="utf-8")
    await restarted.ingest(str(batch))
    await restarted.drop_table("consumed_batch.csv")
    assert await _consumed_edge_count(rag) == 0
    again = real_falkordb_rag_factory(llm=llm, resolver=None, connection=rag._conn)
    assert (await again.get_ontology()).relationship_tables == []


class _Interleaved:
    """Delegate to a graph store, holding the named calls until an event is set.

    ``before[name]`` is awaited before that call runs and ``after[name]`` is
    set once it returns, so two pipelines on one graph run in a fixed order.
    """

    def __init__(self, store, *, before=None, after=None) -> None:
        self._store = store
        self._before = before or {}
        self._after = after or {}

    def __getattr__(self, name):
        attr = getattr(self._store, name)
        if name not in self._before and name not in self._after:
            return attr

        async def call(*args, **kwargs):
            if name in self._before:
                await self._before[name].wait()
            result = await attr(*args, **kwargs)
            if name in self._after:
                self._after[name].set()
            return result

        return call


@pytest.mark.integration
async def test_two_runs_of_one_file_at_once_keep_every_edge_on_real_falkordb(
    real_falkordb_rag_factory, llm, tmp_path
):
    """The run that started first and writes last cannot strip the other's edges.

    Run B writes, run A rewrites the same edges, then B cleans up: with a
    random token per run B's cleanup saw A's token on every edge and deleted
    them all, and both runs reported success.
    """
    import asyncio

    rag = real_falkordb_rag_factory(
        llm=llm,
        resolver=None,
        ontology=Ontology(
            entities=[Entity(label="Unit"), Entity(label="Batch")],
            tables=_unit_and_batch_tables(),
            relationship_tables=[consumed_mapping()],
        ),
    )
    path = await _ingest_units_batches_and_edges(rag, tmp_path)
    a_started, a_wrote, b_wrote = asyncio.Event(), asyncio.Event(), asyncio.Event()
    run_a = RelationshipIngestionPipeline(
        CsvRecordLoader(),
        _Interleaved(
            rag._graph_store,
            before={"upsert_relationships": b_wrote},
            after={"next_relationship_snapshot": a_started, "upsert_relationships": a_wrote},
        ),
    )
    run_b = RelationshipIngestionPipeline(
        CsvRecordLoader(),
        _Interleaved(
            rag._graph_store,
            before={
                "next_relationship_snapshot": a_started,
                "finalize_relationship_snapshot": a_wrote,
            },
            after={"upsert_relationships": b_wrote},
        ),
    )
    results = await asyncio.gather(
        run_a.run(str(path), consumed_mapping(), Context()),
        run_b.run(str(path), consumed_mapping(), Context()),
    )
    assert [
        (r.relationships_written, r.relationships_deleted, r.incomplete_writes) for r in results
    ] == [(2, 0, []), (2, 0, [])]
    edges = await rag._graph_store.query_raw(
        "MATCH (u:Unit)-[r:RELATES {rel_type:'CONSUMED_BATCH'}]->(b:Batch) "
        "RETURN u.unit__unit_id, b.batch__batch_id, r.structured_sources, "
        "r.consumed_batch__consumed_on ORDER BY u.unit__unit_id"
    )
    assert edges.result_set == [
        ["U-1", "B-1", ["consumed_batch"], "2026-09-20"],
        ["U-2", "B-2", ["consumed_batch"], "2026-09-21"],
    ]
    # One run after another still removes a row the file no longer has.
    write_csv(tmp_path, ["U-1,B-1,CONSUMED_BATCH,2026-09-20"])
    assert (await rag.ingest(str(path))).relationships_deleted == 1
    assert await _consumed_edge_count(rag) == 1


@pytest.mark.integration
async def test_rerun_outranks_a_stored_snapshot_ahead_of_the_clock_on_real_falkordb(
    real_falkordb_rag_factory, llm, tmp_path
):
    """Two runs in one millisecond, or a server clock set back, still clean up."""
    rag = real_falkordb_rag_factory(
        llm=llm,
        resolver=None,
        ontology=Ontology(
            entities=[Entity(label="Unit"), Entity(label="Batch")],
            tables=_unit_and_batch_tables(),
            relationship_tables=[consumed_mapping()],
        ),
    )
    path = await _ingest_units_batches_and_edges(rag, tmp_path)
    # The last run's number is a day ahead of the clock the next run reads.
    await rag._graph_store.query_raw(
        "MATCH ()-[r:RELATES]->() "
        "SET r.relationship_mapping_consumed_batch = timestamp() + 86400000"
    )
    write_csv(tmp_path, ["U-1,B-1,CONSUMED_BATCH,2026-09-20"])
    assert (await rag.ingest(str(path))).relationships_deleted == 1
    assert await _consumed_edge_count(rag) == 1


@pytest.mark.integration
async def test_merged_nodes_keep_both_relationship_tables_ownership_on_real_falkordb(
    real_falkordb_rag_factory, llm, tmp_path
):
    """Folding one Unit into another keeps every table's claim on the shared edge."""
    from graphrag_sdk import TableMapping

    by_unit = consumed_mapping(source="consumed_a.csv")
    by_legacy = consumed_mapping(
        source="consumed_b.csv", start=EndpointMapping("Unit", "legacy__legacy_id", "source")
    )
    rag = real_falkordb_rag_factory(
        llm=llm,
        resolver=None,
        ontology=Ontology(
            entities=[Entity(label="Unit"), Entity(label="Batch")],
            tables=[
                TableMapping(source="unit.csv", label="Unit", key="unit_id", name="name"),
                TableMapping(source="legacy.csv", label="Unit", key="legacy_id", name="name"),
                TableMapping(source="batch.csv", label="Batch", key="batch_id", name="batch_id"),
            ],
            relationship_tables=[by_unit, by_legacy],
        ),
    )
    for filename, content in (
        ("unit.csv", "unit_id,name\nU-1,Alpha\n"),
        ("legacy.csv", "legacy_id,name\nL-9,Gamma\n"),
        ("batch.csv", "batch_id\nB-1\n"),
        ("consumed_a.csv", "source,target,consumed_on\nU-1,B-1,2026-01-01\n"),
        ("consumed_b.csv", "source,target,consumed_on\nL-9,B-1,2026-02-02\n"),
    ):
        (tmp_path / filename).write_text(content, encoding="utf-8")
        await rag.ingest(str(tmp_path / filename))

    async def edges():
        result = await rag._graph_store.query_raw(
            "MATCH (u:Unit)-[r:RELATES {rel_type:'CONSUMED_BATCH'}]->(:Batch) "
            "RETURN u.name, r.structured_sources, r.consumed_a__consumed_on, "
            "r.consumed_b__consumed_on ORDER BY u.name"
        )
        return result.result_set

    assert await edges() == [
        ["Alpha", ["consumed_a"], "2026-01-01", None],
        ["Gamma", ["consumed_b"], None, "2026-02-02"],
    ]
    # U-1 is renamed to Gamma, so its node is folded into L-9's.
    (tmp_path / "unit.csv").write_text("unit_id,name\nU-1,Gamma\n", encoding="utf-8")
    await rag.ingest(str(tmp_path / "unit.csv"))
    merged = await edges()
    assert len(merged) == 1
    assert sorted(merged[0][1]) == ["consumed_a", "consumed_b"]
    assert merged[0][2:] == ["2026-01-01", "2026-02-02"]
    # consumed_b.csv still lists L-9 -> B-1, so the edge outlives consumed_a.csv.
    await rag.drop_table("consumed_a.csv")
    assert await edges() == [["Gamma", ["consumed_b"], None, "2026-02-02"]]


@pytest.mark.integration
async def test_apply_changes_in_git_diff_order_writes_every_edge_on_real_falkordb(
    real_falkordb_rag_factory, llm, tmp_path
):
    """A git diff lists consumed_batch.csv before unit.csv; the edges are written anyway."""
    rag = real_falkordb_rag_factory(
        llm=llm,
        resolver=None,
        ontology=Ontology(
            entities=[Entity(label="Unit"), Entity(label="Batch")],
            tables=_unit_and_batch_tables(),
            relationship_tables=[consumed_mapping()],
        ),
    )
    unit, batch = tmp_path / "unit.csv", tmp_path / "batch.csv"
    unit.write_text("unit_id\nU-1\nU-2\n", encoding="utf-8")
    batch.write_text("batch_id\nB-1\nB-2\n", encoding="utf-8")
    edges = write_csv(
        tmp_path, ["U-1,B-1,CONSUMED_BATCH,2026-09-20", "U-2,B-2,CONSUMED_BATCH,2026-09-21"]
    )
    added = await rag.apply_changes(added=sorted([str(unit), str(batch), str(edges)]))
    assert [entry.error for entry in added.added] == [None, None, None]
    assert added.added[1].result.metadata["relationships_written"] == 2
    assert await _consumed_edge_count(rag) == 2

    # Both change: the edge file now names a unit only the other file adds.
    unit.write_text("unit_id\nU-1\nU-2\nU-3\n", encoding="utf-8")
    write_csv(
        tmp_path,
        [
            "U-1,B-1,CONSUMED_BATCH,2026-09-20",
            "U-2,B-2,CONSUMED_BATCH,2026-09-21",
            "U-3,B-1,CONSUMED_BATCH,2026-09-22",
        ],
    )
    modified = await rag.apply_changes(modified=sorted([str(unit), str(edges)]))
    assert [entry.error for entry in modified.modified] == [None, None]
    assert modified.modified[0].result.metadata["relationships_written"] == 3
    assert await _consumed_edge_count(rag) == 3


async def _node_counts(rag):
    result = await rag._graph_store.query_raw(
        "MATCH (n) RETURN count(n), count(CASE WHEN n:Document THEN 1 END), "
        "count(CASE WHEN n:Chunk THEN 1 END)"
    )
    return result.result_set


@pytest.mark.integration
@pytest.mark.parametrize(
    "relationship_source, via_set_ontology",
    [("consumed_batch.csv", False), ("exports/consumed_batch.csv", True)],
)
async def test_a_relationship_declaration_replaces_a_proposed_node_table_on_real_falkordb(
    real_falkordb_rag_factory, llm, tmp_path, relationship_source, via_set_ontology
):
    """The file was ingested before anyone declared it, then declared as edges.

    get_ontology(), ingest() and drop_table() all raised that the source was
    registered under the other mapping kind.
    """
    entities = [Entity(label="Unit"), Entity(label="Batch")]
    rag = real_falkordb_rag_factory(
        llm=llm,
        resolver=None,
        ontology=Ontology(entities=entities, tables=_unit_and_batch_tables()),
    )
    for filename, content in (
        ("unit.csv", "unit_id\nU-1\nU-2\n"),
        ("batch.csv", "batch_id\nB-1\nB-2\n"),
    ):
        table = tmp_path / filename
        table.write_text(content, encoding="utf-8")
        await rag.ingest(str(table))
    before = await _node_counts(rag)
    path = write_csv(
        tmp_path, ["U-1,B-1,CONSUMED_BATCH,2026-09-20", "U-2,B-2,CONSUMED_BATCH,2026-09-21"]
    )
    await rag.ingest(str(path))
    stored = (await rag._ontology_store.load()).tables
    assert [(m.source, m.derived) for m in stored if m.signature == "consumed_batch"] == [
        ("consumed_batch.csv", True)
    ]
    assert await _node_counts(rag) != before

    declared = Ontology(
        entities=entities,
        tables=_unit_and_batch_tables(),
        relationship_tables=[consumed_mapping(source=relationship_source)],
    )
    if via_set_ontology:
        await rag.set_ontology(declared)
        edges = rag
    else:
        edges = real_falkordb_rag_factory(
            llm=llm, resolver=None, ontology=declared, connection=rag._conn
        )
    ontology = await edges.get_ontology()
    assert sorted(m.source for m in ontology.tables) == ["batch.csv", "unit.csv"]
    assert [m.source for m in ontology.relationship_tables] == [relationship_source]
    # The proposal's Document, record chunks and row went with it, and its values.
    assert await _node_counts(rag) == before
    signed = await rag._graph_store.query_raw(
        "MATCH (n) WHERE any(k IN keys(n) WHERE k STARTS WITH 'consumed_batch__') RETURN count(n)"
    )
    assert signed.result_set == [[0]]

    assert (await edges.ingest(str(path))).relationships_written == 2
    assert await _consumed_edge_count(rag) == 2
    assert await _node_counts(rag) == before
    await edges.drop_table(relationship_source)
    assert await _consumed_edge_count(rag) == 0


@pytest.mark.integration
async def test_a_proposal_whose_key_an_edge_table_uses_is_not_replaced_on_real_falkordb(
    real_falkordb_rag_factory, llm, tmp_path
):
    """Replacing a proposed node table removes its signed key from the nodes;
    when a relationship table resolves an endpoint by that key, the
    declaration is refused and nothing is removed."""
    from graphrag_sdk import TableMapping

    batch = TableMapping(source="batch.csv", label="Batch", key="batch_id", name="batch_id")
    base = Ontology(entities=[Entity(label="Batch")], tables=[batch])
    rag = real_falkordb_rag_factory(llm=llm, resolver=None, ontology=base)
    (tmp_path / "batch.csv").write_text("batch_id\nB-1\nB-2\n", encoding="utf-8")
    await rag.ingest(str(tmp_path / "batch.csv"))
    (tmp_path / "unit.csv").write_text("unit_id\nU-1\nU-2\n", encoding="utf-8")
    await rag.ingest(str(tmp_path / "unit.csv"))  # undeclared: stored as a proposal
    consumed = RelationshipMapping(
        source="consumed_batch.csv",
        start=EndpointMapping("unit", "unit__unit_id", "source"),
        end=EndpointMapping("Batch", "batch__batch_id", "target"),
        type="CONSUMED_BATCH",
    )
    edges = tmp_path / "consumed_batch.csv"
    edges.write_text("source,target\nU-1,B-1\nU-2,B-2\n", encoding="utf-8")
    with_edges = real_falkordb_rag_factory(
        llm=llm,
        resolver=None,
        connection=rag._conn,
        ontology=base.model_copy(update={"relationship_tables": [consumed]}),
    )
    assert (await with_edges.ingest(str(edges))).relationships_written == 2

    unit_as_edges = RelationshipMapping(
        source="unit.csv",
        start=EndpointMapping("Batch", "batch__batch_id", "unit_id"),
        end=EndpointMapping("Batch", "batch__batch_id", "unit_id"),
        type="SAME",
    )
    replacing = real_falkordb_rag_factory(
        llm=llm,
        resolver=None,
        connection=rag._conn,
        ontology=base.model_copy(update={"relationship_tables": [unit_as_edges]}),
    )
    with pytest.raises(
        ValueError,
        match=r"declaring 'unit\.csv' as a relationship table.*'consumed_batch\.csv' uses "
        r"unit\.unit__unit_id as its start endpoint key",
    ):
        await replacing.get_ontology()
    stored = await rag._ontology_store.load()
    assert [(m.source, m.derived) for m in stored.tables if m.source == "unit.csv"] == [
        ("unit.csv", True)
    ]
    assert [m.source for m in stored.relationship_tables] == ["consumed_batch.csv"]
    keys = {(e.label, p.name) for e in stored.entities for p in e.properties}
    assert ("unit", "unit__unit_id") in keys
    held = await rag._graph_store.query_raw(
        "MATCH (u:unit) WHERE u.unit__unit_id IS NOT NULL RETURN count(u)"
    )
    assert held.result_set == [[2]]
    assert await _consumed_edge_count(rag) == 2


@pytest.mark.integration
async def test_changing_a_tables_key_column_is_refused_under_a_relationship_table_on_real_falkordb(
    real_falkordb_rag_factory, llm, tmp_path
):
    """Redeclaring unit.csv with another key column takes unit__unit_id off
    every Unit; the relationship table keyed on it then could not register,
    after the values were already gone. Refused before anything is written."""
    from graphrag_sdk import TableMapping

    rag = real_falkordb_rag_factory(
        llm=llm,
        resolver=None,
        ontology=Ontology(
            entities=[Entity(label="Unit"), Entity(label="Batch")],
            tables=_unit_and_batch_tables(),
            relationship_tables=[consumed_mapping()],
        ),
    )
    await _ingest_units_batches_and_edges(rag, tmp_path)
    rekeyed = real_falkordb_rag_factory(
        llm=llm,
        resolver=None,
        connection=rag._conn,
        ontology=Ontology(
            entities=[Entity(label="Unit"), Entity(label="Batch")],
            tables=[
                TableMapping(source="unit.csv", label="Unit", key="code", name="code"),
                _unit_and_batch_tables()[1],
            ],
        ),
    )
    with pytest.raises(
        ValueError,
        match=r"redeclaring 'unit\.csv'.*'consumed_batch\.csv' uses Unit\.unit__unit_id "
        r"as its start endpoint key.*drop_table\('consumed_batch\.csv'\) first",
    ):
        await rekeyed.get_ontology()
    held = await rag._graph_store.query_raw(
        "MATCH (u:Unit) WHERE u.unit__unit_id IS NOT NULL RETURN count(u)"
    )
    assert held.result_set == [[2]]
    stored = await rag._ontology_store.load()
    assert [m.key for m in stored.tables if m.source == "unit.csv"] == ["unit_id"]
    assert await _consumed_edge_count(rag) == 2


@pytest.mark.integration
async def test_a_relationship_table_keyed_on_the_proposal_it_replaces_is_refused_on_real_falkordb(
    real_falkordb_rag_factory, llm, tmp_path
):
    """The incoming declaration is checked too: a relationship table that
    replaces a proposed node table but resolves an endpoint by that
    proposal's key would delete the key it needs."""
    from graphrag_sdk import TableMapping

    batch = TableMapping(source="batch.csv", label="Batch", key="batch_id", name="batch_id")
    base = Ontology(entities=[Entity(label="Batch")], tables=[batch])
    rag = real_falkordb_rag_factory(llm=llm, resolver=None, ontology=base)
    (tmp_path / "batch.csv").write_text("batch_id\nB-1\nB-2\n", encoding="utf-8")
    await rag.ingest(str(tmp_path / "batch.csv"))
    (tmp_path / "unit.csv").write_text("unit_id\nU-1\nU-2\n", encoding="utf-8")
    await rag.ingest(str(tmp_path / "unit.csv"))  # undeclared: stored as a proposal
    self_keyed = RelationshipMapping(
        source="unit.csv",
        start=EndpointMapping("unit", "unit__unit_id", "unit_id"),
        end=EndpointMapping("Batch", "batch__batch_id", "unit_id"),
        type="SAME",
    )
    replacing = real_falkordb_rag_factory(
        llm=llm,
        resolver=None,
        connection=rag._conn,
        ontology=base.model_copy(update={"relationship_tables": [self_keyed]}),
    )
    with pytest.raises(
        ValueError,
        match=r"declaring 'unit\.csv' as a relationship table.*'unit\.csv' uses "
        r"unit\.unit__unit_id as its start endpoint key",
    ):
        await replacing.get_ontology()
    stored = await rag._ontology_store.load()
    assert [(m.source, m.derived) for m in stored.tables if m.source == "unit.csv"] == [
        ("unit.csv", True)
    ]
    held = await rag._graph_store.query_raw(
        "MATCH (u:unit) WHERE u.unit__unit_id IS NOT NULL RETURN count(u)"
    )
    assert held.result_set == [[2]]


@pytest.mark.integration
async def test_relationship_table_edges_are_not_embedded_on_real_falkordb(
    real_falkordb_rag_factory, llm, tmp_path
):
    """Each table edge carries its mapping's one template fact and no endpoint
    names: embedded, they took relationship-search slots that retrieval then
    discarded. Only edges a document also supports are embedded."""
    rag = real_falkordb_rag_factory(
        llm=llm,
        resolver=None,
        ontology=Ontology(
            entities=[Entity(label="Unit"), Entity(label="Batch")],
            tables=_unit_and_batch_tables(),
            relationship_tables=[consumed_mapping()],
        ),
    )
    await _ingest_units_batches_and_edges(rag, tmp_path)
    # One of the two edges is also a document's fact.
    await rag._graph_store.query_raw(
        "MATCH (:Unit {unit__unit_id: 'U-2'})-[r:RELATES {rel_type:'CONSUMED_BATCH'}]->() "
        "SET r.source_chunk_ids = ['chunk-1']"
    )

    assert await rag._vector_store.embed_relationships() == 1

    embedded = await rag._graph_store.query_raw(
        "MATCH (u:Unit)-[r:RELATES {rel_type:'CONSUMED_BATCH'}]->() "
        "RETURN u.unit__unit_id, r.embedding IS NOT NULL ORDER BY u.unit__unit_id"
    )
    assert embedded.result_set == [["U-1", False], ["U-2", True]]


async def _units_batches_rag(real_falkordb_rag_factory, llm, tmp_path, mapping=None):
    rag = real_falkordb_rag_factory(
        llm=llm,
        resolver=None,
        ontology=Ontology(
            entities=[Entity(label="Unit"), Entity(label="Batch")],
            tables=_unit_and_batch_tables(),
            relationship_tables=[mapping or consumed_mapping()],
        ),
    )
    for filename, content in (
        ("unit.csv", "unit_id\nU-1\nU-2\n"),
        ("batch.csv", "batch_id\nB-1\nB-2\n"),
    ):
        (tmp_path / filename).write_text(content, encoding="utf-8")
        await rag.ingest(str(tmp_path / filename))
    return rag


@pytest.mark.integration
async def test_a_table_write_keeps_a_documents_fact_on_a_shared_edge_on_real_falkordb(
    real_falkordb_rag_factory, llm, tmp_path
):
    """The table's fact is one template per mapping; replacing a document's
    left the document's embedding on an edge whose fact no longer said it."""
    rag = await _units_batches_rag(real_falkordb_rag_factory, llm, tmp_path)
    await rag._graph_store.query_raw(
        "MATCH (u:Unit {unit__unit_id:'U-1'}), (b:Batch {batch__batch_id:'B-1'}) "
        "CREATE (u)-[:RELATES {rel_type:'CONSUMED_BATCH', fact:'U-1 used batch B-1 on Monday', "
        "source_chunk_ids:['chunk-1']}]->(b)"
    )
    path = write_csv(
        tmp_path, ["U-1,B-1,CONSUMED_BATCH,2026-09-20", "U-2,B-2,CONSUMED_BATCH,2026-09-21"]
    )
    assert (await rag.ingest(str(path))).relationships_written == 2
    facts = await rag._graph_store.query_raw(
        "MATCH (u:Unit)-[r:RELATES {rel_type:'CONSUMED_BATCH'}]->() "
        "RETURN u.unit__unit_id, r.fact ORDER BY u.unit__unit_id"
    )
    assert facts.result_set == [
        ["U-1", "U-1 used batch B-1 on Monday"],
        ["U-2", "(Unit, CONSUMED_BATCH, Batch)"],
    ]


@pytest.mark.integration
async def test_an_edge_left_to_a_table_loses_its_embedding_on_real_falkordb(
    real_falkordb_rag_factory, llm, tmp_path
):
    """When a document's cleanup leaves an embedded edge with table ownership
    only, its vector goes: such edges are not embedded, and the old vector
    kept taking relationship-search slots."""
    rag = await _units_batches_rag(real_falkordb_rag_factory, llm, tmp_path)
    path = write_csv(tmp_path, ["U-1,B-1,CONSUMED_BATCH,2026-09-20"])
    await rag.ingest(str(path))
    await rag._graph_store.query_raw(
        "MATCH ()-[r:RELATES {rel_type:'CONSUMED_BATCH'}]->() "
        "SET r.source_chunk_ids = ['chunk-1'], r.embedding = vecf32($v)",
        {"v": [0.1] * rag._vector_store.embedding_dimension},
    )
    unit = await rag._graph_store.query_raw("MATCH (u:Unit {unit__unit_id:'U-1'}) RETURN u.id")

    await rag._graph_store.delete_stale_relationships([unit.result_set[0][0]], ["chunk-1"])

    left = await rag._graph_store.query_raw(
        "MATCH ()-[r:RELATES {rel_type:'CONSUMED_BATCH'}]->() "
        "RETURN size(r.source_chunk_ids), r.embedding IS NULL"
    )
    assert left.result_set == [[0, True]]


@pytest.mark.integration
async def test_a_dropped_column_is_cleared_in_bounded_writes_and_published_on_real_falkordb(
    real_falkordb_rag_factory, llm, tmp_path
):
    """Dropping a mapping column cleared it in one write over every edge, and
    the running instance still offered it to text-to-Cypher."""
    with_shift = consumed_mapping(
        properties={"consumed_on": Column("consumed_on", "DATE"), "shift": "shift"}
    )
    rag = await _units_batches_rag(real_falkordb_rag_factory, llm, tmp_path, with_shift)
    path = tmp_path / "consumed_batch.csv"
    path.write_text(
        "source,target,type,consumed_on,shift\n"
        "U-1,B-1,CONSUMED_BATCH,2026-09-20,night\nU-2,B-2,CONSUMED_BATCH,2026-09-21,day\n",
        encoding="utf-8",
    )
    await rag.ingest(str(path))
    without_shift = real_falkordb_rag_factory(
        llm=llm,
        resolver=None,
        connection=rag._conn,
        ontology=Ontology(
            entities=[Entity(label="Unit"), Entity(label="Batch")],
            tables=_unit_and_batch_tables(),
            relationship_tables=[consumed_mapping()],
        ),
    )
    await without_shift.get_ontology()
    without_shift._graph_store._RELATIONSHIP_CLEANUP_BATCH = 1
    watched = _Watched(without_shift._graph_store._conn)
    without_shift._graph_store._conn = watched

    await without_shift.ingest(str(path))

    cleared = await rag._graph_store.query_raw(
        "MATCH ()-[r:RELATES {rel_type:'CONSUMED_BATCH'}]->() "
        "WHERE r.consumed_batch__shift IS NOT NULL RETURN count(r)"
    )
    assert cleared.result_set == [[0]]
    writes = [c for c, _ in watched.sent if "REMOVE r.`consumed_batch__shift`" in c]
    assert len(writes) >= 2, "one bounded write per batch, not one over every edge"
    published = {
        (relation.label, prop.name)
        for relation in without_shift._global_ontology.relations
        for prop in relation.properties
    }
    assert ("CONSUMED_BATCH", "consumed_batch__shift") not in published


@pytest.mark.integration
async def test_two_nodes_sharing_an_id_are_an_ambiguous_endpoint_on_real_falkordb(
    real_falkordb_rag_factory, llm, tmp_path
):
    """Nothing makes entity ids unique; two nodes with one id and one key were
    one candidate, so the default 'error' policy accepted the row and the
    write matched both nodes."""
    rag = await _units_batches_rag(real_falkordb_rag_factory, llm, tmp_path)
    await rag._graph_store.query_raw(
        "MATCH (u:Unit {unit__unit_id:'U-1'}) "
        "CREATE (:__Entity__:Unit {id: u.id, unit__unit_id: 'U-1', name: 'U-1'})"
    )
    path = write_csv(tmp_path, ["U-1,B-1,CONSUMED_BATCH,2026-09-20"])
    with pytest.raises(RelationshipResolutionError):
        await rag.ingest(str(path))
    assert await _consumed_edge_count(rag) == 0


@pytest.mark.integration
async def test_cleanup_of_many_edges_from_one_node_on_real_falkordb(
    real_falkordb_rag_factory, llm, tmp_path
):
    """Cleanup found each stale edge by its start node only, so every row
    walked all of that node's edges: 20,000 edges from one unit took 38 s
    per cleanup and a re-ingest timed out. This pins the result on such a
    hub; the query binds both ends (see test_graph_store)."""
    n = 3_000
    rag = real_falkordb_rag_factory(
        llm=llm,
        resolver=None,
        ontology=Ontology(
            entities=[Entity(label="Unit"), Entity(label="Batch")],
            tables=_unit_and_batch_tables(),
            relationship_tables=[consumed_mapping()],
        ),
    )
    (tmp_path / "unit.csv").write_text("unit_id\nU-1\n", encoding="utf-8")
    await rag.ingest(str(tmp_path / "unit.csv"))
    (tmp_path / "batch.csv").write_text(
        "batch_id\n" + "\n".join(f"B-{i}" for i in range(n)) + "\n", encoding="utf-8"
    )
    await rag.ingest(str(tmp_path / "batch.csv"))
    rows = [f"U-1,B-{i},CONSUMED_BATCH,2026-09-20" for i in range(n)]
    assert (await rag.ingest(str(write_csv(tmp_path, rows)))).relationships_written == n

    await rag.ingest(str(write_csv(tmp_path, rows[: n // 2])))
    assert await _consumed_edge_count(rag) == n // 2
    kept = await rag._graph_store.query_raw(
        "MATCH ()-[r:RELATES {rel_type:'CONSUMED_BATCH'}]->(b) "
        "RETURN min(b.batch__batch_id), count(DISTINCT b)"
    )
    assert kept.result_set == [["B-0", n // 2]]

    await rag.drop_table("consumed_batch.csv")
    assert await _consumed_edge_count(rag) == 0


@pytest.mark.integration
async def test_a_declared_node_table_is_not_replaced_by_a_relationship_declaration_on_real_falkordb(
    real_falkordb_rag_factory, llm, tmp_path
):
    from graphrag_sdk import TableMapping
    from graphrag_sdk.storage.ontology_store import OntologyContradictionError

    entities = [Entity(label="Unit"), Entity(label="Batch"), Entity(label="Consumption")]
    node_table = TableMapping(source="consumed_batch.csv", label="Consumption", key="source")
    rag = real_falkordb_rag_factory(
        llm=llm,
        resolver=None,
        ontology=Ontology(entities=entities, tables=[*_unit_and_batch_tables(), node_table]),
    )
    for filename, content in (
        ("unit.csv", "unit_id\nU-1\nU-2\n"),
        ("batch.csv", "batch_id\nB-1\nB-2\n"),
    ):
        table = tmp_path / filename
        table.write_text(content, encoding="utf-8")
        await rag.ingest(str(table))
    await rag.ingest(str(write_csv(tmp_path, ["U-1,B-1,CONSUMED_BATCH,2026-09-20"])))
    before = await _node_counts(rag)
    edges = real_falkordb_rag_factory(
        llm=llm,
        resolver=None,
        connection=rag._conn,
        ontology=Ontology(
            entities=entities,
            tables=_unit_and_batch_tables(),
            relationship_tables=[consumed_mapping()],
        ),
    )
    with pytest.raises(OntologyContradictionError, match="other structured mapping kind"):
        await edges.get_ontology()
    assert await _node_counts(rag) == before
    stored = (await rag._ontology_store.load()).tables
    assert [(m.source, m.derived) for m in stored if m.signature == "consumed_batch"] == [
        ("consumed_batch.csv", False)
    ]


class _Watched:
    """Delegate to a connection, recording every query sent through it."""

    def __init__(self, conn, on_query=None) -> None:
        self._conn = conn
        self._on_query = on_query
        self.sent: list[tuple[str, Any]] = []

    async def query(self, cypher, params=None, **kwargs):
        self.sent.append((cypher, params))
        if self._on_query is not None:
            await self._on_query(cypher)
        return await self._conn.query(cypher, params, **kwargs)

    def __getattr__(self, name):
        return getattr(self._conn, name)


@pytest.mark.integration
async def test_a_new_key_index_is_built_before_the_first_lookup_on_real_falkordb(
    real_falkordb_rag_factory, llm
):
    """Entities Cypher wrote have no key index, and the first lookup created it.

    That lookup ran while FalkorDB was still building the index, so it scanned
    the label once per key: on 1M nodes the first ingest timed out with
    nothing written.
    """
    from graphrag_sdk.storage.graph_store import GraphStore

    rag = real_falkordb_rag_factory(llm=llm, resolver=None)
    await rag._conn.query(
        "UNWIND range(0, 99999) AS i "
        "CREATE (:Unit:__Entity__ {id: 'unit-' + toString(i), code: 'U-' + toString(i)})"
    )
    statuses: list[list[str]] = []

    async def note_status(cypher):
        if "UNWIND $keys AS k" in cypher:
            listed = await rag._conn.query(
                "CALL db.indexes() YIELD label, status WHERE label = 'Unit' RETURN status"
            )
            statuses.append([row[0] for row in listed.result_set])

    keys = [f"U-{index}" for index in range(0, 100_000, 200)]
    found = await GraphStore(_Watched(rag._conn, note_status)).resolve_by_property(
        "Unit", "code", keys
    )
    assert statuses == [["OPERATIONAL"]]
    assert found == {key: [f"unit-{key[2:]}"] for key in keys}


@pytest.mark.integration
async def test_an_empty_list_cell_clears_the_edges_list_on_real_falkordb(
    real_falkordb_rag_factory, llm, tmp_path
):
    """A LIST cell with no items ("," parses to no items) clears the list an
    earlier run wrote; it used to be dropped from the write and kept."""
    mapping = consumed_mapping(properties={"tags": Column("tags", "LIST")})
    rag = real_falkordb_rag_factory(
        llm=llm,
        resolver=None,
        ontology=Ontology(
            entities=[Entity(label="Unit"), Entity(label="Batch")],
            tables=_unit_and_batch_tables(),
            relationship_tables=[mapping],
        ),
    )
    for filename, content in (("unit.csv", "unit_id\nU-1\n"), ("batch.csv", "batch_id\nB-1\n")):
        (tmp_path / filename).write_text(content, encoding="utf-8")
        await rag.ingest(str(tmp_path / filename))
    path = tmp_path / "consumed_batch.csv"

    async def tags(cell: str):
        path.write_text(
            f"source,target,type,tags\nU-1,B-1,CONSUMED_BATCH,{cell}\n", encoding="utf-8"
        )
        assert (await rag.ingest(str(path))).relationships_written == 1
        result = await rag._graph_store.query_raw(
            "MATCH ()-[r:RELATES {rel_type:'CONSUMED_BATCH'}]->() RETURN r.consumed_batch__tags"
        )
        return result.result_set

    assert await tags('"a,b"') == [[["a", "b"]]]
    assert await tags('","') == [[None]]


@pytest.mark.integration
async def test_blank_cells_keep_each_batch_in_one_write_on_real_falkordb(
    real_falkordb_rag_factory, llm, tmp_path
):
    """A blank cell clears the edge's value through null, in the batch's one write.

    A REMOVE clause per combination of blank cells split each 500-row batch
    into up to one query per row.
    """
    mapping = consumed_mapping(
        properties={"consumed_on": Column("consumed_on", "DATE"), "shift": "shift"}
    )
    rag = real_falkordb_rag_factory(
        llm=llm,
        resolver=None,
        ontology=Ontology(
            entities=[Entity(label="Unit"), Entity(label="Batch")],
            tables=_unit_and_batch_tables(),
            relationship_tables=[mapping],
        ),
    )
    for filename, content in (
        ("unit.csv", "unit_id\nU-1\nU-2\n"),
        ("batch.csv", "batch_id\nB-1\nB-2\n"),
    ):
        table = tmp_path / filename
        table.write_text(content, encoding="utf-8")
        await rag.ingest(str(table))
    path = tmp_path / "consumed_batch.csv"
    watched = _Watched(rag._graph_store._conn)
    rag._graph_store._conn = watched

    async def ingest(rows):
        path.write_text(
            "source,target,type,consumed_on,shift\n" + "\n".join(rows) + "\n", encoding="utf-8"
        )
        watched.sent.clear()
        assert (await rag.ingest(str(path))).relationships_written == len(rows)
        writes = [cypher for cypher, _ in watched.sent if "UNWIND $batch AS item" in cypher]
        assert len(writes) == 1
        edges = await rag._graph_store.query_raw(
            "MATCH (u:Unit)-[r:RELATES {rel_type:'CONSUMED_BATCH'}]->(b:Batch) "
            "RETURN u.unit__unit_id, b.batch__batch_id, "
            "r.consumed_batch__consumed_on, r.consumed_batch__shift "
            "ORDER BY u.unit__unit_id, b.batch__batch_id"
        )
        return edges.result_set

    assert await ingest(
        [
            "U-1,B-1,CONSUMED_BATCH,2026-09-20,",
            "U-1,B-2,CONSUMED_BATCH,,night",
            "U-2,B-1,CONSUMED_BATCH,,",
            "U-2,B-2,CONSUMED_BATCH,2026-09-21,day",
        ]
    ) == [
        ["U-1", "B-1", "2026-09-20", None],
        ["U-1", "B-2", None, "night"],
        ["U-2", "B-1", None, None],
        ["U-2", "B-2", "2026-09-21", "day"],
    ]
    assert await ingest(
        [
            "U-1,B-1,CONSUMED_BATCH,,day",
            "U-1,B-2,CONSUMED_BATCH,2026-09-22,",
            "U-2,B-1,CONSUMED_BATCH,2026-09-23,night",
            "U-2,B-2,CONSUMED_BATCH,,",
        ]
    ) == [
        ["U-1", "B-1", None, "day"],
        ["U-1", "B-2", "2026-09-22", None],
        ["U-2", "B-1", "2026-09-23", "night"],
        ["U-2", "B-2", None, None],
    ]


@pytest.mark.integration
async def test_cleanup_reads_only_the_tables_own_edges_in_chunks_on_real_falkordb(
    real_falkordb_rag_factory, llm, tmp_path, monkeypatch
):
    """Snapshot and cleanup queries read every RELATES edge in the graph,
    document edges included, 3 + one per property times per ingest, and changed
    every stale edge in one write. They now go through one index shared by all
    relationship tables, a bounded number of edges per write."""
    from graphrag_sdk.storage.graph_store import GraphStore

    monkeypatch.setattr(GraphStore, "_RELATIONSHIP_CLEANUP_BATCH", 1)
    rag = real_falkordb_rag_factory(
        llm=llm,
        resolver=None,
        ontology=Ontology(
            entities=[Entity(label="Unit"), Entity(label="Batch")],
            tables=_unit_and_batch_tables(),
            relationship_tables=[consumed_mapping()],
        ),
    )
    watched = _Watched(rag._graph_store._conn)
    rag._graph_store._conn = watched
    path = await _ingest_units_batches_and_edges(rag, tmp_path)
    # A document's edge, and a document chunk that also supports U-2 -> B-2.
    await rag._graph_store.query_raw(
        "MATCH (u:Unit {unit__unit_id: 'U-1'}), (b:Batch {batch__batch_id: 'B-1'}) "
        "CREATE (u)-[:RELATES {rel_type: 'MENTIONED_WITH', source_chunk_ids: ['c1']}]->(b)"
    )
    await rag._graph_store.query_raw(
        "MATCH ()-[r:RELATES {rel_type: 'CONSUMED_BATCH'}]->(:Batch {batch__batch_id: 'B-2'}) "
        "SET r.source_chunk_ids = ['c2']"
    )
    watched.sent.clear()
    write_csv(tmp_path, ["U-1,B-1,CONSUMED_BATCH,2026-09-20"])
    assert (await rag.ingest(str(path))).relationships_deleted == 1
    await rag.drop_table("consumed_batch.csv")
    reads = [
        (cypher, params)
        for cypher, params in watched.sent
        if "structured_sources" in cypher and "UNWIND" not in cypher
    ]
    changes = [(cypher, params) for cypher, params in watched.sent if "UNWIND $edges" in cypher]

    edges = await rag._graph_store.query_raw(
        "MATCH (u:Unit)-[r:RELATES]->(b:Batch) "
        "RETURN u.unit__unit_id, b.batch__batch_id, r.rel_type, r.source_chunk_ids, "
        "r.structured_sources, r.consumed_batch__consumed_on, "
        "r.relationship_mapping_consumed_batch ORDER BY r.rel_type, u.unit__unit_id"
    )
    assert edges.result_set == [
        ["U-2", "B-2", "CONSUMED_BATCH", ["c2"], [], None, None],
        ["U-1", "B-1", "MENTIONED_WITH", ["c1"], None, None, None],
    ]
    # The run's snapshot number and its cleanup's reads, then drop_table's.
    assert len(reads) >= 4 and len(changes) == 2, "one edge per change"
    for cypher, params in reads:
        plan = str(await rag._conn._graph.explain(cypher, params))
        assert "Edge By Index Scan" in plan, (cypher, plan)
    for cypher, params in changes:
        # FalkorDB 6.0.2 crashed setting structured_sources on edges a query
        # was still reading through that index; the change goes by node id.
        plan = str(await rag._conn._graph.explain(cypher, params))
        assert "NodeByIdSeek" in plan and "Index Scan" not in plan, (cypher, plan)


@pytest.mark.integration
async def test_a_shared_edge_keeps_every_tables_ownership_on_real_falkordb(
    real_falkordb_rag_factory, llm, tmp_path
):
    """Two relationship tables list U-1 -> B-1; one dropping the row keeps the edge.

    The second table is written through the per-row fallback a failed batch
    write takes, which has to union ownership as the batch write does.
    """
    rag = real_falkordb_rag_factory(
        llm=llm,
        resolver=None,
        ontology=Ontology(
            entities=[Entity(label="Unit"), Entity(label="Batch")],
            tables=_unit_and_batch_tables(),
            relationship_tables=[
                consumed_mapping(source="consumed_a.csv"),
                consumed_mapping(source="consumed_b.csv"),
            ],
        ),
    )
    for filename, content in (
        ("unit.csv", "unit_id\nU-1\nU-2\n"),
        ("batch.csv", "batch_id\nB-1\nB-2\n"),
        ("consumed_a.csv", "source,target,consumed_on\nU-1,B-1,2026-01-01\nU-2,B-2,2026-01-02\n"),
    ):
        (tmp_path / filename).write_text(content, encoding="utf-8")
        await rag.ingest(str(tmp_path / filename))

    async def lose_the_batch_write(cypher):
        if "UNWIND $batch AS item" in cypher and "MERGE (a)-[r:`RELATES`" in cypher:
            raise ConnectionError("batch write lost")

    conn = rag._graph_store._conn
    watched = _Watched(conn, on_query=lose_the_batch_write)
    rag._graph_store._conn = watched
    (tmp_path / "consumed_b.csv").write_text(
        "source,target,consumed_on\nU-1,B-1,2026-02-02\n", encoding="utf-8"
    )
    result = await rag.ingest(str(tmp_path / "consumed_b.csv"))
    rag._graph_store._conn = conn
    assert (result.relationships_written, result.incomplete_writes) == (1, [])
    assert any("{id: $start_id}" in cypher for cypher, _ in watched.sent), "per-row fallback ran"

    async def edges():
        rows = await rag._graph_store.query_raw(
            "MATCH (u:Unit)-[r:RELATES {rel_type:'CONSUMED_BATCH'}]->(b:Batch) "
            "RETURN u.unit__unit_id, b.batch__batch_id, r.structured_sources, "
            "r.consumed_a__consumed_on, r.consumed_b__consumed_on "
            "ORDER BY u.unit__unit_id"
        )
        return rows.result_set

    assert await edges() == [
        ["U-1", "B-1", ["consumed_a", "consumed_b"], "2026-01-01", "2026-02-02"],
        ["U-2", "B-2", ["consumed_a"], "2026-01-02", None],
    ]
    # consumed_a.csv drops U-1 -> B-1, which consumed_b.csv still lists.
    (tmp_path / "consumed_a.csv").write_text(
        "source,target,consumed_on\nU-2,B-2,2026-01-02\n", encoding="utf-8"
    )
    await rag.ingest(str(tmp_path / "consumed_a.csv"))
    assert await edges() == [
        ["U-1", "B-1", ["consumed_b"], None, "2026-02-02"],
        ["U-2", "B-2", ["consumed_a"], "2026-01-02", None],
    ]
    # Once no table lists it, it goes.
    await rag.drop_table("consumed_b.csv")
    assert await edges() == [["U-2", "B-2", ["consumed_a"], "2026-01-02", None]]


@pytest.mark.integration
async def test_document_cleanup_keeps_edges_a_relationship_table_lists_on_real_falkordb(
    real_falkordb_rag_factory, llm, tmp_path
):
    """Deleting a document's chunks deletes an edge only when no table lists it.

    A table-only edge has ``source_chunk_ids = []``, so any document cleanup
    touching one of its endpoints deleted it, as well as an edge the table and
    the document both supported.
    """
    rag = real_falkordb_rag_factory(
        llm=llm,
        resolver=None,
        ontology=Ontology(
            entities=[Entity(label="Unit"), Entity(label="Batch")],
            tables=_unit_and_batch_tables(),
            relationship_tables=[consumed_mapping()],
        ),
    )
    await _ingest_units_batches_and_edges(rag, tmp_path)
    # A document chunk also supports U-1 -> B-1, and wrote an edge of its own.
    await rag._graph_store.query_raw(
        "MATCH ()-[r:RELATES {rel_type:'CONSUMED_BATCH'}]->(:Batch {batch__batch_id: 'B-1'}) "
        "SET r.source_chunk_ids = ['chunk-1']"
    )
    await rag._graph_store.query_raw(
        "MATCH (u:Unit {unit__unit_id: 'U-2'}), (b:Batch {batch__batch_id: 'B-1'}) "
        "CREATE (u)-[:RELATES {rel_type: 'MENTIONED_WITH', source_chunk_ids: ['chunk-1']}]->(b)"
    )
    units = await rag._graph_store.query_raw("MATCH (u:Unit) RETURN u.id")
    deleted = await rag._graph_store.delete_stale_relationships(
        [row[0] for row in units.result_set], ["chunk-1"]
    )
    assert deleted == 1, "only the document's own edge"
    edges = await rag._graph_store.query_raw(
        "MATCH (u:Unit)-[r:RELATES]->(b:Batch) "
        "RETURN u.unit__unit_id, b.batch__batch_id, r.rel_type, r.source_chunk_ids, "
        "r.structured_sources ORDER BY u.unit__unit_id"
    )
    assert edges.result_set == [
        ["U-1", "B-1", "CONSUMED_BATCH", [], ["consumed_batch"]],
        ["U-2", "B-2", "CONSUMED_BATCH", [], ["consumed_batch"]],
    ]


def consumed_mapping(**changes: Any) -> RelationshipMapping:
    values: dict[str, Any] = {
        "source": "consumed_batch.csv",
        "start": EndpointMapping("Unit", "unit__unit_id", "source"),
        "end": EndpointMapping("Batch", "batch__batch_id", "target"),
        "type": "CONSUMED_BATCH",
        "properties": {"consumed_on": Column("consumed_on", "DATE")},
    }
    values.update(changes)
    return RelationshipMapping(**values)


class RelationshipStore:
    _BATCH_SIZE = 500

    def __init__(self) -> None:
        self.entities = {
            ("Unit", "unit__unit_id"): {"U-1": ["unit-1"], "U-2": ["unit-2"]},
            ("Batch", "batch__batch_id"): {"B-1": ["batch-1"], "B-2": ["batch-2"]},
        }
        self.resolve_sizes: list[int] = []
        self.relationships: dict[tuple[str, str, str], Any] = {}
        self.write_calls = 0
        self.snapshots = 0
        self.snapshot_properties: list[str | None] = []

    async def resolve_by_property(self, label: str, prop: str, keys: list[str]):
        self.resolve_sizes.append(len(keys))
        values = self.entities.get((label, prop), {})
        return {key: values[key] for key in dict.fromkeys(keys) if key in values}

    async def next_relationship_snapshot(self, marker_property: str, signature: str) -> int:
        self.snapshots += 1
        return self.snapshots

    async def upsert_relationships(self, relationships, *, snapshot_property=None):
        self.write_calls += 1
        self.snapshot_properties.append(snapshot_property)
        for relationship in relationships:
            key = (
                relationship.start_node_id,
                relationship.end_node_id,
                relationship.properties["rel_type"],
            )
            self.relationships[key] = relationship
        return len(relationships)

    async def finalize_relationship_snapshot(
        self,
        signature: str,
        marker_property: str,
        snapshot_token: int,
        signed_properties: list[str],
    ):
        stale = [
            key
            for key, relationship in self.relationships.items()
            if signature in relationship.properties.get("structured_sources", [])
            and relationship.properties.get(marker_property, 0) < snapshot_token
        ]
        for key in stale:
            del self.relationships[key]
        return len(stale)


def write_csv(tmp_path, rows: list[str]):
    path = tmp_path / "consumed_batch.csv"
    path.write_text(
        "source,target,type,consumed_on\n" + "\n".join(rows) + "\n",
        encoding="utf-8",
    )
    return path


class TestRelationshipMappingModel:
    def test_a_blank_fixed_type_does_not_bypass_allowed_types(self):
        """type="" passed validation as "no fixed type" but was read back as
        one, so every row got the empty type and allowed_types was skipped."""
        mapping = consumed_mapping(
            type="", type_column="type", allowed_types=["CONSUMED_BATCH"], properties={}
        )
        assert mapping.type is None
        assert mapping.relationship_types == ["CONSUMED_BATCH"]
        assert mapping.type_for({"type": "CONSUMED_BATCH"}) == "CONSUMED_BATCH"
        with pytest.raises(MappingError, match="expected one of CONSUMED_BATCH"):
            mapping.type_for({"type": "SOMETHING_ELSE"})
        with pytest.raises(MappingError, match="exactly one of type= or type_column="):
            consumed_mapping(type="  ", properties={})

    def test_json_round_trip(self):
        ontology = Ontology(relationship_tables=[consumed_mapping()])
        restored = Ontology.model_validate_json(ontology.model_dump_json())
        assert restored.relationship_tables == ontology.relationship_tables

    def test_ontology_merge_preserves_relationship_mappings(self):
        merged = Ontology().merge(Ontology(relationship_tables=[consumed_mapping()]))
        assert merged.relationship_tables == [consumed_mapping()]

    def test_requires_exactly_one_type_source(self):
        with pytest.raises(MappingError, match="exactly one"):
            consumed_mapping(type=None)
        with pytest.raises(MappingError, match="exactly one"):
            consumed_mapping(type_column="type")

    def test_dynamic_type_is_allow_listed(self):
        mapping = consumed_mapping(
            type=None,
            type_column="type",
            allowed_types=["CONSUMED_BATCH", "RETURNED_BATCH"],
        )
        assert mapping.type_for({"type": "RETURNED_BATCH"}) == "RETURNED_BATCH"
        with pytest.raises(MappingError, match="expected one of"):
            mapping.type_for({"type": "DELETES"})

    def test_relationship_ontology_has_direction_and_typed_signed_properties(self):
        mapping = consumed_mapping(direction="INCOMING")
        ontology = ontology_for_relationship(mapping)
        relation = ontology.relations[0]
        assert relation.patterns == [("Batch", "Unit")]
        assert [(prop.name, prop.type) for prop in relation.properties] == [
            ("consumed_batch__consumed_on", "DATE")
        ]

    def test_duplicate_source_across_mapping_kinds_is_rejected(self):
        from graphrag_sdk import TableMapping

        with pytest.raises(ValueError, match="declared twice"):
            Ontology(
                entities=[],
                tables=[
                    TableMapping(
                        source="consumed_batch.csv",
                        label="Event",
                        key="source",
                        standalone=True,
                    )
                ],
                relationship_tables=[consumed_mapping()],
            )

    def test_endpoint_keys_must_exist_in_registered_schema(self):
        properties = {
            entity.label: {prop.name: prop.type for prop in entity.properties}
            for entity in [
                Entity(
                    label="Unit",
                    properties=[Attribute(name="unit__unit_id", type="STRING")],
                ),
                Entity(label="Batch"),
            ]
        }
        assert GraphRAG._relationship_endpoint_schema_problems(consumed_mapping(), properties) == [
            "end endpoint key Batch.batch__batch_id is not a declared ontology property"
        ]


class TestRelationshipIngestion:
    async def test_writes_typed_relationship_without_nodes(self, tmp_path):
        path = write_csv(tmp_path, ["U-1,B-1,CONSUMED_BATCH,2026-09-20"])
        store = RelationshipStore()
        result = await RelationshipIngestionPipeline(CsvRecordLoader(), store).run(
            str(path), consumed_mapping(), Context()
        )
        assert result.as_dict()["relationships_written"] == 1
        relationship = next(iter(store.relationships.values()))
        assert relationship.start_node_id == "unit-1"
        assert relationship.end_node_id == "batch-1"
        assert relationship.properties["consumed_batch__consumed_on"] == "2026-09-20"
        assert relationship.properties["structured_sources"] == ["consumed_batch"]

    async def test_incoming_direction_reverses_written_endpoints(self, tmp_path):
        path = write_csv(tmp_path, ["U-1,B-1,CONSUMED_BATCH,2026-09-20"])
        store = RelationshipStore()
        await RelationshipIngestionPipeline(CsvRecordLoader(), store).run(
            str(path), consumed_mapping(direction="INCOMING"), Context()
        )
        relationship = next(iter(store.relationships.values()))
        assert relationship.start_node_id == "batch-1"
        assert relationship.end_node_id == "unit-1"

    async def test_blank_property_clears_prior_value(self, tmp_path):
        path = write_csv(tmp_path, ["U-1,B-1,CONSUMED_BATCH,"])
        store = RelationshipStore()
        await RelationshipIngestionPipeline(CsvRecordLoader(), store).run(
            str(path), consumed_mapping(), Context()
        )
        relationship = next(iter(store.relationships.values()))
        assert relationship.properties["consumed_batch__consumed_on"] is None

    async def test_invalid_dynamic_type_fails_before_resolution_or_writes(self, tmp_path):
        path = write_csv(tmp_path, ["U-1,B-1,DELETES,2026-09-20"])
        store = RelationshipStore()
        with pytest.raises(MappingError, match="row 1.*expected one of"):
            await RelationshipIngestionPipeline(CsvRecordLoader(), store).run(
                str(path),
                consumed_mapping(
                    type=None,
                    type_column="type",
                    allowed_types=["CONSUMED_BATCH"],
                ),
                Context(),
            )
        assert store.resolve_sizes == []
        assert store.write_calls == 0

    async def test_retry_and_reordered_rows_are_idempotent(self, tmp_path):
        first = write_csv(
            tmp_path,
            [
                "U-1,B-1,CONSUMED_BATCH,2026-09-20",
                "U-2,B-2,CONSUMED_BATCH,2026-09-21",
            ],
        )
        store = RelationshipStore()
        pipeline = RelationshipIngestionPipeline(CsvRecordLoader(), store)
        await pipeline.run(str(first), consumed_mapping(), Context())
        second = write_csv(
            tmp_path,
            [
                "U-2,B-2,CONSUMED_BATCH,2026-09-21",
                "U-1,B-1,CONSUMED_BATCH,2026-09-20",
            ],
        )
        await pipeline.run(str(second), consumed_mapping(), Context())
        assert len(store.relationships) == 2

    async def test_reingest_removes_relationships_absent_from_new_snapshot(self, tmp_path):
        path = write_csv(
            tmp_path,
            [
                "U-1,B-1,CONSUMED_BATCH,2026-09-20",
                "U-2,B-2,CONSUMED_BATCH,2026-09-21",
            ],
        )
        store = RelationshipStore()
        pipeline = RelationshipIngestionPipeline(CsvRecordLoader(), store)
        await pipeline.run(str(path), consumed_mapping(), Context())
        write_csv(tmp_path, ["U-1,B-1,CONSUMED_BATCH,2026-09-20"])
        result = await pipeline.run(str(path), consumed_mapping(), Context())
        assert len(store.relationships) == 1
        assert result.relationships_deleted == 1

    async def test_rows_carry_the_runs_snapshot_number_from_the_store(self, tmp_path):
        """The number orders runs, and the store is told which property holds it."""
        path = write_csv(tmp_path, ["U-1,B-1,CONSUMED_BATCH,2026-09-20"])
        store = RelationshipStore()
        pipeline = RelationshipIngestionPipeline(CsvRecordLoader(), store)
        await pipeline.run(str(path), consumed_mapping(), Context())
        await pipeline.run(str(path), consumed_mapping(), Context())
        relationship = next(iter(store.relationships.values()))
        assert relationship.properties["relationship_mapping_consumed_batch"] == 2
        assert store.snapshot_properties == ["relationship_mapping_consumed_batch"] * 2

    async def test_conflicting_duplicate_is_rejected_before_writes(self, tmp_path):
        path = write_csv(
            tmp_path,
            [
                "U-1,B-1,CONSUMED_BATCH,2026-09-20",
                "U-1,B-1,CONSUMED_BATCH,2026-09-21",
            ],
        )
        store = RelationshipStore()
        with pytest.raises(MappingError, match="order-dependent"):
            await RelationshipIngestionPipeline(CsvRecordLoader(), store).run(
                str(path), consumed_mapping(), Context()
            )
        assert store.write_calls == 0

    @pytest.mark.parametrize(
        "start_key, policy",
        [
            ("", {"missing_endpoint": "skip"}),
            ("U-404", {"missing_endpoint": "skip"}),
            ("U-2", {"ambiguous_endpoint": "skip"}),
        ],
    )
    async def test_conflicting_duplicates_a_skip_policy_drops_do_not_fail_the_file(
        self, tmp_path, start_key, policy
    ):
        """Rows that are skipped are never written, so their order cannot matter."""
        path = write_csv(
            tmp_path,
            [
                "U-1,B-1,CONSUMED_BATCH,2026-01-01",
                f"{start_key},B-2,CONSUMED_BATCH,2026-01-02",
                f"{start_key},B-2,CONSUMED_BATCH,2026-01-03",
            ],
        )
        store = RelationshipStore()
        store.entities[("Unit", "unit__unit_id")]["U-2"] = ["unit-2a", "unit-2b"]
        result = await RelationshipIngestionPipeline(CsvRecordLoader(), store).run(
            str(path), consumed_mapping(**policy), Context()
        )
        assert result.relationships_written == 1
        assert (result.rows, result.duplicate_rows, result.rows_skipped) == (3, 1, 1)
        relationship = next(iter(store.relationships.values()))
        assert relationship.properties["consumed_batch__consumed_on"] == "2026-01-01"

    async def test_conflicting_duplicates_with_a_missing_endpoint_report_resolution(self, tmp_path):
        """Under the error policy the caller gets the diagnostics, not a bare MappingError."""
        path = write_csv(
            tmp_path,
            ["U-404,B-1,CONSUMED_BATCH,2026-01-02", "U-404,B-1,CONSUMED_BATCH,2026-01-03"],
        )
        store = RelationshipStore()
        with pytest.raises(RelationshipResolutionError) as raised:
            await RelationshipIngestionPipeline(CsvRecordLoader(), store).run(
                str(path), consumed_mapping(), Context()
            )
        assert raised.value.result.missing_start == 1
        assert raised.value.result.endpoint_errors[0]["key_value"] == "U-404"
        assert store.write_calls == 0

    async def test_missing_endpoint_errors_before_writes(self, tmp_path):
        path = write_csv(tmp_path, ["U-1,B-404,CONSUMED_BATCH,2026-09-20"])
        store = RelationshipStore()
        with pytest.raises(RelationshipResolutionError) as raised:
            await RelationshipIngestionPipeline(CsvRecordLoader(), store).run(
                str(path), consumed_mapping(), Context()
            )
        assert raised.value.result.missing_end == 1
        assert raised.value.result.endpoint_errors[0]["key_value"] == "B-404"
        assert store.write_calls == 0

    @pytest.mark.parametrize(
        "entity, key, value, counter",
        [
            ("Unit", "unit__unit_id", "U-1", "ambiguous_start"),
            ("Batch", "batch__batch_id", "B-1", "ambiguous_end"),
        ],
    )
    async def test_ambiguous_endpoint_errors_before_writes_by_default(
        self, tmp_path, entity, key, value, counter
    ):
        """The default policy is fail-closed for a key two nodes hold, too."""
        path = write_csv(tmp_path, ["U-1,B-1,CONSUMED_BATCH,2026-09-20"])
        store = RelationshipStore()
        store.entities[(entity, key)][value] = ["node-a", "node-b"]
        with pytest.raises(RelationshipResolutionError) as raised:
            await RelationshipIngestionPipeline(CsvRecordLoader(), store).run(
                str(path), consumed_mapping(), Context()
            )
        assert getattr(raised.value.result, counter) == 1
        (sample,) = raised.value.result.endpoint_errors
        assert (sample["status"], sample["key_value"], sample["candidate_ids"]) == (
            "ambiguous",
            value,
            ["node-a", "node-b"],
        )
        assert store.write_calls == 0

    async def test_a_key_that_becomes_ambiguous_keeps_its_edge_by_default(self, tmp_path):
        """Raising before the write also skips the snapshot cleanup that would drop it."""
        path = write_csv(
            tmp_path, ["U-1,B-1,CONSUMED_BATCH,2026-09-20", "U-2,B-2,CONSUMED_BATCH,2026-09-21"]
        )
        store = RelationshipStore()
        pipeline = RelationshipIngestionPipeline(CsvRecordLoader(), store)
        await pipeline.run(str(path), consumed_mapping(), Context())
        before = dict(store.relationships)
        store.entities[("Unit", "unit__unit_id")]["U-2"] = ["unit-2", "duplicate-unit"]
        with pytest.raises(RelationshipResolutionError):
            await pipeline.run(str(path), consumed_mapping(), Context())
        assert store.relationships == before
        assert store.write_calls == 1

    async def test_a_short_write_keeps_old_edges_until_a_clean_retry(self, tmp_path):
        """Cleanup after a short write deleted rows the file still has.

        ``GraphStore.upsert_relationships`` logs a failed write and returns a
        smaller count rather than raising, so the count is the only sign. A row
        whose rewrite was lost still carries the previous run's number, and the
        snapshot cleanup would treat it as gone from the file.
        """
        path = write_csv(
            tmp_path,
            [
                "U-1,B-1,CONSUMED_BATCH,2026-09-20",
                "U-2,B-2,CONSUMED_BATCH,2026-09-21",
                "U-1,B-2,CONSUMED_BATCH,2026-09-22",
            ],
        )
        store = RelationshipStore()
        pipeline = RelationshipIngestionPipeline(CsvRecordLoader(), store)
        await pipeline.run(str(path), consumed_mapping(), Context())
        first = dict(store.relationships)
        assert len(first) == 3

        write = store.upsert_relationships

        async def lose_unit_2(relationships, *, snapshot_property=None):
            kept = [rel for rel in relationships if rel.start_node_id != "unit-2"]
            return await write(kept, snapshot_property=snapshot_property)

        store.upsert_relationships = lose_unit_2
        # U-1 -> B-2 left the file; U-2 -> B-2 is still in it but is not rewritten.
        write_csv(tmp_path, ["U-1,B-1,CONSUMED_BATCH,2026-09-20", "U-2,B-2,CONSUMED_BATCH,"])
        result = await pipeline.run(str(path), consumed_mapping(), Context())
        assert result.incomplete_writes == ["relationships 1/2 in row batch 1-2"]
        assert result.relationships_deleted == 0
        assert store.relationships.keys() == first.keys(), "cleanup waits for a full write"

        store.upsert_relationships = write
        result = await pipeline.run(str(path), consumed_mapping(), Context())
        assert result.incomplete_writes == []
        assert result.relationships_deleted == 1
        assert sorted(store.relationships) == [
            ("unit-1", "batch-1", "CONSUMED_BATCH"),
            ("unit-2", "batch-2", "CONSUMED_BATCH"),
        ]
        assert (
            store.relationships[("unit-2", "batch-2", "CONSUMED_BATCH")].properties[
                "consumed_batch__consumed_on"
            ]
            is None
        )

    async def test_skip_policy_reports_missing_and_ambiguous(self, tmp_path):
        path = write_csv(
            tmp_path,
            [
                "U-1,B-404,CONSUMED_BATCH,2026-09-20",
                "U-2,B-2,CONSUMED_BATCH,2026-09-21",
            ],
        )
        store = RelationshipStore()
        store.entities[("Unit", "unit__unit_id")]["U-2"] = ["unit-2a", "unit-2b"]
        result = await RelationshipIngestionPipeline(CsvRecordLoader(), store).run(
            str(path),
            consumed_mapping(missing_endpoint="skip", ambiguous_endpoint="skip"),
            Context(),
        )
        assert result.missing_end == 1
        assert result.ambiguous_start == 1
        assert result.relationships_written == 0
        assert len(result.endpoint_errors) == 2

    async def test_more_than_ten_thousand_rows_resolve_in_bounded_batches(self, tmp_path):
        count = 10_001
        rows = [f"U-{index},B-{index},CONSUMED_BATCH,2026-09-20" for index in range(count)]
        path = write_csv(tmp_path, rows)
        store = RelationshipStore()
        store.entities = {
            ("Unit", "unit__unit_id"): {f"U-{index}": [f"unit-{index}"] for index in range(count)},
            ("Batch", "batch__batch_id"): {
                f"B-{index}": [f"batch-{index}"] for index in range(count)
            },
        }
        result = await RelationshipIngestionPipeline(CsvRecordLoader(), store).run(
            str(path), consumed_mapping(), Context()
        )
        assert result.relationships_written == count
        assert max(store.resolve_sizes) <= store._BATCH_SIZE
        assert store.write_calls == 21

    async def test_each_endpoint_key_is_looked_up_once(self, tmp_path):
        """Not once per 500-row batch in each of two passes: 40 lookups for 150 keys.

        The write pass also uses exactly the answers the check pass approved.
        """
        rows = [
            f"U-{index % 100},B-{index // 100},CONSUMED_BATCH,2026-09-20" for index in range(5_000)
        ]
        path = write_csv(tmp_path, rows)
        store = RelationshipStore()
        store.entities = {
            ("Unit", "unit__unit_id"): {f"U-{index}": [f"unit-{index}"] for index in range(100)},
            ("Batch", "batch__batch_id"): {f"B-{index}": [f"batch-{index}"] for index in range(50)},
        }
        result = await RelationshipIngestionPipeline(CsvRecordLoader(), store).run(
            str(path), consumed_mapping(), Context()
        )
        assert result.relationships_written == 5_000
        assert store.resolve_sizes == [100, 50]


class TestRelationshipReviewRegressions:
    @pytest.mark.parametrize("kind", ["INTEGER", "FLOAT", "BOOLEAN", "DATE", "LIST"])
    def test_non_string_endpoint_keys_are_refused_before_writes(self, kind):
        problems = GraphRAG._relationship_endpoint_schema_problems(
            consumed_mapping(),
            {"Unit": {"unit__unit_id": kind}, "Batch": {"batch__batch_id": "STRING"}},
        )
        assert len(problems) == 1
        assert "must be declared STRING" in problems[0]

    def test_incoming_description_matches_pattern(self):
        relation = ontology_for_relationship(consumed_mapping(direction="INCOMING")).relations[0]
        assert relation.description.startswith("Batch to Unit")
        assert relation.patterns == [("Batch", "Unit")]

    def test_endpoint_types_are_protected_from_drop(self):
        ontology = Ontology(relationship_tables=[consumed_mapping()])
        assert ontology.tables_naming("Unit") == {
            "consumed_batch.csv": "relationship start endpoint"
        }
        assert ontology.tables_naming("Batch") == {
            "consumed_batch.csv": "relationship end endpoint"
        }

    def test_merge_refuses_source_declared_under_two_kinds(self):
        from graphrag_sdk import TableMapping

        nodes = Ontology(
            tables=[
                TableMapping(source="consumed_batch.csv", label="Event", key="id", standalone=True)
            ]
        )
        edges = Ontology(relationship_tables=[consumed_mapping()])
        for left, right in ((nodes, edges), (edges, nodes)):
            with pytest.raises(ValueError, match="declared twice"):
                left.merge(right)

    def test_node_and_edge_signatures_cannot_overlap(self):
        from graphrag_sdk import TableMapping
        from graphrag_sdk.storage.ontology_store import OntologyStore

        nodes = Ontology(
            tables=[TableMapping(source="HR.csv", label="Event", key="id", standalone=True)]
        )
        edges = Ontology(relationship_tables=[consumed_mapping(source="hr.csv")])
        assert nodes.tables[0].signature == edges.relationship_tables[0].signature
        for left, right in ((nodes, edges), (edges, nodes)):
            with pytest.raises(ValueError, match="property signature"):
                left.merge(right)
            from graphrag_sdk.storage.ontology_store import OntologyContradictionError

            with pytest.raises(OntologyContradictionError, match="property signature"):
                OntologyStore._check_no_signature_collision(left, right)

    async def test_duplicate_invalid_rows_count_once(self, tmp_path):
        path = write_csv(tmp_path, ["U-1,B-404,CONSUMED_BATCH,", "U-1,B-404,CONSUMED_BATCH,"])
        result = await RelationshipIngestionPipeline(CsvRecordLoader(), RelationshipStore()).run(
            str(path), consumed_mapping(missing_endpoint="skip"), Context()
        )
        assert (result.rows, result.duplicate_rows, result.rows_skipped, result.missing_end) == (
            2,
            1,
            1,
            1,
        )

    async def test_unchanged_file_removes_rows_that_become_ambiguous(self, tmp_path):
        path = write_csv(tmp_path, ["U-1,B-1,CONSUMED_BATCH,"])
        store = RelationshipStore()
        pipeline = RelationshipIngestionPipeline(CsvRecordLoader(), store)
        mapping = consumed_mapping(ambiguous_endpoint="skip")
        await pipeline.run(str(path), mapping, Context())
        assert len(store.relationships) == 1
        store.entities[("Unit", "unit__unit_id")]["U-1"] = ["unit-1", "duplicate-unit"]
        result = await pipeline.run(str(path), mapping, Context())
        assert result.rows_skipped == result.relationships_deleted == 1
        assert store.relationships == {}

    async def test_batch_delete_failure_is_attributable_and_does_not_abort(
        self, mock_connection, llm, embedder
    ):
        from unittest.mock import AsyncMock

        from graphrag_sdk.core.connection import ConnectionConfig

        mock_connection.config = ConnectionConfig(graph_name="unit-test")
        rag = GraphRAG(connection=mock_connection, llm=llm, embedder=embedder)
        rag._global_ontology = Ontology(relationship_tables=[consumed_mapping()])
        rag._ensure_ontology_initialized = AsyncMock()
        rag._graph_store.drop_relationship_source = AsyncMock(
            side_effect=[RuntimeError("database busy"), 0]
        )
        # No Document has either id, so both names reach the relationship table.
        rag.delete_document = AsyncMock(side_effect=DocumentNotFoundError("no document exists"))
        results = await rag.apply_changes(
            deleted=["first/consumed_batch.csv", "second/consumed_batch.csv"]
        )
        assert results.deleted[0].error_type == "RuntimeError"
        assert results.deleted[0].error == "database busy"
        assert results.deleted[1].is_success
        assert results.deleted[1].result.chunks_deleted == 0
        assert rag._graph_store.drop_relationship_source.await_count == 2

    async def test_delete_uses_exact_relationship_path_before_entity_basename(
        self, mock_connection, llm, embedder
    ):
        from unittest.mock import AsyncMock
        from graphrag_sdk import TableMapping
        from graphrag_sdk.core.connection import ConnectionConfig

        mock_connection.config = ConnectionConfig(graph_name="unit-test")
        rag = GraphRAG(connection=mock_connection, llm=llm, embedder=embedder)
        mapping = consumed_mapping(source="relations/links.csv")
        # Simulate a legacy stored collision: new declarations now refuse this.
        rag._global_ontology = Ontology(relationship_tables=[mapping]).model_copy(
            update={
                "tables": [
                    TableMapping(
                        source="people/links.csv", label="Event", key="id", standalone=True
                    )
                ]
            }
        )
        rag._ensure_ontology_initialized = AsyncMock()
        rag._graph_store.drop_relationship_source = AsyncMock()
        rag._ontology_store.drop_relationship_mapping = AsyncMock()
        rag.delete_document = AsyncMock(side_effect=DocumentNotFoundError("no document"))
        result = await rag.apply_changes(deleted=["relations/links.csv"])
        assert result.deleted[0].is_success
        rag._graph_store.drop_relationship_source.assert_awaited_once_with(mapping.signature, [])
        # Only the Document named exactly that is tried, never the entity table's.
        rag.delete_document.assert_awaited_once_with("relations/links.csv")
        rag._ontology_store.drop_relationship_mapping.assert_not_awaited()

    @pytest.mark.parametrize(
        "relationship_source, doc_id",
        [
            ("consumed_batch.csv", "consumed_batch.csv"),
            ("exports/consumed_batch.csv", "notes/consumed_batch.csv"),
        ],
    )
    async def test_delete_prefers_an_existing_document_over_a_relationship_table(
        self, mock_connection, llm, embedder, relationship_source, doc_id
    ):
        from unittest.mock import AsyncMock

        from graphrag_sdk import DeleteDocumentResult
        from graphrag_sdk.core.connection import ConnectionConfig

        mock_connection.config = ConnectionConfig(graph_name="unit-test")
        rag = GraphRAG(connection=mock_connection, llm=llm, embedder=embedder)
        rag._global_ontology = Ontology(
            relationship_tables=[consumed_mapping(source=relationship_source)]
        )
        rag._ensure_ontology_initialized = AsyncMock()
        # A Document with this exact id exists: prose ingested under the name.
        rag.delete_document = AsyncMock(
            return_value=DeleteDocumentResult(document_uid=doc_id, chunks_deleted=1)
        )
        rag.drop_table = AsyncMock(side_effect=AssertionError("relationship table dropped"))
        rag._graph_store.drop_relationship_source = AsyncMock(
            side_effect=AssertionError("relationship edges removed")
        )

        result = await rag.apply_changes(deleted=[doc_id])

        assert result.deleted[0].is_success, result.deleted[0].error
        assert result.deleted[0].result.chunks_deleted == 1
        rag.delete_document.assert_awaited_once_with(doc_id)
        rag.drop_table.assert_not_awaited()
        rag._graph_store.drop_relationship_source.assert_not_awaited()

    async def test_delete_keeps_the_relationship_mapping(self, mock_connection, llm, embedder):
        """A deleted relationship CSV loses its edges, not its mapping, like a node table.

        Otherwise the file added back later has no declaration and is written
        as a proposed node table. ``drop_table()`` still removes the mapping.
        """
        from unittest.mock import AsyncMock

        from graphrag_sdk.core.connection import ConnectionConfig

        mock_connection.config = ConnectionConfig(graph_name="unit-test")
        rag = GraphRAG(connection=mock_connection, llm=llm, embedder=embedder)
        mapping = consumed_mapping()
        rag._global_ontology = ontology_for_relationship(mapping)
        rag._ensure_ontology_initialized = AsyncMock()
        rag._refresh_global_ontology = AsyncMock(return_value=Ontology())
        rag.delete_document = AsyncMock(side_effect=DocumentNotFoundError("no document"))
        rag._graph_store.drop_relationship_source = AsyncMock(return_value=2)
        rag._ontology_store.drop_relation_property = AsyncMock()
        rag._ontology_store.drop_relationship_mapping = AsyncMock()

        result = await rag.apply_changes(deleted=["consumed_batch.csv"])

        assert result.deleted[0].is_success, result.deleted[0].error
        rag._graph_store.drop_relationship_source.assert_awaited_once_with(
            mapping.signature, ["consumed_batch__consumed_on"]
        )
        rag._ontology_store.drop_relationship_mapping.assert_not_awaited()
        rag._ontology_store.drop_relation_property.assert_not_awaited()
        assert rag._global_ontology.relationship_tables == [mapping]
        assert await rag._mapping_for("consumed_batch.csv") == mapping

        await rag.drop_table("consumed_batch.csv")
        rag._ontology_store.drop_relationship_mapping.assert_awaited_once_with(mapping.source)
        rag._ontology_store.drop_relation_property.assert_awaited_once_with(
            "CONSUMED_BATCH", "consumed_batch__consumed_on"
        )


class TestRedeclarationMovesKeyAndEndpointTogether:
    """A declaration that moves a node table's key and the relationship
    endpoint keyed on it together leaves nothing depending on the old key; the
    stored relationship mapping it replaces must not be counted."""

    async def test_a_combined_redeclaration_is_allowed(self, mock_connection, llm, embedder):
        from graphrag_sdk import TableMapping
        from graphrag_sdk.core.connection import ConnectionConfig

        mock_connection.config = ConnectionConfig(graph_name="unit-test")
        rag = GraphRAG(connection=mock_connection, llm=llm, embedder=embedder)
        batch = TableMapping(source="batch.csv", label="Batch", key="batch_id", name="batch_id")
        entities = [Entity(label="Unit"), Entity(label="Batch")]
        stored = Ontology(
            entities=entities,
            tables=[
                TableMapping(source="unit.csv", label="Unit", key="unit_id", name="unit_id"),
                batch,
            ],
            relationship_tables=[consumed_mapping()],
        )
        rekeyed_unit = TableMapping(source="unit.csv", label="Unit", key="code", name="code")
        moved = consumed_mapping(start=EndpointMapping("Unit", "unit__code", "source"))
        both = Ontology(
            entities=entities, tables=[rekeyed_unit, batch], relationship_tables=[moved]
        )
        only_the_table = Ontology(entities=entities, tables=[rekeyed_unit, batch])

        rag._refuse_a_redeclaration_that_drops_an_endpoint_key(both, stored)
        with pytest.raises(ValueError, match=r"redeclaring 'unit\.csv'"):
            rag._refuse_a_redeclaration_that_drops_an_endpoint_key(only_the_table, stored)


class TestRelationshipEndpointKeyGuard:
    """An endpoint key cannot be taken from under a relationship table.

    The stored mapping is registered again on every first touch, and one whose
    key is no longer declared failed that: every new GraphRAG on the graph
    raised ``MappingError``, including the ``drop_table()`` that removes it.
    """

    @staticmethod
    def _rag(mock_connection, llm, embedder, ontology: Ontology) -> GraphRAG:
        from unittest.mock import AsyncMock

        from graphrag_sdk.core.connection import ConnectionConfig

        mock_connection.config = ConnectionConfig(graph_name="unit-test")
        rag = GraphRAG(connection=mock_connection, llm=llm, embedder=embedder)
        rag._global_ontology = ontology
        rag._ensure_ontology_initialized = AsyncMock()
        rag._refresh_global_ontology = AsyncMock(return_value=ontology)
        return rag

    @pytest.mark.parametrize(
        "table, key",
        [
            ("unit.csv", r"Unit\.unit__unit_id as its start"),
            ("batch.csv", r"Batch\.batch__batch_id as its end"),
        ],
    )
    async def test_drop_table_refuses_a_node_table_whose_key_an_edge_table_uses(
        self, mock_connection, llm, embedder, table, key
    ):
        from unittest.mock import AsyncMock

        from graphrag_sdk import DeleteDocumentResult, TableMapping

        site = TableMapping(source="site.csv", label="Site", key="site_id", name="site_id")
        rag = self._rag(
            mock_connection,
            llm,
            embedder,
            Ontology(
                entities=[Entity(label="Unit"), Entity(label="Batch"), Entity(label="Site")],
                tables=[
                    TableMapping(source="unit.csv", label="Unit", key="unit_id", name="unit_id"),
                    TableMapping(
                        source="batch.csv", label="Batch", key="batch_id", name="batch_id"
                    ),
                    site,
                ],
                relationship_tables=[consumed_mapping()],
            ),
        )
        rag._graph_store.documents_of_table = AsyncMock(return_value=[])
        rag.delete_document = AsyncMock(return_value=DeleteDocumentResult(document_uid=table))
        rag._retract_table = AsyncMock(return_value=(0, 0))
        rag._ontology_store.drop_entity_property = AsyncMock()
        rag._ontology_store.drop_table_mapping = AsyncMock()

        with pytest.raises(
            ValueError,
            match=rf"'consumed_batch\.csv' uses {key} endpoint key.*"
            r"drop_table\('consumed_batch\.csv'\) first",
        ):
            await rag.drop_table(table)
        rag.delete_document.assert_not_awaited()
        rag._retract_table.assert_not_awaited()
        rag._ontology_store.drop_entity_property.assert_not_awaited()
        rag._ontology_store.drop_table_mapping.assert_not_awaited()

        # A table no relationship table resolves by still drops.
        await rag.drop_table("site.csv")
        rag._ontology_store.drop_table_mapping.assert_awaited_once_with("site.csv")

    @pytest.mark.parametrize("call", ["drop_attribute", "rename_attribute"])
    async def test_a_plain_endpoint_key_cannot_be_dropped_or_renamed(
        self, mock_connection, llm, embedder, call
    ):
        from unittest.mock import AsyncMock

        rag = self._rag(
            mock_connection,
            llm,
            embedder,
            Ontology(
                entities=[
                    Entity(
                        label="Unit",
                        properties=[
                            Attribute(name="code", type="STRING"),
                            Attribute(name="nickname", type="STRING"),
                        ],
                    ),
                    Entity(
                        label="Batch",
                        properties=[Attribute(name="batch__batch_id", type="STRING")],
                    ),
                ],
                relationship_tables=[
                    consumed_mapping(start=EndpointMapping("Unit", "code", "source"))
                ],
            ),
        )
        rag._graph_store.drop_node_property = AsyncMock(return_value=1)
        rag._graph_store.rename_node_property = AsyncMock(return_value=1)
        rag._ontology_store.drop_entity_property = AsyncMock()
        rag._ontology_store.rename_property_label = AsyncMock()
        renaming = call == "rename_attribute"
        extra = ("unit_code",) if renaming else ()

        with pytest.raises(
            ValueError,
            match=r"'consumed_batch\.csv' uses Unit\.code as its start endpoint key.*"
            r"drop_table\('consumed_batch\.csv'\) first",
        ):
            await getattr(rag, call)("Unit", "code", *extra)
        for mock in (
            rag._graph_store.drop_node_property,
            rag._graph_store.rename_node_property,
            rag._ontology_store.drop_entity_property,
            rag._ontology_store.rename_property_label,
        ):
            mock.assert_not_awaited()

        # A property no relationship table resolves by is unaffected.
        await getattr(rag, call)("Unit", "nickname", *extra)
        changed = (
            rag._ontology_store.rename_property_label
            if renaming
            else rag._ontology_store.drop_entity_property
        )
        changed.assert_awaited_once()

    @staticmethod
    def _stored_without_the_start_key() -> Ontology:
        """What an unguarded ``drop_table('unit.csv')`` left in the ontology graph."""
        return Ontology(
            entities=[
                Entity(label="Unit"),
                Entity(
                    label="Batch", properties=[Attribute(name="batch__batch_id", type="STRING")]
                ),
            ],
            relationship_tables=[consumed_mapping()],
        )

    async def test_stored_mapping_with_a_missing_key_is_skipped_at_startup(
        self, mock_connection, llm, embedder, caplog
    ):
        import logging
        from unittest.mock import AsyncMock

        from graphrag_sdk.core.connection import ConnectionConfig

        mock_connection.config = ConnectionConfig(graph_name="unit-test")
        rag = GraphRAG(connection=mock_connection, llm=llm, embedder=embedder)
        stored = self._stored_without_the_start_key()
        rag._ontology_store.load = AsyncMock(return_value=stored)
        rag._ontology_store.register = AsyncMock(side_effect=AssertionError("re-registered"))

        with caplog.at_level(logging.WARNING, logger="graphrag_sdk.api.main"):
            ontology = await rag.get_ontology()

        assert ontology.relationship_tables == [consumed_mapping()]
        assert "'consumed_batch.csv'" in caplog.text
        assert "start endpoint key Unit.unit__unit_id" in caplog.text
        assert "drop_table('consumed_batch.csv')" in caplog.text
        rag._ontology_store.register.assert_not_awaited()

        # Still stored, so drop_table() can take it out.
        rag._graph_store.drop_relationship_source = AsyncMock(return_value=0)
        rag._ontology_store.drop_relationship_mapping = AsyncMock()
        await rag.drop_table("consumed_batch.csv")
        rag._ontology_store.drop_relationship_mapping.assert_awaited_once_with("consumed_batch.csv")

    async def test_declared_mapping_with_a_missing_key_still_raises_at_startup(
        self, mock_connection, llm, embedder
    ):
        """Only the stored copy is skipped; a caller's own declaration is checked."""
        from unittest.mock import AsyncMock

        from graphrag_sdk.core.connection import ConnectionConfig

        mock_connection.config = ConnectionConfig(graph_name="unit-test")
        rag = GraphRAG(
            connection=mock_connection,
            llm=llm,
            embedder=embedder,
            ontology=Ontology(relationship_tables=[consumed_mapping()]),
        )
        stored = self._stored_without_the_start_key()
        rag._ontology_store.load = AsyncMock(return_value=stored)
        rag._ontology_store.register = AsyncMock(return_value=stored)

        with pytest.raises(MappingError, match=r"start endpoint key Unit\.unit__unit_id"):
            await rag.get_ontology()


class TestApplyChangesRunsRelationshipTablesLast:
    """A relationship table waits for the entity tables of the same batch.

    Its endpoints resolve against nodes those tables write, and a git diff lists
    files alphabetically: consumed_batch.csv ran before unit.csv and found its
    start endpoints missing.
    """

    @staticmethod
    def _rag(mock_connection, llm, embedder, calls: list[tuple[str, str]], fail=()):
        import os
        from unittest.mock import AsyncMock

        from graphrag_sdk.core.connection import ConnectionConfig
        from graphrag_sdk.core.models import DocumentInfo, UpdateResult
        from graphrag_sdk.ingestion.structured_pipeline import (
            RelationshipIngestionResult,
            StructuredIngestionResult,
        )

        mock_connection.config = ConnectionConfig(graph_name="unit-test")
        rag = GraphRAG(connection=mock_connection, llm=llm, embedder=embedder)
        rag._global_ontology = Ontology(
            entities=[Entity(label="Unit"), Entity(label="Batch")],
            tables=_unit_and_batch_tables(),
            relationship_tables=[consumed_mapping()],
        )
        rag._ensure_ontology_initialized = AsyncMock()

        async def ingest(path, **kwargs):
            name = os.path.basename(path)
            calls.append(("ingest", name))
            if name in fail:
                raise RuntimeError(f"{name} failed")
            if name == "consumed_batch.csv":
                result = RelationshipIngestionResult(path)
                result.relationships_written = 2
                return result
            return StructuredIngestionResult(name)

        async def update(path, **kwargs):
            name = os.path.basename(path)
            calls.append(("update", name))
            return UpdateResult(document_info=DocumentInfo(uid=name, path=path))

        rag.ingest = ingest
        rag.update = update
        return rag

    async def test_added_in_sorted_order_writes_the_edges_last(
        self, mock_connection, llm, embedder
    ):
        calls: list[tuple[str, str]] = []
        rag = self._rag(mock_connection, llm, embedder, calls)
        result = await rag.apply_changes(added=["batch.csv", "consumed_batch.csv", "unit.csv"])
        assert calls == [
            ("ingest", "batch.csv"),
            ("ingest", "unit.csv"),
            ("ingest", "consumed_batch.csv"),
        ]
        # Each entry stays where its file was in the list.
        assert [entry.result.document_info.uid for entry in result.added] == [
            "batch.csv",
            "consumed_batch.csv",
            "unit.csv",
        ]
        assert result.added[1].result.metadata["relationships_written"] == 2

    async def test_a_modified_relationship_table_waits_for_added_and_modified_tables(
        self, mock_connection, llm, embedder
    ):
        calls: list[tuple[str, str]] = []
        rag = self._rag(mock_connection, llm, embedder, calls)
        result = await rag.apply_changes(
            modified=["consumed_batch.csv", "unit.csv"], added=["batch.csv"]
        )
        assert calls == [
            ("update", "unit.csv"),
            ("ingest", "batch.csv"),
            ("ingest", "consumed_batch.csv"),
        ]
        assert result.modified[0].result.metadata["relationships_written"] == 2
        assert result.modified[1].result.document_info.uid == "unit.csv"

    async def test_each_file_keeps_its_own_error(self, mock_connection, llm, embedder):
        calls: list[tuple[str, str]] = []
        rag = self._rag(
            mock_connection, llm, embedder, calls, fail=("batch.csv", "consumed_batch.csv")
        )
        result = await rag.apply_changes(added=["batch.csv", "consumed_batch.csv", "unit.csv"])
        assert [entry.error for entry in result.added] == [
            "batch.csv failed",
            "consumed_batch.csv failed",
            None,
        ]
        assert calls[-1] == ("ingest", "consumed_batch.csv")


class TestOntologyInitializationReentry:
    """Replacing a proposal mid-initialization deletes its Document, and that asks
    for the ontology again from the task holding the initialization lock."""

    @staticmethod
    def _rag(mock_connection, llm, embedder) -> GraphRAG:
        from graphrag_sdk.core.connection import ConnectionConfig

        mock_connection.config = ConnectionConfig(graph_name="unit-test")
        return GraphRAG(connection=mock_connection, llm=llm, embedder=embedder)

    async def test_the_initializing_task_passes_through(self, mock_connection, llm, embedder):
        import asyncio

        rag = self._rag(mock_connection, llm, embedder)
        runs: list[str] = []

        async def initialize():
            runs.append("initialize")
            # What delete_document() reaches while a proposal is being replaced.
            await rag._ensure_ontology_initialized()
            rag._ontology_initialized = True

        rag._initialize_ontology = initialize
        await asyncio.wait_for(rag._ensure_ontology_initialized(), timeout=5)
        assert runs == ["initialize"]
        assert rag._ontology_init_task is None

    async def test_another_task_still_waits_for_initialization(
        self, mock_connection, llm, embedder
    ):
        import asyncio

        rag = self._rag(mock_connection, llm, embedder)
        started, release = asyncio.Event(), asyncio.Event()

        async def initialize():
            started.set()
            await release.wait()
            rag._ontology_initialized = True

        rag._initialize_ontology = initialize
        first = asyncio.create_task(rag._ensure_ontology_initialized())
        await started.wait()
        second = asyncio.create_task(rag._ensure_ontology_initialized())
        await asyncio.sleep(0.01)
        assert not second.done()
        release.set()
        await asyncio.wait_for(asyncio.gather(first, second), timeout=5)
