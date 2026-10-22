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

    async def resolve_by_property(self, label: str, prop: str, keys: list[str]):
        self.resolve_sizes.append(len(keys))
        values = self.entities.get((label, prop), {})
        return {key: values[key] for key in dict.fromkeys(keys) if key in values}

    async def upsert_relationships(self, relationships):
        self.write_calls += 1
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
        snapshot_token: str,
        signed_properties: list[str],
    ):
        stale = [
            key
            for key, relationship in self.relationships.items()
            if signature in relationship.properties.get("structured_sources", [])
            and relationship.properties.get(marker_property) != snapshot_token
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


class TestRelationshipReviewRegressions:
    @pytest.mark.parametrize("kind", ["INTEGER", "FLOAT", "BOOLEAN", "LIST"])
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
