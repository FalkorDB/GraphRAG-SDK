from __future__ import annotations

from typing import Any

import pytest

from graphrag_sdk import (
    Attribute,
    Column,
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
    assert (await rag.ingest(str(path))).relationships_written == 2
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
    result = await rag.ingest(str(path))
    assert result.relationships_deleted == 1
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
    await rag.drop_table("consumed_batch.csv")
    assert (
        await rag._graph_store.query_raw(
            "MATCH ()-[r:RELATES {rel_type:'CONSUMED_BATCH'}]->() RETURN count(r)"
        )
    ).result_set == [[0]]
    assert (await rag._ontology_store.load()).relationship_tables == []
    assert await counts() == before


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
            entity.label: {prop.name for prop in entity.properties}
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
