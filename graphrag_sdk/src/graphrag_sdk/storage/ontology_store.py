"""Persistent ontology storage in a dedicated FalkorDB graph.

The ontology lives in a separate FalkorDB graph named ``<data_graph>__ontology``
so it survives drops of the data graph and can be inspected via Cypher.

Ingest passes call :py:meth:`OntologyStore.register` with the run's local schema;
each register call is an idempotent union into the persisted ontology. Retrieval
calls :py:meth:`OntologyStore.load` to fetch the **global** ontology (union of
every schema ever registered) and feeds it into the Cypher generation prompt.
"""

from __future__ import annotations

import logging
from typing import Any

from graphrag_sdk.core.connection import FalkorDBConnection
from graphrag_sdk.core.models import (
    EntityType,
    GraphSchema,
    PropertyType,
    RelationType,
)

logger = logging.getLogger(__name__)


def _encode_patterns(patterns: list[tuple[str, str]]) -> list[str]:
    return [f"{src}|{tgt}" for src, tgt in patterns]


def _decode_patterns(encoded: list[str] | None) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for s in encoded or []:
        if not isinstance(s, str) or "|" not in s:
            continue
        src, tgt = s.split("|", 1)
        out.append((src, tgt))
    return out


class OntologyStore:
    """Persists and loads :py:class:`GraphSchema` in a dedicated FalkorDB graph.

    The store owns its own graph handle, derived from the same FalkorDB driver
    as the data-graph connection. Queries go directly to the FalkorDB driver
    and bypass the connection's retry / circuit-breaker layer — ontology
    operations are infrequent, idempotent, and tolerant of a single failure
    (the caller can retry by re-registering).
    """

    ONTOLOGY_GRAPH_SUFFIX = "__ontology"

    def __init__(self, connection: FalkorDBConnection, data_graph_name: str) -> None:
        self._conn = connection
        self._graph_name = f"{data_graph_name}{self.ONTOLOGY_GRAPH_SUFFIX}"
        self._graph: Any | None = None

    def _ensure_graph(self) -> Any:
        if self._graph is not None:
            return self._graph
        self._conn._ensure_client()
        driver = self._conn._driver
        if driver is None:
            raise RuntimeError("FalkorDB driver not initialised on connection")
        self._graph = driver.select_graph(self._graph_name)
        return self._graph

    @property
    def graph_name(self) -> str:
        return self._graph_name

    async def _query(self, cypher: str, params: dict[str, Any] | None = None) -> Any:
        graph = self._ensure_graph()
        return await graph.query(cypher, params=params)

    async def load(self) -> GraphSchema:
        """Read the ontology graph and reconstruct a :py:class:`GraphSchema`.

        Returns an empty schema if the ontology graph does not yet exist.
        """
        try:
            ent_result = await self._query(
                "MATCH (e:OntologyEntityType) "
                "OPTIONAL MATCH (e)-[:HAS_PROPERTY]->(p:OntologyProperty) "
                "RETURN e.label AS label, e.description AS description, "
                "collect({name: p.name, type: p.type, description: p.description, "
                "required: p.required}) AS properties"
            )
            rel_result = await self._query(
                "MATCH (r:OntologyRelationType) "
                "OPTIONAL MATCH (r)-[:HAS_PROPERTY]->(p:OntologyProperty) "
                "RETURN r.label AS label, r.description AS description, "
                "r.patterns AS patterns, "
                "collect({name: p.name, type: p.type, description: p.description, "
                "required: p.required}) AS properties"
            )
        except Exception as exc:
            logger.debug("Ontology load failed (returning empty schema): %s", exc)
            return GraphSchema()

        entities = [
            EntityType(
                label=row[0],
                description=row[1],
                properties=_props_from_rows(row[2]),
            )
            for row in (ent_result.result_set or [])
            if row[0]
        ]
        relations = [
            RelationType(
                label=row[0],
                description=row[1],
                patterns=_decode_patterns(row[2]),
                properties=_props_from_rows(row[3]),
            )
            for row in (rel_result.result_set or [])
            if row[0]
        ]
        return GraphSchema(entities=entities, relations=relations)

    async def register(self, schema: GraphSchema) -> GraphSchema:
        """Merge ``schema`` into the persisted ontology; return the new global ontology.

        Idempotent. ``MERGE`` keys on ``(label, name)``; descriptions/types use
        last-write-wins; relation ``patterns`` are union-merged.
        """
        if not schema.entities and not schema.relations:
            return await self.load()

        for et in schema.entities:
            await self._upsert_entity_type(et)
        for rt in schema.relations:
            await self._upsert_relation_type(rt)

        return await self.load()

    async def _upsert_entity_type(self, et: EntityType) -> None:
        await self._query(
            "MERGE (e:OntologyEntityType {label: $label}) "
            "SET e.description = coalesce($description, e.description)",
            {"label": et.label, "description": et.description},
        )
        for prop in et.properties:
            await self._upsert_property(et.label, prop, owner_label="OntologyEntityType")

    async def _upsert_relation_type(self, rt: RelationType) -> None:
        new_patterns = _encode_patterns(rt.patterns)
        result = await self._query(
            "MATCH (r:OntologyRelationType {label: $label}) "
            "RETURN r.patterns AS patterns",
            {"label": rt.label},
        )
        existing: list[str] = []
        if result.result_set:
            existing = list(result.result_set[0][0] or [])
        seen: set[str] = set()
        merged: list[str] = []
        for s in existing + new_patterns:
            if s not in seen:
                seen.add(s)
                merged.append(s)
        await self._query(
            "MERGE (r:OntologyRelationType {label: $label}) "
            "SET r.description = coalesce($description, r.description), "
            "r.patterns = $patterns",
            {"label": rt.label, "description": rt.description, "patterns": merged},
        )
        for prop in rt.properties:
            await self._upsert_property(rt.label, prop, owner_label="OntologyRelationType")

    async def _upsert_property(
        self, owner_label_value: str, prop: PropertyType, *, owner_label: str
    ) -> None:
        await self._query(
            f"MATCH (o:{owner_label} {{label: $owner}}) "
            "MERGE (o)-[:HAS_PROPERTY]->(p:OntologyProperty {name: $name}) "
            "SET p.type = $type, "
            "p.description = coalesce($description, p.description), "
            "p.required = $required",
            {
                "owner": owner_label_value,
                "name": prop.name,
                "type": prop.type,
                "description": prop.description,
                "required": prop.required,
            },
        )

    async def clear(self) -> None:
        """Drop the ontology graph (``GRAPH.DELETE``). Idempotent for empty graphs."""
        self._conn._ensure_client()
        from redis.asyncio import Redis

        redis: Redis = Redis(connection_pool=self._conn._pool)
        try:
            await redis.execute_command("GRAPH.DELETE", self._graph_name)
        except Exception as exc:
            if "empty" in str(exc).lower() or "invalid" in str(exc).lower():
                logger.debug("Ontology graph '%s' already empty", self._graph_name)
            else:
                raise
        self._graph = None


def _props_from_rows(rows: list[Any] | None) -> list[PropertyType]:
    """Convert a ``collect(...)`` result into ``PropertyType`` objects.

    FalkorDB ``OPTIONAL MATCH`` with ``collect`` yields a list containing one
    null-keyed dict when there are no matches; we filter those out.
    """
    out: list[PropertyType] = []
    for row in rows or []:
        if not row or not isinstance(row, dict):
            continue
        name = row.get("name")
        if not name:
            continue
        out.append(
            PropertyType(
                name=name,
                type=row.get("type") or "STRING",
                description=row.get("description"),
                required=bool(row.get("required")),
            )
        )
    return out
