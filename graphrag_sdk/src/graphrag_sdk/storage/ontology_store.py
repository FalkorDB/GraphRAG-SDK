"""Persistent ontology storage in a dedicated FalkorDB graph.

The ontology lives in a separate FalkorDB graph named ``<data_graph>__ontology``
and is the **anchor** for the working schema:

- Always-on: every :py:class:`GraphRAG` has exactly one ontology graph,
  created lazily on first use, dropped on ``delete_all()``.
- Single source of truth: retrieval, ``get_ontology()``, and any cross-process
  worker all read from the same graph.
- Additive only: :py:meth:`register` validates incoming schema against what's
  already persisted and refuses **type contradictions** on existing properties.
  New entity types, relation types, properties, and relation patterns are all
  welcome. Re-typing an existing property is not.

Users who want a curated, declarative schema (descriptions, future flags,
properties not yet observed in the data) supply a ``schema`` to ``GraphRAG``;
it gets registered into the ontology graph on first connection. JSON
import/export via :py:meth:`GraphSchema.save_to_file` / ``from_file`` is a
review / version-control bridge — the ontology graph is the canonical copy.
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


class OntologyContradictionError(ValueError):
    """Raised when an incoming schema redefines an existing property's type.

    The ontology is additive: new labels, properties, and relation patterns are
    welcome, but re-typing a property already registered on a label is
    explicitly rejected so downstream Cypher queries don't break silently.
    """


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


def _props_from_rows(rows: list[Any] | None) -> list[PropertyType]:
    """Reconstruct PropertyType objects from a ``collect(...)`` query result.

    Filters out the null-keyed dict FalkorDB returns for an OPTIONAL MATCH
    with no matches.
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
            )
        )
    return out


class OntologyStore:
    """Persists and loads :py:class:`GraphSchema` in a dedicated FalkorDB graph.

    Owns its own graph handle, derived from the data-graph connection's
    driver. Queries go directly to the FalkorDB driver and bypass the
    connection's retry / circuit-breaker — ontology operations are
    infrequent, idempotent, and tolerant of a single failure.
    """

    ONTOLOGY_GRAPH_SUFFIX = "__ontology"

    def __init__(self, connection: FalkorDBConnection, data_graph_name: str) -> None:
        self._conn = connection
        self._graph_name = f"{data_graph_name}{self.ONTOLOGY_GRAPH_SUFFIX}"
        self._graph: Any | None = None

    @property
    def graph_name(self) -> str:
        return self._graph_name

    def _ensure_graph(self) -> Any:
        if self._graph is not None:
            return self._graph
        self._conn._ensure_client()
        driver = self._conn._driver
        if driver is None:
            raise RuntimeError("FalkorDB driver not initialised on connection")
        self._graph = driver.select_graph(self._graph_name)
        return self._graph

    async def _query(self, cypher: str, params: dict[str, Any] | None = None) -> Any:
        graph = self._ensure_graph()
        return await graph.query(cypher, params=params)

    # ── Load ─────────────────────────────────────────────────────

    async def load(self) -> GraphSchema:
        """Read the ontology graph and reconstruct a :py:class:`GraphSchema`.

        Returns an empty schema if the ontology graph does not yet exist or
        introspection fails. Failure is logged at DEBUG so we don't spam an
        unconfigured GraphRAG instance with warnings.
        """
        try:
            ent_result = await self._query(
                "MATCH (e:OntologyEntityType) "
                "OPTIONAL MATCH (e)-[:HAS_PROPERTY]->(p:OntologyProperty) "
                "RETURN e.label AS label, e.description AS description, "
                "collect({name: p.name, type: p.type, description: p.description}) AS properties"
            )
            rel_result = await self._query(
                "MATCH (r:OntologyRelationType) "
                "OPTIONAL MATCH (r)-[:HAS_PROPERTY]->(p:OntologyProperty) "
                "RETURN r.label AS label, r.description AS description, "
                "r.patterns AS patterns, "
                "collect({name: p.name, type: p.type, description: p.description}) AS properties"
            )
        except Exception as exc:
            logger.debug("Ontology load failed (returning empty schema): %s", exc)
            return GraphSchema()

        ent_rows = getattr(ent_result, "result_set", None) or []
        rel_rows = getattr(rel_result, "result_set", None) or []
        if not isinstance(ent_rows, list):
            ent_rows = []
        if not isinstance(rel_rows, list):
            rel_rows = []

        entities = [
            EntityType(
                label=row[0],
                description=row[1],
                properties=_props_from_rows(row[2]),
            )
            for row in ent_rows
            if isinstance(row, list) and len(row) >= 3 and row[0]
        ]
        relations = [
            RelationType(
                label=row[0],
                description=row[1],
                patterns=_decode_patterns(row[2]),
                properties=_props_from_rows(row[3]),
            )
            for row in rel_rows
            if isinstance(row, list) and len(row) >= 4 and row[0]
        ]
        return GraphSchema(entities=entities, relations=relations)

    # ── Register ─────────────────────────────────────────────────

    async def register(self, schema: GraphSchema) -> GraphSchema:
        """Merge ``schema`` into the persisted ontology and return the union.

        Validates first: if ``schema`` redefines the type of a property
        already registered on the same entity/relation label, raises
        :py:class:`OntologyContradictionError` before any partial state is
        persisted.

        Additive operations — new entity types, new relations, new properties,
        new relation patterns — go through unchanged.
        """
        if not schema.entities and not schema.relations:
            return await self.load()

        existing = await self.load()
        self._check_no_contradictions(existing, schema)

        for et in schema.entities:
            await self._upsert_entity_type(et)
        for rt in schema.relations:
            await self._upsert_relation_type(rt)

        return await self.load()

    @staticmethod
    def _check_no_contradictions(existing: GraphSchema, incoming: GraphSchema) -> None:
        """Raise :py:class:`OntologyContradictionError` on any type re-declaration."""
        existing_ent_types: dict[tuple[str, str], str] = {
            (e.label, p.name): p.type for e in existing.entities for p in e.properties
        }
        for et in incoming.entities:
            for p in et.properties:
                prior = existing_ent_types.get((et.label, p.name))
                if prior is not None and prior != p.type:
                    raise OntologyContradictionError(
                        f"Property '{et.label}.{p.name}' is already registered as "
                        f"{prior}; refusing to redefine as {p.type}. The ontology "
                        f"is additive — drop the data graph and start fresh if you "
                        f"need to change a property's type."
                    )

        existing_rel_types: dict[tuple[str, str], str] = {
            (r.label, p.name): p.type for r in existing.relations for p in r.properties
        }
        for rt in incoming.relations:
            for p in rt.properties:
                prior = existing_rel_types.get((rt.label, p.name))
                if prior is not None and prior != p.type:
                    raise OntologyContradictionError(
                        f"Property '{rt.label}.{p.name}' (on relation) is already "
                        f"registered as {prior}; refusing to redefine as {p.type}."
                    )

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
            "MATCH (r:OntologyRelationType {label: $label}) RETURN r.patterns AS patterns",
            {"label": rt.label},
        )
        existing_patterns: list[str] = []
        rows = getattr(result, "result_set", None) or []
        if isinstance(rows, list) and rows and isinstance(rows[0], list) and rows[0]:
            existing_patterns = list(rows[0][0] or [])
        seen: set[str] = set()
        merged: list[str] = []
        for s in existing_patterns + new_patterns:
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
        # Property nodes are keyed by ``(owner_label_kind, owner_label, name)``
        # so two different types can declare the same property name without
        # trampling each other's metadata.
        owner_alias = "ent" if owner_label == "OntologyEntityType" else "rel"
        await self._query(
            f"MATCH ({owner_alias}:{owner_label} {{label: $owner}}) "
            f"MERGE ({owner_alias})-[:HAS_PROPERTY]->"
            f"(p:OntologyProperty {{name: $name, owner: $owner_kind, owner_label: $owner}}) "
            "SET p.type = $type, "
            "p.description = coalesce($description, p.description)",
            {
                "owner": owner_label_value,
                "owner_kind": owner_label,
                "name": prop.name,
                "type": prop.type,
                "description": prop.description,
            },
        )

    # ── Delete ───────────────────────────────────────────────────

    async def delete_property(
        self, label: str, prop_name: str, *, on_relation: bool = False
    ) -> None:
        """Remove a property declaration from an entity or relation type.

        Existing values for this property on data-graph nodes are **not**
        touched — schema deletions are forward-only. After this returns,
        future extraction will not try to fill ``prop_name`` and Cypher
        generation will not list it.
        """
        owner_label = "OntologyRelationType" if on_relation else "OntologyEntityType"
        await self._query(
            f"MATCH (o:{owner_label} {{label: $label}})"
            "-[:HAS_PROPERTY]->(p:OntologyProperty {name: $name}) "
            "DETACH DELETE p",
            {"label": label, "name": prop_name},
        )

    async def delete_entity_type(self, label: str) -> None:
        """Remove an entity type and all of its declared properties from the
        ontology graph. Data-graph nodes with this label are untouched.
        """
        await self._query(
            "MATCH (e:OntologyEntityType {label: $label}) "
            "OPTIONAL MATCH (e)-[:HAS_PROPERTY]->(p:OntologyProperty) "
            "DETACH DELETE e, p",
            {"label": label},
        )

    async def delete_relation_type(self, label: str) -> None:
        """Remove a relation type and all of its declared properties from
        the ontology graph. Data-graph relationships are untouched.
        """
        await self._query(
            "MATCH (r:OntologyRelationType {label: $label}) "
            "OPTIONAL MATCH (r)-[:HAS_PROPERTY]->(p:OntologyProperty) "
            "DETACH DELETE r, p",
            {"label": label},
        )

    # ── Clear ────────────────────────────────────────────────────

    async def clear(self) -> None:
        """Drop the ontology graph (``GRAPH.DELETE``). Idempotent.

        Called from ``GraphRAG.delete_all()`` so the ontology graph never
        outlives the data graph.
        """
        self._conn._ensure_client()
        from redis.asyncio import Redis

        redis: Redis = Redis(connection_pool=self._conn._pool)
        try:
            await redis.execute_command("GRAPH.DELETE", self._graph_name)
        except Exception as exc:
            msg = str(exc).lower()
            if "empty" in msg or "invalid" in msg or "key" in msg:
                logger.debug("Ontology graph '%s' already empty", self._graph_name)
            else:
                raise
        self._graph = None
