"""Ontology inference from the live data graph.

The schema is **derived** from what's in the data graph, not maintained in a
separate persistent graph. This keeps the architecture honest: the source of
truth for "what entities and relations exist" is the data itself.

Two consumers:
- Retrieval reads the inferred schema each session to build the Cypher prompt.
- ``GraphRAG.get_ontology()`` returns it for inspection.

Users who want a curated, declarative schema (descriptions, not-yet-extracted
properties) pass a ``local_schema`` to ``GraphRAG`` — it's unioned with the
inferred schema at retrieval time so declared metadata survives.
``GraphSchema.save_to_file`` / ``GraphSchema.from_file`` cover the
schema-as-config workflow.
"""

from __future__ import annotations

import logging
from typing import Any

from graphrag_sdk.core.connection import FalkorDBConnection
from graphrag_sdk.core.models import (
    RESERVED_PROPERTY_NAMES,
    EntityType,
    GraphSchema,
    PropertyType,
    RelationType,
)

logger = logging.getLogger(__name__)


# Labels created by the SDK that are not user entities.
_STRUCTURAL_LABELS: frozenset[str] = frozenset({"Chunk", "Document", "__Entity__"})

# Edge labels created by the SDK that are not user relations.
_STRUCTURAL_REL_TYPES: frozenset[str] = frozenset({"PART_OF", "NEXT_CHUNK", "MENTIONED_IN"})

# Property keys we never want to expose to the LLM as "custom attributes".
# These are SDK-internal or reserved meanings; the Cypher prompt already
# emits the reserved ones it cares about (``name``, ``description``, etc.).
_INFER_SKIP_KEYS: frozenset[str] = RESERVED_PROPERTY_NAMES | frozenset(
    {"content_hash", "path", "text", "uid", "index", "metadata", "embedding"}
)


# FalkorDB ``typeof()`` returns lowercase strings; map to our PropertyType vocabulary.
_TYPE_MAP: dict[str, str] = {
    "string": "STRING",
    "integer": "INTEGER",
    "double": "FLOAT",
    "float": "FLOAT",
    "boolean": "BOOLEAN",
    "array": "LIST",
    "list": "LIST",
}


def _normalize_type(raw: str | None) -> str | None:
    if not raw:
        return None
    return _TYPE_MAP.get(raw.strip().lower())


class OntologyStore:
    """Infers the working :py:class:`GraphSchema` from the data graph.

    No separate FalkorDB graph: this introspects the data graph directly via
    ``db.labels()`` / ``db.relationshipTypes()`` plus per-label sampling of
    property keys and types.
    """

    def __init__(self, connection: FalkorDBConnection) -> None:
        self._conn = connection

    async def infer(self, *, sample_size: int = 200) -> GraphSchema:
        """Build a :py:class:`GraphSchema` from what is currently in the data graph.

        ``sample_size`` caps the per-label scan used to discover property keys
        and types. Returns an empty schema on any introspection failure.
        """
        try:
            labels_result = await self._conn.query("CALL db.labels()")
            rel_types_result = await self._conn.query("CALL db.relationshipTypes()")
        except Exception as exc:
            logger.debug("Ontology inference: labels/types query failed: %s", exc)
            return GraphSchema()

        labels: list[str] = [
            row[0]
            for row in (labels_result.result_set or [])
            if row and row[0] and row[0] not in _STRUCTURAL_LABELS
        ]
        rel_types: list[str] = [
            row[0]
            for row in (rel_types_result.result_set or [])
            if row and row[0] and row[0] not in _STRUCTURAL_REL_TYPES
        ]

        entities = [
            EntityType(
                label=label,
                properties=await self._properties_for_node(label, sample_size),
            )
            for label in labels
        ]
        # The unified data model writes every user relation as a ``RELATES``
        # edge whose ``rel_type`` property carries the original label; the
        # SDK's structural edges (PART_OF/NEXT_CHUNK/MENTIONED_IN) are excluded
        # above. We surface the distinct ``rel_type`` values as RelationTypes
        # so the Cypher prompt knows the allowed values, and expose their
        # property keys + endpoint patterns.
        relations: list[RelationType] = []
        if "RELATES" in rel_types:
            relations = await self._infer_relates_subtypes(sample_size)

        return GraphSchema(entities=entities, relations=relations)

    async def _properties_for_node(self, label: str, sample_size: int) -> list[PropertyType]:
        try:
            result = await self._conn.query(
                f"MATCH (n:`{label}`) "
                "WITH n LIMIT $limit "
                "UNWIND keys(n) AS k "
                "WITH k, typeof(n[k]) AS t "
                "RETURN k AS key, t AS type, count(*) AS c "
                "ORDER BY c DESC",
                {"limit": sample_size},
            )
        except Exception as exc:
            logger.debug("Ontology inference: properties query failed for %s: %s", label, exc)
            return []
        return _props_from_rows(result.result_set)

    async def _infer_relates_subtypes(self, sample_size: int) -> list[RelationType]:
        """Group ``RELATES`` edges by ``rel_type`` and infer per-subtype properties."""
        try:
            subtypes_result = await self._conn.query(
                "MATCH ()-[r:RELATES]->() "
                "WITH r LIMIT $limit "
                "WITH DISTINCT r.rel_type AS rel_type "
                "WHERE rel_type IS NOT NULL "
                "RETURN rel_type",
                {"limit": sample_size * 5},  # broader pool to capture rare subtypes
            )
        except Exception as exc:
            logger.debug("Ontology inference: RELATES subtypes query failed: %s", exc)
            return []

        relations: list[RelationType] = []
        for row in subtypes_result.result_set or []:
            subtype = row[0]
            if not subtype:
                continue
            properties = await self._properties_for_relates_subtype(subtype, sample_size)
            patterns = await self._patterns_for_relates_subtype(subtype)
            relations.append(RelationType(label=subtype, patterns=patterns, properties=properties))
        return relations

    async def _properties_for_relates_subtype(
        self, subtype: str, sample_size: int
    ) -> list[PropertyType]:
        try:
            result = await self._conn.query(
                "MATCH ()-[r:RELATES {rel_type: $sub}]->() "
                "WITH r LIMIT $limit "
                "UNWIND keys(r) AS k "
                "WITH k, typeof(r[k]) AS t "
                "RETURN k AS key, t AS type, count(*) AS c "
                "ORDER BY c DESC",
                {"sub": subtype, "limit": sample_size},
            )
        except Exception as exc:
            logger.debug(
                "Ontology inference: relation properties query failed for %s: %s",
                subtype,
                exc,
            )
            return []
        return _props_from_rows(result.result_set)

    async def _patterns_for_relates_subtype(self, subtype: str) -> list[tuple[str, str]]:
        try:
            result = await self._conn.query(
                "MATCH (a)-[r:RELATES {rel_type: $sub}]->(b) "
                "WITH labels(a) AS la, labels(b) AS lb "
                "RETURN DISTINCT la, lb LIMIT 25",
                {"sub": subtype},
            )
        except Exception as exc:
            logger.debug(
                "Ontology inference: endpoint patterns query failed for %s: %s",
                subtype,
                exc,
            )
            return []
        patterns: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for row in result.result_set or []:
            src_labels, tgt_labels = row[0] or [], row[1] or []
            src = next(
                (lbl for lbl in src_labels if lbl not in _STRUCTURAL_LABELS),
                None,
            )
            tgt = next(
                (lbl for lbl in tgt_labels if lbl not in _STRUCTURAL_LABELS),
                None,
            )
            if not src or not tgt:
                continue
            key = (src, tgt)
            if key in seen:
                continue
            seen.add(key)
            patterns.append(key)
        return patterns


def _props_from_rows(rows: list[list[Any]] | None) -> list[PropertyType]:
    """Turn ``(key, typeof, count)`` rows into :py:class:`PropertyType` objects.

    Skips reserved/system keys and unmappable types so they never leak into the
    LLM-facing schema.
    """
    out: list[PropertyType] = []
    seen: set[str] = set()
    for row in rows or []:
        if not row or len(row) < 2:
            continue
        key, raw_type = row[0], row[1]
        if not isinstance(key, str) or key in _INFER_SKIP_KEYS or key in seen:
            continue
        normalized = _normalize_type(raw_type if isinstance(raw_type, str) else None)
        if not normalized:
            continue
        seen.add(key)
        out.append(PropertyType(name=key, type=normalized))
    return out
