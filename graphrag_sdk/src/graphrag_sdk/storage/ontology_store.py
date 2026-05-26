"""Persistent ontology storage in a dedicated FalkorDB graph.

The ontology lives in a separate FalkorDB graph named ``<data_graph>__ontology``
and is the **anchor** for the working ontology:

- **Always-on**: every :py:class:`GraphRAG` has exactly one ontology graph,
  created lazily on first use, dropped on ``delete_all()``.
- **Single source of truth**: retrieval, ``get_ontology()``, and any
  cross-process worker all read from the same graph.
- **Ingest path is constrained**. :py:meth:`register` admits:
    - new entity / relation labels with their declared properties and patterns,
    - re-declarations of existing labels with the *same* properties / patterns
      (or a strict subset — treated as "use the persisted definition").

  It refuses:
    - **type contradictions** on existing properties
      (:py:class:`OntologyContradictionError`), and
    - **modifications** to existing labels — adding properties or patterns
      (:py:class:`OntologyModificationNotAllowedError`).

  The latter is reserved for a future ontology-evolution API that updates the
  data graph in lockstep with the ontology, keeping the two aligned.

Users who want a curated, declarative ontology (descriptions, future flags,
properties not yet observed in the data) supply a ``ontology`` to ``GraphRAG``;
it gets registered into the ontology graph on first connection. JSON
import/export via :py:meth:`Ontology.save_to_file` / ``from_file`` is a
review / version-control bridge — the ontology graph is the canonical copy.
"""

from __future__ import annotations

import logging
from typing import Any

from graphrag_sdk.core.connection import FalkorDBConnection
from graphrag_sdk.core.models import (
    Attribute,
    Entity,
    Ontology,
    Relation,
)

logger = logging.getLogger(__name__)


class OntologyContradictionError(ValueError):
    """Raised when an incoming ontology re-types an existing property.

    Re-typing (e.g., ``Person.age`` STRING → INTEGER) is rejected before any
    partial state is persisted, so downstream Cypher queries can never break
    silently because someone changed a property's type out from under them.
    """


class OntologyModificationNotAllowedError(ValueError):
    """Raised when an incoming ontology tries to MODIFY an existing label.

    The ingest path is intentionally constrained: it can add brand-new labels
    (with whatever attributes the user wants) and re-declare existing labels
    *exactly* (or with a strict subset of their persisted properties / patterns).
    It cannot add attributes / patterns / fields to an already-registered
    label — that's a ontology-evolution operation that has to also update
    existing graph data to keep the two in sync, which is handled by a separate
    API path (not yet implemented).

    Workaround for v1: ``await rag.delete_all()`` and re-ingest with the new
    ontology, OR rely solely on the constructor's first ``register`` to lock in
    your full ontology from the start.
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


def _props_from_rows(rows: list[Any] | None) -> list[Attribute]:
    """Reconstruct Attribute objects from a ``collect(...)`` query result.

    Filters out the null-keyed dict FalkorDB returns for an OPTIONAL MATCH
    with no matches.
    """
    out: list[Attribute] = []
    for row in rows or []:
        if not row or not isinstance(row, dict):
            continue
        name = row.get("name")
        if not name:
            continue
        out.append(
            Attribute(
                name=name,
                type=row.get("type") or "STRING",
                description=row.get("description"),
            )
        )
    return out


class OntologyStore:
    """Persists and loads :py:class:`Ontology` in a dedicated FalkorDB graph.

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

    async def load(self) -> Ontology:
        """Read the ontology graph and reconstruct a :py:class:`Ontology`.

        Returns an empty ontology if the ontology graph does not yet exist or
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
            logger.debug("Ontology load failed (returning empty ontology): %s", exc)
            return Ontology()

        ent_rows = getattr(ent_result, "result_set", None) or []
        rel_rows = getattr(rel_result, "result_set", None) or []
        if not isinstance(ent_rows, list):
            ent_rows = []
        if not isinstance(rel_rows, list):
            rel_rows = []

        entities = [
            Entity(
                label=row[0],
                description=row[1],
                properties=_props_from_rows(row[2]),
            )
            for row in ent_rows
            if isinstance(row, list) and len(row) >= 3 and row[0]
        ]
        relations = [
            Relation(
                label=row[0],
                description=row[1],
                patterns=_decode_patterns(row[2]),
                properties=_props_from_rows(row[3]),
            )
            for row in rel_rows
            if isinstance(row, list) and len(row) >= 4 and row[0]
        ]
        return Ontology(entities=entities, relations=relations)

    # ── Register ─────────────────────────────────────────────────

    async def register(self, ontology: Ontology | None = None, **legacy: Any) -> Ontology:
        """Register ``ontology`` into the persisted ontology and return the union.

        The ingest path is constrained:

        - **New labels** — registered with their declared properties and
          (for relations) patterns.
        - **Existing labels** — incoming declaration must be a *strict subset*
          of what's already persisted. The same property names, the same
          types, the same (or fewer) relation patterns. Re-declaring a label
          with no properties is fine; it just means "I know this label exists,
          use the persisted definition."
        - Trying to **add a property** to an existing label →
          :py:class:`OntologyModificationNotAllowedError`.
        - Trying to **change the type** of an existing property →
          :py:class:`OntologyContradictionError`.

        Both errors are raised before any partial state is persisted.

        .. deprecated:: 1.2
            Passing the ``schema=`` keyword argument is deprecated; use
            ``ontology=`` (or pass positionally) instead.
        """
        if "schema" in legacy:
            import warnings

            legacy_schema = legacy.pop("schema")
            if legacy:
                raise TypeError(f"register() got unexpected keyword arguments: {sorted(legacy)}")
            if ontology is not None:
                raise TypeError(
                    "OntologyStore.register() received both `ontology=` and "
                    "`schema=`. Use `ontology=` only; `schema=` is deprecated."
                )
            warnings.warn(
                "The `schema=` keyword argument on OntologyStore.register() has "
                "been renamed to `ontology=` (graphrag_sdk v1.2+). Update your "
                "call site — the alias will be removed in a future release.",
                DeprecationWarning,
                stacklevel=2,
            )
            ontology = legacy_schema
        elif legacy:
            raise TypeError(f"register() got unexpected keyword arguments: {sorted(legacy)}")

        if ontology is None:
            raise TypeError("register() missing required argument: 'ontology'")
        if not ontology.entities and not ontology.relations:
            return await self.load()

        existing = await self.load()
        self._check_no_contradictions(existing, ontology)
        self._check_no_modifications_to_existing(existing, ontology)

        for et in ontology.entities:
            await self._upsert_entity_type(et)
        for rt in ontology.relations:
            await self._upsert_relation_type(rt)

        return await self.load()

    @staticmethod
    def _check_no_contradictions(existing: Ontology, incoming: Ontology) -> None:
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
                        f"{prior}; refusing to redefine as {p.type}. Type changes "
                        f"on existing properties are not supported via the ingest "
                        f"path; drop the data graph and start fresh if you need to."
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

    @staticmethod
    def _check_no_modifications_to_existing(existing: Ontology, incoming: Ontology) -> None:
        """Raise :py:class:`OntologyModificationNotAllowedError` when incoming
        tries to add new properties / patterns to a label that's already
        registered. New labels are unaffected.

        Subset re-declarations are allowed: an existing label may appear in
        the incoming ontology with fewer (or zero) properties — that's treated
        as "use the persisted definition" rather than "remove things."
        """
        existing_ent_by_label = {e.label: e for e in existing.entities}
        for et in incoming.entities:
            prior_et = existing_ent_by_label.get(et.label)
            if prior_et is None:
                continue
            prior_prop_names = {p.name for p in prior_et.properties}
            new_prop_names = {p.name for p in et.properties} - prior_prop_names
            if new_prop_names:
                raise OntologyModificationNotAllowedError(
                    f"Refusing to add new attribute(s) {sorted(new_prop_names)} "
                    f"to existing label '{et.label}'. The ingest path only "
                    f"accepts new labels; modifying an existing label requires "
                    f"a ontology-evolution operation (not yet supported). "
                    f"Workaround: `await rag.delete_all()` and re-ingest with "
                    f"the updated ontology."
                )

        existing_rel_by_label = {r.label: r for r in existing.relations}
        for rt in incoming.relations:
            prior_rt = existing_rel_by_label.get(rt.label)
            if prior_rt is None:
                continue
            prior_prop_names = {p.name for p in prior_rt.properties}
            new_prop_names = {p.name for p in rt.properties} - prior_prop_names
            prior_patterns = set(prior_rt.patterns)
            new_patterns = set(rt.patterns) - prior_patterns
            if new_prop_names or new_patterns:
                diffs: list[str] = []
                if new_prop_names:
                    diffs.append(f"new properties {sorted(new_prop_names)}")
                if new_patterns:
                    diffs.append(f"new patterns {sorted(new_patterns)}")
                raise OntologyModificationNotAllowedError(
                    f"Refusing to add {' and '.join(diffs)} to existing relation "
                    f"'{rt.label}'. The ingest path only accepts new relation "
                    f"types; modifying an existing one requires a ontology-evolution "
                    f"operation (not yet supported)."
                )

    async def _upsert_entity_type(self, et: Entity) -> None:
        await self._query(
            "MERGE (e:OntologyEntityType {label: $label}) "
            "SET e.description = coalesce($description, e.description)",
            {"label": et.label, "description": et.description},
        )
        for prop in et.properties:
            await self._upsert_property(et.label, prop, owner_label="OntologyEntityType")

    async def _upsert_relation_type(self, rt: Relation) -> None:
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
        self, owner_label_value: str, prop: Attribute, *, owner_label: str
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


# ── Deprecation aliases ──────────────────────────────────────────


def __getattr__(name: str):  # PEP 562
    if name == "SchemaModificationNotAllowedError":
        import warnings

        warnings.warn(
            "`SchemaModificationNotAllowedError` has been renamed to "
            "`OntologyModificationNotAllowedError` (graphrag_sdk v1.2+). "
            "Update your imports — the alias will be removed in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )
        return OntologyModificationNotAllowedError
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
