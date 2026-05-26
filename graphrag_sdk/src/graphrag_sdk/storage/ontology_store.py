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

The ontology graph **accumulates new labels** across ingest passes, but it is
not free-form additive: existing labels are frozen by :py:meth:`register` to
keep the ontology and the data graph aligned.

Users who want a curated, declarative ontology (descriptions, future flags,
properties not yet observed in the data) supply an ``ontology`` to
``GraphRAG``; it gets registered into the ontology graph on first connection.
JSON import/export via :py:meth:`Ontology.save_to_file` / ``from_file`` is a
review / version-control bridge — the ontology graph is the canonical copy.

On-graph shape
--------------
Three node types, connected like a schema diagram::

    (:Entity   {label, description})
    (:Relation {label, description})
    (:Property {label, type, description})

    (:Entity)-[:HAS_PROPERTY]->(:Property)
    (:Relation)-[:SOURCE]->(:Entity)
    (:Relation)-[:TARGET]->(:Entity)
    (:Relation)-[:HAS_PROPERTY]->(:Property)

A relation with N declared patterns materialises as N ``Relation`` nodes (one
per ``(src, tgt)`` triple), each carrying its own ``SOURCE``/``TARGET`` /
``HAS_PROPERTY`` edges. A relation declared without patterns (open mode)
materialises as a single ``Relation`` node with no ``SOURCE`` / ``TARGET``
edges. ``Property`` nodes are scoped to the specific owner that declares
them — no sharing across labels.
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
    label — that's an ontology-evolution operation that has to also update
    existing graph data to keep the two in sync, which is handled by a separate
    API path (not yet implemented).

    Workaround for v1: ``await rag.delete_all()`` and re-ingest with the new
    ontology, OR rely solely on the constructor's first ``register`` to lock in
    your full ontology from the start.
    """


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
        """Read the ontology graph and reconstruct an :py:class:`Ontology`.

        Returns an empty ontology if the graph doesn't exist or introspection
        fails. Failure is logged at DEBUG so an unconfigured ``GraphRAG``
        instance doesn't emit warnings.
        """
        try:
            ent_result = await self._query(
                "MATCH (e:Entity) RETURN e.label AS label, e.description AS description"
            )
            ent_prop_result = await self._query(
                "MATCH (e:Entity)-[:HAS_PROPERTY]->(p:Property) "
                "RETURN e.label AS owner, p.label AS name, p.type AS type, "
                "p.description AS description"
            )
            patterned_rel_result = await self._query(
                "MATCH (r:Relation)-[:SOURCE]->(s:Entity), (r)-[:TARGET]->(t:Entity) "
                "RETURN r.label AS rel_label, r.description AS description, "
                "s.label AS src, t.label AS tgt"
            )
            open_rel_result = await self._query(
                "MATCH (r:Relation) WHERE NOT (r)-[:SOURCE]->() "
                "RETURN r.label AS rel_label, r.description AS description"
            )
            rel_prop_result = await self._query(
                "MATCH (r:Relation)-[:HAS_PROPERTY]->(p:Property) "
                "RETURN r.label AS owner, p.label AS name, p.type AS type, "
                "p.description AS description"
            )
        except Exception as exc:
            logger.debug("Ontology load failed (returning empty ontology): %s", exc)
            return Ontology()

        def _rows(result: Any) -> list[Any]:
            rows = getattr(result, "result_set", None) or []
            return rows if isinstance(rows, list) else []

        # Entity properties, deduplicated per (owner, name)
        ent_props_by_owner: dict[str, dict[str, Attribute]] = {}
        for row in _rows(ent_prop_result):
            if not (isinstance(row, list) and len(row) >= 4 and row[0] and row[1]):
                continue
            owner, name, type_, desc = row
            bucket = ent_props_by_owner.setdefault(owner, {})
            if name not in bucket:
                bucket[name] = Attribute(name=name, type=type_ or "STRING", description=desc)

        entities = [
            Entity(
                label=row[0],
                description=row[1],
                properties=list(ent_props_by_owner.get(row[0], {}).values()),
            )
            for row in _rows(ent_result)
            if isinstance(row, list) and len(row) >= 2 and row[0]
        ]

        # Relation properties, deduplicated per (owner, name) across all
        # Relation nodes with the same label
        rel_props_by_owner: dict[str, dict[str, Attribute]] = {}
        for row in _rows(rel_prop_result):
            if not (isinstance(row, list) and len(row) >= 4 and row[0] and row[1]):
                continue
            owner, name, type_, desc = row
            bucket = rel_props_by_owner.setdefault(owner, {})
            if name not in bucket:
                bucket[name] = Attribute(name=name, type=type_ or "STRING", description=desc)

        # Group patterned relations by label, collecting (src, tgt) pairs
        rel_by_label: dict[str, dict[str, Any]] = {}
        for row in _rows(patterned_rel_result):
            if not (isinstance(row, list) and len(row) >= 4 and row[0]):
                continue
            rel_label, description, src, tgt = row
            entry = rel_by_label.setdefault(rel_label, {"description": description, "patterns": []})
            if src and tgt:
                entry["patterns"].append((src, tgt))
            if description:
                entry["description"] = description

        # Open-mode relations (Relation node with no SOURCE edge)
        for row in _rows(open_rel_result):
            if not (isinstance(row, list) and len(row) >= 2 and row[0]):
                continue
            rel_label, description = row
            rel_by_label.setdefault(rel_label, {"description": description, "patterns": []})

        relations = [
            Relation(
                label=label,
                description=entry["description"],
                patterns=entry["patterns"],
                properties=list(rel_props_by_owner.get(label, {}).values()),
            )
            for label, entry in rel_by_label.items()
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
                    f"an ontology-evolution operation (not yet supported). "
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
                    f"types; modifying an existing one requires an ontology-evolution "
                    f"operation (not yet supported)."
                )

    async def _upsert_entity_type(self, et: Entity) -> None:
        """Upsert an ``:Entity`` node + its ``:Property`` children.

        Each declared attribute becomes a separate ``:Property`` node attached
        via ``HAS_PROPERTY``. Property nodes are scoped to this entity (the
        MERGE pattern includes the owning edge), so ``Person.age`` and
        ``Mountain.age`` are distinct nodes — there's no accidental sharing
        across labels.
        """
        await self._query(
            "MERGE (e:Entity {label: $label}) "
            "SET e.description = coalesce($description, e.description)",
            {"label": et.label, "description": et.description},
        )
        for prop in et.properties:
            await self._upsert_entity_property(et.label, prop)

    async def _upsert_entity_property(self, entity_label: str, prop: Attribute) -> None:
        await self._query(
            "MATCH (e:Entity {label: $owner}) "
            "MERGE (e)-[:HAS_PROPERTY]->(p:Property {label: $name}) "
            "SET p.type = $type, "
            "p.description = coalesce($description, p.description)",
            {
                "owner": entity_label,
                "name": prop.name,
                "type": prop.type,
                "description": prop.description,
            },
        )

    async def _upsert_relation_type(self, rt: Relation) -> None:
        """Upsert one ``:Relation`` node per declared pattern.

        Each pattern materialises as a distinct ``:Relation`` node with
        ``SOURCE``/``TARGET`` edges to the relevant ``:Entity`` nodes. Each
        ``:Relation`` node carries its own ``HAS_PROPERTY`` edges to its
        ``:Property`` children — a small amount of duplication, but it keeps
        every pattern self-contained and the load query simple.

        An open-mode relation (no patterns) materialises as a single
        ``:Relation`` node with no ``SOURCE``/``TARGET`` edges.
        """
        if rt.patterns:
            for src, tgt in rt.patterns:
                await self._query(
                    "MERGE (s:Entity {label: $src}) "
                    "MERGE (t:Entity {label: $tgt}) "
                    "MERGE (s)<-[:SOURCE]-(r:Relation {label: $rel_label})-[:TARGET]->(t) "
                    "SET r.description = coalesce($description, r.description)",
                    {
                        "src": src,
                        "tgt": tgt,
                        "rel_label": rt.label,
                        "description": rt.description,
                    },
                )
                for prop in rt.properties:
                    await self._upsert_relation_property(rt.label, src, tgt, prop)
        else:
            await self._query(
                "MERGE (r:Relation {label: $rel_label}) "
                "SET r.description = coalesce($description, r.description)",
                {"rel_label": rt.label, "description": rt.description},
            )
            for prop in rt.properties:
                await self._upsert_open_relation_property(rt.label, prop)

    async def _upsert_relation_property(
        self, rel_label: str, src: str, tgt: str, prop: Attribute
    ) -> None:
        await self._query(
            "MATCH (s:Entity {label: $src})<-[:SOURCE]-(r:Relation {label: $rel_label})"
            "-[:TARGET]->(t:Entity {label: $tgt}) "
            "MERGE (r)-[:HAS_PROPERTY]->(p:Property {label: $name}) "
            "SET p.type = $type, "
            "p.description = coalesce($description, p.description)",
            {
                "rel_label": rel_label,
                "src": src,
                "tgt": tgt,
                "name": prop.name,
                "type": prop.type,
                "description": prop.description,
            },
        )

    async def _upsert_open_relation_property(self, rel_label: str, prop: Attribute) -> None:
        """Attach a Property to an open-mode Relation node (no SOURCE/TARGET)."""
        await self._query(
            "MATCH (r:Relation {label: $rel_label}) "
            "WHERE NOT (r)-[:SOURCE]->() "
            "MERGE (r)-[:HAS_PROPERTY]->(p:Property {label: $name}) "
            "SET p.type = $type, "
            "p.description = coalesce($description, p.description)",
            {
                "rel_label": rel_label,
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
