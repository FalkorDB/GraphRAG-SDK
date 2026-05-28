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
from typing import Any, Literal

from graphrag_sdk.core.connection import FalkorDBConnection
from graphrag_sdk.core.models import (
    Attribute,
    Entity,
    Ontology,
    Relation,
)

OwnerKind = Literal["entity", "relation"]
DescriptionKind = Literal["entity", "relation", "entity_property", "relation_property"]

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

    # ── Evolution primitives (bypass register() strictness) ──────
    #
    # These lower-level methods are called by ``GraphRAG`` ontology-evolution
    # APIs (``rename_entity``, ``add_attribute``, ``drop_relation``, …). They
    # deliberately mutate already-registered labels in ways ``register()``
    # rejects, because the public ``GraphRAG`` methods are responsible for
    # keeping the data graph in lockstep (or for explicitly running a backfill
    # to populate new structure on existing data).

    async def _check_no_property_retype(
        self,
        owner_kind: OwnerKind,
        owner_label: str,
        attribute: Attribute,
    ) -> None:
        """Refuse to silently change the declared type of an existing property.

        Raised pre-write so a stray ``add_entity_property(... type="STRING")``
        on an existing INTEGER property doesn't silently retype the schema
        (and leave the data graph mistyped). To deliberately change a
        property's type, callers must go through
        ``GraphRAG.drop_attribute`` + ``GraphRAG.add_attribute`` with the
        new type — the LLM re-derives values from chunks, keeping the
        data and ontology aligned.
        """
        owner_lbl = "Entity" if owner_kind == "entity" else "Relation"
        result = await self._query(
            f"MATCH (o:{owner_lbl} {{label: $owner}})-[:HAS_PROPERTY]->"
            "(p:Property {label: $name}) "
            "RETURN p.type AS type LIMIT 1",
            {"owner": owner_label, "name": attribute.name},
        )
        rows = getattr(result, "result_set", None) or []
        if rows and rows[0] and rows[0][0] and rows[0][0] != attribute.type:
            raise OntologyContradictionError(
                f"Property '{owner_label}.{attribute.name}' is already registered "
                f"as {rows[0][0]}; refusing to redefine as {attribute.type}. "
                f"To change the type, call drop_attribute() then add_attribute() "
                f"with the new type — the LLM will re-derive values."
            )

    async def add_entity_property(self, entity_label: str, attribute: Attribute) -> None:
        """Attach a new ``Property`` node to an existing ``Entity``.

        Idempotent on identical re-declarations. Raises
        :py:class:`OntologyContradictionError` if the same property name
        already exists with a different type — the caller must explicitly
        ``GraphRAG.drop_attribute`` + ``GraphRAG.add_attribute`` with the
        new type to change a type. Description is coalesced (existing
        wins unless the caller supplies a non-null one).
        """
        await self._check_no_property_retype("entity", entity_label, attribute)
        await self._upsert_entity_property(entity_label, attribute)

    async def add_relation_property(self, rel_label: str, attribute: Attribute) -> None:
        """Attach a new ``Property`` node to every ``Relation`` node carrying
        ``rel_label`` — i.e. every pattern of that relation, plus the
        open-mode node if any.

        Mirrors how :py:meth:`_upsert_relation_type` lays out properties
        per-pattern, so the property appears uniformly across all patterns
        of the relation. Raises :py:class:`OntologyContradictionError` on
        type re-declaration (same semantics as
        :py:meth:`add_entity_property`).
        """
        await self._check_no_property_retype("relation", rel_label, attribute)
        await self._query(
            "MATCH (r:Relation {label: $rel_label}) "
            "MERGE (r)-[:HAS_PROPERTY]->(p:Property {label: $name}) "
            "ON CREATE SET p.type = $type "
            "SET p.description = coalesce($description, p.description)",
            {
                "rel_label": rel_label,
                "name": attribute.name,
                "type": attribute.type,
                "description": attribute.description,
            },
        )

    async def add_relation_pattern_node(
        self, rel_label: str, src: str, tgt: str, *, description: str | None = None
    ) -> None:
        """Add a new ``(src, tgt)`` pattern to an existing relation label.

        MERGEs source/target ``Entity`` nodes (idempotent — if they exist they
        are reused), creates a fresh ``:Relation`` pattern node, and copies
        properties from any existing pattern node of the same label so the
        new pattern lines up with siblings.
        """
        await self._query(
            "MERGE (s:Entity {label: $src}) "
            "MERGE (t:Entity {label: $tgt}) "
            "MERGE (s)<-[:SOURCE]-(r:Relation {label: $rel_label})-[:TARGET]->(t) "
            "SET r.description = coalesce($description, r.description)",
            {"src": src, "tgt": tgt, "rel_label": rel_label, "description": description},
        )
        # Copy properties from any sibling pattern node, if one exists.
        # The MERGE above may have created a fresh Relation node; ensure
        # any properties declared on the original carry over to the new
        # pattern so retrieval-time prompts surface identical attrs.
        await self._query(
            "MATCH (existing:Relation {label: $rel_label})-[:HAS_PROPERTY]->(p:Property) "
            "MATCH (new:Relation {label: $rel_label})-[:SOURCE]->(s:Entity {label: $src}) "
            "MATCH (new)-[:TARGET]->(t:Entity {label: $tgt}) "
            "WHERE existing <> new "
            "MERGE (new)-[:HAS_PROPERTY]->(p2:Property {label: p.label}) "
            "SET p2.type = p.type, p2.description = p.description",
            {"rel_label": rel_label, "src": src, "tgt": tgt},
        )

    async def rename_entity_label(self, old: str, new: str) -> None:
        """Rename the ``:Entity`` label on the ontology graph.

        Only updates the ``label`` property on the ontology graph's
        ``:Entity`` node — the underlying data-graph rename is the
        caller's responsibility (handled by ``GraphRAG.rename_entity``).
        """
        await self._query(
            "MATCH (e:Entity {label: $old}) SET e.label = $new",
            {"old": old, "new": new},
        )

    async def rename_relation_label(self, old: str, new: str) -> None:
        """Rename the ``:Relation`` label on the ontology graph.

        Updates every ``:Relation`` node carrying this label (i.e. all
        declared patterns of the relation, plus the open-mode node if any).
        """
        await self._query(
            "MATCH (r:Relation {label: $old}) SET r.label = $new",
            {"old": old, "new": new},
        )

    async def rename_property_label(
        self,
        owner_kind: OwnerKind,
        owner_label: str,
        old_name: str,
        new_name: str,
    ) -> None:
        """Rename a property hanging off an Entity or every Relation node
        with the given label.

        Properties are scoped per-owner in the schema graph (one ``:Property``
        node per owner via the ``HAS_PROPERTY`` edge), so the MATCH pattern
        ensures we only touch the right node and never collide with
        same-named properties on a different owner.
        """
        owner_label_cypher = "Entity" if owner_kind == "entity" else "Relation"
        await self._query(
            f"MATCH (o:{owner_label_cypher} {{label: $owner}})-[:HAS_PROPERTY]->"
            "(p:Property {label: $old_name}) "
            "SET p.label = $new_name",
            {"owner": owner_label, "old_name": old_name, "new_name": new_name},
        )

    async def set_description(
        self,
        kind: DescriptionKind,
        label: str,
        description: str | None,
        *,
        owner_label: str | None = None,
    ) -> None:
        """Update the ``description`` on an Entity, Relation, or Property node.

        For ``kind in {"entity", "relation"}``, ``label`` identifies the
        node directly. For ``kind in {"entity_property", "relation_property"}``,
        ``label`` is the property name and ``owner_label`` MUST be passed —
        property nodes are scoped per owner so we always disambiguate.
        Setting ``description=None`` clears the description.
        """
        if kind == "entity":
            await self._query(
                "MATCH (e:Entity {label: $label}) SET e.description = $description",
                {"label": label, "description": description},
            )
        elif kind == "relation":
            await self._query(
                "MATCH (r:Relation {label: $label}) SET r.description = $description",
                {"label": label, "description": description},
            )
        elif kind in ("entity_property", "relation_property"):
            if owner_label is None:
                raise ValueError(f"set_description(kind='{kind}', ...) requires owner_label")
            owner_lbl = "Entity" if kind == "entity_property" else "Relation"
            await self._query(
                f"MATCH (o:{owner_lbl} {{label: $owner}})-[:HAS_PROPERTY]->"
                "(p:Property {label: $name}) "
                "SET p.description = $description",
                {"owner": owner_label, "name": label, "description": description},
            )
        else:
            raise ValueError(f"Unknown description kind: {kind}")

    async def drop_entity_property(self, entity_label: str, prop_name: str) -> None:
        """Remove a property declaration from an Entity in the ontology graph.

        DETACH DELETEs the ``:Property`` node scoped to this entity. Properties
        are owner-scoped so a same-named property on another entity is
        untouched.
        """
        await self._query(
            "MATCH (e:Entity {label: $owner})-[:HAS_PROPERTY]->(p:Property {label: $name}) "
            "DETACH DELETE p",
            {"owner": entity_label, "name": prop_name},
        )

    async def drop_relation_property(self, rel_label: str, prop_name: str) -> None:
        """Remove a property declaration from every Relation node with this label."""
        await self._query(
            "MATCH (r:Relation {label: $owner})-[:HAS_PROPERTY]->(p:Property {label: $name}) "
            "DETACH DELETE p",
            {"owner": rel_label, "name": prop_name},
        )

    async def drop_entity_label(self, label: str) -> None:
        """Remove an entity from the ontology graph entirely.

        Cascades:
        - Property nodes attached to this Entity → deleted.
        - Relation pattern nodes that referenced this Entity as SOURCE or
          TARGET → deleted along with their property children (the pattern is
          no longer expressible without one of its endpoints).
        """
        # Drop property children first (they're owner-scoped).
        await self._query(
            "MATCH (e:Entity {label: $label})-[:HAS_PROPERTY]->(p:Property) DETACH DELETE p",
            {"label": label},
        )
        # Drop Relation pattern nodes that pointed at this entity, and their
        # property children.  Two passes because the property children hang
        # off the Relation node, not the Entity.
        await self._query(
            "MATCH (r:Relation)-[:SOURCE|TARGET]->(e:Entity {label: $label}) "
            "OPTIONAL MATCH (r)-[:HAS_PROPERTY]->(p:Property) "
            "DETACH DELETE r, p",
            {"label": label},
        )
        await self._query(
            "MATCH (e:Entity {label: $label}) DETACH DELETE e",
            {"label": label},
        )

    async def drop_relation_label(self, label: str) -> None:
        """Remove a relation from the ontology graph entirely.

        Cascades all Relation pattern nodes carrying this label plus their
        property children.
        """
        await self._query(
            "MATCH (r:Relation {label: $label}) "
            "OPTIONAL MATCH (r)-[:HAS_PROPERTY]->(p:Property) "
            "DETACH DELETE r, p",
            {"label": label},
        )

    async def drop_relation_pattern_node(self, rel_label: str, src: str, tgt: str) -> None:
        """Remove a single ``(src, tgt)`` pattern of a relation.

        Other patterns of the same relation label are untouched.
        """
        await self._query(
            "MATCH (s:Entity {label: $src})<-[:SOURCE]-(r:Relation {label: $rel_label})"
            "-[:TARGET]->(t:Entity {label: $tgt}) "
            "OPTIONAL MATCH (r)-[:HAS_PROPERTY]->(p:Property) "
            "DETACH DELETE r, p",
            {"rel_label": rel_label, "src": src, "tgt": tgt},
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
