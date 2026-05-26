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
keep the ontology and the data graph aligned. A future ontology-evolution
API will support extending existing labels in lockstep with data updates.

Users who want a curated, declarative ontology (descriptions, future flags,
properties not yet observed in the data) supply an ``ontology`` to
``GraphRAG``; it gets registered into the ontology graph on first
connection. JSON import/export via :py:meth:`Ontology.save_to_file` /
``from_file`` is a review / version-control bridge — the ontology graph is
the canonical copy.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from graphrag_sdk.core.connection import FalkorDBConnection
from graphrag_sdk.core.models import (
    Attribute,
    Entity,
    Ontology,
    Relation,
)
from graphrag_sdk.utils.cypher import sanitize_cypher_label

# Marker label every ontology-graph node carries, so MATCH queries can pick
# them out independently of the user-declared label. Defensive — the ontology
# graph is in its own FalkorDB graph already, but the marker lets us evolve
# the shape without scanning everything.
_ONTOLOGY_LABEL = "__Ontology"

# A relation type can be declared without any patterns (open mode — applies
# to any entity-type pair). We materialise those as self-loops on a single
# placeholder node so every relation is still an edge in the ontology graph.
_OPEN_RELATION_LABEL = "__OpenRelation"

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


def _encode_attributes(props: list[Attribute]) -> str:
    """JSON-encode declared attributes for storage as a single node/edge property.

    FalkorDB allows scalar / list properties on nodes and edges; nested maps
    aren't durable across all versions, so we serialise to a JSON string and
    decode on read. Empty ``props`` → ``"{}"`` so the property is always
    present (lets ``coalesce()`` work in MERGE upserts).
    """
    if not props:
        return "{}"
    return json.dumps(
        {
            p.name: (
                {"type": p.type, "description": p.description}
                if p.description
                else {"type": p.type}
            )
            for p in props
        }
    )


def _decode_attributes(raw: Any) -> list[Attribute]:
    """Reconstruct ``Attribute`` objects from the JSON-encoded ``attributes`` prop."""
    if not raw or not isinstance(raw, str):
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(data, dict):
        return []
    out: list[Attribute] = []
    for name, info in data.items():
        if not name:
            continue
        if isinstance(info, dict):
            out.append(
                Attribute(
                    name=name,
                    type=info.get("type") or "STRING",
                    description=info.get("description"),
                )
            )
        elif isinstance(info, str):
            # Tolerant of an earlier shape where the value was just a type.
            out.append(Attribute(name=name, type=info))
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
        """Read the ontology graph and reconstruct an :py:class:`Ontology`.

        Returns an empty ontology if the graph doesn't exist or introspection
        fails. Failure is logged at DEBUG so an unconfigured ``GraphRAG``
        instance doesn't emit warnings.

        The on-graph shape (Option B):

        - Each entity type is a node carrying both its user label and
          ``:__Ontology``. The declared :py:class:`Attribute`\\s sit in a
          single JSON-encoded ``attributes`` property on the node.
        - Each relation type is a real edge between two entity-type nodes;
          one edge per declared ``(src, tgt)`` pattern. The relation's own
          declared attributes are JSON-encoded into an ``attributes`` prop
          on the edge.
        - Relations declared without patterns (open mode) appear as self-loops
          on a single ``:__OpenRelation:__Ontology`` placeholder node.
        """
        try:
            ent_result = await self._query(
                f"MATCH (e:`{_ONTOLOGY_LABEL}`) "
                f"WHERE NOT e:`{_OPEN_RELATION_LABEL}` "
                "RETURN e.label AS label, e.description AS description, "
                "e.attributes AS attributes"
            )
            rel_result = await self._query(
                f"MATCH (s:`{_ONTOLOGY_LABEL}`)-[r]->(t:`{_ONTOLOGY_LABEL}`) "
                f"WHERE NOT s:`{_OPEN_RELATION_LABEL}` "
                f"AND NOT t:`{_OPEN_RELATION_LABEL}` "
                "RETURN s.label AS src, t.label AS tgt, type(r) AS rel_label, "
                "r.description AS description, r.attributes AS attributes"
            )
            open_rel_result = await self._query(
                f"MATCH (o:`{_OPEN_RELATION_LABEL}`)-[r]->(o) "
                "RETURN type(r) AS rel_label, "
                "r.description AS description, r.attributes AS attributes"
            )
        except Exception as exc:
            logger.debug("Ontology load failed (returning empty ontology): %s", exc)
            return Ontology()

        def _rows(result: Any) -> list[Any]:
            rows = getattr(result, "result_set", None) or []
            return rows if isinstance(rows, list) else []

        # Entity types
        entities = [
            Entity(
                label=row[0],
                description=row[1],
                properties=_decode_attributes(row[2]),
            )
            for row in _rows(ent_result)
            if isinstance(row, list) and len(row) >= 3 and row[0]
        ]

        # Relation types — group edges by rel_label, collect endpoint patterns.
        rel_by_label: dict[str, dict[str, Any]] = {}
        for row in _rows(rel_result):
            if not (isinstance(row, list) and len(row) >= 5 and row[2]):
                continue
            src, tgt, rel_label, description, attributes = row
            entry = rel_by_label.setdefault(
                rel_label,
                {
                    "description": description,
                    "attributes": attributes,
                    "patterns": [],
                },
            )
            entry["patterns"].append((src, tgt))
            # Last write wins for description / attributes — every edge for a
            # given relation type carries the same metadata under the current
            # contract.
            if description:
                entry["description"] = description
            if attributes:
                entry["attributes"] = attributes
        for row in _rows(open_rel_result):
            if not (isinstance(row, list) and len(row) >= 3 and row[0]):
                continue
            rel_label, description, attributes = row
            rel_by_label.setdefault(
                rel_label,
                {"description": description, "attributes": attributes, "patterns": []},
            )

        relations = [
            Relation(
                label=label,
                description=entry["description"],
                patterns=entry["patterns"],
                properties=_decode_attributes(entry["attributes"]),
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
        """Upsert an entity-type node carrying the user label + ``__Ontology``.

        Declared attributes are JSON-encoded into a single ``attributes``
        property so the node looks like a miniature data-graph node would.
        """
        label = sanitize_cypher_label(et.label)
        await self._query(
            f"MERGE (e:`{label}`:`{_ONTOLOGY_LABEL}` {{label: $label}}) "
            "SET e.description = coalesce($description, e.description), "
            "e.attributes = $attributes",
            {
                "label": et.label,
                "description": et.description,
                "attributes": _encode_attributes(et.properties),
            },
        )

    async def _upsert_relation_type(self, rt: Relation) -> None:
        """Upsert a relation type as one or more edges.

        - Each declared ``(src, tgt)`` pattern materialises as a MERGE on
          an edge of the user-declared relation label between the two
          entity-type nodes.
        - A relation without patterns (open mode) materialises as a self-loop
          on the ``:__OpenRelation:__Ontology`` placeholder.

        Both flavours carry the relation's description and JSON-encoded
        attribute declarations on the edge itself.
        """
        rel_label = sanitize_cypher_label(rt.label)
        attributes = _encode_attributes(rt.properties)
        if rt.patterns:
            for src, tgt in rt.patterns:
                src_label = sanitize_cypher_label(src)
                tgt_label = sanitize_cypher_label(tgt)
                await self._query(
                    f"MERGE (s:`{src_label}`:`{_ONTOLOGY_LABEL}` {{label: $src}}) "
                    f"MERGE (t:`{tgt_label}`:`{_ONTOLOGY_LABEL}` {{label: $tgt}}) "
                    f"MERGE (s)-[r:`{rel_label}`]->(t) "
                    "SET r.description = coalesce($description, r.description), "
                    "r.attributes = $attributes",
                    {
                        "src": src,
                        "tgt": tgt,
                        "description": rt.description,
                        "attributes": attributes,
                    },
                )
        else:
            await self._query(
                f"MERGE (o:`{_OPEN_RELATION_LABEL}`:`{_ONTOLOGY_LABEL}`) "
                f"MERGE (o)-[r:`{rel_label}`]->(o) "
                "SET r.description = coalesce($description, r.description), "
                "r.attributes = $attributes",
                {"description": rt.description, "attributes": attributes},
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
