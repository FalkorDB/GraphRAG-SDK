# GraphRAG SDK — Ingestion: Structured Source Mapping
# A declaration that says how a record becomes graph nodes and edges.
#
# The mapping is the whole contract for structured input. It is authored once
# per source and never consulted by a model at ingest time, which is what makes
# a structured write deterministic: the same file always produces the same graph.

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field

from graphrag_sdk.core.models import Attribute, Entity, Ontology, Relation
from graphrag_sdk.core.tables import (
    COLUMN_TYPES,
    RESERVED_PROPERTY_NAMES,
    Column,
    Link,
    MappingError,
    TableMapping,
    _as_columns,
    _check_identifier,
    _check_label,
    safe_property_name,
    signature_for,
)

# The declaration value types moved to ``core.tables`` so that ``Ontology`` can
# hold them without ``core`` importing ``ingestion``. Re-exported here because
# this is where users and the rest of the package have always imported them from.
__all__ = [
    "COLUMN_TYPES",
    "RESERVED_PROPERTY_NAMES",
    "Column",
    "EdgeMapping",
    "Link",
    "MappingError",
    "NodeMapping",
    "RecordMapping",
    "TableMapping",
    "safe_property_name",
    "signature_for",
]


@dataclass
class NodeMapping:
    """One entity produced from each record.

    Args:
        label: The entity label, e.g. ``"Person"``.
        key: The column whose value identifies the entity, kept on the node as
            ``entity_key`` for links and re-sync. It derives the node id only
            when the record has no ``name``.
        name: The column carrying the display name, when the record has one.
            The name is the entity's **identity**: its id is derived from it the
            same way a prose mention's is, so a row and a document describing one
            thing land on one node with no merge step. The key is kept beside it
            as ``entity_key`` for links and re-sync.
        reference: ``True`` when the record only points at the entity by id and
            does not describe it, as a foreign key does. A reference writes its
            key and nothing else onto a node that already exists, so it can
            never overwrite a name or a property that a dimension source
            supplied.
        alias: A handle unique within the record, so one record can carry two
            entities of the same label. Edges address aliases, never labels.
            Defaults to ``label``.
        description: Optional prose, carried into the generated ontology.

    Example::

        NodeMapping(label="Person", key="employee_id", name="full_name",
                    properties={"age": Column("age", "INTEGER")})
        NodeMapping(label="Organization", key="org_id", reference=True)
    """

    label: str
    key: str
    name: str | None = None
    properties: dict[str, Column | str] = field(default_factory=dict)
    reference: bool = False
    alias: str | None = None
    description: str | None = None
    signature: str | None = None
    """The source that declared this, if it came from a :class:`TableMapping`.

    Present whenever a source declared the mapping, which
    is what keeps every property this writes attributable to one table: two
    tables describing one entity write ``hr__age`` and ``finance__age`` rather
    than racing on ``age``.
    """

    def __post_init__(self) -> None:
        if not self.label or not self.label.strip():
            raise MappingError("NodeMapping.label must be a non-empty label")
        _check_label(self.label)
        if not self.key or not self.key.strip():
            raise MappingError(f"NodeMapping({self.label!r}) must declare a key column")
        if self.reference and self.properties:
            raise MappingError(
                f"NodeMapping({self.label!r}) is a reference and cannot declare "
                "properties: a reference claims the entity exists, never what it "
                "looks like. Describe it in the source that owns it."
            )
        # The field's declared type is the *input* contract (a bare string is
        # accepted as STRING); what it holds after this line is always a Column.
        # ``typed_properties`` is the accessor that states that.
        normalised: dict[str, Column | str] = dict(_as_columns(self.properties))
        self.properties = normalised
        if self.key_property in normalised:
            raise MappingError(
                f"NodeMapping({self.label!r}) maps a property {self.key_property!r}, "
                f"but that is where the key column {self.key!r} is stored, and the "
                f"property would overwrite the row's identity. Store it under "
                f"another name."
            )
        if self.alias is None:
            self.alias = self.label

    def signed(self, prop: str) -> str:
        """``prop`` as it is written on the node.

        Prefixed with the declaring source when there is one, so no two tables
        can occupy one property name. Unsigned for SDK-owned names, and unsigned
        throughout when no signature was supplied.
        """
        if not self.signature or prop in RESERVED_PROPERTY_NAMES:
            return prop
        return f"{self.signature}__{prop}"

    @property
    def key_property(self) -> str:
        """The graph property the key column's value is written to.

        Usually the column's own name. A header the SDK owns or cannot address —
        ``id``, ``HQ Country`` — is stored under a ``col_`` name instead, because
        writing it verbatim would either overwrite a system value or reach the
        driver as something it cannot serialise.
        """
        return safe_property_name(self.key)

    @property
    def handle(self) -> str:
        """The alias, which ``__post_init__`` guarantees is set.

        ``alias`` is declared optional because the caller may omit it, but it is
        never ``None`` once constructed. This says so in a way a type checker can
        follow, without asking every caller to prove it again.
        """
        return self.alias or self.label

    @property
    def typed_properties(self) -> dict[str, Column]:
        """``properties`` as columns. ``__post_init__`` already normalised them.

        Hoist this out of a per-record loop rather than calling it per row.
        """
        return {
            name: spec if isinstance(spec, Column) else Column(str(spec))
            for name, spec in self.properties.items()
        }

    @property
    def columns(self) -> set[str]:
        """Every source column this node reads."""
        used = {self.key}
        if self.name:
            used.add(self.name)
        used.update(col.name for col in self.typed_properties.values())
        return used


@dataclass
class EdgeMapping:
    """An edge between two aliases in the same record.

    ``source`` and ``target`` address :attr:`NodeMapping.alias`, not labels,
    which is what lets a record hold a buyer and a seller that are both
    ``Organization``. Addressing by label would silently produce a self loop.

    Args:
        type: The semantic edge type, e.g. ``"WORKS_AT"``. Written as the
            ``rel_type`` property on a ``RELATES`` edge, which is the shape every
            retrieval path already expects.
        source: The alias the edge starts at.
        target: The alias the edge ends at.
        properties: ``{property_name: Column}`` written onto the edge.
        description: Optional prose, carried into the generated ontology.
    """

    type: str
    source: str
    target: str
    properties: dict[str, Column | str] = field(default_factory=dict)
    description: str | None = None
    signature: str | None = None
    """The table that declared this edge, recorded on it so a re-load of one
    source can leave another source's edges alone."""

    def __post_init__(self) -> None:
        for label, value in (("type", self.type), ("source", self.source), ("target", self.target)):
            if not value or not str(value).strip():
                raise MappingError(f"EdgeMapping.{label} must be non-empty")
        _check_identifier("relationship type", self.type)
        normalised: dict[str, Column | str] = dict(_as_columns(self.properties))
        self.properties = normalised

    def signed(self, prop: str) -> str:
        """``prop`` as it is written on the edge.

        Unlike :meth:`NodeMapping.signed`, **nothing is exempt**. A node leaves
        ``name`` and the SDK's own keys unsigned because they are the join: an
        extracted node and a keyed node have to meet on them. An edge has no such
        join, so every declared property is signed, which also puts the SDK's own
        ``rel_type``, ``fact`` and ``source_chunk_ids`` permanently out of a
        declaration's reach. Two tables declaring ``WORKS_AT.since`` therefore
        write two properties and the conflict is kept, rather than whichever
        loaded last winning silently.
        """
        if not self.signature:
            return prop
        return f"{self.signature}__{prop}"

    @property
    def typed_properties(self) -> dict[str, Column]:
        """``properties`` as columns, normalised in ``__post_init__``."""
        return {
            name: spec if isinstance(spec, Column) else Column(str(spec))
            for name, spec in self.properties.items()
        }

    @property
    def columns(self) -> set[str]:
        return {col.name for col in self.typed_properties.values()}


@dataclass
class RecordMapping:
    """How one structured source becomes nodes and edges.

    Authored once per source, by a person or by a model whose proposal a person
    approved. Either way it is fixed before ingest runs, so no model is consulted
    per record.

    Example::

        RecordMapping(
            nodes=[
                NodeMapping(label="Person", key="employee_id", name="full_name",
                            properties={"age": Column("age", "INTEGER")}),
                NodeMapping(label="Organization", key="org_id", reference=True),
            ],
            edges=[EdgeMapping(type="WORKS_AT", source="Person", target="Organization")],
        )
    """

    nodes: list[NodeMapping]
    edges: list[EdgeMapping] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.nodes:
            raise MappingError("a RecordMapping must declare at least one node")
        aliases = [node.handle for node in self.nodes]
        duplicates = {a for a in aliases if aliases.count(a) > 1}
        if duplicates:
            raise MappingError(
                f"duplicate node aliases {sorted(duplicates)}; give each node a "
                "distinct alias so edges can address them unambiguously"
            )
        known = set(aliases)
        for edge in self.edges:
            for end in ("source", "target"):
                alias = getattr(edge, end)
                if alias not in known:
                    raise MappingError(
                        f"edge {edge.type!r} has {end}={alias!r}, which is not a "
                        f"declared node alias; declared aliases are {sorted(known)}"
                    )
        if all(node.reference for node in self.nodes):
            raise MappingError(
                "every node in this mapping is a reference, so the source would "
                "describe nothing. At least one node must own its record."
            )

    @property
    def anchor(self) -> NodeMapping:
        """The node whose key identifies the record itself.

        The first non reference node. Its key value names the record's chunk, so
        the chunk id is stable across runs.
        """
        for node in self.nodes:
            if not node.reference:
                return node
        raise MappingError("no non-reference node to anchor the record on")

    @property
    def columns(self) -> set[str]:
        """Every source column this mapping reads."""
        used: set[str] = set()
        for node in self.nodes:
            used |= node.columns
        for edge in self.edges:
            used |= edge.columns
        return used

    def validate_against(self, columns: list[str], *, strict: bool = False) -> list[str]:
        """Check the mapping against a source's real header.

        Args:
            columns: The column names the source actually has.
            strict: When ``True``, a column the mapping never reads is also
                reported. Off by default because ignoring a column is a
                legitimate choice; on when a model authored the mapping, where a
                dropped column is usually a mistake.

        Returns:
            Human readable problems, empty when the mapping fits.
        """
        available = set(columns)
        problems: list[str] = []
        for node in self.nodes:
            if node.key not in available:
                problems.append(
                    f"node {node.handle!r}: key column {node.key!r} is not in the source"
                )
            if node.name and node.name not in available:
                problems.append(
                    f"node {node.handle!r}: name column {node.name!r} is not in the source"
                )
            for prop, col in node.typed_properties.items():
                if col.name not in available:
                    problems.append(
                        f"node {node.handle!r}: property {prop!r} reads missing column {col.name!r}"
                    )
        for edge in self.edges:
            for prop, col in edge.typed_properties.items():
                if col.name not in available:
                    problems.append(
                        f"edge {edge.type!r}: property {prop!r} reads missing column {col.name!r}"
                    )
        if strict:
            unused = sorted(available - self.columns)
            if unused:
                problems.append("these columns are not mapped anywhere: " + ", ".join(unused))
        return problems

    @property
    def fingerprint(self) -> str:
        """A stable digest of the declaration itself.

        Structured ``update()`` short-circuits when the source's content hash is
        unchanged, and identical rows under a *different* mapping produce a
        different graph. Folding this into that hash is what stops a re-declared
        mapping from being mistaken for unchanged data and silently skipped.

        Order-independent by construction, so reordering nodes in a declaration
        is not treated as a change.
        """
        parts: list[str] = []
        for node in sorted(self.nodes, key=lambda n: n.handle):
            columns = ",".join(
                f"{prop}:{col.name}:{col.type}"
                for prop, col in sorted(node.typed_properties.items())
            )
            parts.append(
                f"node({node.handle}|{node.label}|{node.key}|{node.name or ''}"
                f"|{'ref' if node.reference else 'own'}|{columns})"
            )
        for edge in sorted(self.edges, key=lambda e: (e.type, e.source, e.target)):
            columns = ",".join(
                f"{prop}:{col.name}:{col.type}"
                for prop, col in sorted(edge.typed_properties.items())
            )
            parts.append(f"edge({edge.type}|{edge.source}|{edge.target}|{columns})")
        return hashlib.sha256("".join(parts).encode("utf-8")).hexdigest()

    def to_ontology(self) -> Ontology:
        """Project the mapping into an ontology fragment.

        This is what makes a structured source queryable. Registered into the
        ontology store, it tells text-to-Cypher that ``Person.age`` is an
        ``INTEGER`` and that ``PARTY_TO`` runs from an organization to a
        contract. Without it a generated query cannot see typed columns at all
        and falls back to guessing that everything is a described entity.

        Reference-only labels still get a stub entry, so the ontology validator
        does not warn about an edge pointing at an undeclared label.
        """
        entities: list[Entity] = []
        for node in self.nodes:
            attributes: list[Attribute] = []
            # ``name`` is deliberately NOT declared. The extraction path merges
            # ontology-declared attributes over the system ones it just built
            # (see GraphExtraction._entities_to_nodes), so declaring an
            # attribute named ``name`` invites the extractor to answer it with a
            # null for every prose mention and blank out the real display name.
            # Every entity carries ``name`` already; declaring it adds nothing
            # and costs the label its names.
            #
            # The key survives as a queryable property in its own right, so a
            # later source can still join on it.
            attributes.append(
                Attribute(
                    # The property the value is actually written to, which is
                    # not always the column's own name. Publishing the raw
                    # header here put `- Org ID (STRING)` in the text-to-Cypher
                    # schema block, inviting `WHERE o.Org ID = ...` — not valid
                    # Cypher, on the one property every mapping declares.
                    # Signed to match what the write path actually puts on the
                    # node. Unsigned here while the node carried hr__employee_id
                    # meant the schema handed to text-to-Cypher named a property
                    # that does not exist.
                    name=node.signed(node.key_property),
                    type="STRING",
                    description=f"key from {node.key}",  # the source header, verbatim
                    structured=True,
                )
            )
            for prop, col in node.typed_properties.items():
                attributes.append(
                    Attribute(
                        name=node.signed(prop),
                        type=col.type,
                        # Load bearing, not decoration. The property is named
                        # hr__age, and this description is the only thing that
                        # tells text-to-Cypher the word "age" means that column
                        # and that hr.csv is where it came from, so "what grade
                        # does finance say?" can generate p.finance__grade.
                        description=col.description
                        or (
                            f"{col.name}, from table {node.signature}"
                            if node.signature
                            else f"from column {col.name}"
                        ),
                        structured=True,
                    )
                )
            # A key column may itself be called "name" (or another system key),
            # which would reintroduce the shadowing above through the back door.
            attributes = [a for a in attributes if a.name not in RESERVED_PROPERTY_NAMES]
            entities.append(
                Entity(
                    label=node.label,
                    description=node.description
                    or (
                        "Referenced by key from a structured source"
                        if node.reference
                        else f"Declared by a structured source, keyed on {node.key}"
                    ),
                    properties=_dedupe_attributes(attributes),
                )
            )

        by_alias = {node.handle: node.label for node in self.nodes}
        relations = [
            Relation(
                label=edge.type,
                description=edge.description or f"{edge.source} to {edge.target}",
                patterns=[(by_alias[edge.source], by_alias[edge.target])],
                properties=_dedupe_attributes(
                    [
                        Attribute(
                            name=edge.signed(prop),
                            type=col.type,
                            # Same job as the node-side description above: the
                            # property is named hr__since, and this is the only
                            # thing telling text-to-Cypher that "since" means
                            # that column and hr.csv is where it came from.
                            description=col.description
                            or (
                                f"{col.name}, from table {edge.signature}"
                                if edge.signature
                                else f"from column {col.name}"
                            ),
                            structured=True,
                        )
                        for prop, col in edge.typed_properties.items()
                    ]
                ),
            )
            for edge in self.edges
        ]
        return Ontology(entities=_merge_entities(entities), relations=relations)


def _dedupe_attributes(attributes: list[Attribute]) -> list[Attribute]:
    """First declaration of a name wins, so a key column named like a property
    does not appear twice."""
    seen: dict[str, Attribute] = {}
    for attribute in attributes:
        seen.setdefault(attribute.name, attribute)
    return list(seen.values())


def _merge_entities(entities: list[Entity]) -> list[Entity]:
    """Fold repeated labels into one entry, unioning their properties.

    One record can produce two nodes of the same label under different aliases;
    the ontology has one entry per label.
    """
    merged: dict[str, Entity] = {}
    for entity in entities:
        existing = merged.get(entity.label)
        if existing is None:
            merged[entity.label] = entity
            continue
        merged[entity.label] = Entity(
            label=entity.label,
            description=existing.description or entity.description,
            properties=_dedupe_attributes(list(existing.properties) + list(entity.properties)),
        )
    return list(merged.values())


def _normalise(
    node: str,
    key: str,
    name: str | None,
    properties: dict[str, Column | str],
    links: Sequence[Link],
    description: str | None,
    signature: str | None,
) -> tuple[list[NodeMapping], list[EdgeMapping]]:
    """Turn a declaration into the nodes and edges the write path wants.

    Used by :func:`record_mapping_for`, so the handle
    derivation and the two errors it can raise are stated once. ``signature``
    is stamped onto everything produced, which is how a property written here
    stays attributable to the table that declared it.
    """
    subject = NodeMapping(
        label=node,
        key=key,
        name=name,
        properties=dict(properties),
        description=description,
        signature=signature,
    )
    nodes = [subject]
    edges: list[EdgeMapping] = []
    for link in links or ():
        if not isinstance(link, Link):
            raise MappingError(
                f"links must contain Link objects, got {type(link).__name__}. "
                'Write links=[Link("WORKS_AT", to="Organization", by="org_id")].'
            )
        # A target keyed by the same column as the subject would be the subject,
        # and an edge from a thing to itself says nothing.
        if link.by == key and link.to == node:
            raise MappingError(
                f"link {link.type!r} points at {link.to!r} by column {link.by!r}, "
                "which is this record's own key, so it would link the record to "
                "itself. Point it at the column holding the other entity's key."
            )
        # Handles are internal, so they are derived rather than asked for. The
        # first link to a label takes the label; any later one is told apart by
        # its column, which is the thing that actually differs. Disambiguating by
        # relationship type instead would collide as soon as two links share a
        # type, and the failure surfaced as a complaint about "duplicate aliases"
        # — a word the caller never wrote.
        alias = link.to if link.to != node else f"{link.to}__{link.by}"
        if any(existing.handle == alias for existing in nodes):
            alias = f"{link.to}__{link.by}"
        if any(existing.handle == alias for existing in nodes):
            raise MappingError(
                f"two links both point at {link.to!r} by column {link.by!r}, "
                "so they describe the same target twice. Drop one, or point "
                "them at the different columns holding each target's key."
            )
        nodes.append(
            NodeMapping(
                label=link.to,
                key=link.by,
                name=link.name,
                reference=True,
                alias=alias,
                description=link.description,
                signature=signature,
            )
        )
        edges.append(
            EdgeMapping(
                type=link.type,
                source=subject.handle,
                target=alias,
                properties=dict(link.properties),
                description=link.description,
                signature=signature,
            )
        )
    return nodes, edges


def ontology_for(mapping: TableMapping) -> Ontology:
    """The ontology contribution of a declared mapping, mapping included.

    ``RecordMapping.to_ontology()`` derives the entity and relation types a
    mapping implies. This adds the mapping itself, because it belongs in the
    schema too: without it the ontology would describe the columns and not where
    they came from, and a second load would have nothing to reuse or diff against.
    """
    contribution = record_mapping_for(mapping).to_ontology()
    return contribution.model_copy(update={"tables": [mapping]})


def record_mapping_for(mapping: TableMapping) -> RecordMapping:
    """The normalised write-path form of a declared mapping.

    ``TableMapping`` holds the *declaration* — a label, a key, a property map and
    a list of links — because that is the shape a user writes and the shape the
    ontology stores. The write path wants it flattened into nodes and edges, with
    each link's target as a reference node. This is that translation, and it is
    where the source's signature is stamped on.
    """
    nodes, edges = _normalise(
        node=mapping.label,
        key=mapping.key_column,
        name=mapping.name,
        properties=mapping.properties,
        links=mapping.links,
        description=mapping.description,
        signature=mapping.signature,
    )
    return RecordMapping(nodes=nodes, edges=edges)
