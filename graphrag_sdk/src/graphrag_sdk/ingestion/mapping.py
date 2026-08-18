# GraphRAG SDK — Ingestion: Structured Source Mapping
# A declaration that says how a record becomes graph nodes and edges.
#
# The mapping is the whole contract for structured input. It is authored once
# per source and never consulted by a model at ingest time, which is what makes
# a structured write deterministic: the same file always produces the same graph.

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

from graphrag_sdk.core.models import Attribute, Entity, Ontology, Relation

# Property types a column may declare. Deliberately small, and matching the
# uppercase convention the ontology already uses for Attribute.type.
COLUMN_TYPES: frozenset[str] = frozenset({"STRING", "INTEGER", "FLOAT", "BOOLEAN", "DATE", "LIST"})

# Keys the SDK writes on every entity node. A mapping that declared one of these
# as a property name would shadow a system value, so they are rejected.
RESERVED_PROPERTY_NAMES: frozenset[str] = frozenset(
    {"id", "type", "description", "source_chunk_ids", "spans", "embedding", "alias_ids"}
)

_TRUE = frozenset({"1", "true", "t", "yes", "y", "on"})
_FALSE = frozenset({"0", "false", "f", "no", "n", "off"})


class MappingError(ValueError):
    """A mapping is malformed, or does not fit the source it was applied to."""


@dataclass(frozen=True)
class Column:
    """A property: which column it reads, and the type it becomes in the graph.

    The type is required rather than inferred. Sniffing types from a sample of
    rows makes the resulting schema depend on which rows happened to be read
    first, which is the same reason identity is never inferred.

    A bare string is accepted wherever a ``Column`` is expected and means
    ``STRING``, so the common case stays short.

    Args:
        name: The column in the source record.
        type: One of :data:`COLUMN_TYPES`.
        description: Optional prose, carried into the generated ontology.

    Example::

        Column("age", "INTEGER")
        Column("signed_on", "DATE", "date the contract was signed")
    """

    name: str
    type: str = "STRING"
    description: str | None = None

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise MappingError("Column.name must be a non-empty column name")
        if self.type not in COLUMN_TYPES:
            raise MappingError(
                f"Column({self.name!r}) declares unknown type {self.type!r}; "
                f"expected one of {', '.join(sorted(COLUMN_TYPES))}"
            )

    def cast(self, raw: Any) -> Any:
        """Convert a raw cell to the declared type.

        An empty cell becomes ``None`` and the caller omits the property, rather
        than writing a falsy value that a query cannot distinguish from a real
        zero or empty string.

        Raises:
            MappingError: If the cell cannot be read as the declared type. A
                declared type that does not hold is a fault in the declaration
                or the data, and silently coercing it would hide both.
        """
        if raw is None:
            return None
        if isinstance(raw, str):
            raw = raw.strip()
            if not raw:
                return None
        if self.type == "STRING":
            return str(raw)
        if self.type == "LIST":
            if isinstance(raw, (list, tuple)):
                return list(raw)
            return [part.strip() for part in str(raw).split(",") if part.strip()]
        try:
            if self.type == "INTEGER":
                return int(raw) if not isinstance(raw, str) else int(raw.replace(",", ""))
            if self.type == "FLOAT":
                return float(raw) if not isinstance(raw, str) else float(raw.replace(",", ""))
            if self.type == "BOOLEAN":
                if isinstance(raw, bool):
                    return raw
                lowered = str(raw).lower()
                if lowered in _TRUE:
                    return True
                if lowered in _FALSE:
                    return False
                raise ValueError(f"not a boolean: {raw!r}")
            if self.type == "DATE":
                if isinstance(raw, (date, datetime)):
                    return raw.isoformat()
                # Stored as an ISO string: FalkorDB has no date type, and an
                # ISO string sorts and compares correctly.
                return datetime.fromisoformat(str(raw)).date().isoformat()
        except (TypeError, ValueError) as exc:
            raise MappingError(
                f"column {self.name!r} declares {self.type} but holds {raw!r}"
            ) from exc
        raise MappingError(f"unhandled column type {self.type!r}")  # pragma: no cover


def _as_columns(properties: dict[str, Column | str] | None) -> dict[str, Column]:
    """Normalise the property map, accepting a bare string as STRING."""
    out: dict[str, Column] = {}
    for name, spec in (properties or {}).items():
        if name in RESERVED_PROPERTY_NAMES:
            raise MappingError(
                f"property {name!r} is written by the SDK and cannot be mapped; "
                f"reserved names are {', '.join(sorted(RESERVED_PROPERTY_NAMES))}"
            )
        out[name] = spec if isinstance(spec, Column) else Column(str(spec))
    return out


@dataclass
class NodeMapping:
    """One entity produced from each record.

    Args:
        label: The entity label, e.g. ``"Person"``.
        key: The column whose value identifies the entity. Its value becomes the
            node id via the same derivation the extraction path uses, so a
            structured node and an extracted node can be the same node.
        name: The column carrying the display name, when the record has one.
            A source that carries both ``key`` and ``name`` also publishes the
            name-derived id in ``alias_ids``, which is what lets a keyed node and
            a node extracted from prose resolve to one.
        properties: ``{property_name: Column}``. A bare string means ``STRING``.
        reference: ``True`` when the record only points at the entity by id and
            does not describe it, as a foreign key does. Reference nodes are
            written ON CREATE only, so they can never overwrite a name or a
            property that a dimension source supplied.
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

    def __post_init__(self) -> None:
        if not self.label or not self.label.strip():
            raise MappingError("NodeMapping.label must be a non-empty label")
        if not self.key or not self.key.strip():
            raise MappingError(f"NodeMapping({self.label!r}) must declare a key column")
        if self.reference and self.properties:
            raise MappingError(
                f"NodeMapping({self.label!r}) is a reference and cannot declare "
                "properties: a reference claims the entity exists, never what it "
                "looks like. Describe it in the source that owns it."
            )
        self.properties = _as_columns(self.properties)
        if self.alias is None:
            self.alias = self.label

    @property
    def columns(self) -> set[str]:
        """Every source column this node reads."""
        used = {self.key}
        if self.name:
            used.add(self.name)
        used.update(col.name for col in self.properties.values())
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

    def __post_init__(self) -> None:
        for label, value in (("type", self.type), ("source", self.source), ("target", self.target)):
            if not value or not str(value).strip():
                raise MappingError(f"EdgeMapping.{label} must be non-empty")
        self.properties = _as_columns(self.properties)

    @property
    def columns(self) -> set[str]:
        return {col.name for col in self.properties.values()}


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
        aliases = [node.alias for node in self.nodes]
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
                    f"node {node.alias!r}: key column {node.key!r} is not in the source"
                )
            if node.name and node.name not in available:
                problems.append(
                    f"node {node.alias!r}: name column {node.name!r} is not in the source"
                )
            for prop, col in node.properties.items():
                if col.name not in available:
                    problems.append(
                        f"node {node.alias!r}: property {prop!r} reads missing column {col.name!r}"
                    )
        for edge in self.edges:
            for prop, col in edge.properties.items():
                if col.name not in available:
                    problems.append(
                        f"edge {edge.type!r}: property {prop!r} reads missing column {col.name!r}"
                    )
        if strict:
            unused = sorted(available - self.columns)
            if unused:
                problems.append("these columns are not mapped anywhere: " + ", ".join(unused))
        return problems

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
            if node.name:
                attributes.append(Attribute(name="name", type="STRING", description="display name"))
            # The key survives as a queryable property in its own right, so a
            # later source can still join on it.
            attributes.append(
                Attribute(name=node.key, type="STRING", description=f"key from {node.key}")
            )
            for prop, col in node.properties.items():
                attributes.append(
                    Attribute(
                        name=prop,
                        type=col.type,
                        description=col.description or f"from column {col.name}",
                    )
                )
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

        by_alias = {node.alias: node.label for node in self.nodes}
        relations = [
            Relation(
                label=edge.type,
                description=edge.description or f"{edge.source} to {edge.target}",
                patterns=[(by_alias[edge.source], by_alias[edge.target])],
                properties=_dedupe_attributes(
                    [
                        Attribute(
                            name=prop,
                            type=col.type,
                            description=col.description or f"from column {col.name}",
                        )
                        for prop, col in edge.properties.items()
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


def Table(  # noqa: N802 - reads as a declaration, not a function call
    node: str,
    key: str,
    name: str | None = None,
    *,
    description: str | None = None,
    **properties: Column | str,
) -> RecordMapping:
    """Shorthand for the common case: one record is one entity.

    Example::

        Table("Organization", key="org_id", name="org_name",
              hq_country="hq_country",
              employee_count=Column("employee_count", "INTEGER"))
    """
    return RecordMapping(
        nodes=[
            NodeMapping(
                label=node,
                key=key,
                name=name,
                properties=dict(properties),
                description=description,
            )
        ]
    )
