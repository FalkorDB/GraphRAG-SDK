# GraphRAG SDK — Core: table declaration value types
#
# The vocabulary a user writes to say how a table becomes graph nodes: a Column
# with a declared type, a Link that turns a foreign key into an edge, and the
# TableMapping that binds one source to one label.
#
# These live in ``core`` rather than ``ingestion`` because ``Ontology`` holds a
# list of them — a mapping is part of the schema, not a separate artifact — and
# ``core`` must not import ``ingestion``. The behaviour that turns a row into
# nodes and edges stays in ``ingestion.mapping``, which re-exports these names
# so existing imports keep working.

from __future__ import annotations

import csv
import hashlib
import math
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

from graphrag_sdk.utils.cypher import sanitize_cypher_label

# Property types a column may declare. Deliberately small, and matching the
# uppercase convention the ontology already uses for Attribute.type.
COLUMN_TYPES: frozenset[str] = frozenset({"STRING", "INTEGER", "FLOAT", "BOOLEAN", "DATE", "LIST"})

# Keys the SDK writes on every entity node. A mapping that declared one of these
# as a property name would shadow a system value, so they are rejected. ``name``
# is here because it has its own slot: use ``NodeMapping(name="full_name")``.
RESERVED_PROPERTY_NAMES: frozenset[str] = frozenset(
    {
        "id",
        "name",
        "type",
        "description",
        "source_chunk_ids",
        "spans",
        "embedding",
        "entity_key",
        "is_stub",
    }
)
# A graph property name and an entity label both end up in Cypher, one as a
# parameter key and one interpolated into the query after sanitisation. Names are
# therefore restricted to identifiers, which is both what a graph can address
# and what keeps a declaration from reaching the driver as something it cannot
# serialise. Measured without this: a property named with a backtick and a
# comment marker surfaced as `DatabaseError: Invalid input at end of input`, from
# a query the caller never wrote and cannot see.
_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# Prefix for a source column whose own name cannot be a graph property, either
# because the SDK owns that name or because it is not an identifier. One rule,
# used for both the entity property a key column writes and the per-cell
# properties of a record chunk, so ``HQ Country`` addresses the same way
# wherever it lands.
_SAFE_PREFIX = "col_"


def safe_property_name(column: str, owned: frozenset[str] | None = None) -> str:
    """The graph property name a source column is stored under.

    A header is whatever the exporting system wrote — ``HQ Country``,
    ``Revenue (M USD)``, ``id``. A graph property name is written into generated
    Cypher as a bare name and, when it is one the SDK owns, silently overwrites a
    system value. Measured before this existed: a header with a space reached the
    driver inside a parameter map and surfaced as ``DatabaseError: Invalid input
    at end of input``, from a query the caller never wrote; and ``key="id"``
    overwrote the node's graph id, which cost every row its provenance edges.

    Identifiers the SDK does not own pass through unchanged, so existing graphs
    do not move. ``owned`` overrides which names are the SDK's, because a record
    chunk owns different ones than an entity does.
    """
    reserved = RESERVED_PROPERTY_NAMES if owned is None else owned
    if _IDENTIFIER.match(column) and column not in reserved:
        return column
    slug = re.sub(r"[^A-Za-z0-9_]+", "_", column.strip()).strip("_").lower()
    return f"{_SAFE_PREFIX}{slug}" if slug else f"{_SAFE_PREFIX}unnamed"


_GROUPED = re.compile(r"^-?\d{1,3}(,\d{3})+$")


def _normalise_number(raw: str, integral: bool = False) -> str:
    """Return ``raw`` with grouping separators removed and ``.`` as the decimal.

    A comma means opposite things either side of the Atlantic, so stripping every
    comma is wrong half the time and silently: measured, a German export's
    ``880,5`` was stored as ``8805.0`` — every figure in the column out by a
    factor of ten, with nothing to point at. The decidable cases are decided and
    the one that is not is refused rather than guessed.

    - Both separators present: the rightmost is the decimal point, so
      ``1,234.56`` and ``1.234,56`` both read as ``1234.56``.
    - Only commas, in more than one group of three: grouping. ``1,234,567``
      reads as ``1234567``.
    - One comma not followed by exactly three digits: a decimal comma. ``880,5``
      reads as ``880.5``.
    - One comma followed by exactly three digits: ambiguous. ``1,234`` is either
      one thousand or one-point-two-three-four and nothing in the cell says
      which, so it is refused — unless the column is ``INTEGER``, where a
      fractional part is not on offer and grouping is the only reading left.
      ``1,000`` in an INTEGER column is a thousand.
    """
    text = raw.strip().replace(" ", "").replace("\u00a0", "")
    if "," not in text:
        return text
    if "." in text:
        decimal = "," if text.rindex(",") > text.rindex(".") else "."
        grouping = "." if decimal == "," else ","
        return text.replace(grouping, "").replace(decimal, ".")
    if text.count(",") > 1 and _GROUPED.match(text):
        return text.replace(",", "")
    head, _, tail = text.rpartition(",")
    if len(tail) == 3 and head.lstrip("-").isdigit():
        if integral:
            return text.replace(",", "")
        raise ValueError(
            f"{raw!r} is ambiguous: {head + tail!r} if the comma groups thousands, "
            f"{head + '.' + tail!r} if it is a decimal comma. Clean the column, or "
            f"declare it STRING and convert it yourself."
        )
    return text.replace(",", ".")


def _check_identifier(kind: str, value: str) -> str:
    """Reject a name that cannot safely be a graph identifier.

    Applied to property names and relationship types, because both are written
    into generated Cypher as bare names (``n.age``, ``rel_type``). A name that
    would need quoting there is not usable even though a graph would store it.
    """
    if not _IDENTIFIER.match(value):
        raise MappingError(
            f"{kind} {value!r} is not a usable name: it must start with a letter "
            "or underscore and contain only letters, digits and underscores. "
            "Give it a usable name in the mapping and point it at the column, "
            'e.g. properties={"hq_country": Column("HQ Country")}.'
        )
    return value


def _check_label(label: str) -> str:
    """Reject a label that would not survive being written to the graph.

    Deliberately looser than :func:`_check_identifier`: ``Legal Entity``,
    ``Org-Unit`` and ``Ünïcode`` are all fine as labels, because the write path
    quotes them. What is not fine is a label the sanitiser has to *change*, since
    the graph would then silently hold something other than what was declared:
    ``Org`) DETACH DELETE (n) //`` was written as a label reading
    ``Org) DETACH DELETE (n) //``, harmless but nonsense.
    """
    if sanitize_cypher_label(label) != label:
        raise MappingError(
            f"label {label!r} contains characters that cannot be written as a "
            "label, so the graph would hold a different name than the one "
            "declared. Remove them from the label."
        )
    return label


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
            # Parsed as one CSV row rather than split on commas, so a quoted
            # element containing a comma survives: `"a,b",c` is two items, not
            # three. A naive split turns one value into several silently.
            try:
                parts = next(csv.reader([str(raw)]))
            except (csv.Error, StopIteration):
                parts = str(raw).split(",")
            return [part.strip() for part in parts if part.strip()]
        try:
            if self.type == "INTEGER":
                if not isinstance(raw, str):
                    return int(raw)
                return int(_normalise_number(raw, integral=True))
            if self.type == "FLOAT":
                value = float(raw) if not isinstance(raw, str) else float(_normalise_number(raw))
                if not math.isfinite(value):
                    # "nan" and "inf" are valid float literals and poison every
                    # aggregate they reach: one NaN turns an avg() over the whole
                    # column into NaN, with nothing to point at.
                    raise ValueError(f"not a finite number: {raw!r}")
                return value
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
            # Keep the cause's own words when it has any: the number and boolean
            # paths raise messages that say which reading was ambiguous and what
            # to do, and a bare "declares FLOAT but holds '1,234'" throws that
            # away at the point the reader most needs it.
            detail = str(exc).strip()
            because = f". {detail}" if detail and not detail.startswith("invalid literal") else ""
            raise MappingError(
                f"column {self.name!r} declares {self.type} but holds {raw!r}{because}"
            ) from exc
        raise MappingError(f"unhandled column type {self.type!r}")  # pragma: no cover


@dataclass
class Link:
    """A column that points at another entity, and the edge to it.

    A foreign key is the whole reason a table is worth putting in a graph: the
    ``org_id`` sitting in an employee export is not text, it is an edge. This is
    how you say so.

    Args:
        type: The relationship, e.g. ``"WORKS_AT"``. Written onto a ``RELATES``
            edge as its ``rel_type``, which is how every data edge is stored.
        to: The label of the entity being pointed at.
        by: The column holding its key.
        name: Optional column holding the target's display name, when the row
            denormalises it. Without one the placeholder is named by its key, so
            a node reads as "ORG-42" until the source that owns it arrives.
        properties: Optional columns written onto the edge itself, e.g. the date
            an employment started.

    The target is created if it is missing and **keyed, never named** if it is
    there already. This row says the organization
    exists and gives its key; it does not claim to describe it, so it can never
    overwrite what the source that owns the organization supplied. That is what
    makes the order of two files irrelevant.

    Example::

        TableMapping(source="hr.csv", label="Person", key="employee_id",
                     name="full_name",
                     properties={"age": Column("age", "INTEGER")},
                     links=[Link("WORKS_AT", to="Organization", by="org_id")])
    """

    type: str
    to: str
    by: str
    name: str | None = None
    properties: dict[str, Column | str] = field(default_factory=dict)
    description: str | None = None

    def __post_init__(self) -> None:
        for label, value in (("type", self.type), ("to", self.to), ("by", self.by)):
            if not value or not str(value).strip():
                raise MappingError(f"Link.{label} must be non-empty")
        _check_identifier("relationship type", self.type)
        _check_label(self.to)

    @property
    def typed_properties(self) -> dict[str, Column]:
        """``properties`` as columns, the same accessor ``TableMapping`` has."""
        return _as_columns(self.properties)


def _as_columns(properties: dict[str, Column | str] | None) -> dict[str, Column]:
    """Normalise the property map, accepting a bare string as STRING."""
    out: dict[str, Column] = {}
    for name, spec in (properties or {}).items():
        if name in RESERVED_PROPERTY_NAMES:
            raise MappingError(
                f"property {name!r} is written by the SDK and cannot be mapped; "
                f"reserved names are {', '.join(sorted(RESERVED_PROPERTY_NAMES))}"
            )
        _check_identifier("property", name)
        out[name] = spec if isinstance(spec, Column) else Column(str(spec))
    return out


def signature_for(source: str) -> str:
    """The property-name prefix a source signs its values with.

    Derived from the file name with the directory and one extension removed, so
    ``/data/2026-08/hr.csv`` and ``./hr.csv`` both sign ``hr``. Lower-cased and
    reduced to an identifier, because the result is joined onto every property
    name this source writes and the whole name must be a bare Cypher identifier.

    **Not injective, deliberately.** ``hr.csv``, ``HR.CSV`` and ``hr csv`` all
    reduce to ``hr``. Two sources that reduce alike would silently overwrite each
    other's properties, so the collision is refused where every mapping is
    visible at once — when the ontology is registered — rather than papered over
    with a disambiguating suffix that no user could predict.
    """
    stem = source.replace("\\", "/").rsplit("/", 1)[-1]
    if "." in stem:
        stem = stem[: stem.rindex(".")]
    slug = re.sub(r"[^A-Za-z0-9_]+", "_", stem).strip("_").lower()
    if not slug:
        return "source"
    return slug if _IDENTIFIER.match(slug) else f"s_{slug}"


@dataclass
class TableMapping:
    """How one table becomes part of the graph. The whole declaration.

    A member of the ontology, alongside entity and relation types: a mapping is
    part of the schema, so it is stored with the schema and a second load needs
    no mapping argument at all.

    Args:
        source: The table's file name. Also the **signature** — every property
            this mapping writes is stored as ``<signature>__<property>``, so two
            tables describing one entity can disagree without either overwriting
            the other. See :attr:`signature`.
        label: The label each row becomes. Need not already exist; a mapping may
            add it to the ontology, in which case it must also say how the new
            label connects — see ``links`` and ``standalone``.
        key: The column identifying the row. Its value derives the node id, so
            re-loading a corrected export updates in place instead of
            duplicating. **Optional: defaults to** ``name``. Most tables have one
            column that is both the identifier and the display name, and then
            the node id is exactly what prose extraction would compute for that
            name — the two halves share an id outright. Declare a separate key
            when names can repeat (two John Smiths), when a name can change
            while the row persists (a rename), or when other tables link to this
            one by an id column rather than by name.
        name: The column holding the display name. This is what joins a row to a
            mention of the same thing in prose.
        properties: ``property_name -> Column``. A bare string means STRING.
        links: Columns that point at other entities. See :class:`Link`.
        standalone: Declare a new label that genuinely relates to nothing. Only
            meaningful when the label is new and there are no links; it exists so
            that an unconnected island is something you *said*, rather than
            something a typo produced.
        description: Optional prose, carried into the generated ontology.

    Example::

        TableMapping(
            source="hr.csv",
            label="Person",
            key="employee_id",
            name="full_name",
            properties={"age": Column("age", "INTEGER")},
            links=[Link("WORKS_AT", to="Organization", by="org_id")],
        )

    which writes ``hr__age`` onto each Person, leaving ``finance.csv`` free to
    write ``finance__age`` without a collision.
    """

    source: str
    label: str
    key: str | None = None
    name: str | None = None
    properties: dict[str, Column | str] = field(default_factory=dict)
    links: list[Link] = field(default_factory=list)
    standalone: bool = False
    derived: bool = False
    """True when nobody declared this — it is the natural reading of the file.

    Reported by ``finalize()`` so a table that was loaded without a mapping, and
    is therefore unjoined to anything in the documents, is visible rather than
    quietly inert.
    """
    description: str | None = None

    def __post_init__(self) -> None:
        if self.key is None:
            # The common case: one column is both the identifier and the display
            # name. The node id is then compute_entity_id(name, label) — the same
            # id a document mention of that name produces — so the row and the
            # prose meet on the id itself. A duplicate name is refused at ingest
            # by the uniqueness check, exactly as a duplicate key would be.
            if not self.name or not str(self.name).strip():
                raise MappingError(
                    f"TableMapping({self.source!r}) needs a name= column (used as the "
                    f"row's identity) or an explicit key= column."
                )
            self.key = self.name
        for field_name, value in (
            ("source", self.source),
            ("label", self.label),
            ("key", self.key),
        ):
            if not value or not str(value).strip():
                raise MappingError(f"TableMapping.{field_name} must be non-empty")
        _check_label(self.label)
        if self.standalone and self.links:
            raise MappingError(
                f"TableMapping({self.source!r}) is standalone but declares links. "
                "standalone=True means the label relates to nothing; drop it, or "
                "drop the links."
            )
        # Normalise the input contract (a bare string means STRING) exactly once,
        # so every reader after this point sees Columns.
        self.properties = dict(_as_columns(self.properties))

    @property
    def key_column(self) -> str:
        """The column the row's identity comes from — ``key``, or ``name`` when no
        key was declared. Always set after construction; this exists so readers
        get a ``str`` without re-deriving the default."""
        assert self.key is not None  # resolved in __post_init__
        return self.key

    @property
    def signature(self) -> str:
        """The prefix every property this mapping writes is stored under.

        Derived from ``source`` and required to be a bare Cypher identifier: the
        SDK sends property maps as query parameters, and the driver serialises
        their keys **unquoted** into the query text, so a name needing backticks
        never reaches the server intact. Measured before this rule existed: a
        header with a space surfaced as ``DatabaseError: Invalid input at end of
        input``, from a query the caller never wrote.

        Not derived through :func:`safe_property_name`, which is not injective —
        ``hr.csv``, ``HR.CSV`` and ``hr csv`` all collapse to one name there, and
        two sources sharing a signature would silently overwrite each other.
        """
        return signature_for(self.source)

    @property
    def fingerprint_of_declaration(self) -> str:
        """A stable digest of everything this mapping declares.

        Used to tell a re-declaration apart from the same declaration passed
        again. Order-independent, so reordering a property map is not a change.
        """
        parts = [self.label, self.key_column, self.name or "", str(self.standalone)]
        parts += sorted(
            f"{name}={column.name}:{column.type}" for name, column in self.typed_properties.items()
        )
        parts += sorted(
            f"{link.type}->{link.to}:{link.by}:{link.name or ''}" for link in self.links
        )
        return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()

    def signed_name(self, prop: str) -> str:
        """``prop`` as this mapping writes it on the node.

        The same rule ``NodeMapping.signed`` applies on the write path, available
        here so a caller holding only the declaration — reconciling a stored
        mapping against a new one, say — can name the properties without
        normalising it first.
        """
        if prop in RESERVED_PROPERTY_NAMES:
            return prop
        return f"{self.signature}__{prop}"

    @property
    def typed_properties(self) -> dict[str, Column]:
        """``properties`` as Columns. ``__post_init__`` already normalised them."""
        return {
            name: spec if isinstance(spec, Column) else Column(str(spec))
            for name, spec in self.properties.items()
        }

    @property
    def columns(self) -> set[str]:
        """Every source column this mapping reads."""
        used = {self.key_column}
        if self.name:
            used.add(self.name)
        used.update(column.name for column in self.typed_properties.values())
        for link in self.links:
            used.add(link.by)
            if link.name:
                used.add(link.name)
            used.update(
                spec.name if isinstance(spec, Column) else str(spec)
                for spec in link.properties.values()
            )
        return used
