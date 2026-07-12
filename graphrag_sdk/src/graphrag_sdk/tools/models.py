# GraphRAG SDK — Tools: result models + LLM-text rendering
# Typed results for the agent toolkit. Every model is pydantic v2 and
# renders a deterministic, budget-bounded plain-text form via to_llm_text().

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from graphrag_sdk.storage.graph_store import GraphStore

_ELLIPSIS = "…"
_SNIPPET_CHARS = 240


def _clean(value: Any) -> str:
    """Sanitize a value for LLM output: strip control chars (reuses the
    ingestion sanitizer) and collapse all whitespace runs to single spaces."""
    return " ".join(GraphStore._sanitize_string(str(value)).split())


def _snippet(value: Any, limit: int = _SNIPPET_CHARS) -> str:
    """A cleaned, length-capped one-line excerpt ending with an ellipsis."""
    text = _clean(value)
    return text if len(text) <= limit else text[: limit - 1] + _ELLIPSIS


def _render(
    preamble: list[str],
    sections: list[tuple[str, list[str]]],
    *,
    max_chars: int,
) -> str:
    """Assemble preamble lines + (header, items) sections into text ≤ max_chars.

    Truncation happens only at item boundaries; a dropped tail is marked
    with the exact marker ``…(N more)``. Deterministic for equal inputs.
    """
    if max_chars < 1:
        return ""
    lines: list[str] = []
    used = 0

    def try_add(line: str) -> bool:
        nonlocal used
        cost = len(line) + (1 if lines else 0)
        if used + cost > max_chars:
            return False
        lines.append(line)
        used += cost
        return True

    for line in preamble:
        if not try_add(line):
            if not lines:  # always say something
                lines.append(line[: max_chars - 1] + _ELLIPSIS)
            return "\n".join(lines)

    for header, items in sections:
        if not items:
            continue
        header_line = f"{header} ({len(items)}):"
        full_marker = f"  {_ELLIPSIS}({len(items)} more)"
        # Only start a section if the header plus at least a drop-marker fit —
        # otherwise a bare header would dangle with nothing under it. Marker
        # strings never grow as the remaining count shrinks, so this reserve
        # guarantees every emitted header is followed by an item or a marker.
        needed = used + (1 if lines else 0) + len(header_line) + 1 + len(full_marker)
        if needed > max_chars:
            try_add(f"{header}: {_ELLIPSIS}({len(items)} items)")
            continue
        try_add(header_line)
        for idx, item in enumerate(items):
            marker = f"  {_ELLIPSIS}({len(items) - idx} more)"
            reserve = (len(marker) + 1) if idx < len(items) - 1 else 0
            if used + len(item) + 1 + reserve <= max_chars:
                try_add(item)
            else:
                try_add(marker)
                break
    return "\n".join(lines)


class ToolResult(BaseModel):
    """Base class for toolkit results: strict fields + LLM-text rendering."""

    model_config = ConfigDict(extra="forbid")

    def to_llm_text(self, *, max_chars: int = 4000) -> str:
        """Render a compact, deterministic plain-text form bounded by max_chars."""
        return _render(self._preamble(), self._sections(), max_chars=max_chars)

    def _preamble(self) -> list[str]:  # pragma: no cover - overridden
        return []

    def _sections(self) -> list[tuple[str, list[str]]]:  # pragma: no cover
        return []


class DocumentRef(BaseModel):
    """A source document reference."""

    model_config = ConfigDict(extra="forbid")
    document_id: str
    document_path: str = ""


class EntityCard(BaseModel):
    """A knowledge-graph entity with its user-facing properties."""

    model_config = ConfigDict(extra="forbid")
    name: str
    label: str = ""
    description: str | None = None
    properties: dict[str, Any] = Field(default_factory=dict)


class RelationTriple(BaseModel):
    """A directed relationship ``source -[type]-> target`` with optional evidence."""

    model_config = ConfigDict(extra="forbid")
    source: str
    type: str
    target: str
    fact: str | None = None


class ChunkRef(BaseModel):
    """A retrieved source chunk with provenance ids."""

    model_config = ConfigDict(extra="forbid")
    chunk_id: str
    document_id: str = ""
    document_path: str = ""
    text: str


class Citation(BaseModel):
    """A provenance citation attached to a generated answer."""

    model_config = ConfigDict(extra="forbid")
    document_id: str
    document_path: str = ""
    chunk_id: str
    snippet: str


class EntityTypeInfo(BaseModel):
    """A declared or observed entity label with a live node count."""

    model_config = ConfigDict(extra="forbid")
    label: str
    description: str | None = None
    count: int = 0
    properties: list[str] = Field(default_factory=list)


class RelationTypeInfo(BaseModel):
    """A declared or observed relation type with patterns and a live count."""

    model_config = ConfigDict(extra="forbid")
    label: str
    description: str | None = None
    patterns: list[tuple[str, str]] = Field(default_factory=list)
    count: int = 0


def _entity_line(e: EntityCard) -> str:
    line = f"- {_clean(e.name)}"
    if e.label:
        line += f" [{_clean(e.label)}]"
    if e.description:
        line += f": {_snippet(e.description, 160)}"
    return line


def _relation_line(r: RelationTriple) -> str:
    line = f"- {_clean(r.source)} -[{_clean(r.type)}]-> {_clean(r.target)}"
    if r.fact:
        line += f": {_snippet(r.fact, 160)}"
    return line


class SearchResult(ToolResult):
    """Ranked, typed retrieval context for a query (no LLM generation)."""

    query: str
    entities: list[EntityCard] = Field(default_factory=list)
    relations: list[RelationTriple] = Field(default_factory=list)
    facts: list[str] = Field(default_factory=list)
    chunks: list[ChunkRef] = Field(default_factory=list)
    documents: list[DocumentRef] = Field(default_factory=list)

    def _preamble(self) -> list[str]:
        return [f"Query: {_clean(self.query)}"]

    def _sections(self) -> list[tuple[str, list[str]]]:
        return [
            ("Entities", [_entity_line(e) for e in self.entities]),
            ("Relations", [_relation_line(r) for r in self.relations]),
            ("Facts", [f"- {_snippet(f)}" for f in self.facts]),
            (
                "Chunks",
                [
                    f"- [{_clean(c.document_path or c.document_id)}#{_clean(c.chunk_id)}] "
                    f"{_snippet(c.text)}"
                    for c in self.chunks
                ],
            ),
            (
                "Documents",
                [
                    f"- {_clean(d.document_id)} ({_clean(d.document_path)})"
                    if d.document_path
                    else f"- {_clean(d.document_id)}"
                    for d in self.documents
                ],
            ),
        ]


class AnswerResult(ToolResult):
    """A generated answer plus provenance citations."""

    answer: str
    citations: list[Citation] = Field(default_factory=list)
    entities_touched: list[str] = Field(default_factory=list)
    cypher_used: str | None = None

    def _preamble(self) -> list[str]:
        return [_clean(line) for line in self.answer.splitlines() if line.strip()]

    def _sections(self) -> list[tuple[str, list[str]]]:
        sections = [
            (
                "Citations",
                [
                    f"- [{_clean(c.document_path or c.document_id)}#{_clean(c.chunk_id)}] "
                    f"{_snippet(c.snippet)}"
                    for c in self.citations
                ],
            ),
            ("Entities", [f"- {_clean(n)}" for n in self.entities_touched]),
        ]
        if self.cypher_used:
            sections.append(("Cypher used", [f"- {_clean(self.cypher_used)}"]))
        return sections


class SchemaResult(ToolResult):
    """The graph's entity labels and relation types with live counts."""

    entities: list[EntityTypeInfo] = Field(default_factory=list)
    relations: list[RelationTypeInfo] = Field(default_factory=list)
    node_count: int = 0
    edge_count: int = 0

    def _preamble(self) -> list[str]:
        return [f"Nodes: {self.node_count} | Edges: {self.edge_count}"]

    def _sections(self) -> list[tuple[str, list[str]]]:
        ent_lines = []
        for e in self.entities:
            line = f"- {_clean(e.label)}: {e.count}"
            if e.description:
                line += f" — {_snippet(e.description, 100)}"
            if e.properties:
                line += f" (props: {', '.join(_clean(p) for p in e.properties)})"
            ent_lines.append(line)
        rel_lines = []
        for r in self.relations:
            line = f"- {_clean(r.label)}: {r.count}"
            if r.patterns:
                pats = ", ".join(f"{_clean(a)}->{_clean(b)}" for a, b in r.patterns)
                line += f" [{pats}]"
            rel_lines.append(line)
        return [("Entity labels", ent_lines), ("Relation types", rel_lines)]


class CypherResult(ToolResult):
    """Rows returned by a guarded read-only Cypher query."""

    columns: list[str] = Field(default_factory=list)
    rows: list[list[Any]] = Field(default_factory=list)
    row_count: int = 0
    truncated: bool = False

    def _preamble(self) -> list[str]:
        suffix = " (truncated)" if self.truncated else ""
        return [
            f"Columns: {', '.join(_clean(c) for c in self.columns)}",
            f"Rows: {self.row_count}{suffix}",
        ]

    def _sections(self) -> list[tuple[str, list[str]]]:
        return [
            (
                "Rows",
                [
                    f"- {_clean(json.dumps(row, ensure_ascii=False, default=str))}"
                    for row in self.rows
                ],
            )
        ]


class EntityResult(ToolResult):
    """Entity card: best-match node, neighbors, and source documents."""

    query: str
    found: bool = False
    entity: EntityCard | None = None
    neighbors: list[RelationTriple] = Field(default_factory=list)
    nearby: list[str] = Field(default_factory=list)
    documents: list[DocumentRef] = Field(default_factory=list)

    def _preamble(self) -> list[str]:
        if not self.found or self.entity is None:
            return [
                f"No entity found matching '{_clean(self.query)}'. "
                f"Try graph_search for fuzzy discovery."
            ]
        e = self.entity
        lines = [f"Entity: {_clean(e.name)}" + (f" [{_clean(e.label)}]" if e.label else "")]
        if e.description:
            lines.append(f"Description: {_snippet(e.description)}")
        if e.properties:
            props = "; ".join(f"{_clean(k)}={_clean(v)}" for k, v in sorted(e.properties.items()))
            lines.append(f"Properties: {props}")
        return lines

    def _sections(self) -> list[tuple[str, list[str]]]:
        return [
            ("Neighbors", [_relation_line(r) for r in self.neighbors]),
            ("Documents", [f"- {_clean(d.document_id)}" for d in self.documents]),
            ("Nearby", [f"- {_clean(n)}" for n in self.nearby]),
        ]


class RememberResult(ToolResult):
    """Outcome of storing text into the graph via graph_remember."""

    document_id: str
    chunks_indexed: int = 0
    nodes_created: int = 0
    relationships_created: int = 0
    finalized: bool = False

    def _preamble(self) -> list[str]:
        lines = [
            f"Stored document '{_clean(self.document_id)}' "
            f"({self.chunks_indexed} chunks, {self.nodes_created} nodes, "
            f"{self.relationships_created} relations)."
        ]
        lines.append(
            "Finalized."
            if self.finalized
            else "Pending finalize — call graph_flush (or GraphRAG.finalize())."
        )
        return lines
