# GraphRAG SDK — Tools: GraphRAGToolkit
# The graph tools an outside agent calls (through MCP or its own framework).
# The agent plans, calls tools and writes the answer; the toolkit runs each
# call bounded and read-only, numbers the evidence it returns and checks the
# agent's citations against it. There is no agent loop in the SDK.

from __future__ import annotations

import asyncio
import copy
import dataclasses
import json
import os
import re
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ValidationError

from graphrag_sdk.core.connection import ConnectionConfig
from graphrag_sdk.core.context import Context
from graphrag_sdk.core.exceptions import ConfigError, DatabaseError, ReadOnlyViolation
from graphrag_sdk.core.models import Entity, Ontology, Relation
from graphrag_sdk.core.providers import Embedder, LLMInterface
from graphrag_sdk.retrieval.graph_walk import DynamicGraphWalk
from graphrag_sdk.retrieval.strategies.multi_path import MultiPathRetrieval
from graphrag_sdk.tools import files as fs
from graphrag_sdk.tools import graph_ops as ops
from graphrag_sdk.tools.cypher_guard import (
    GraphSchema,
    enforce_row_cap,
    introspect_schema,
    mask_internal_labels,
    validate_read_query,
)
from graphrag_sdk.tools.errors import ToolError
from graphrag_sdk.tools.evidence import DEFAULT_LEDGER_CAPACITY, EvidenceLedger, ground_answer
from graphrag_sdk.tools.models import (
    ChunkRef,
    CitationCheckResult,
    ClaimCheck,
    CypherResult,
    CypherRow,
    DocSample,
    DocumentRef,
    EntityCard,
    EntityMatch,
    EntityTypeInfo,
    EvidenceRef,
    Fact,
    FinalizeReport,
    IngestedDocument,
    IngestResult,
    LookupResult,
    OntologyResult,
    PreviewFilesResult,
    PrunedTypes,
    RelationTriple,
    RelationTypeInfo,
    RetrieveResult,
    SkillRunResult,
    StatusResult,
    TraverseNode,
    TraverseResult,
    WalkPath,
    WalkResult,
)
from graphrag_sdk.tools.specs import (
    RETRIEVE_PATHS,
    TOOL_REGISTRY,
    CheckCitationsInput,
    CypherQueryInput,
    GraphWalkInput,
    IngestInput,
    LookupEntityInput,
    PlanRuleInput,
    PreviewFilesInput,
    RetrieveInput,
    RunSkillInput,
    SetOntologyInput,
    ToolSpec,
    TraverseInput,
    build_tool_specs,
    tool_enabled,
)
from graphrag_sdk.utils.cypher import sanitize_cypher_label

if TYPE_CHECKING:
    from graphrag_sdk.api.main import GraphRAG

FinalizePolicy = Literal["manual", "on_write", "never"]
_POLICIES = ("manual", "on_write", "never")
_TENANT_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
#: Seconds a cached schema is trusted when the graph's topology has not changed.
SCHEMA_CACHE_SECONDS = 60.0
#: Time allowed to resolve graph_walk's start / goal names.
_RESOLVE_BUDGET_SECONDS = 2.0
#: Longest evidence text kept for one item.
_EVIDENCE_CHARS = 2000
#: Most names listed in a hint.
_HINT_NAMES = 30
#: Total characters preview_files returns in one call (split across the files).
SAMPLE_BUDGET_CHARS = 30_000
#: Files loaded or ingested at once by preview_files / ingest(files=...).
_FILE_CONCURRENCY = 3
# Files whose last plan is remembered for reextract (oldest forgotten first).
_PLAN_MEMORY = 10_000
#: retrieve's search paths and the MultiPath path behind each. ``links`` has
#: none: retrieve walks the relations itself, so MultiPath's own expansion,
#: whose output it never reads, is not run.
_MULTIPATH_PATH = {
    "facts": "relates",
    "exact_names": "entity_cypher",
    "fuzzy_names": "entity_fulltext",
    "passages": "chunks",
}
_ALL_MULTIPATH_PATHS = frozenset(_MULTIPATH_PATH.values())
#: Paths that find the entities links walk from.
_SEED_PATHS = frozenset({"facts", "exact_names", "fuzzy_names"})


class _FixedPlan:
    """A MultiPath path router that always runs the same paths."""

    def __init__(self, paths: frozenset[str]) -> None:
        self._paths = sorted(paths)

    async def plan(self, query: str, ctx: Context) -> list[str]:
        return list(self._paths)


_FACT_RE = re.compile(r"^(.+?) —\[([^\]]+)\]→ (.+?)(?:: (.*))?$", re.S)


def _short_fact(raw: str) -> tuple[str, tuple[str, str, str] | None]:
    """A retrieval fact line without the repeated triple, and its triple key.

    Retrieval writes ``A —[T]→ B: (A, T, B): description``; the bracketed
    triple only repeats ``A —[T]→ B``. Lines in other shapes pass through.
    """
    raw = raw.strip()
    m = _FACT_RE.match(raw)
    if not m:
        return raw, None
    src, rel_type, tgt, rest = m.group(1), m.group(2), m.group(3), m.group(4) or ""
    body = ops.fact_body(rest, rel_type)
    text = f"{src} —[{rel_type}]→ {tgt}" + (f": {body}" if body else "")
    return text, (src.lower(), rel_type, tgt.lower())


_PATTERN_HINT_RELATIONS = 6
_PATTERN_HINT_EXAMPLES = 8


def _pattern_hint(pruned: PrunedTypes, redo: str) -> str:
    """Say how many relations an undeclared pattern dropped, and how to keep them."""
    mismatches = pruned.pattern_mismatches
    dropped = sum(n for pairs in mismatches.values() for n in pairs.values())
    ranked = sorted(mismatches.items(), key=lambda kv: (-sum(kv[1].values()), kv[0]))
    add = []
    examples = []
    for rel, pairs in ranked[:_PATTERN_HINT_RELATIONS]:
        ordered = sorted(pairs, key=lambda p: (-pairs[p], p))
        ends = [pair.split("->", 1) for pair in ordered]
        add.append({"label": rel, "patterns": [e for e in ends if len(e) == 2]})
        for pair in ordered:
            example = pruned.pattern_examples.get(rel, {}).get(pair)
            if example and len(examples) < _PATTERN_HINT_EXAMPLES:
                examples.append(f"{rel} {pair}: {example}")
    more = " (and more, see pruned)" if len(ranked) > _PATTERN_HINT_RELATIONS else ""
    seen = f" Dropped, e.g.: {'; '.join(examples)}." if examples else ""
    return (
        f"{dropped} extracted relation(s) were dropped only because their [source, target] "
        f"pair is not declared.{seen} Keep only the pairs whose facts the relation's meaning "
        "fits; for the others, declare a better relation type. To keep them all: "
        f"set_ontology(relations={json.dumps(add)}){more}, then {redo}, then finalize."
    )


def _names_hint(names: Sequence[str]) -> str:
    shown = list(names)[:_HINT_NAMES]
    more = f" (+{len(names) - len(shown)} more)" if len(names) > len(shown) else ""
    return ", ".join(shown) + more if shown else "none"


def _short(text: str, limit: int = _EVIDENCE_CHARS) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _mask(value: Any) -> Any:
    """Hide SDK-internal ``__Label__`` names in a JSON-safe value."""
    if isinstance(value, str):
        return mask_internal_labels(value)
    if isinstance(value, list):
        return [_mask(v) for v in value]
    if isinstance(value, dict):
        return {k: _mask(v) for k, v in value.items()}
    return value


def _as_dict(value: Any) -> Any:
    return value.model_dump() if isinstance(value, BaseModel) else value


def _ingested(path: str, res: Any, status: str, plan: dict[str, Any] | None) -> IngestedDocument:
    meta = getattr(res, "metadata", None) or {}
    return IngestedDocument(
        path=path,
        document_id=res.document_info.uid,
        status=status,
        chunks_indexed=res.chunks_indexed,
        nodes_created=res.nodes_created,
        relationships_created=res.relationships_created,
        plan=plan,
        pruned=PrunedTypes.from_report(meta.get("pruned")),
    )


class GraphRAGToolkit:
    """The graph tools for an outside agent, over one :class:`GraphRAG`.

    Read tools: ``status``, ``get_ontology``, ``lookup_entity``,
    ``traverse``, ``graph_walk``, ``cypher_query``, ``retrieve``,
    ``run_skill``, ``check_citations``. Ingestion tools: ``preview_files``
    (with a ``files_root``), ``set_ontology``, ``ingest``, ``finalize``.
    Each is an async method; :meth:`tool_specs` describes them for an agent
    framework and :meth:`call` runs one by name with validated arguments.

    Every tool result an answer may rely on carries an evidence number
    ``[n]``. Numbers run across all calls of one session (one toolkit, or one
    :meth:`new_session` copy), and :meth:`check_citations` verifies an
    answer's ``[n]`` against them.

    Args:
        rag: The GraphRAG instance to wrap.
        finalize_policy: ``"manual"`` (default: call :meth:`finalize`
            yourself after a batch of ingests), ``"on_write"`` (finalize
            after every ingest; for demos, finalize cost grows with the
            graph) or ``"never"`` (you call ``rag.finalize()``).
        read_only: Drop the ingestion tools and refuse them.
        include: Optional subset of tool names to advertise and accept in
            :meth:`call`. Direct method calls are not affected.
        tenant_id: Stamped into the Context of every operation.
        owns_rag: When True, :meth:`aclose` closes the wrapped GraphRAG.
        ledger_capacity: Evidence items kept per session.
        files_root: Folder ``preview_files`` and ``ingest(files=...)`` read
            from; nothing outside it is read. ``None`` (default) disables
            file ingestion.
        plan_limits: The strategies an agent's ingest plan may use, as
            :class:`~graphrag_sdk.ingestion.ingestion_planner.RuleBasedIngestionPlanner`
            keyword arguments (``chunkers``, ``extractors``, ``resolvers``,
            ``max_llm_pairs``, ``tune_params``). Default
            :data:`~graphrag_sdk.ingestion.ingestion_planner.CHEAP_PLAN_LIMITS`:
            no strategy that adds LLM calls.
    """

    def __init__(
        self,
        rag: GraphRAG,
        *,
        finalize_policy: FinalizePolicy = "manual",
        read_only: bool = False,
        include: Sequence[str] | None = None,
        tenant_id: str = "default",
        owns_rag: bool = False,
        ledger_capacity: int = DEFAULT_LEDGER_CAPACITY,
        files_root: str | os.PathLike[str] | None = None,
        plan_limits: Mapping[str, Any] | None = None,
    ) -> None:
        from graphrag_sdk.ingestion.ingestion_planner import (
            CHEAP_PLAN_LIMITS,
            RuleBasedIngestionPlanner,
        )

        if finalize_policy not in _POLICIES:
            raise ValueError(f"finalize_policy must be one of {_POLICIES}, got {finalize_policy!r}")
        valid = {td.name for td in TOOL_REGISTRY}
        if include is not None:
            unknown = sorted(set(include) - valid)
            if unknown:
                raise ValueError(f"Unknown tool names in include={unknown}; valid: {sorted(valid)}")
        self._rag = rag
        self._finalize_policy: FinalizePolicy = finalize_policy
        self._read_only = read_only
        self._include = frozenset(include) if include is not None else None
        self._tenant_id = tenant_id
        self._owns_rag = owns_rag
        self._ledger_capacity = ledger_capacity
        self._ledger = EvidenceLedger(ledger_capacity)
        self._files_root: Path | None = (
            fs.resolve_root(files_root) if files_root is not None else None
        )
        limits = dict(CHEAP_PLAN_LIMITS if plan_limits is None else plan_limits)
        RuleBasedIngestionPlanner((), **limits)  # validate now, not at the first ingest
        self._plan_limits: dict[str, Any] = limits
        # Shared by every session copy of this toolkit.
        self._shared: dict[str, Any] = {}

    # ── Lifecycle ────────────────────────────────────────────────

    @property
    def rag(self) -> GraphRAG:
        """The wrapped GraphRAG instance."""
        return self._rag

    @property
    def files_root(self) -> Path | None:
        """The folder file ingestion reads from (``None``: disabled)."""
        return self._files_root

    @property
    def ledger(self) -> EvidenceLedger:
        """This session's numbered evidence."""
        return self._ledger

    def new_session(self) -> GraphRAGToolkit:
        """A toolkit on the same graph with its own, empty evidence ledger.

        Use one per conversation (MCP does this per client session) so
        evidence numbers from different conversations never mix. The copy
        never closes the wrapped GraphRAG.
        """
        clone = copy.copy(self)
        clone._ledger = EvidenceLedger(self._ledger_capacity)
        clone._owns_rag = False
        return clone

    @classmethod
    def for_tenant(
        cls,
        base_config: ConnectionConfig,
        tenant_id: str,
        *,
        llm: LLMInterface,
        embedder: Embedder,
        ontology: Ontology | None = None,
        embedding_dimension: int = 256,
        **toolkit_kwargs: Any,
    ) -> GraphRAGToolkit:
        """Build a toolkit bound to a tenant-scoped graph.

        Derives ``graph_name = f"{base_config.graph_name}__{tenant_id}"`` and
        constructs a dedicated GraphRAG the toolkit owns (closed by
        :meth:`aclose` / ``async with``).
        """
        if not _TENANT_RE.match(tenant_id):
            raise ValueError(
                "tenant_id must match ^[A-Za-z0-9_-]{1,64}$ (it becomes part "
                f"of the graph name); got {tenant_id!r}"
            )
        from graphrag_sdk.api.main import GraphRAG  # local import: avoid cycle

        config = dataclasses.replace(
            base_config, graph_name=f"{base_config.graph_name}__{tenant_id}"
        )
        rag = GraphRAG(
            connection=config,
            llm=llm,
            embedder=embedder,
            ontology=ontology,
            embedding_dimension=embedding_dimension,
        )
        return cls(rag, tenant_id=tenant_id, owns_rag=True, **toolkit_kwargs)

    async def aclose(self) -> None:
        """Close the wrapped GraphRAG if this toolkit owns it."""
        if self._owns_rag:
            await self._rag.close()

    async def __aenter__(self) -> GraphRAGToolkit:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    # ── Specs & dispatch ─────────────────────────────────────────

    def tool_specs(self) -> list[ToolSpec]:
        """Tool definitions for this toolkit configuration."""
        return build_tool_specs(
            read_only=self._read_only,
            finalize_policy=self._finalize_policy,
            include=self._include,
            files=self._files_root is not None,
            plan_limits=self._plan_limits,
        )

    async def call(self, name: str, arguments: dict[str, Any] | None = None) -> BaseModel:
        """Validate ``arguments`` for tool ``name`` and run it.

        The entry point for adapters. Applies the same ``read_only`` /
        ``include`` / finalize-policy gates as :meth:`tool_specs`. Bad
        arguments and unknown or disabled tools raise :class:`ToolError`
        with a hint; a write on a read-only toolkit raises
        :class:`ReadOnlyViolation`.
        """
        by_name = {td.name: td for td in TOOL_REGISTRY}
        td = by_name.get(name)
        if td is None:
            enabled = [t.name for t in TOOL_REGISTRY if self._enabled(t)]
            raise ToolError(f"Unknown tool {name!r}.", hint=f"Tools: {', '.join(enabled)}.")
        if self._read_only and (not td.read_only or td.ingest_only):
            raise ReadOnlyViolation(f"{name} is disabled: the toolkit is read-only.")
        if not self._enabled(td):
            enabled = [t.name for t in TOOL_REGISTRY if self._enabled(t)]
            hint = f"Tools: {', '.join(enabled)}."
            if td.needs_files and self._files_root is None:
                hint = "No files root is configured; ingest text instead. " + hint
            raise ToolError(f"Tool {name!r} is not enabled.", hint=hint)
        try:
            model = td.input_model(**(arguments or {}))
        except ValidationError as exc:
            problems = "; ".join(
                f"{'.'.join(str(p) for p in err['loc']) or 'arguments'}: {err['msg']}"
                for err in exc.errors()
            )
            raise ToolError(
                f"Invalid arguments for {name}: {problems}.",
                hint="Fix the arguments to match the tool's input schema.",
            ) from None
        return await getattr(self, name)(**model.model_dump())

    def _enabled(self, td: Any) -> bool:
        return tool_enabled(
            td,
            read_only=self._read_only,
            finalize_policy=self._finalize_policy,
            include=self._include,
            files=self._files_root is not None,
        )

    def _new_ctx(self) -> Context:
        return Context(tenant_id=self._tenant_id)

    @property
    def _store(self) -> Any:
        return self._rag._graph_store

    def _evidence(self, tool: str, kind: str, content: str, source: str = "", **meta: Any) -> int:
        return self._ledger.add(
            tool=tool, kind=kind, content=_short(content), source=source, metadata=meta
        ).n

    async def _schema(self) -> GraphSchema:
        """Live labels / relationship types / property keys plus every
        ``RELATES`` rel_type (declared or stored), cached until the graph's
        topology changes or :data:`SCHEMA_CACHE_SECONDS` pass."""
        version = getattr(self._rag._conn, "topology_version", 0)
        now = time.monotonic()
        cached = self._shared.get("schema")
        if cached and cached[0] == version and now - cached[1] < SCHEMA_CACHE_SECONDS:
            return cached[2]
        ontology: Ontology | None = None
        try:
            ontology = await self._declared_ontology()
        except Exception:
            ontology = None
        schema = await introspect_schema(self._store, ontology)
        try:
            _, rel_counts = await ops.schema_counts(self._store)
            schema.semantic_relations = sorted(set(schema.semantic_relations) | set(rel_counts))
        except Exception:
            pass
        self._shared["schema"] = (version, now, schema)
        return schema

    def _schema_hint(self, schema: GraphSchema) -> str:
        labels = sorted(
            lbl for lbl in schema.labels if not lbl.startswith("__") and lbl not in ("Chunk",)
        )
        return (
            f"Entity labels: {_names_hint(labels)}. Relations are "
            f"[:RELATES {{rel_type: ...}}] with rel_type in: "
            f"{_names_hint(schema.semantic_relations)}. "
            "Use lookup_entity for exact entity names."
        )

    # ── Read tools ───────────────────────────────────────────────

    async def status(self) -> StatusResult:
        """What the graph holds: counts, entity labels and relation types."""
        counts = await ops.graph_counts(self._store)
        label_counts, rel_counts = await ops.schema_counts(self._store)
        hint = None
        if counts["entities"] == 0:
            hint = "The graph is empty: ingest documents (and finalize) before querying."
        elif counts["relations"] == 0:
            hint = "The graph has entities but no relations; traverse will find nothing."
        return StatusResult(
            entities=counts["entities"],
            relations=counts["relations"],
            documents=counts["documents"],
            chunks=counts["chunks"],
            entity_types=[EntityTypeInfo(label=k, count=v) for k, v in label_counts.items()],
            relation_types=[RelationTypeInfo(label=k, count=v) for k, v in rel_counts.items()],
            hint=hint,
        )

    async def _declared_ontology(self) -> Ontology:
        """The persisted ontology, read without registering anything.

        ``GraphRAG.get_ontology()`` registers the default entity types into
        an empty ontology graph; a read tool must not write, and those types
        would then be part of every ontology declared afterwards. Before
        anything is persisted this is the ontology the GraphRAG was built with.
        """
        store = getattr(self._rag, "_ontology_store", None)
        if store is None:
            return await self._rag.get_ontology()
        loaded = await store.load()
        if loaded.entities or loaded.relations:
            return loaded
        return getattr(self._rag, "ontology", None) or Ontology()

    async def get_ontology(self) -> OntologyResult:
        """The declared ontology with live counts (undeclared stored types too)."""
        ontology = await self._declared_ontology()
        label_counts, rel_counts = await ops.schema_counts(self._store)
        declared_e = {e.label: e for e in ontology.entities}
        declared_r = {r.label: r for r in ontology.relations}
        entity_types = [
            EntityTypeInfo(
                label=label,
                description=declared_e[label].description if label in declared_e else None,
                count=count,
                properties=[a.name for a in declared_e[label].properties]
                if label in declared_e
                else [],
            )
            for label, count in sorted({**{e: 0 for e in declared_e}, **label_counts}.items())
        ]
        relation_types = [
            RelationTypeInfo(
                label=label,
                description=declared_r[label].description if label in declared_r else None,
                patterns=list(declared_r[label].patterns) if label in declared_r else [],
                count=count,
            )
            for label, count in sorted({**{r: 0 for r in declared_r}, **rel_counts}.items())
        ]
        return OntologyResult(entity_types=entity_types, relation_types=relation_types)

    async def lookup_entity(
        self, name: str, *, limit: int = ops.LOOKUP_DEFAULT_LIMIT
    ) -> LookupResult:
        """Entities whose name matches ``name`` (or its words), best first.

        Not evidence: it finds names for the other tools, it does not state facts.
        """
        inp = LookupEntityInput(name=name, limit=limit)
        found = await ops.find_entities(self._store, inp.name.strip(), limit=inp.limit)
        if found is None:
            raise ToolError(
                "The entity lookup failed.",
                hint="Retry, or use retrieve to search by meaning.",
            )
        hint = None
        if not found:
            hint = (
                "No entity name matches. Try a shorter or different part of the name, "
                "or retrieve to search by meaning."
            )
        return LookupResult(query=inp.name, matches=[EntityMatch(**c) for c in found], hint=hint)

    async def traverse(
        self,
        start: str,
        *,
        via: list[str] | None = None,
        max_depth: int = ops.TRAVERSE_DEFAULT_DEPTH,
        max_expansions: int = ops.TRAVERSE_DEFAULT_EXPANSIONS,
        timeout_ms: int = ops.TRAVERSE_DEFAULT_TIMEOUT_MS,
    ) -> TraverseResult:
        """Walk the graph from ``start``, breadth-first or along ``via``.

        Bounded by ``max_depth`` hops, ``max_expansions`` edges read and
        ``timeout_ms`` in total; ``stopped`` names the budget that ended the
        walk (``"complete"`` when none did). Each edge (or, with ``via``,
        each entity reached) is a piece of evidence.
        """
        inp = TraverseInput(
            start=start,
            via=via,
            max_depth=max_depth,
            max_expansions=max_expansions,
            timeout_ms=timeout_ms,
        )
        deadline = time.monotonic() + inp.timeout_ms / 1000

        def remaining_ms() -> int:
            return int((deadline - time.monotonic()) * 1000)

        start_id, start_name = await ops.resolve_start(self._store, inp.start.strip(), deadline)

        if inp.via:
            path = await self._resolve_via(inp.via)
            nodes, stopped = await ops.typed_path(
                self._store, start_id, path, max(1, remaining_ms())
            )
            arrow = " -> ".join(path)
            reached = [
                TraverseNode(
                    n=self._evidence(
                        "traverse",
                        "path",
                        f"{start_name} -[{arrow}]-> {x['name']}",
                        entity_id=x["id"],
                    ),
                    id=x["id"],
                    name=x["name"],
                    depth=x["depth"],
                    via=x["relation"],
                )
                for x in nodes
            ]
            hint = None
            if stopped == "time_budget":
                hint = "The path query timed out; raise timeout_ms or start from a narrower entity."
            elif stopped == "return_limit":
                hint = (
                    f"More than {ops.TRAVERSE_PATH_LIMIT} entities matched; only those are shown."
                )
            elif not reached:
                hint = (
                    f"Nothing is reached from {start_name} by {arrow}. Relations are followed "
                    "in either direction; check the names with get_ontology, or traverse "
                    "without via to see what is connected."
                )
            return TraverseResult(
                start=start_name,
                start_id=start_id,
                via=path,
                nodes=reached,
                stopped=stopped,
                truncated=stopped != "complete",
                hint=hint,
            )

        nodes, edges, stopped = await ops.expand(
            self._store,
            start_id,
            start_name,
            inp.max_depth,
            inp.max_expansions,
            remaining_ms,
        )
        triples = []
        for e in edges:
            text = f"{e['source']} -[{e['relation']}]-> {e['target']}"
            if e["fact"]:
                text += f": {e['fact']}"
            triples.append(
                RelationTriple(
                    n=self._evidence(
                        "traverse",
                        "relation",
                        text,
                        source_id=e["source_id"],
                        target_id=e["target_id"],
                    ),
                    source=e["source"],
                    type=e["relation"],
                    target=e["target"],
                    fact=e["fact"] or None,
                )
            )
        hint = None
        if stopped == "expansion_budget":
            hint = "Stopped at max_expansions; raise it, or follow one path with via."
        elif stopped == "time_budget":
            hint = "Stopped at timeout_ms; raise it, or follow one path with via."
        elif stopped == "depth_budget":
            hint = "More lies beyond max_depth; raise it if you need further hops."
        elif not triples:
            hint = f"{start_name} has no relations."
        return TraverseResult(
            start=start_name,
            start_id=start_id,
            nodes=[
                TraverseNode(id=x["id"], name=x["name"], depth=x["depth"], via=x["relation"])
                for x in nodes
            ],
            edges=triples,
            stopped=stopped,
            truncated=stopped != "complete",
            hint=hint,
        )

    async def _resolve_via(self, via: list[str]) -> list[str]:
        """Map relation names to the stored spelling, or raise with the valid ones."""
        schema = await self._schema()
        known = list(schema.semantic_relations)
        if not known:
            return list(via)
        by_lower = {k.lower(): k for k in known}
        resolved, unknown = [], []
        for v in via:
            hit = by_lower.get(v.strip().lower())
            if hit is None:
                unknown.append(v)
            else:
                resolved.append(hit)
        if unknown:
            raise ToolError(
                f"Unknown relation(s): {', '.join(unknown)}.",
                hint=f"Relations in this graph: {_names_hint(known)}.",
            )
        return resolved

    async def graph_walk(
        self,
        start: str,
        *,
        goal: str | None = None,
        beam_width: int = 5,
        max_depth: int = 3,
        ctx: Context | None = None,
    ) -> WalkResult:
        """The strongest paths out of ``start`` (or from ``start`` to ``goal``).

        Paths are scored by relation weight and PageRank importance (when
        FalkorDB provides PageRank); each path is a piece of evidence.
        """
        inp = GraphWalkInput(start=start, goal=goal, beam_width=beam_width, max_depth=max_depth)
        ctx = ctx or self._new_ctx()
        start_id, start_name = await ops.resolve_start(
            self._store, inp.start.strip(), time.monotonic() + _RESOLVE_BUDGET_SECONDS
        )
        goal_id = goal_name = None
        if inp.goal and inp.goal.strip():
            goal_id, goal_name = await ops.resolve_start(
                self._store, inp.goal.strip(), time.monotonic() + _RESOLVE_BUDGET_SECONDS
            )
        try:
            weights = await self._store.pagerank()
        except Exception:
            weights = {}

        async def neighbor_fn(node_id: str) -> list[tuple[str, float, str]]:
            return await self._store.weighted_neighbors(node_id)

        walk = DynamicGraphWalk(
            neighbor_fn,
            node_weights=weights,
            beam_width=inp.beam_width,
            max_depth=inp.max_depth,
        )
        if goal_id is not None:
            found = await walk.bidirectional_search(start_id, goal_id, ctx=ctx)
            scored = [found] if found is not None else []
        else:
            scored = [p for p in await walk.beam_search(start_id, ctx=ctx) if p.length >= 1]
        names = await ops.entity_names(self._store, [n for p in scored for n in p.nodes])
        paths = []
        for p in scored:
            path = WalkPath(
                node_ids=list(p.nodes),
                nodes=[names.get(n, n) for n in p.nodes],
                edges=list(p.edges),
                score=round(float(p.score), 4),
            )
            text = " ".join(
                [path.nodes[0]]
                + [
                    f"-[{path.edges[i] if i < len(path.edges) else '?'}]-> {name}"
                    for i, name in enumerate(path.nodes[1:])
                ]
            )
            path.n = self._evidence("graph_walk", "path", text, node_ids=path.node_ids)
            paths.append(path)
        hint = None
        if not paths:
            hint = (
                f"No path connects {start_name} and {goal_name} within {inp.max_depth} hops."
                if goal_id is not None
                else f"{start_name} has no relations to walk."
            )
        return WalkResult(start=start_name, goal=goal_name, paths=paths, hint=hint)

    async def cypher_query(
        self,
        cypher: str,
        params: dict[str, Any] | None = None,
        *,
        max_rows: int = 25,
        timeout_ms: int = 5000,
    ) -> CypherResult:
        """Run one read-only Cypher query, checked against the live schema.

        Writes and procedure calls are rejected, unknown labels, relationship
        types and properties are rejected with the valid names, every
        ``LIMIT`` is capped to ``max_rows`` and the query runs as
        ``GRAPH.RO_QUERY`` with a database timeout. Each row is a piece of
        evidence; ``truncated`` means more rows matched than were returned.
        """
        inp = CypherQueryInput(
            cypher=cypher, params=params, max_rows=max_rows, timeout_ms=timeout_ms
        )
        # Structural checks (writes, procedures) need no schema or database.
        problems = validate_read_query(inp.cypher, None, require_rel_type=False)
        if problems:
            raise ToolError(
                f"The query was rejected: {'; '.join(problems)}.",
                hint="Only read queries are allowed: MATCH ... RETURN, with no CALL.",
            )
        schema = await self._schema()
        problems = validate_read_query(inp.cypher, schema, require_rel_type=False)
        if problems:
            raise ToolError(
                f"The query was rejected: {'; '.join(problems)}.",
                hint=self._schema_hint(schema),
            )
        try:
            capped = enforce_row_cap(inp.cypher, inp.max_rows + 1)
        except ValueError as exc:
            raise ToolError(f"The query could not be read: {exc}.") from None
        try:
            result = await ops.bounded_query(self._store, capped, inp.params, inp.timeout_ms)
        except ops.TIMEOUTS:
            raise ToolError(
                f"The query timed out after {inp.timeout_ms} ms.",
                hint="Filter earlier, start from a specific entity, or aggregate with count().",
            ) from None
        except DatabaseError as exc:
            raise ToolError(
                f"The query failed: {exc}",
                hint=self._schema_hint(schema),
            ) from None
        columns = [mask_internal_labels(c) for c in ops.column_names(result)]
        raw_rows = list(getattr(result, "result_set", None) or [])
        truncated = len(raw_rows) > inp.max_rows
        rows = []
        for raw in raw_rows[: inp.max_rows]:
            values = [_mask(ops.to_jsonable(v)) for v in raw]
            record = dict(zip(columns, values, strict=False)) if columns else values
            n = self._evidence(
                "cypher_query",
                "row",
                json.dumps(record, ensure_ascii=False, default=str),
                cypher=inp.cypher,
            )
            rows.append(CypherRow(n=n, values=values))
        hint = None
        if truncated:
            hint = (
                f"More than {inp.max_rows} rows matched; do not count or rank from these rows. "
                "Aggregate in Cypher (count(), collect()) or add filters."
            )
        elif not rows:
            hint = (
                "No rows. Check exact names with lookup_entity; rel_type values are "
                "case-sensitive (see get_ontology)."
            )
        return CypherResult(
            columns=columns,
            rows=rows,
            row_count=len(rows),
            truncated=truncated,
            hint=hint,
        )

    def _make_strategy(
        self,
        top_k: int,
        ontology: Ontology | None = None,
        plan: frozenset[str] | None = None,
    ) -> MultiPathRetrieval:
        """A MultiPathRetrieval sized to ``top_k`` (its constructor only assigns).

        ``plan`` names the MultiPath paths to run (default: every path
        retrieve reads).
        """
        return MultiPathRetrieval(
            graph_store=self._rag._graph_store,
            vector_store=self._rag._vector_store,
            embedder=self._rag.embedder,
            llm=self._rag.llm,
            chunk_top_k=top_k,
            rel_top_k=top_k,
            max_entities=max(2 * top_k, 10),
            router=_FixedPlan(plan or _ALL_MULTIPATH_PATHS),
            ontology=ontology or self._rag._global_ontology,
        )

    async def _search(
        self, question: str, top_k: int, ctx: Context, plan: frozenset[str] | None = None
    ) -> Any:
        """Run MultiPath retrieval without writing to the graph.

        ``GraphRAG.retrieve`` first initialises the ontology, which on a graph
        nothing was declared on yet registers the default entity types: a
        write, from a read tool. Until the GraphRAG has done that itself, run
        the strategy directly on the declared ontology instead.
        """
        if getattr(self._rag, "_ontology_initialized", True):
            return await self._rag.retrieve(
                question, strategy=self._make_strategy(top_k, plan=plan), ctx=ctx
            )
        declared = await self._declared_ontology()
        ontology = declared if (declared.entities or declared.relations) else None
        await self._rag._validate_graph_config(ctx=ctx)
        ctx.ensure_budget("retrieval strategy search")
        return await self._make_strategy(top_k, ontology, plan).search(question, ctx)

    async def retrieve(
        self,
        question: str,
        *,
        top_k: int = 8,
        expand_hops: int = 1,
        include_chunks: bool = True,
        paths: list[str] | None = None,
        ctx: Context | None = None,
    ) -> RetrieveResult:
        """Hybrid retrieval for ``question``: entities, relations, facts, passages.

        Runs the SDK's MultiPath retrieval (vector, full-text and graph) and
        returns the ranked context, without generating an answer. Each
        entity, relation, fact and passage is a piece of evidence. ``paths``
        runs only the named search paths (see ``RETRIEVE_PATHS``); by default
        every path runs.
        """
        inp = RetrieveInput(
            question=question,
            top_k=top_k,
            expand_hops=expand_hops,
            include_chunks=include_chunks,
            paths=paths,  # type: ignore[arg-type]
        )
        wanted = set(inp.paths or RETRIEVE_PATHS)
        if not inp.include_chunks:
            wanted.discard("passages")
        if not wanted:
            raise ToolError(
                "paths names only passages, but include_chunks is false.",
                hint="Drop include_chunks, or add other paths.",
            )
        if "links" in wanted and not wanted & _SEED_PATHS:
            wanted.add("exact_names")  # links walk from found entities
        plan = frozenset(_MULTIPATH_PATH[p] for p in wanted if p in _MULTIPATH_PATH)
        ctx = ctx or self._new_ctx()
        store = self._store
        rr = await self._search(inp.question, inp.top_k, ctx, plan)
        prov: dict[str, Any] = rr.metadata.get("provenance") or {}
        seeds = [e for e in (prov.get("entities") or []) if e.get("id")][: inp.top_k]
        seed_ids = [e["id"] for e in seeds]
        cards = await ops.enrich_entities(store, seed_ids)

        entities = []
        for seed in seeds:
            card = cards.get(seed["id"]) or {
                "id": seed["id"],
                "name": seed.get("name", ""),
                "label": "",
                "description": seed.get("description") or None,
                "properties": {},
            }
            text = card["name"] + (f" ({card['label']})" if card["label"] else "")
            if card.get("description"):
                text += f": {card['description']}"
            entities.append(
                EntityCard(
                    n=self._evidence("retrieve", "entity", text, entity_id=card["id"]), **card
                )
            )

        relations = []
        shown: set[tuple[str, str, str]] = set()
        triples = (
            await ops.expand_triples(
                store, seed_ids, hops=inp.expand_hops, cap=min(4 * inp.top_k, 40)
            )
            if "links" in wanted
            else []
        )
        for t in triples:
            text = f"{t['source']} -[{t['type']}]-> {t['target']}"
            if t["fact"]:
                text += f": {t['fact']}"
            shown.add((t["source"].lower(), t["type"], t["target"].lower()))
            relations.append(
                RelationTriple(
                    n=self._evidence("retrieve", "relation", text),
                    source=t["source"],
                    type=t["type"],
                    target=t["target"],
                    fact=t["fact"] or None,
                )
            )

        facts = []
        for raw in (prov.get("facts") or [])[: 2 * inp.top_k]:
            text, key = _short_fact(str(raw))
            if not text or (key and key in shown):
                continue
            facts.append(Fact(n=self._evidence("retrieve", "fact", text), text=text))

        chunks: list[ChunkRef] = []
        documents: list[DocumentRef] = []
        if inp.include_chunks:
            prov_chunks = [c for c in (prov.get("chunks") or []) if c.get("id")][: inp.top_k]
            doc_map = await ops.chunk_documents(store, [c["id"] for c in prov_chunks])
            seen_docs: set[str] = set()
            for c in prov_chunks:
                doc_id, doc_path = doc_map.get(c["id"], ("", c.get("document_path", "")))
                text = str(c.get("text", ""))
                chunks.append(
                    ChunkRef(
                        n=self._evidence(
                            "retrieve",
                            "passage",
                            text,
                            source=doc_path or doc_id,
                            chunk_id=c["id"],
                        ),
                        chunk_id=c["id"],
                        document_id=doc_id,
                        document_path=doc_path if doc_path != doc_id else None,
                        text=text,
                    )
                )
                if doc_id and doc_id not in seen_docs:
                    seen_docs.add(doc_id)
                    documents.append(DocumentRef(document_id=doc_id, document_path=doc_path))
        hint = None
        if not (entities or relations or facts or chunks):
            hint = (
                "Nothing relevant was found. Try lookup_entity with a name from the "
                "question, or rephrase."
            )
            if inp.paths:
                hint = (
                    "Nothing relevant was found on the paths you chose. Run retrieve "
                    "again without paths."
                )
        return RetrieveResult(
            question=inp.question,
            entities=entities,
            relations=relations,
            facts=facts,
            chunks=chunks,
            documents=documents,
            hint=hint,
        )

    async def run_skill(
        self,
        skill: str,
        params: dict[str, Any] | None = None,
        *,
        ctx: Context | None = None,
    ) -> SkillRunResult:
        """Run a registered skill; its findings are one piece of evidence."""
        from graphrag_sdk.skills import SKILL_REGISTRY, build_skill
        from graphrag_sdk.skills.base import check_arguments

        inp = RunSkillInput(skill=skill, params=params or {})
        if inp.skill not in SKILL_REGISTRY:
            raise ToolError(
                f"Unknown skill {inp.skill!r}.",
                hint=f"Skills: {', '.join(sorted(SKILL_REGISTRY))}.",
            )
        impl = build_skill(inp.skill, self._store, self._rag.llm)
        problems = check_arguments(impl.parameters, inp.params)
        if problems:
            accepted = ", ".join((impl.parameters.get("properties") or {}).keys()) or "none"
            raise ToolError(
                f"Invalid arguments for skill {inp.skill!r}: {'; '.join(problems)}.",
                hint=f"{inp.skill} accepts: {accepted}.",
            )
        try:
            result = await impl.run(ctx or self._new_ctx(), **inp.params)
        except ValueError as exc:
            raise ToolError(
                f"Skill {inp.skill!r} could not run: {exc}.",
                hint="Get exact entity names with lookup_entity, then call the skill again.",
            ) from None
        data = _mask(ops.to_jsonable(dict(result.data)))
        content = result.summary or json.dumps(data, ensure_ascii=False, default=str)
        n = self._evidence("run_skill", "skill", f"{inp.skill}: {content}", skill=inp.skill)
        return SkillRunResult(
            skill=inp.skill,
            n=n,
            summary=result.summary,
            data=data if isinstance(data, dict) else {"value": data},
        )

    async def check_citations(self, answer: str) -> CitationCheckResult:
        """Check ``answer``'s ``[n]`` citations against this session's evidence.

        Markers that match no evidence are removed from the returned answer.
        Grounded means at least one real piece of evidence is cited. On a
        grounded answer it also lists claims (sentences stating a figure or a
        name) with no [n] of their own, and cited claims whose figures or names
        the cited evidence does not contain; neither changes ``grounded``.
        """
        inp = CheckCitationsInput(answer=answer)
        checked = ground_answer(inp.answer, self._ledger)
        sources = [
            EvidenceRef(n=ev.n, tool=ev.tool, kind=ev.kind, content=ev.content, source=ev.source)
            for ev in checked.citations(self._ledger)
        ]
        hints: list[str] = []
        if checked.reason == "no_citations":
            hints.append("Cite the evidence numbers [n] from tool results after each claim.")
        elif checked.reason == "invalid_citations":
            hints.append(
                "None of those numbers were returned in this session; cite only [n] "
                "from tool results."
            )
        elif checked.invalid:
            hints.append("Some numbers were not returned in this session and were removed.")
        if checked.uncited:
            hints.append("Give each listed claim the [n] that states it, or drop it.")
        if checked.unsupported:
            hints.append(
                "The cited evidence does not contain the missing figures or names: cite "
                "the [n] that states them, or correct the claim."
            )
        hint = " ".join(hints) or None
        return CitationCheckResult(
            grounded=checked.grounded,
            reason=checked.reason,
            cited=checked.cited,
            invalid=checked.invalid,
            uncited=checked.uncited,
            unsupported=[
                ClaimCheck(sentence=c.sentence, missing=c.missing) for c in checked.unsupported
            ],
            answer=checked.answer,
            sources=sources,
            hint=hint,
        )

    # ── Write tools ──────────────────────────────────────────────

    def _ensure_writable(self, tool: str) -> None:
        if self._read_only:
            raise ReadOnlyViolation(f"{tool} is disabled: the toolkit is read-only.")

    def _require_root(self, tool: str) -> Path:
        if self._files_root is None:
            raise ToolError(
                f"{tool} needs a files root, and none is configured.",
                hint="Ingest text instead, or start the toolkit with files_root=...",
            )
        return self._files_root

    async def preview_files(
        self,
        pattern: str = "**/*",
        *,
        offset: int = 0,
        limit: int = 20,
        max_chars: int = 1000,
        ctx: Context | None = None,
    ) -> PreviewFilesResult:
        """A page of the files under the files root, with a sample of each.

        Each sample is taken from the middle of the file's text, where the
        subject matter is rather than the title page or the references. Page
        through every file (``next_offset``) to see the whole corpus before
        designing an ontology. Samples are not evidence: nothing here is
        stored, and they get no ``[n]``.
        """
        self._ensure_writable("preview_files")
        root = self._require_root("preview_files")
        inp = PreviewFilesInput(pattern=pattern, offset=offset, limit=limit, max_chars=max_chars)
        try:
            rels = await asyncio.to_thread(fs.list_files, root, inp.pattern)
        except ValueError as exc:
            raise ToolError(str(exc), hint="Use a glob relative to the files root.") from None
        types: dict[str, int] = {}
        for rel in rels:
            kind = Path(rel).suffix.lower().lstrip(".")
            types[kind] = types.get(kind, 0) + 1
        page = rels[inp.offset : inp.offset + inp.limit]
        per_file = min(inp.max_chars, SAMPLE_BUDGET_CHARS // max(len(page), 1))
        ctx = ctx or self._new_ctx()
        gate = asyncio.Semaphore(_FILE_CONCURRENCY)

        async def sample(rel: str) -> DocSample:
            path = root / rel
            kind = path.suffix.lower().lstrip(".")
            try:
                size = path.stat().st_size
            except OSError:
                size = 0
            async with gate:
                try:
                    from graphrag_sdk.api.main import GraphRAG  # local import: avoid cycle

                    doc = await GraphRAG._default_loader_for(str(path)).load(str(path), ctx)
                except Exception as exc:  # one unreadable file must not fail the page
                    return DocSample(
                        path=rel, type=kind, size_bytes=size, error=_short(str(exc), 300)
                    )
            text = doc.text or ""
            return DocSample(
                path=rel,
                type=kind,
                size_bytes=size,
                chars=len(text),
                sample=fs.middle_sample(text, per_file),
            )

        samples = list(await asyncio.gather(*(sample(rel) for rel in page)))
        end = inp.offset + len(page)
        hint = None
        if not rels:
            hint = (
                f"No supported files match {inp.pattern!r} "
                f"({', '.join(fs.SUPPORTED_SUFFIXES)}). Try pattern='**/*'."
            )
        elif end < len(rels):
            hint = "Keep paging until next_offset is null, then design the ontology."
        return PreviewFilesResult(
            pattern=inp.pattern,
            total=len(rels),
            offset=inp.offset,
            next_offset=end if end < len(rels) else None,
            types=types,
            files=samples,
            hint=hint,
        )

    async def set_ontology(
        self,
        entities: Sequence[Mapping[str, Any] | BaseModel] = (),
        relations: Sequence[Mapping[str, Any] | BaseModel] = (),
    ) -> OntologyResult:
        """Declare entity and relation types; extraction keeps only these.

        Adds to what is declared: new types are added, a given description
        replaces the old one, and new ``[source, target]`` patterns are added
        to a relation. Nothing is removed or renamed. Call it before the first
        ingest; types added later apply to files ingested (or re-extracted)
        afterwards.
        """
        self._ensure_writable("set_ontology")
        inp = SetOntologyInput(
            entities=[_as_dict(e) for e in entities],
            relations=[_as_dict(r) for r in relations],
        )
        if not inp.entities and not inp.relations:
            raise ToolError(
                "set_ontology needs at least one entity or relation type.",
                hint="Pass entities=[{label, description}] and/or relations=[...].",
            )
        for label in [e.label for e in inp.entities] + [r.label for r in inp.relations]:
            if "|" in label or sanitize_cypher_label(label) != label:
                raise ToolError(
                    f"{label!r} is not a usable type name.",
                    hint="Use letters, digits, spaces, '-' or '_' (no '|', quotes or backticks).",
                )

        store = self._rag._ontology_store
        stored = await store.load()
        persisted = bool(stored.entities or stored.relations)
        base = stored if persisted else (getattr(self._rag, "ontology", None) or Ontology())
        entities_by = {e.label: e for e in base.entities}
        relations_by = {r.label: r for r in base.relations}
        known = set(entities_by) | {e.label for e in inp.entities}
        undeclared = sorted(
            {end for r in inp.relations for pair in r.patterns for end in pair} - known
        )
        if undeclared:
            raise ToolError(
                f"Relation patterns use undeclared entity types: {', '.join(undeclared)}.",
                hint="Declare them in entities (in this call or an earlier one).",
            )

        new_entities: list[Entity] = []
        new_relations: list[Relation] = []
        descriptions: list[tuple[str, str, str]] = []  # (kind, label, description)
        patterns: list[tuple[str, str, str]] = []  # (relation, source, target)
        for e in inp.entities:
            old = entities_by.get(e.label)
            if old is None:
                entity = Entity(label=e.label, description=e.description)
                entities_by[e.label] = entity
                new_entities.append(entity)
            elif e.description and e.description != old.description:
                descriptions.append(("entity", e.label, e.description))
        for r in inp.relations:
            pairs = list(dict.fromkeys((src, tgt) for src, tgt in r.patterns))
            old_rel = relations_by.get(r.label)
            if old_rel is None:
                relation = Relation(label=r.label, description=r.description, patterns=pairs)
                relations_by[r.label] = relation
                new_relations.append(relation)
                continue
            if r.description and r.description != old_rel.description:
                descriptions.append(("relation", r.label, r.description))
            have = {tuple(p) for p in old_rel.patterns}
            patterns.extend((r.label, s, t) for s, t in pairs if (s, t) not in have)

        if not persisted:
            # Nothing stored yet: fold every change into one declaration.
            for kind, label, text in descriptions:
                if kind == "entity":
                    entities_by[label] = entities_by[label].model_copy(update={"description": text})
                else:
                    relations_by[label] = relations_by[label].model_copy(
                        update={"description": text}
                    )
            for rel, src, tgt in patterns:
                old_rel = relations_by[rel]
                relations_by[rel] = old_rel.model_copy(
                    update={"patterns": [*old_rel.patterns, (src, tgt)]}
                )
            descriptions, patterns = [], []
        try:
            if new_entities or new_relations or not persisted:
                merged = base.model_copy(
                    update={
                        "entities": list(entities_by.values()),
                        "relations": list(relations_by.values()),
                    }
                )
                await self._rag.set_ontology(merged)
            for kind, label, text in descriptions:
                if kind == "entity":
                    await self._rag.set_entity_description(label, text)
                else:
                    await self._rag.set_relation_description(label, text)
            for rel, src, tgt in patterns:
                await self._rag.add_relation_pattern(rel, src, tgt)
        except ValueError as exc:
            raise ToolError(
                f"The ontology was not accepted: {_short(str(exc), 500)}",
                hint="Existing types cannot change their properties; add new types instead.",
            ) from None
        self._shared.pop("schema", None)
        return await self.get_ontology()

    async def ingest(
        self,
        text: str | None = None,
        *,
        document_id: str | None = None,
        files: Sequence[str] | None = None,
        plan: Sequence[Mapping[str, Any] | BaseModel] | None = None,
        reextract: bool = False,
        ctx: Context | None = None,
    ) -> IngestResult:
        """Extract ``text``, or ``files`` under the files root, into the graph.

        ``files`` takes paths and globs relative to the files root (at most
        50 files per call). ``plan`` is a list of rules, each a glob
        (``match``) with the chunker / extractor / resolver for the files it
        matches; the first matching rule wins, and the allowed options are
        those of ``plan_limits``. A file whose content is unchanged is
        skipped unless ``reextract`` is set, which re-extracts it in place
        (after the ontology has grown, for instance); without a ``plan``, a
        file re-extracted this way keeps the plan of its last ingest by
        this toolkit.

        Under ``finalize_policy="on_write"`` this also finalizes, whose cost
        grows with the graph; prefer "manual" and one :meth:`finalize` per batch.
        """
        self._ensure_writable("ingest")
        inp = IngestInput(
            text=text,
            document_id=document_id,
            files=list(files) if files is not None else None,
            plan=[_as_dict(r) for r in plan] if plan is not None else None,
            reextract=reextract,
        )
        ctx = ctx or self._new_ctx()
        if inp.text is not None:
            if inp.reextract:
                result = await self._rag.update(
                    text=inp.text,
                    document_id=inp.document_id,
                    force=True,
                    if_missing="ingest",
                    ctx=ctx,
                )
                status = "reextracted" if result.replaced_existing else "ingested"
            else:
                result = await self._rag.ingest(text=inp.text, document_id=inp.document_id, ctx=ctx)
                unchanged = (result.metadata or {}).get("skipped_unchanged")
                status = "unchanged" if unchanged else "ingested"
            out = IngestResult(
                document_id=result.document_info.uid,
                status=status,
                chunks_indexed=result.chunks_indexed,
                nodes_created=result.nodes_created,
                relationships_created=result.relationships_created,
                pruned=PrunedTypes.from_report(result.metadata.get("pruned")),
            )
        else:
            out = await self._ingest_files(inp.files or [], inp.plan or [], inp.reextract, ctx)
        if self._finalize_policy == "on_write":
            await self._rag.finalize()
            out.finalized = True
        out.hint = self._ingest_hint(out)
        return out

    def _remember_plan(self, path: str, plan: Mapping[str, Any]) -> None:
        plans: dict[str, dict[str, Any]] = self._shared.setdefault("file_plans", {})
        plans.pop(path, None)
        plans[path] = dict(plan)
        while len(plans) > _PLAN_MEMORY:
            del plans[next(iter(plans))]

    def _last_plan(self, path: str) -> Any:
        """The plan ``path`` was last ingested with by this toolkit, or ``None``."""
        from graphrag_sdk.ingestion.ingestion_planner import IngestionPlan

        used = self._shared.get("file_plans", {}).get(path)
        if not used:
            return None
        try:
            return IngestionPlan(
                chunker=used["chunker"],
                extractor=used["extractor"],
                resolver=used["resolver"],
                chunker_params=dict(used.get("chunker_params") or {}),
                extractor_params=dict(used.get("extractor_params") or {}),
                resolver_params=dict(used.get("resolver_params") or {}),
                reason="the plan of this file's last ingest",
            )
        except (KeyError, ValueError):
            return None

    async def _ingest_files(
        self,
        entries: list[str],
        rules: list[PlanRuleInput],
        reextract: bool,
        ctx: Context,
    ) -> IngestResult:
        from graphrag_sdk.ingestion.ingestion_planner import (
            PlanRule,
            RuleBasedIngestionPlanner,
            build_ingestion_strategies,
        )

        root = self._require_root("ingest(files=...)")
        try:
            rels = await asyncio.to_thread(fs.resolve_files, root, entries)
            planner = RuleBasedIngestionPlanner(
                [PlanRule.from_dict(r.model_dump(exclude_none=True)) for r in rules],
                root=root,
                **self._plan_limits,
            )
        except ValueError as exc:
            raise ToolError(str(exc), hint="Fix the plan rules.") from None
        paths = [str(root / rel) for rel in rels]
        docs: list[IngestedDocument] = []
        if not reextract:
            results = await self._rag.ingest(
                paths, planner=planner, max_concurrency=_FILE_CONCURRENCY, ctx=ctx
            )
            for rel, res in zip(rels, results):
                if isinstance(res, BaseException):
                    docs.append(
                        IngestedDocument(path=rel, status="failed", error=_short(str(res), 300))
                    )
                    continue
                meta = res.metadata or {}
                used = (meta.get("plan") or {}).get("plan")
                if used:
                    self._remember_plan(str(root / rel), used)
                docs.append(
                    _ingested(
                        rel,
                        res,
                        "unchanged" if meta.get("skipped_unchanged") else "ingested",
                        used,
                    )
                )
        else:
            gate = asyncio.Semaphore(_FILE_CONCURRENCY)
            entity_types = self._rag._ontology_entity_types()

            async def redo(rel: str, path: str) -> IngestedDocument:
                async with gate:
                    try:
                        p = None if rules else self._last_plan(path)
                        if p is None:
                            p = await planner.plan(None, source=path, ctx=ctx)
                        chunker, extractor, resolver = build_ingestion_strategies(
                            p,
                            llm=self._rag.llm,
                            embedder=self._rag.embedder,
                            entity_types=entity_types,
                        )
                        res = await self._rag.update(
                            path,
                            force=True,
                            if_missing="ingest",
                            chunker=chunker,
                            extractor=extractor,
                            resolver=resolver,
                            ctx=ctx,
                        )
                    except Exception as exc:  # one bad file must not fail the batch
                        return IngestedDocument(
                            path=rel, status="failed", error=_short(str(exc), 300)
                        )
                plan_info = {
                    "chunker": p.chunker,
                    "extractor": p.extractor,
                    "resolver": p.resolver,
                    "chunker_params": dict(p.chunker_params),
                    "extractor_params": dict(p.extractor_params),
                    "resolver_params": dict(p.resolver_params),
                    "reason": p.reason,
                }
                self._remember_plan(path, plan_info)
                status = "reextracted" if res.replaced_existing else "ingested"
                return _ingested(rel, res, status, plan_info)

            docs = list(await asyncio.gather(*(redo(r, p) for r, p in zip(rels, paths))))

        total = PrunedTypes()
        for doc in docs:
            if doc.pruned is not None:
                total.add(doc.pruned)
        return IngestResult(
            chunks_indexed=sum(d.chunks_indexed for d in docs),
            nodes_created=sum(d.nodes_created for d in docs),
            relationships_created=sum(d.relationships_created for d in docs),
            documents=docs,
            failed=sum(d.status == "failed" for d in docs),
            pruned=total if (total.missing_types or total.dangling) else None,
        )

    def _ingest_hint(self, out: IngestResult) -> str | None:
        hints: list[str] = []
        if out.failed:
            hints.append(f"{out.failed} file(s) failed; see each document's error.")
        if out.pruned is not None and out.pruned.missing_types:
            affected = [
                d.path for d in out.documents if d.pruned is not None and d.pruned.missing_types
            ]
            redo = (
                f"ingest(files={affected[:10]}{' ...' if len(affected) > 10 else ''}, "
                "reextract=true)"
                if affected
                else f"ingest(text=..., document_id={out.document_id!r}, reextract=true)"
            )
            look = (
                " To see the Unknown entities: cypher_query MATCH (n:Unknown) "
                "RETURN n.name, n.description LIMIT 25."
                if out.pruned.unknown
                else ""
            )
            pruned = out.pruned
            if pruned.pattern_mismatches:
                hints.append(_pattern_hint(pruned, redo))
            if pruned.entities or pruned.relations or pruned.unknown:
                hints.append(
                    "Extraction found types the ontology does not declare (see pruned)."
                    f"{look} If they matter, add them with set_ontology, then {redo}."
                )
        if out.documents and all(d.status == "unchanged" for d in out.documents):
            hints.append("Every file was unchanged; pass reextract=true to extract them again.")
        if out.status == "unchanged":
            hints.append(
                "This text is already stored; to extract it again, pass "
                f"document_id={out.document_id!r} and reextract=true."
            )
        if self._finalize_policy == "manual" and not out.finalized:
            hints.append("Call finalize once after this batch of ingests.")
        return " ".join(hints) or None

    async def finalize(self) -> FinalizeReport:
        """Run ``GraphRAG.finalize()`` (dedup, embeddings, indexes).

        Under "on_write" each ingest already finalized, so this does
        nothing; under "never" it raises :class:`ConfigError`.
        """
        self._ensure_writable("finalize")
        if self._finalize_policy == "on_write":
            return FinalizeReport(finalized=True, hint="Each ingest already finalized.")
        if self._finalize_policy == "never":
            raise ConfigError(
                'finalize_policy="never": the toolkit never finalizes; call '
                "GraphRAG.finalize() yourself."
            )
        result = await self._rag.finalize()
        return FinalizeReport(
            finalized=True,
            entities_deduplicated=int(getattr(result, "entities_deduplicated", 0) or 0),
            entities_embedded=int(getattr(result, "entities_embedded", 0) or 0),
            relationships_embedded=int(getattr(result, "relationships_embedded", 0) or 0),
            null_stubs_removed=int(getattr(result, "null_stubs_removed", 0) or 0),
        )
