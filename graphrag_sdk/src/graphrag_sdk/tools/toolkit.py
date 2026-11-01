# GraphRAG SDK — Tools: GraphRAGToolkit
# The graph tools an outside agent calls (through MCP or its own framework).
# The agent plans, calls tools and writes the answer; the toolkit runs each
# call bounded and read-only, numbers the evidence it returns and checks the
# agent's citations against it. There is no agent loop in the SDK.

from __future__ import annotations

import copy
import dataclasses
import logging
import re
from collections.abc import Sequence
from types import TracebackType
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ValidationError

from graphrag_sdk.core.connection import ConnectionConfig
from graphrag_sdk.core.context import Context
from graphrag_sdk.core.exceptions import ConfigError, ReadOnlyViolation
from graphrag_sdk.core.models import Ontology
from graphrag_sdk.core.providers import Embedder, LLMInterface
from graphrag_sdk.retrieval.strategies.multi_path import MultiPathRetrieval
from graphrag_sdk.tools import graph_ops as ops
from graphrag_sdk.tools.errors import ToolError
from graphrag_sdk.tools.evidence import DEFAULT_LEDGER_CAPACITY, EvidenceLedger, ground_answer
from graphrag_sdk.tools.models import (
    ChunkRef,
    CitationCheckResult,
    ClaimCheck,
    DocumentRef,
    EntityCard,
    EntityTypeInfo,
    EvidenceRef,
    Fact,
    FinalizeReport,
    IngestResult,
    OntologyResult,
    RelationTriple,
    RelationTypeInfo,
    RetrieveResult,
    StatusResult,
    properties_text,
)
from graphrag_sdk.tools.specs import (
    RETRIEVE_PATHS,
    TOOL_REGISTRY,
    CheckCitationsInput,
    IngestInput,
    RetrieveInput,
    ToolSpec,
    build_tool_specs,
)

if TYPE_CHECKING:
    from graphrag_sdk.api.main import GraphRAG

FinalizePolicy = Literal["manual", "on_write", "never"]
_POLICIES = ("manual", "on_write", "never")
_TENANT_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
logger = logging.getLogger(__name__)

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


class GraphRAGToolkit:
    """The graph tools for an outside agent, over one :class:`GraphRAG`.

    Read tools: ``status``, ``get_ontology``, ``retrieve``,
    ``check_citations``. Write tools:
    ``ingest``, ``finalize``. Each is an async method; :meth:`tool_specs`
    describes them for an agent framework and :meth:`call` runs one by name
    with validated arguments.

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
        read_only: Drop ``ingest`` / ``finalize`` and refuse them.
        include: Optional subset of tool names to advertise and accept in
            :meth:`call`. Direct method calls are not affected.
        tenant_id: Stamped into the Context of every operation.
        owns_rag: When True, :meth:`aclose` closes the wrapped GraphRAG.
        ledger_capacity: Evidence items kept per session.
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
    ) -> None:
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
        # Shared by every session copy of this toolkit.
        self._shared: dict[str, Any] = {}

    # ── Lifecycle ────────────────────────────────────────────────

    @property
    def rag(self) -> GraphRAG:
        """The wrapped GraphRAG instance."""
        return self._rag

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
        enabled = [s.name for s in self.tool_specs()]
        if td is None:
            raise ToolError(f"Unknown tool {name!r}.", hint=f"Tools: {', '.join(enabled)}.")
        if self._read_only and not td.read_only:
            raise ReadOnlyViolation(f"{name} is disabled: the toolkit is read-only.")
        if name not in enabled:
            raise ToolError(f"Tool {name!r} is not enabled.", hint=f"Tools: {', '.join(enabled)}.")
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

    def _new_ctx(self) -> Context:
        return Context(tenant_id=self._tenant_id)

    @property
    def _store(self) -> Any:
        return self._rag._graph_store

    def _evidence(self, tool: str, kind: str, content: str, source: str = "", **meta: Any) -> int:
        # The full text the agent was shown, so check_citations checks the
        # same text the agent cited from.
        return self._ledger.add(
            tool=tool, kind=kind, content=content, source=source, metadata=meta
        ).n

    @property
    def _graph_name(self) -> str:
        config = getattr(self._rag._conn, "config", None)
        return str(getattr(config, "graph_name", None) or "default")

    def _missing_graph(self) -> ToolError:
        return ToolError(
            f"Graph '{self._graph_name}' does not exist.",
            hint=(
                "Ingest documents (and finalize) first, or check the graph name "
                "the server was started with."
            ),
        )

    async def _require_graph(self) -> None:
        """Raise a ToolError if the graph was never written. Checked
        read-only (asking never creates it) until it is found to exist."""
        if self._shared.get("graph_exists"):
            return
        exists = await ops.graph_exists(self._store)
        if exists is False:
            raise self._missing_graph()
        if exists:
            self._shared["graph_exists"] = True

    # ── Read tools ───────────────────────────────────────────────

    async def status(self) -> StatusResult:
        """What the graph holds: counts, entity labels and relation types."""
        counts = await ops.graph_counts(self._store)
        label_counts, rel_counts = await ops.schema_counts(self._store)
        hint = None
        if counts["entities"] == 0 and await ops.graph_exists(self._store) is False:
            hint = f"{self._missing_graph()}"
        elif counts["entities"] == 0:
            hint = "The graph is empty: ingest documents (and finalize) before querying."
        elif counts["relations"] == 0:
            hint = "The graph has entities but no relations."
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
        await self._require_graph()
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
            if props := properties_text(card.get("properties") or {}):
                text += f" ({props})"
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
                "Nothing relevant was found on any path. Rephrase the question with "
                "other words, or call status to check the graph holds data."
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

    async def ingest(
        self,
        text: str,
        *,
        document_id: str | None = None,
        ctx: Context | None = None,
    ) -> IngestResult:
        """Extract ``text`` into the graph.

        Under ``finalize_policy="on_write"`` this also finalizes, whose cost
        grows with the graph; prefer "manual" and one :meth:`finalize` per batch.
        """
        self._ensure_writable("ingest")
        inp = IngestInput(text=text, document_id=document_id)
        result = await self._rag.ingest(
            text=inp.text, document_id=inp.document_id, ctx=ctx or self._new_ctx()
        )
        finalized = False
        if self._finalize_policy == "on_write":
            await self._rag.finalize()
            finalized = True
        hint = None
        if self._finalize_policy == "manual":
            hint = "Call finalize once after this batch of ingests."
        return IngestResult(
            document_id=result.document_info.uid,
            chunks_indexed=result.chunks_indexed,
            nodes_created=result.nodes_created,
            relationships_created=result.relationships_created,
            finalized=finalized,
            hint=hint,
        )

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
