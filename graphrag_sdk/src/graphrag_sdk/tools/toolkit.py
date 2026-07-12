# GraphRAG SDK — Tools: GraphRAGToolkit facade
# Framework-neutral agent surface over GraphRAG. Retrieval rides the real
# MultiPathRetrieval pipeline (with provenance metadata); QA rides completion().

from __future__ import annotations

import dataclasses
import re
from collections.abc import Sequence
from types import TracebackType
from typing import TYPE_CHECKING, Any, Literal

from graphrag_sdk.core.connection import ConnectionConfig
from graphrag_sdk.core.context import Context
from graphrag_sdk.core.exceptions import ConfigError, ReadOnlyViolation
from graphrag_sdk.core.models import Ontology
from graphrag_sdk.core.providers import Embedder, LLMInterface
from graphrag_sdk.retrieval.strategies.multi_path import MultiPathRetrieval
from graphrag_sdk.tools import graph_ops
from graphrag_sdk.tools.cypher_guard import apply_limit, ensure_read_only
from graphrag_sdk.tools.models import (
    AnswerResult,
    ChunkRef,
    Citation,
    CypherResult,
    DocumentRef,
    EntityCard,
    EntityResult,
    EntityTypeInfo,
    RelationTypeInfo,
    RememberResult,
    SchemaResult,
    SearchResult,
)
from graphrag_sdk.tools.specs import (
    _TOOL_REGISTRY,
    AnswerInput,
    CypherReadInput,
    EntityInput,
    RememberInput,
    SearchInput,
    ToolSpec,
    build_tool_specs,
)

if TYPE_CHECKING:
    from graphrag_sdk.api.main import GraphRAG

FinalizePolicy = Literal["manual", "on_write", "never"]
_POLICIES = ("manual", "on_write", "never")
_TENANT_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_SNIPPET_CHARS = 200


class GraphRAGToolkit:
    """Framework-neutral agent toolkit over an async :class:`GraphRAG` instance.

    Exposes a small, stable set of async operations with LLM-friendly typed
    results, plus :meth:`tool_specs` — the machine-readable contract adapters
    (pydantic-ai, LangGraph, MCP) generate their tool definitions from.

    Args:
        rag: The GraphRAG instance to wrap (binds to its graph/tenant).
        finalize_policy: ``"manual"`` (default — call :meth:`flush` yourself),
            ``"on_write"`` (finalize after every remember; demos only —
            finalize is O(graph size)), or ``"never"`` (you call
            ``rag.finalize()`` yourself).
        read_only: Disable remember/flush and drop them from tool_specs().
        include: Optional subset of tool names to advertise via
            tool_specs()/call(). Direct method calls are not affected.
        tenant_id: Stamped into the Context of every operation.
        owns_rag: When True, :meth:`aclose` closes the wrapped GraphRAG.
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
    ) -> None:
        if finalize_policy not in _POLICIES:
            raise ValueError(f"finalize_policy must be one of {_POLICIES}, got {finalize_policy!r}")
        valid = {td.name for td in _TOOL_REGISTRY}
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

    # ── Lifecycle ────────────────────────────────────────────────

    @property
    def rag(self) -> GraphRAG:
        """The wrapped GraphRAG instance."""
        return self._rag

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
        """Machine-readable tool definitions for this toolkit configuration."""
        return build_tool_specs(
            read_only=self._read_only,
            finalize_policy=self._finalize_policy,
            include=self._include,
        )

    async def call(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        """Validate *arguments* against the tool's input model and invoke it.

        The generic entry point for adapters/MCP servers. Enforces the same
        read_only/include gates as :meth:`tool_specs`.
        """
        by_name = {td.name: td for td in _TOOL_REGISTRY}
        td = by_name.get(name)
        if td is None:
            raise ValueError(f"Unknown tool {name!r}; valid: {sorted(by_name)}")
        if self._include is not None and name not in self._include:
            enabled = [s.name for s in self.tool_specs()]
            raise ValueError(f"Tool {name!r} is not enabled; enabled: {enabled}")
        if self._read_only and td.is_write:
            raise ReadOnlyViolation(f"{name} is disabled: toolkit is read-only.")
        if td.manual_only and self._finalize_policy != "manual":
            raise ValueError(
                f"Tool {name!r} is unavailable under finalize_policy={self._finalize_policy!r}."
            )
        model = td.input_model(**(arguments or {}))
        method = getattr(self, td.method)
        return await method(**model.model_dump())

    def _new_ctx(self) -> Context:
        return Context(tenant_id=self._tenant_id)

    def _make_strategy(self, top_k: int) -> MultiPathRetrieval:
        """A per-call MultiPathRetrieval tuned to top_k (ctor is pure assignment)."""
        return MultiPathRetrieval(
            graph_store=self._rag._graph_store,
            vector_store=self._rag._vector_store,
            embedder=self._rag.embedder,
            llm=self._rag.llm,
            chunk_top_k=top_k,
            rel_top_k=top_k,
            max_entities=max(2 * top_k, 10),
            ontology=self._rag._global_ontology,
        )

    # ── Read path ────────────────────────────────────────────────

    async def search(
        self,
        query: str,
        *,
        top_k: int = 8,
        expand_hops: int = 1,
        include_chunks: bool = True,
        ctx: Context | None = None,
    ) -> SearchResult:
        """Typed retrieval only — no generation. The default agent mode:
        returns ranked entities/relations/facts/chunks the host LLM composes
        from, with document/chunk ids for citations."""
        inp = SearchInput(
            query=query, top_k=top_k, expand_hops=expand_hops, include_chunks=include_chunks
        )
        ctx = ctx or self._new_ctx()
        store = self._rag._graph_store
        rr = await self._rag.retrieve(inp.query, strategy=self._make_strategy(inp.top_k), ctx=ctx)
        prov: dict[str, Any] = rr.metadata.get("provenance") or {}
        seeds = [e for e in (prov.get("entities") or []) if e.get("id")][: inp.top_k]
        seed_ids = [e["id"] for e in seeds]
        cards_by_id = await graph_ops.enrich_entities(store, seed_ids)
        entities = [
            cards_by_id.get(
                e["id"],
                EntityCard(name=e.get("name", ""), description=e.get("description") or None),
            )
            for e in seeds
        ]
        relations = await graph_ops.expand_triples(
            store, seed_ids, hops=inp.expand_hops, cap=min(4 * inp.top_k, 40)
        )
        chunks: list[ChunkRef] = []
        documents: list[DocumentRef] = []
        if inp.include_chunks:
            prov_chunks = (prov.get("chunks") or [])[: inp.top_k]
            doc_map = await graph_ops.chunk_documents(store, [c["id"] for c in prov_chunks])
            seen_docs: set[str] = set()
            for c in prov_chunks:
                doc_id, doc_path = doc_map.get(c["id"], ("", c.get("document_path", "")))
                chunks.append(
                    ChunkRef(
                        chunk_id=c["id"],
                        document_id=doc_id,
                        document_path=doc_path,
                        text=c.get("text", ""),
                    )
                )
                if doc_id and doc_id not in seen_docs:
                    seen_docs.add(doc_id)
                    documents.append(DocumentRef(document_id=doc_id, document_path=doc_path))
        return SearchResult(
            query=inp.query,
            entities=entities,
            relations=relations,
            facts=(prov.get("facts") or [])[: 2 * inp.top_k],
            chunks=chunks,
            documents=documents,
        )

    async def answer(
        self, question: str, *, top_k: int = 8, ctx: Context | None = None
    ) -> AnswerResult:
        """Full RAG: the real completion() pipeline plus provenance citations.

        ``cypher_used`` is currently always None (the experimental
        text-to-Cypher path does not surface its query); the field exists
        for forward compatibility.
        """
        inp = AnswerInput(question=question, top_k=top_k)
        ctx = ctx or self._new_ctx()
        rag_result = await self._rag.completion(
            inp.question,
            strategy=self._make_strategy(inp.top_k),
            return_context=True,
            ctx=ctx,
        )
        prov: dict[str, Any] = {}
        if rag_result.retriever_result is not None:
            prov = rag_result.retriever_result.metadata.get("provenance") or {}
        prov_chunks = (prov.get("chunks") or [])[: inp.top_k]
        doc_map = await graph_ops.chunk_documents(
            self._rag._graph_store, [c["id"] for c in prov_chunks]
        )
        citations = []
        for c in prov_chunks:
            doc_id, doc_path = doc_map.get(c["id"], ("", c.get("document_path", "")))
            text = c.get("text", "")
            snippet = text if len(text) <= _SNIPPET_CHARS else text[: _SNIPPET_CHARS - 1] + "…"
            citations.append(
                Citation(
                    document_id=doc_id,
                    document_path=doc_path,
                    chunk_id=c["id"],
                    snippet=snippet,
                )
            )
        seen: set[str] = set()
        touched: list[str] = []
        for e in prov.get("entities") or []:
            n = e.get("name", "")
            if n and n.lower() not in seen:
                seen.add(n.lower())
                touched.append(n)
        return AnswerResult(
            answer=rag_result.answer,
            citations=citations,
            entities_touched=touched,
            cypher_used=None,
        )

    async def schema(self, *, ctx: Context | None = None) -> SchemaResult:
        """Entity labels + relation types with declared metadata and live counts."""
        ontology = await self._rag.get_ontology()
        label_counts, rel_counts = await graph_ops.schema_counts(self._rag._graph_store)
        stats = await self._rag.get_statistics()
        declared_e = {e.label: e for e in ontology.entities}
        declared_r = {r.label: r for r in ontology.relations}
        entity_infos = [
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
        relation_infos = [
            RelationTypeInfo(
                label=label,
                description=declared_r[label].description if label in declared_r else None,
                patterns=list(declared_r[label].patterns) if label in declared_r else [],
                count=count,
            )
            for label, count in sorted({**{r: 0 for r in declared_r}, **rel_counts}.items())
        ]
        return SchemaResult(
            entities=entity_infos,
            relations=relation_infos,
            node_count=int(stats.get("node_count", 0)),
            edge_count=int(stats.get("edge_count", 0)),
        )

    async def cypher_read(
        self,
        query: str,
        params: dict[str, Any] | None = None,
        *,
        limit: int = 100,
        timeout_ms: int = 5000,
        ctx: Context | None = None,
    ) -> CypherResult:
        """Guarded read-only Cypher. Rejects writes (:class:`ReadOnlyViolation`),
        injects LIMIT when absent, enforces a server-side timeout.

        The connection retries transient failures up to ``retry_count`` times,
        so worst-case wall time is about ``retry_count * timeout_ms``.
        """
        inp = CypherReadInput(query=query, params=params, limit=limit, timeout_ms=timeout_ms)
        ensure_read_only(inp.query)
        final_query, injected = apply_limit(inp.query, inp.limit)
        result = await self._rag._conn.query(final_query, params=inp.params, timeout=inp.timeout_ms)
        return graph_ops.convert_query_result(result, limit=inp.limit, limit_injected=injected)

    async def entity(self, name: str, *, hops: int = 1, ctx: Context | None = None) -> EntityResult:
        """Entity card: best name match, neighbors up to *hops*, source documents."""
        inp = EntityInput(name=name, hops=hops)
        store = self._rag._graph_store
        matches = await graph_ops.find_entity_matches(store, inp.name)
        if not matches:
            return EntityResult(query=inp.name, found=False)
        (eid, card), rest = matches[0], matches[1:]
        neighbors = await graph_ops.expand_triples(store, [eid], hops=inp.hops)
        documents = await graph_ops.entity_documents(store, eid)
        return EntityResult(
            query=inp.name,
            found=True,
            entity=card,
            neighbors=neighbors,
            nearby=[c.name for _, c in rest],
            documents=documents,
        )

    # ── Write path ───────────────────────────────────────────────

    def _ensure_writable(self, tool: str) -> None:
        if self._read_only:
            raise ReadOnlyViolation(f"{tool} is disabled: toolkit is read-only.")

    async def remember(
        self, text: str, *, document_id: str | None = None, ctx: Context | None = None
    ) -> RememberResult:
        """Ingest raw text into the graph (agent memory / fact capture).

        Under ``finalize_policy="on_write"`` this also runs finalize —
        which is O(graph size); use "manual" + :meth:`flush` in production.
        """
        self._ensure_writable("graph_remember")
        inp = RememberInput(text=text, document_id=document_id)
        ctx = ctx or self._new_ctx()
        result = await self._rag.ingest(text=inp.text, document_id=inp.document_id, ctx=ctx)
        finalized = False
        if self._finalize_policy == "on_write":
            await self._rag.finalize()
            finalized = True
        return RememberResult(
            document_id=result.document_info.uid,
            chunks_indexed=result.chunks_indexed,
            nodes_created=result.nodes_created,
            relationships_created=result.relationships_created,
            finalized=finalized,
        )

    async def flush(self, *, ctx: Context | None = None) -> None:
        """Run ``GraphRAG.finalize()`` under the "manual" policy.

        No-op under "on_write" (each remember already finalized); raises
        :class:`ConfigError` under "never" (call ``rag.finalize()`` yourself).
        """
        self._ensure_writable("graph_flush")
        if self._finalize_policy == "manual":
            await self._rag.finalize()
            return
        if self._finalize_policy == "on_write":
            return
        raise ConfigError(
            'finalize_policy="never": the toolkit never finalizes — call '
            "GraphRAG.finalize() yourself."
        )
