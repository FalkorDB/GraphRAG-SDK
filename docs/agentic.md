# Agentic GraphRAG

`graphrag_sdk.tools` gives any agent framework — pydantic-ai, LangGraph, CrewAI,
MCP clients — a small, stable set of async operations over a knowledge graph.
Every operation returns a typed pydantic model that renders itself into
compact, deterministic plain text for LLM consumption, and the whole surface
is described by machine-readable [`tool_specs()`](#tool_specs-for-adapter-authors)
so adapters never hand-copy names, descriptions, or schemas.

## Quickstart

```python
import asyncio

from graphrag_sdk import ConnectionConfig, GraphRAG, LiteLLM, LiteLLMEmbedder
from graphrag_sdk.tools import GraphRAGToolkit


async def main():
    rag = GraphRAG(
        connection=ConnectionConfig(host="localhost", graph_name="agent_demo"),
        llm=LiteLLM(model="openai/gpt-5.5"),
        embedder=LiteLLMEmbedder(model="openai/text-embedding-3-large", dimensions=256),
        embedding_dimension=256,
    )
    toolkit = GraphRAGToolkit(rag)  # finalize_policy="manual" by default

    await toolkit.remember("Alice is a software engineer at Acme Corp.")
    await toolkit.flush()  # dedup + embeddings + indexes (expensive — see policies)

    result = await toolkit.search("Who works at Acme?")
    print(result.to_llm_text())        # prompt-ready text with ids for citations
    print(result.model_dump_json())    # or full structured output

    answer = await toolkit.answer("Who works at Acme?")
    print(answer.answer, answer.citations)

    await rag.close()


asyncio.run(main())
```

## Tools

| Tool name | Method | When to use |
|---|---|---|
| `graph_search` | `search(query, *, top_k=8, expand_hops=1, include_chunks=True)` | Retrieval only — ranked entities, relations, facts, and source chunks the host LLM composes from. The default agent mode; cite with the returned `document_id`/`chunk_id`. |
| `graph_answer` | `answer(question, *, top_k=8)` | Full RAG: the SDK's `completion()` pipeline plus provenance citations. One-shot Q&A. |
| `graph_schema` | `schema()` | Entity labels, relation types, directional patterns, and live counts. Call once up front to plan queries. |
| `graph_entity` | `entity(name, *, hops=1)` | Entity card: best name match, properties, neighbors up to `hops`, source documents, and "nearby" alternative matches. |
| `cypher_read` | `cypher_read(query, params=None, *, limit=100, timeout_ms=5000)` | Guarded read-only Cypher for aggregations and precise filters search cannot express. |
| `graph_remember` | `remember(text, *, document_id=None)` | Store new text into the graph (agent memory, fact capture). |
| `graph_flush` | `flush()` | Run finalization after a write session (only advertised under the `"manual"` policy). |

All methods are async and validate their arguments through the same pydantic
input models that generate the JSON Schemas in `tool_specs()`. Every result
model supports `model_dump()` / `model_dump_json()` for structured consumers
and `to_llm_text(max_chars=4000)` for prompt building — deterministic,
truncated only at item boundaries with an explicit `…(N more)` marker, and
control-character-stripped.

## `tool_specs()` for adapter authors

`tool_specs()` is the single source of truth. Adapters and MCP servers must
generate their tool definitions from it — names, LLM-facing descriptions, and
input JSON Schemas are never duplicated downstream.

```python
for spec in toolkit.tool_specs():
    print(spec.name)          # "graph_search"
    print(spec.description)   # when-to-use guidance written for the model
    print(spec.input_schema)  # JSON Schema (draft 2020-12, additionalProperties: false)
    print(spec.output_hint)   # one-line shape hint for the result
```

Dispatch generically with `call()` — it validates arguments against the
tool's input model and enforces the same gates as `tool_specs()`:

```python
result = await toolkit.call("graph_search", {"query": "Who works at Acme?", "top_k": 5})
```

Two knobs shape the advertised surface:

- `read_only=True` — `graph_remember`/`graph_flush` are removed from
  `tool_specs()` and raise `ReadOnlyViolation` if invoked anyway.
- `include=["graph_search", "graph_schema"]` — advertise a subset. `include`
  filters `tool_specs()` and `call()`; direct method calls still work.

## Finalize policies

`GraphRAG.finalize()` runs cross-document entity deduplication, entity and
relationship embeddings, and index creation.

!!! warning "finalize() is O(graph size)"
    Finalization cost grows with the whole graph, not with the size of the
    last write. `finalize_policy="on_write"` (finalize after **every**
    `remember`) is for demos and tiny graphs only.

| Policy | Behavior |
|---|---|
| `"manual"` (default) | Writes accumulate; call `flush()` once at the end of a write session. `graph_flush` is advertised in `tool_specs()`. |
| `"on_write"` | Every `remember()` finalizes (`RememberResult.finalized=True`). `flush()` is a no-op and is not advertised. |
| `"never"` | The toolkit never finalizes; `flush()` raises `ConfigError`. You own the `rag.finalize()` lifecycle. |

## Security: treat agent Cypher as untrusted input

Natural-language-to-Cypher — whether generated by your model or typed by an
agent — is **model-generated, untrusted input**. `cypher_read` is guarded:

- Write clauses (`CREATE`, `MERGE`, `DELETE`, `DETACH`, `SET`, `REMOVE`,
  `DROP`, `FOREACH`, `LOAD CSV`) are rejected with `ReadOnlyViolation`
  naming the offending token. Detection is case-insensitive, tolerant of
  comments/whitespace, immune to string-literal smuggling (a single-pass
  lexer masks literals and strips comments), and runs on both the raw and
  the NFKC-normalized query text.
- `CALL` is allowed only for read-safe procedures:
  `db.labels`, `db.relationshipTypes`, `db.propertyKeys`, `db.indexes`,
  `db.idx.fulltext.queryNodes`, `db.idx.fulltext.queryRelationships`,
  `db.idx.vector.queryNodes`, `db.idx.vector.queryRelationships`
  (full-name match — `db.idx.fulltext.createNodeIndex` is a write and is
  rejected). `CALL { … }` subqueries are permitted; their contents are
  scanned like everything else.
- A `LIMIT {limit}` is appended when the query has none, and `timeout_ms`
  is enforced server-side per query. The connection layer retries transient
  failures (`ConnectionConfig.retry_count`, default 3), so worst-case wall
  time is about `retry_count × timeout_ms`.
- Node/edge values returned by `cypher_read` are converted to JSON-safe
  data with bulky internals (`embedding`, `source_chunk_ids`) stripped.

!!! tip "Defense in depth"
    The guard is one layer. In production, point agent toolkits at read-only
    replicas or construct them with `read_only=True`; give write access only
    to toolkits that need it.

All strings rendered by `to_llm_text()` pass the SDK's control-character
sanitizer, so tool output cannot smuggle terminal escapes into prompts.

## Multi-tenancy

Bind a toolkit to a tenant-scoped graph with `for_tenant`. It derives
`graph_name = f"{base.graph_name}__{tenant_id}"` (tenant ids are validated
against `^[A-Za-z0-9_-]{1,64}$`), builds a dedicated `GraphRAG` the toolkit
owns, and closes it on `aclose()` / `async with`:

```python
async with GraphRAGToolkit.for_tenant(
    base_config, "acme", llm=llm, embedder=embedder
) as toolkit:
    await toolkit.search("...")
```

## Limitations

- `AnswerResult.cypher_used` is currently always `None`; the experimental
  text-to-Cypher retrieval path does not yet surface its generated query.
  The field exists for forward compatibility.
- `search()`/`answer()` drive a toolkit-tuned `MultiPathRetrieval` so that
  `top_k` and citations behave predictably. A custom `retrieval_strategy`
  configured on the `GraphRAG` instance is **not** used by the toolkit —
  call `rag.completion()` / `rag.retrieve()` directly when you need it.
- Citations come from the retrieval pipeline's provenance
  (`RetrieverResult.metadata["provenance"]`), which lists the context the
  model actually saw — in relevance order, capped at `top_k`.

See [`examples/11_agent_toolkit.py`](https://github.com/FalkorDB/GraphRAG-SDK/blob/main/graphrag_sdk/examples/11_agent_toolkit.py)
for a runnable end-to-end script.
