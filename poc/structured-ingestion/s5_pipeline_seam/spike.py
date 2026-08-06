"""s5 — can `StructuredIngestionPipeline` reuse the existing pipeline's steps?

Proposal #6 says steps 3 (lexical graph), 6 (prune) and 9 (mentions) are
"♻ existing implementation, factored into a shared base, **not** copied",
because step 9's ordering is load-bearing for concurrent-update correctness.

That is an assertion about whether the real method signatures allow it. This
spike tries both factorings against the real `IngestionPipeline` and a real
FalkorDB, and reports which one actually works and what it costs in `src`.

  A  subclass `IngestionPipeline` and call its methods
  B  a mixin holding only the three reusable methods
"""

from __future__ import annotations

import asyncio
import csv
import hashlib
import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _harness.env import FIXTURES, Report, connection, falkor_available, reset_graph  # noqa: E402

from graphrag_sdk.core.context import Context  # noqa: E402
from graphrag_sdk.core.models import (  # noqa: E402
    DocumentInfo,
    Entity,
    EntityMention,
    GraphData,
    GraphNode,
    GraphRelationship,
    Ontology,
    Relation,
    TextChunk,
    TextChunks,
)
from graphrag_sdk.ingestion.extraction_strategies.entity_extractors import (  # noqa: E402
    compute_entity_id,
)
from graphrag_sdk.ingestion.pipeline import IngestionPipeline  # noqa: E402
from graphrag_sdk.storage.graph_store import GraphStore  # noqa: E402


def rows() -> list[dict[str, str]]:
    with open(FIXTURES / "employees.csv", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def records_to_chunks(doc_uid: str, records: list[dict[str, str]]) -> TextChunks:
    """Proposal #5: a record IS a chunk. Deterministic uid on the EFFECTIVE doc id (s4)."""
    return TextChunks(
        chunks=[
            TextChunk(
                text=f"{r['full_name']} · age {r['age']} · {r['job_title']} at {r['org_id']}",
                index=i,
                uid=hashlib.sha256(f"{doc_uid}::{r['employee_id']}".encode()).hexdigest()[:32],
                metadata={"kind": "record", "record_key": r["employee_id"]},
            )
            for i, r in enumerate(records)
        ]
    )


def records_to_graph_data(records: list[dict[str, str]], chunks: TextChunks) -> GraphData:
    """Proposal #6 step 4: pure mapping function, no LLM."""
    nodes, rels, mentions = [], [], []
    for r, chunk in zip(records, chunks.chunks):
        pid = compute_entity_id(r["employee_id"], "Person")
        oid = compute_entity_id(r["org_id"], "Organization")
        nodes += [
            GraphNode(
                id=pid,
                label="Person",
                properties={"name": r["full_name"], "title": r["job_title"], "age": int(r["age"])},
            ),
            GraphNode(id=oid, label="Organization", properties={"name": r["org_id"]}),
        ]
        rels.append(
            GraphRelationship(
                start_node_id=pid,
                end_node_id=oid,
                type="RELATES",
                properties={
                    "rel_type": "WORKS_AT",
                    "fact": f"({r['full_name']}, WORKS_AT, {r['org_id']})",
                    "source_chunk_ids": [chunk.uid],
                    "src_name": r["full_name"],
                    "tgt_name": r["org_id"],
                },
            )
        )
        mentions += [
            EntityMention(chunk_id=chunk.uid, entity_id=pid),
            EntityMention(chunk_id=chunk.uid, entity_id=oid),
        ]
    return GraphData(nodes=nodes, relationships=rels, mentions=mentions)


ONTOLOGY = Ontology(
    entities=[Entity(label="Person"), Entity(label="Organization")],
    relations=[Relation(label="WORKS_AT", patterns=[("Person", "Organization")])],
)


# ── Factoring A: subclass the real pipeline ──────────────────────


class SubclassStructuredPipeline(IngestionPipeline):
    async def run_structured(self, records, doc_info, ctx):  # type: ignore[no-untyped-def]
        chunks = records_to_chunks(doc_info.uid, records)
        await self._build_lexical_graph(doc_info, chunks, ctx)  # step 3 ♻
        data = records_to_graph_data(records, chunks)  # step 4
        data = self._prune(data, self.ontology)  # step 6 ♻
        await self.graph_store.upsert_nodes(data.nodes)  # step 8
        await self.graph_store.upsert_relationships(data.relationships)
        return await self._write_mentions(data, ctx)  # step 9 ♻


# ── Factoring B: a mixin carrying only the reusable steps ────────


class LexicalGraphMixin:
    """What the shared base would look like: depends on graph_store, nothing else."""

    graph_store: GraphStore

    _build_lexical_graph = IngestionPipeline._build_lexical_graph
    _prune = IngestionPipeline._prune
    _write_mentions = IngestionPipeline._write_mentions


class MixinStructuredPipeline(LexicalGraphMixin):
    def __init__(self, graph_store: GraphStore, ontology: Ontology) -> None:
        self.graph_store = graph_store
        self.ontology = ontology

    async def run_structured(self, records, doc_info, ctx):  # type: ignore[no-untyped-def]
        chunks = records_to_chunks(doc_info.uid, records)
        await self._build_lexical_graph(doc_info, chunks, ctx)
        data = records_to_graph_data(records, chunks)
        data = self._prune(data, self.ontology)
        await self.graph_store.upsert_nodes(data.nodes)
        await self.graph_store.upsert_relationships(data.relationships)
        return await self._write_mentions(data, ctx)


async def main() -> int:
    r = Report("s5 — pipeline seam")
    if not falkor_available():
        r.note("SKIPPED — no FalkorDB on FALKOR_HOST:FALKOR_PORT")
        return 0

    # What does __init__ demand of a structured pipeline that has no chunker
    # and no LLM extractor?
    params = inspect.signature(IngestionPipeline.__init__).parameters
    required = [
        n
        for n, p in params.items()
        if n != "self" and p.default is inspect.Parameter.empty and p.kind != p.VAR_KEYWORD
    ]
    r.note(f"IngestionPipeline.__init__ required args: {required}")
    r.check(
        {"chunker", "extractor"} <= set(required),
        "subclassing forces a structured pipeline to supply a chunker and an LLM extractor",
        "neither exists on the structured path — they would be dead None placeholders",
    )

    # A — subclass, with None for the strategies it has no use for.
    conn = connection("poc_s5_subclass")
    store = GraphStore(conn)
    await reset_graph(conn)
    ctx = Context()
    doc = DocumentInfo(path="employees.csv", uid="doc-employees-A")
    try:
        pipe_a = SubclassStructuredPipeline(
            loader=None,  # type: ignore[arg-type]
            chunker=None,  # type: ignore[arg-type]
            extractor=None,  # type: ignore[arg-type]
            resolver=None,  # type: ignore[arg-type]
            graph_store=store,
            vector_store=None,
            ontology=ONTOLOGY,
        )
        mentions_a = await pipe_a.run_structured(rows(), doc, ctx)
        r.check(
            mentions_a > 0,
            "A: subclassing works at runtime — the reused steps never touch the unused strategies",
            f"{mentions_a} MENTIONED_IN edges written",
        )
    except Exception as exc:  # noqa: BLE001
        r.check(False, "A: subclassing works at runtime", f"{type(exc).__name__}: {exc}")
    a_stats = await store.get_statistics()
    await conn.close()

    # B — mixin.
    conn = connection("poc_s5_mixin")
    store = GraphStore(conn)
    await reset_graph(conn)
    doc = DocumentInfo(path="employees.csv", uid="doc-employees-B")
    pipe_b = MixinStructuredPipeline(store, ONTOLOGY)
    mentions_b = await pipe_b.run_structured(rows(), doc, ctx)
    r.check(
        mentions_b == mentions_a,
        "B: the mixin factoring produces an identical graph with no dead dependencies",
        f"{mentions_b} MENTIONED_IN edges",
    )
    b_stats = await store.get_statistics()
    r.check(
        a_stats.get("node_count") == b_stats.get("node_count"),
        "A and B agree on the resulting graph",
        f"A={a_stats.get('node_count')} nodes · B={b_stats.get('node_count')} nodes",
    )

    # The three steps are reusable *verbatim* — no signature change needed.
    sig = inspect.signature(IngestionPipeline._build_lexical_graph)
    r.check(
        "chunks" in sig.parameters,
        "_build_lexical_graph() consumes TextChunks, so records need no new type to reuse it",
        f"signature: {sig}",
    )

    # ...but it unconditionally chains NEXT_CHUNK between adjacent chunks.
    nxt = await store.query_raw("MATCH ()-[r:NEXT_CHUNK]->() RETURN count(r) AS n")
    n_next = nxt.result_set[0][0]
    r.check(
        n_next == len(rows()) - 1,
        "reusing it also chains NEXT_CHUNK between unrelated CSV rows",
        f"{n_next} NEXT_CHUNK edges for {len(rows())} rows — N-1 edges asserting a "
        "sequential relationship that does not exist between table rows",
    )
    r.note(
        "cypher_generation.py tells the LLM 'NEXT_CHUNK: connects Chunk to next sequential "
        "Chunk' — false for rows, and 1M rows means 1M meaningless edges"
    )
    await conn.close()
    return r.verdict()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
