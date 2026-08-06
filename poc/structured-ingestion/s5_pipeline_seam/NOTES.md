# s5 — pipeline seam · DECIDED

**Question.** Proposal #6 marks steps 3 / 6 / 9 as "♻ existing implementation, factored into a
shared base, **not** copied". Do the real signatures allow that, and which factoring is right?

**Run:** `python s5_pipeline_seam/spike.py` (needs FalkorDB; no keys). All checks pass.

## The good news: the steps are reusable verbatim

```
_build_lexical_graph(self, doc_info: DocumentInfo, chunks: TextChunks, ctx: Context, *,
                     content_hash: str | None = None) -> None
```

It consumes `TextChunks` and only ever reads `chunk.uid` / `.text` / `.index` / `.metadata`. Since
proposal #5 already says *a record is a chunk*, records map onto `TextChunks` directly —
**no new type and no signature change** is needed to reuse step 3. Same for `_prune` (pure
function over `GraphData` + `Ontology`) and `_write_mentions` (reads `graph_data.mentions`, writes
via `graph_store`). All three depend on `self.graph_store` and nothing else.

Both factorings were run end-to-end against FalkorDB and produced **identical graphs**
(9 nodes, 6 `MENTIONED_IN` edges).

## The bad news: subclassing is the wrong seam

```
IngestionPipeline.__init__ required args:
    ['loader', 'chunker', 'extractor', 'resolver', 'graph_store', 'vector_store']
```

A structured pipeline has a *record* loader, no chunker (records are already chunks), and no LLM
extractor (mapping is deterministic — that is the entire point of the design). Subclassing forces
it to pass `None` for `chunker` and `extractor` and hope nothing ever touches them. It works today
purely by accident of which methods we call, and it converts every future change to
`IngestionPipeline.run()` into a latent `AttributeError` on the structured path.

**Decision: extract the three methods into a `LexicalGraphWriter` base that depends only on
`graph_store`**, and have both pipelines inherit it. Verified in the spike as
`LexicalGraphMixin` — same graph, no dead dependencies, and it preserves the property proposal #6
actually cares about: step 9's ordering lives in exactly one place, so the concurrency invariant
guarded by the warning box at `pipeline.py:235–260` cannot be broken on one path only.

Cost in `src`: move three method bodies to a new base class; `IngestionPipeline` keeps its public
surface unchanged.

## One thing reuse gets wrong: `NEXT_CHUNK`

`_build_lexical_graph` unconditionally chains `prev_chunk -[NEXT_CHUNK]-> chunk`. Reused for
records, that asserts a sequential relationship **between unrelated table rows** — the spike
measured 2 edges for 3 rows, so N-1 for any source. For a 1M-row CSV that is 1M meaningless edges,
and `cypher_generation.py:129` actively tells the LLM "NEXT_CHUNK: connects Chunk to next
sequential Chunk", which is false for a table.

Sort order in a CSV is usually incidental, so these edges are not merely useless — they encode a
claim that isn't true.

**Decision:** add `link_sequential: bool = True` to `_build_lexical_graph` and pass `False` for
record chunks. One keyword-only argument, default preserves today's behaviour exactly. This is the
*only* signature change the whole seam needs.
