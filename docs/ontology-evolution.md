# Ontology Evolution

How to safely evolve the ontology (schema) of an already-populated knowledge graph — adding attributes, renaming labels, dropping types — without drifting away from the data graph.

If you're new to the ontology, start with [Graph Schema](graph-schema.md). This page is about *changing* an existing ontology. If you don't have one yet, see [Ontology Discovery](ontology-discovery.md) — `Ontology.from_sources` drafts one from a corpus and `GraphRAG.suggest_schema_extensions` proposes additions as new docs arrive.

---

## The Invariant

GraphRAG SDK treats the ontology as a **contract**: every instance of a declared attribute's owner entity type is queryable for that attribute. Values the LLM didn't determine read as `null` — Cypher treats absent and explicit-null identically under `WHERE n.attr IS NULL`, so callers get a consistent view regardless of which one underlies a given entity. There is no API that can declare schema your data doesn't match.

This means one thing in practice: when you add an attribute, the SDK runs an LLM backfill across your existing chunks **as part of the same call** and only commits the schema change once the data is aligned. Adding an attribute is therefore expensive — pay attention to corpus size and use `dry_run=True` to preview before running for real.

The invariant covers attributes only. Declaring a new entity type or relation pattern is cheap (no data implication — "this is allowed" is not "this exists"). For those, there are opt-in discovery tools that re-scan the corpus.

---

## The API at a Glance

15 methods on `GraphRAG`, grouped by what they touch.

### Group 1 — Pure declarations (cheap, no LLM)

Return: `Ontology`.

```python
rag.set_entity_description(label, description)
rag.set_relation_description(label, description)
rag.set_attribute_description(owner_label, attribute_name, description)
rag.add_entity(entity: Entity)                    # declaration only
rag.add_relation_pattern(rel_label, source, target)  # declaration only
```

### Group 2 — Mechanical data migration (Cypher, no LLM)

Return: `Ontology`. Data migration runs first; the ontology graph is updated second. A crash between the two leaves the data graph ahead, and re-running the same call is idempotent.

```python
rag.rename_entity(old, new)
rag.rename_attribute(owner_label, old_name, new_name)
rag.rename_relation(old, new)
rag.drop_entity(label)                            # cascades to relation patterns
rag.drop_relation(label)
rag.drop_relation_pattern(rel_label, source, target)
```

### Group 3 — Atomic attribute evolution (LLM, invariant-enforcing)

Return: `EvolutionResult`. The ontology graph write is the **commit point** — backfill runs first, schema change last.

```python
rag.add_attribute(owner_label, attribute, *, concurrency=4, dry_run=False)
# Atomic: LLM extracts values from every chunk mentioning the owner
# type, sets them on matching entities (null where the LLM doesn't
# know), then commits the new :Property to the ontology graph.
# Pass dry_run=True to preview cost without invoking the LLM.

rag.drop_attribute(owner_label, name)
# Atomic: REMOVE n.<name> on every instance, then delete the :Property
# declaration. Cheap, no LLM.
```

Entity owners only in v1. Relation-attribute mutation raises `NotImplementedError`.

### Group 4 — Opportunistic discovery (opt-in, not invariant-enforcing)

Return: `BackfillResult`. Use after `add_entity` / `add_relation_pattern` if you want to populate instances from the existing corpus.

```python
rag.backfill_entity(label, *, scope="all" | list[chunk_id], dry_run=False)
# "Re-scan chunks for any entities of this type I might have missed."

rag.backfill_relation_pattern(rel_label, source, target, *, dry_run=False)
# "Re-scan candidate co-mention chunks for any edges of this pattern."
```

These don't enforce anything. Declaring an entity type or relation pattern just says *the schema allows this* — zero instances is a valid state.

> **Provenance:** `backfill_relation_pattern` does NOT write `source_chunk_ids` on the new edges. `GraphStore.upsert_relationships` only unions provenance for `:RELATES` edges; writing it on arbitrary-typed edges would be silently overwritten by future MERGE writes. If you need chunk-level traceability on these backfilled edges, query the underlying co-mentioned chunks via the source/target entities' `MENTIONED_IN` edges instead.

---

## Why `add_attribute` is Atomic

Two reasons.

**The invariant.** If `add_attribute` were "declare cheap, fill later," the schema would say "Person has `role`" while many Person nodes lacked the property. Querying `MATCH (p:Person) WHERE p.role = "engineer"` would silently miss data. The atomic shape guarantees consistency.

**Honest commit ordering.** The data graph is mutated first. The ontology graph is updated **last**, as the commit point. If anything fails during backfill, the schema stays at its pre-call state — readers see a consistent (old) view of the world. There is no window in which the schema promises an attribute that isn't there.

```
add_attribute("Person", Attribute(name="role", type="STRING"))
        │
        ▼
  ┌─────────────────────────────┐
  │ 1. LLM-extract role from    │
  │    every chunk mentioning   │
  │    a Person.                │
  │                             │
  │ 2. SET p.role on matching   │
  │    entities (null where     │
  │    the LLM doesn't know).   │
  │                             │
  │ 3. Commit :Property node    │
  │    to ontology graph.       │   ← commit point
  └─────────────────────────────┘
        │
        ▼
   EvolutionResult
```

---

## Failure & Retry

If a chunk hard-fails (LLM error / parse error beyond retries), the call raises `OntologyEvolutionError` and the ontology graph is **not** updated. The data graph may be partially mutated — some entities have the new property, some don't. That's safe because the schema doesn't yet promise the property exists.

To recover: fix the underlying cause (rate limits, malformed chunks, etc.), then **call `add_attribute` again with the same arguments**. The call is idempotent:

- Already-processed chunks carry an `extracted_ops` marker; the LLM is not re-invoked for them.
- Already-set entity values are not overwritten.

```python
from graphrag_sdk import OntologyEvolutionError

try:
    result = await rag.add_attribute("Person", Attribute(name="role"))
except OntologyEvolutionError as e:
    print(f"Failed on {len(e.failed_chunks)} chunks: {e.failed_chunks[:5]}")
    # ...investigate and fix...
    result = await rag.add_attribute("Person", Attribute(name="role"))  # idempotent retry
```

---

## Type Changes

There is no `retype_attribute`. To change a type, drop the attribute and add it with the new type — the LLM re-derives values from the chunks.

```python
# Person.age is STRING. We want INTEGER.
await rag.drop_attribute("Person", "age")
result = await rag.add_attribute("Person", Attribute(name="age", type="INTEGER"))
```

This is the only honest move when the source of truth is the corpus. A mechanical `toInteger` coercion of `"around thirty"` would either drop the value or fabricate one; the LLM can re-extract `30` from the surrounding text.

---

## Concurrency

**Evolution calls are not safe to run concurrently with `ingest()` or with each other on the same graph.** Coordinate at the application level — treat evolution as a maintenance operation: pause new ingestion, run the evolution, resume.

The constraint has two sources:

- **`add_attribute` / `drop_attribute`** — the extractor reads the persisted ontology to decide what to extract from new chunks. While `add_attribute` is mid-flight, the ontology hasn't been committed yet, and any concurrent `ingest()` would produce entities without the new attribute. The invariant would silently break the moment the schema commits.
- **Mechanical Group 2 calls and `backfill_*`** — internally use a count-then-mutate pattern across two Cypher statements. Concurrent writes can change the count between the two queries.

For multi-process deployments where this matters, gate evolution behind an application-level lock (Redis, etc.) that also blocks `ingest()`.

---

## Cost Preview (`dry_run=True`)

`add_attribute`, `backfill_entity`, and `backfill_relation_pattern` accept a `dry_run` keyword. When `True`, the scope query runs but the LLM is not invoked and no writes happen. The returned result carries `chunks_in_scope` — the number of chunks this run *would* scan.

```python
preview = await rag.add_attribute(
    "Person", Attribute(name="role", type="STRING"),
    dry_run=True,
)
print(f"Would scan {preview.chunks_in_scope} chunks "
      f"(~{preview.chunks_in_scope} LLM calls)")
# Decide whether to proceed.
if preview.chunks_in_scope < 1000:
    result = await rag.add_attribute("Person", Attribute(name="role", type="STRING"))
```

The count reflects what *this run* would scan: chunks already marked from a prior partial run are already filtered out, so the preview is accurate for the work that remains, not a hypothetical clean slate.

---

## End-to-End Example

```python
import asyncio
from graphrag_sdk import (
    Attribute, ConnectionConfig, Entity, EvolutionResult, GraphRAG,
    LiteLLM, LiteLLMEmbedder, Ontology, Relation,
)


async def main():
    starter = Ontology(
        entities=[
            Entity(label="Person", description="A human"),
            Entity(label="Company", description="A business"),
        ],
        relations=[
            Relation(
                label="WORKS_AT",
                patterns=[("Person", "Company")],
            ),
        ],
    )

    async with GraphRAG(
        connection=ConnectionConfig(host="localhost", graph_name="evolve_demo"),
        llm=LiteLLM(model="openai/gpt-4o-mini"),
        embedder=LiteLLMEmbedder(model="openai/text-embedding-3-small"),
        ontology=starter,
    ) as rag:
        # 1. Ingest with the starter ontology.
        await rag.ingest(
            text="Alice Liddell is a software engineer at Acme Corp.",
            document_id="doc1",
        )

        # 2. Atomic attribute add: declare + LLM backfill + commit.
        result: EvolutionResult = await rag.add_attribute(
            "Person", Attribute(name="role", type="STRING"),
        )
        print(f"role filled on {result.values_filled} entities, "
              f"{result.chunks_scanned} chunks scanned, "
              f"{result.llm_calls} LLM calls")

        # 3. Type change via drop + add.
        await rag.drop_attribute("Person", "role")
        result = await rag.add_attribute(
            "Person", Attribute(name="role", type="STRING",
                                description="Their job title"),
        )

        # 4. Declare a new entity type — cheap, no LLM.
        await rag.add_entity(Entity(label="City"))

        # 5. Opportunistic: scan corpus for any Cities we missed.
        discovery = await rag.backfill_entity("City", scope="all")
        print(f"discovered {discovery.values_filled} City instances")

        # 6. Inspect the final ontology.
        ontology = await rag.get_ontology()
        for e in ontology.entities:
            attrs = ", ".join(f"{p.name}:{p.type}" for p in e.properties) or "—"
            print(f"  {e.label:<10} {attrs}")


asyncio.run(main())
```

A more comprehensive walkthrough lives in `examples/09_ontology_evolution.py`.

---

## EvolutionResult Reference

Returned by `add_attribute`. On a successful (non-dry-run) return, all counters are populated; hard failures raise `OntologyEvolutionError` instead of returning.

| Field | Type | Description |
|-------|------|-------------|
| `ontology` | `Ontology` | Refreshed ontology after the schema commit |
| `chunks_in_scope` | `int` | Total chunks matching the scope query — equal to `chunks_scanned + chunks_skipped` on success; the only meaningful field under `dry_run=True` |
| `chunks_scanned` | `int` | Chunks processed by the LLM this run |
| `chunks_skipped` | `int` | Chunks already marked from a prior run (idempotent re-runs) |
| `llm_calls` | `int` | Total LLM invocations |
| `values_filled` | `int` | Entities for which the LLM returned a value |
| `values_skipped` | `int` | Entities for which the LLM returned `null` |
| `elapsed_s` | `float` | Wall-clock time |

## OntologyEvolutionError Reference

Raised by `add_attribute` when one or more chunks hard-fail. The ontology graph is **not** updated.

| Field | Type | Description |
|-------|------|-------------|
| `failed_chunks` | `list[str]` | Chunk ids that raised after retries — investigate and retry |
| `chunks_scanned` | `int` | Chunks that did succeed before the failure |

The exception subclasses `RuntimeError` so it propagates through `await` paths cleanly.

---

## What's Not in the API (and Why)

| Removed | Why |
|---------|-----|
| `retype_attribute` | Type changes go through `drop_attribute` + `add_attribute`. The LLM re-derives values from chunks — the only honest move when text is the source of truth. |
| `backfill_attribute` (public) | Folded into `add_attribute`. Exposing it as a separate call would re-introduce the drift it's designed to prevent. |
| `backfill_attribute_semantic` | Supported `retype_attribute`. Gone with it. |

Relation-attribute mutation (`add_attribute("WORKS_AT", ...)` etc.) raises `NotImplementedError` in v1. The workaround is `delete_all()` followed by a fresh `ingest()` with the updated ontology — that's a heavy hammer, but it's the only way to keep edge properties aligned without an edge-property migration primitive. A follow-up PR can lift this.
