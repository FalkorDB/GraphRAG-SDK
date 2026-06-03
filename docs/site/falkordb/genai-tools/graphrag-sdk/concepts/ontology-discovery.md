---
title: "Ontology discovery"
nav_order: 3
parent: "Concepts"
grand_parent: "GraphRAG-SDK"
description: "Bootstrap an ontology from a corpus, or propose schema additions when new documents arrive — LLM-driven or grounded against a live catalog like DBpedia. New in v1.2."
---

# Ontology discovery
{: .label .label-green }
New in v1.2
{: .fs-3 }

If you've never built an ontology by hand and you don't know where to start, start here.

There are four places an ontology can come from:

1. **The built-in default.** Construct `GraphRAG` without an `ontology=` and the SDK seeds the ontology with `DEFAULT_ENTITY_TYPES` (`Person`, `Organization`, `Technology`, `Product`, `Location`, `Date`, `Event`, `Concept`, `Law`, `Dataset`, `Method`) so extraction has something to anchor on.
2. **Hand-authored.** You write `Ontology(entities=[...], relations=[...])` because you already know the domain. Fastest when you do.
3. **Discovered from a corpus.** Point the SDK at some documents and ask it to draft one. Use for unfamiliar / domain-specific corpora.
4. **Discovered and evolved.** The realistic workflow: draft once, ingest, then as new documents arrive use `suggest_schema_extensions` to propose additions you review and apply.

This page is about (3) and (4).

## Two entry points

| Goal | API |
|---|---|
| Bootstrap a new ontology from documents | `Ontology.from_sources(sources, llm, ...)` |
| Propose additions to an existing ontology | `GraphRAG.suggest_schema_extensions(sources, ...)` |

Both return inspectable artifacts — nothing is written to the graph until you apply a proposal explicitly via the evolution API.

## Bootstrap — two algorithms

Choose with `method=`:

### `method="llm"` (default) — LLM-driven invention

```python
draft = await Ontology.from_sources(
    sources,
    llm,
    boundaries="biotech papers about CRISPR",   # free-text scope hint
    sample_chunks_per_doc=3,
)
```

For each source: load → chunk → sample N chunks → per-document summary call (anchors the per-chunk prompts) → per-chunk proposal call (constrained Pydantic output with validation-retry) → merge in-document. Across the corpus, all per-doc drafts are merged, then a single normalization pass collapses synonyms (`Org` + `Organization` → one label) and fixes reversed relation directions.

### `method="grounded"` — catalog lookup, zero LLM calls

```python
from graphrag_sdk.discovery.catalog import DBpediaCatalog

draft = await Ontology.from_sources(
    sources,
    method="grounded",
    catalog=DBpediaCatalog(),
)
```

For each source: load → chunk → sample → run NER on each chunk with anchor labels (`["person", "organization", "location", "event"]`) to find entity mention strings. For each unique mention, ask the catalog "what types is this?" — `DBpediaCatalog` SPARQL-queries DBpedia. Aggregate the union of detected types; ask the catalog for each type's full schema (`Entity` with attributes + canonical URI) and for relations between detected types. Merge.

Zero LLM calls when `llm` is not supplied — the schema is entirely catalog-derived, the corpus only chooses the subset. Limited to what the catalog knows; domain-specific types won't appear.

**Optional per-type LLM trim.** Pass `llm` and the pipeline runs one extra LLM call per detected type: it shows the LLM the catalog's full property list (Schema.org's `Person` has 28 properties) plus up to 5 chunks where the type was detected, and asks "which properties are stated or implied?" The result trims each type to a corpus-specific subset. Cost: one call per *type*, not per chunk. Soft-fails to the catalog's full list on LLM error — you never silently lose schema information.

### Choosing between them

|  | `method="llm"` | `method="grounded"` |
|---|---|---|
| LLM cost | `D × (S + 1) + 1` calls per corpus | None (or one per type if `llm` supplied) |
| Catalog needed | No | Yes |
| Determinism | Stochastic | Deterministic for a given corpus + catalog |
| Domain-specific types | Yes — LLM can invent | Only if the catalog has them |
| Best for | Unfamiliar / domain-specific | General web / news / biographies |

Both algorithms accept `existing=current_ontology` — when supplied, `method="llm"` treats it as a soft controlled vocabulary in prompts, `method="grounded"` merges catalog output into it. Either way the result is the merge.

## Catalogs

A `Catalog` answers two questions:

1. Given an entity mention name, what types is it?
2. What's the schema (attributes, URI) for a given type?

The ABC `graphrag_sdk.discovery.Catalog` has three abstract methods — `link_entity`, `lookup`, `relations_among` — so adding a Wikidata catalog, a domain catalog, or a custom SPARQL store is a focused subclass.

The bundled catalog is `DBpediaCatalog`. Fully live, no bundled data:

- `link_entity(name)` → SPARQL DBpedia for `rdfs:label` matches filtered to `http://dbpedia.org/ontology/` types. Cached in-process.
- `lookup(type)` / `relations_among(types)` → live Schema.org JSON-LD with `rdfs:subClassOf` inheritance applied (so `Article` carries `CreativeWork`'s attributes). Cached to `$XDG_CACHE_HOME/graphrag-sdk/schema_org.json` for 30 days by default.

There's no offline fallback. If DBpedia or Schema.org is unreachable and the cache is missing or stale, `DBpediaFetchError` is raised. Pre-warm by running once with connectivity or supply a pre-populated `cache_path`.

## Live-graph delta — `suggest_schema_extensions`

Once an ontology is committed and you've ingested some documents, the question shifts from "what's the schema?" to "what's missing?" A new batch of documents may mention types or relations the committed ontology doesn't cover.

```python
proposal = await rag.suggest_schema_extensions(
    "docs/acquisition_news.md",
    boundaries="company news",
)
print(proposal.summary())
# SchemaExtensionProposal(entities=+2, relations=+1, patterns=+0, attributes=+3, sources_scanned=1)
```

The pipeline runs the same discovery against the new sources, but with the committed ontology as prior, then diffs and returns **only additions**. The graph is untouched.

To apply:

```python
for entity in proposal.new_entities:
    await rag.add_entity(entity)
for rel_label, src, tgt in proposal.new_patterns:
    await rag.add_relation_pattern(rel_label, src, tgt)
for owner, attribute in proposal.new_attributes:
    await rag.add_attribute(owner, attribute)   # atomic, LLM-backfilled
```

Discovery never commits anything itself — the evolution API stays the single commit surface.

## How the LLM is kept honest

Open-vocabulary discovery is brittle by default. Three disciplines make it tractable:

1. **Structured output with validation-retry feedback.** Every LLM call inside discovery parses into a Pydantic model and runs a semantic validator (relation patterns reference declared entities, attribute types are in the allowed set, no system-key shadowing). On failure, the *specific* errors are sent back to the LLM as a follow-up user message and the call is retried — the conversation history is preserved across retries so the model sees its own rejected output.
2. **Anchoring with a per-document summary.** Before any per-chunk proposal, a one-shot per-doc call extracts central entities and a one-sentence "aboutness". That summary prefixes every per-chunk prompt for the same doc, so chunks share the doc's frame.
3. **Normalization as a separate pass.** Per-chunk proposals are merged naively, then a single LLM call canonicalizes labels and direction. Without this, drafts accumulate synonyms across calls. The normalization pass is also where `existing`, if supplied, becomes a hard preference: "prefer these labels."

The wrapper is strict — on exhausted retries it raises `OntologyDiscoveryError`. The pipeline above it is soft-fail: a bad chunk is logged and skipped, a failed summary degrades to weaker anchoring, a failed normalization returns the un-normalized draft. Net effect: one noisy chunk or one rate-limited call doesn't kill the draft.

## Cost and scale

`Ontology.from_sources(method="llm")` with `D` documents and `S = sample_chunks_per_doc`:

| Step | LLM calls |
|---|---|
| Per-document summary | `D` |
| Per-chunk proposal | `D × S` |
| Normalization | `1` |
| **Total** | `D × (S + 1) + 1` |

A 10-document corpus with `sample_chunks_per_doc=3`: 41 LLM calls. Fast enough to run interactively on a `gpt-4o-mini`-class model.

`method="grounded"` (without LLM trim): zero LLM calls. The cost is SPARQL round-trips proportional to unique entity mentions, plus one Schema.org fetch (cached for 30 days).

## See also

- [API Reference → Ontology](../api-reference/ontology) — full `from_sources` signature, all parameters.
- [API Reference → Discovery](../api-reference/discovery) — `Catalog`, `DBpediaCatalog`, `SchemaExtensionProposal`, `OntologyDiscoveryError`.
- [Guides → Auto-discover a schema](../guides/auto-discover-schema) — runnable walkthrough.
- [Guides → Suggest extensions from new docs](../guides/suggest-extensions-from-new-docs) — runnable walkthrough for the evolution loop.
