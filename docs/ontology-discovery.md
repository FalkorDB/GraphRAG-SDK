# Ontology Discovery

How to bootstrap an ontology from a corpus and how to propose schema additions as new documents arrive — the *discovery* side of the ontology lifecycle. The complement to [Ontology Evolution](ontology-evolution.md), which covers safely *changing* an ontology that already exists.

If you've never built an ontology by hand and you don't know where to start, start here.

---

## The Mental Model

There are three places an ontology can come from:

1. **Hand-authored.** You write `Ontology(entities=[...], relations=[...])` because you already know what your knowledge graph should look like. The fastest path when the domain is well understood.
2. **Discovered from a corpus.** You point the SDK at some documents and ask it to draft one. Use when you're exploring an unfamiliar corpus, or when manually enumerating types upfront is brittle.
3. **Discovered and then evolved.** The realistic workflow: draft once with discovery, ingest with that draft, then as new documents arrive use `suggest_schema_extensions` to propose additions you review and apply via the [evolution API](ontology-evolution.md).

This page is about (2) and (3). The same data model (`Ontology`) is used end-to-end — you can mix and match.

---

## The API at a Glance

Two cooperating entry points. Both return artifacts you inspect before any data is touched.

### Bootstrap — `Ontology.from_sources`

A classmethod on the `Ontology` data model. Pure function, no DB connection needed. Two discovery algorithms, selected via the `method=` argument.

#### `method="llm"` (default) — LLM-driven schema invention

```python
draft = await Ontology.from_sources(
    sources,                          # str or list[str], same as ingest()
    llm,                              # any LLMInterface
    method="llm",                     # explicit, but this is the default
    boundaries="biotech papers about CRISPR",   # free-text scope hint
    existing=None,                    # optional structured prior
    sample_chunks_per_doc=3,
    max_retries=3,
    concurrency=4,
)
```

Behavior:

- For each source: load → chunk → sample `sample_chunks_per_doc` chunks → run a per-document summary call (anchors the per-chunk prompts) → run a per-chunk proposal call (constrained Pydantic output, validated with retry) → merge in-document.
- Across the corpus: merge all per-document drafts.
- Single normalization call: collapse synonyms (`Org` + `Organization` → one label), fix obvious direction reversals, drop entity types whose only role was to be a property of another type.
- When `existing` is supplied, its labels are passed into every prompt as a soft controlled vocabulary, and the returned draft is merged with `existing` on the way out.

Returns: a new `Ontology`.

#### `method="grounded"` — catalog lookup, zero LLM calls

Adapts [`barakb/text-to-rdf`](https://github.com/barakb/text-to-rdf)'s "find entities, look up their schemas" technique to schema discovery. The corpus tells you *which* types to include; the catalog tells you *what* their schemas are.

```python
from graphrag_sdk.discovery.catalog import DBpediaCatalog

draft = await Ontology.from_sources(
    sources,
    method="grounded",
    catalog=DBpediaCatalog(),       # required for method="grounded"
    sample_chunks_per_doc=3,          # used for NER, not LLM
    concurrency=4,
)
```

Behavior:

- For each source: load → chunk → sample → run NER on each sampled chunk with a small fixed anchor label list (`["person", "organization", "location", "event"]`) to find entity mention strings. These anchor labels never appear in the output ontology — they only help local NER spot proper nouns.
- For each unique mention name across the corpus, call `catalog.link_entity(name)` — the catalog answers "what types is this entity?" (for `DBpediaCatalog` that's a SPARQL query to DBpedia).
- Aggregate the union of types returned across all linked entities.
- Ask the catalog for each detected type's schema definition (`Entity` with attributes + canonical URI).
- Ask the catalog for every relation whose source and target are both in the detected set (or in `existing` — bridge relations between newly-detected and pre-existing types are surfaced).
- Merge with `existing` if supplied.

Zero LLM calls when `llm` is not supplied. No hallucination, no drift — the schema is entirely catalog-derived; the corpus only chooses the subset. At the cost of being **limited to what the catalog knows** — domain-specific types that aren't in the catalog won't appear.

The default NER backend is `GLiNERExtractor()` (no API calls, local model). Override with `entity_extractor=` to plug in `LLMExtractor` or any custom `EntityExtractor`.

Returns: a new `Ontology`.

##### Optional: per-type property trim with an LLM

The catalog hands back every property Schema.org (or whatever catalog) declares for a type. For `Person` that means ~28 properties — including obscure ones like `duns`, `vatID`, `callSign`. To tailor the schema to your corpus, pass `llm`:

```python
draft = await Ontology.from_sources(
    sources,
    llm=llm,                          # triggers the per-type trim
    method="grounded",
    catalog=DBpediaCatalog(),
)
```

With `llm` provided, after type detection the pipeline runs **one LLM call per detected type**: it shows the LLM the catalog's full property list plus up to 5 chunks where that type was detected, and asks "which of these properties are stated or implied in any chunk?" The result trims each entity to a corpus-specific subset.

Cost: ~1 LLM call per *type*, not per chunk — much cheaper than `method="llm"`. Output is deterministic for label invention (still purely catalog) but stochastic for which properties survive (the LLM decides). On hard failure the per-type trim soft-fails to the catalog's full list — a flaky LLM never silently loses schema information.

#### Catalogs

A `Catalog` is the source of truth for ontology vocabulary. Concrete catalogs that ship today:

| Class | Per-entity source | Per-type source |
|---|---|---|
| `DBpediaCatalog` | DBpedia SPARQL (`https://dbpedia.org/sparql`) | Schema.org JSON-LD (live) |

`DBpediaCatalog` is **fully live** — no bundled data. The work splits across two services:

- **`link_entity(name)`** — given an entity mention NER found in a chunk (e.g. `"Albert Einstein"`), SPARQL DBpedia for entities whose `rdfs:label` matches, filtered to `http://dbpedia.org/ontology/` types. Returns the local names (e.g. `["Person", "Scientist"]`). Results are cached in-process for the lifetime of the catalog instance — repeated mentions across chunks only get one SPARQL call.
- **`lookup(type)` / `relations_among(types)`** — first call downloads Schema.org's full JSON-LD vocabulary, processes it (applies `rdfs:subClassOf` inheritance so e.g. `Article` carries `CreativeWork`'s attributes), and caches the processed result under `$XDG_CACHE_HOME/graphrag-sdk/schema_org.json` (typically `~/.cache/graphrag-sdk/schema_org.json`). Cache TTL defaults to 30 days; pass `cache_ttl_days=None` to disable expiry or `0` to force re-fetch every construction.

There is **no offline fallback**. If DBpedia or Schema.org is unreachable and the Schema.org cache is missing/stale, `DBpediaFetchError` is raised. Pre-warm the cache by running once with connectivity, or supply a pre-populated `cache_path`.

No hardcoded type list: types come from real entities found in your corpus, mapped through DBpedia's ontology, then defined by Schema.org. The corpus drives which types appear in the output.

The `Catalog` ABC has three abstract methods — `link_entity(name) / lookup(type_name) / relations_among(type_names)` — so adding a `WikidataCatalog`, a SPARQL-store catalog, or a domain-specific catalog is a focused subclass. The pipeline runs NER (default: `GLiNERExtractor`) to find mention strings, then asks the catalog the rest.

#### Choosing between `method="llm"` and `method="grounded"`

| | `method="llm"` | `method="grounded"` |
|---|---|---|
| LLM cost | Yes (~`D × (S + 1) + 1` calls per corpus) | None |
| Catalog needed | No | Yes |
| Determinism | Stochastic (LLM) | Fully deterministic for a given corpus + catalog |
| Coverage | Whatever the LLM extracts from the text | Whatever the catalog defines for detected types |
| Domain-specific types | Yes (LLM can invent them) | Only if the catalog has them |
| Best for | Unfamiliar / domain-specific corpora | General web / news / biographies where Schema.org fits |
| Composition | Run `method="grounded"` first to anchor on the catalog, then a second `method="llm"` pass with `existing=draft` to add domain-specific types | Same composition pattern, from either direction |

### Live-graph delta — `GraphRAG.suggest_schema_extensions`

A method on `GraphRAG`. Operates against the currently committed ontology.

```python
proposal = await rag.suggest_schema_extensions(
    sources,                          # new docs that motivate additions
    boundaries=...,
    sample_chunks_per_doc=3,
    max_retries=3,
    concurrency=4,
)
print(proposal.summary())
# SchemaExtensionProposal(entities=+2, relations=+1, patterns=+0, attributes=+3, ...)
```

Internally runs the same discovery pipeline with the committed ontology as the prior, then diffs against the committed ontology and surfaces *only the additions*. Returns a [`SchemaExtensionProposal`](#schemaextensionproposal-reference). Nothing is applied to the graph — you review and apply.

### Apply — existing evolution API

You hand the accepted parts of the proposal to the v1.2.x mutation API:

```python
for entity in proposal.new_entities:
    await rag.add_entity(entity)

for relation in proposal.new_relations:
    if not relation.patterns:
        # Open-mode relations (no patterns) can't be applied via
        # add_relation_pattern in v1 — see "What's Not in the API" below.
        continue
    for src, tgt in relation.patterns:
        await rag.add_relation_pattern(relation.label, src, tgt)
    # Preserve the proposed description once the relation has been
    # declared via its first add_relation_pattern call.
    if relation.description:
        await rag.set_relation_description(relation.label, relation.description)

for rel_label, src, tgt in proposal.new_patterns:
    await rag.add_relation_pattern(rel_label, src, tgt)

for owner, attribute in proposal.new_attributes:
    await rag.add_attribute(owner, attribute)   # atomic, LLM-backfilled
```

See [Ontology Evolution](ontology-evolution.md) for the invariants `add_attribute` enforces. Discovery never commits anything itself — the existing evolution machinery stays the single commit surface.

**On `new_attributes` for relation owners:** the v1 mutation API does not yet apply attributes to relations (see "What's Not in the API" in Ontology Evolution — `add_attribute` raises `NotImplementedError` for relation owners). Proposals may include them so you see what discovery found, but applying them requires either waiting for the relation-attribute path to land or hand-managing it via `delete_all()` + re-`ingest()` with the updated schema. The diff still surfaces them so the proposal is honest about what discovery saw.

---

## How the LLM is Kept Honest

Open-vocabulary discovery is brittle by default. The pipeline applies three disciplines that make it tractable:

**1. Structured output with validation-retry feedback.** Every LLM call inside the discovery pipeline goes through a wrapper that parses the response into a Pydantic model and runs a semantic validator (relation patterns reference declared entities, attribute types are in the allowed set, no system-key shadowing — except for `name`, see below). On failure, the *specific* errors are sent back to the LLM as a new user message and the call is retried. The conversation history is preserved across retries so the model sees its own rejected output and the precise correction it needs to make.

**2. Anchoring with a per-document summary.** Before any per-chunk proposal is made, a one-shot per-document call extracts the document's central entities and a one-sentence "aboutness". That summary is prefixed onto every per-chunk prompt for the same document. Chunks share the doc's frame instead of each independently guessing what the document is about.

**3. Normalization as a separate pass.** Per-chunk proposals are merged naively (label-based union) and then a single cross-draft LLM call canonicalizes labels and direction. Without this, drafts accumulate synonyms across calls (`Org` + `Organization` + `Company` as three types) and the schema drifts. The normalization pass is also where the `existing` prior, if supplied, becomes a hard preference: "prefer these labels."

The wrapper is strict — on exhausted retries it raises [`OntologyDiscoveryError`](#ontologydiscoveryerror-reference). The pipeline above it is soft-fail: a single bad chunk is logged and skipped, a failed summary degrades the doc to weaker anchoring, a failed normalization returns the un-normalized draft. Net effect: one noisy chunk or one rate-limited call does not kill the whole draft.

---

## The `name` Attribute

Every entity type in a discovered ontology carries a `name: STRING` attribute. This is intentional — every node in the graph has a `name` (e.g. `"Alice Liddell"` for a Person, `"Acme Corporation"` for an Organization), and the schema honestly reflects that.

The catch: the SDK *fills* `name` automatically during extraction (from the entity span detected by GLiNER and verified in the LLM step). The LLM is never asked to extract `name` as a per-entity attribute — that would cause the value to be written twice, with possibly conflicting versions.

Practical consequences:

- **In `ontology.json` files**, every entity has `name` listed in `properties`. This is documentation, not a knob — you don't need to do anything to populate it.
- **If discovery doesn't propose `name`** on some entity type (small models sometimes omit it), the pipeline adds it automatically before returning. The schema is consistent regardless of model behavior.
- **`suggest_schema_extensions` will never propose `name` as a new attribute.** It's conceptually always present on every entity, so adding it is a no-op.
- **You should not declare `name` with a non-STRING type** — the SDK writes a string. Discovery rejects non-STRING `name` declarations like any other invalid type.

Other reserved-system names (`id`, `description`, `source_chunk_ids`, `spans`, `rel_type`, `fact`, `src_name`, `tgt_name`, `label`) are *not* schema-visible — they are internal SDK plumbing and the discovery validator still rejects them in proposals.

---

## When to Reach for Which

| Situation | Use |
|---|---|
| Brand-new project, no ontology, unfamiliar / domain-specific corpus | `Ontology.from_sources(sources, llm, method="llm")` |
| General web content / news / biographies — Schema.org vocabulary fits | `Ontology.from_sources(sources, method="grounded", catalog=DBpediaCatalog())` |
| You have a draft and want to refresh it against more docs | `Ontology.from_sources(new_sources, llm, existing=current_draft)` |
| You've ingested with a committed ontology and new docs are coming in | `rag.suggest_schema_extensions(new_sources)` → review → apply with the mutation API |
| You already know the schema | Hand-author `Ontology(...)` — discovery only buys overhead |
| The corpus has hard structural constraints (e.g. a known taxonomy) | Hand-author the core; use `suggest_schema_extensions` to find what you missed |

---

## End-to-End Example

```python
import asyncio
from graphrag_sdk import (
    ConnectionConfig, GraphRAG, LiteLLM, LiteLLMEmbedder, Ontology,
)


async def main():
    llm = LiteLLM(model="gpt-4o-mini")

    # 1. Bootstrap a draft from a small corpus.
    draft = await Ontology.from_sources(
        ["docs/intro.md", "docs/history.pdf"],
        llm,
        boundaries="company history and biographies",
        sample_chunks_per_doc=3,
    )
    draft.save_to_file("ontology.json")
    # Review ontology.json by hand at this point — edit / curate / commit.

    # 2. Ingest with the curated draft.
    rag = GraphRAG(
        connection=ConnectionConfig(host="localhost", graph_name="discover_demo"),
        llm=llm,
        embedder=LiteLLMEmbedder(model="text-embedding-3-large"),
        embedding_dimension=256,
        ontology=Ontology.from_file("ontology.json"),
    )
    async with rag:
        await rag.ingest(source=["docs/intro.md", "docs/history.pdf"])

        # 3. Later — a new document shows up.
        proposal = await rag.suggest_schema_extensions(
            "docs/acquisition_news.md",
            boundaries="company history and biographies",
        )

        # 4. Apply the additions you accept.
        for entity in proposal.new_entities:
            await rag.add_entity(entity)
        for owner, attribute in proposal.new_attributes:
            await rag.add_attribute(owner, attribute)


asyncio.run(main())
```

A more comprehensive walkthrough lives in `examples/10_ontology_discovery.py`.

---

## SchemaExtensionProposal Reference

Returned by `suggest_schema_extensions`. Carries additions only — never modifications or deletions of existing schema.

| Field | Type | Description |
|-------|------|-------------|
| `new_entities` | `list[Entity]` | Entity types not in the committed ontology |
| `new_relations` | `list[Relation]` | Relation types not in the committed ontology |
| `new_patterns` | `list[tuple[str, str, str]]` | Additional `(rel_label, src, tgt)` patterns for relation types that already exist |
| `new_attributes` | `list[tuple[str, Attribute]]` | Additional `(owner_label, attribute)` pairs for entity types that already exist |
| `sources_scanned` | `list[str]` | Source identifiers the proposal was derived from |
| `is_empty` | `bool` (property) | `True` when the proposal has no additions to apply |
| `summary()` | `-> str` | One-line summary for logs / CLI output |

`sources_scanned` is coarse-grained evidence — it tells you which inputs informed the proposal, not which input motivated which specific addition. Per-item evidence (proposal-id → chunk-ids) is a planned upgrade that requires plumbing chunk identifiers through the discovery pipeline; not in v1.

---

## OntologyDiscoveryError Reference

Raised by the validation-retry wrapper inside the discovery pipeline when an individual LLM call exhausts its retry budget. The pipeline itself is soft-fail and catches these to keep going, so most users never see this exception. If you call `extract_with_retry` directly (from `graphrag_sdk.discovery`) you will.

| Field | Type | Description |
|-------|------|-------------|
| `chunk_id` | `str \| None` | Identifier of the unit being processed (chunk uid, `"summary:<source>"`, or `"normalize"`) |
| `attempts` | `int` | How many LLM calls were made before giving up |
| `last_error` | `Exception \| None` | The last validation/parse error encountered |

Subclasses `RuntimeError`.

---

## Cost & Scale

Per call to `Ontology.from_sources(sources)` with `D` documents and `S = sample_chunks_per_doc`:

| Step | LLM calls |
|---|---|
| Per-document summary | `D` |
| Per-chunk proposal | `D × S` |
| Normalization | `1` |
| **Total** | `D × (S + 1) + 1` |

Each call is bounded by `max_retries + 1`. Concurrency is capped by the `concurrency` parameter (default 4). For a 10-document corpus with `sample_chunks_per_doc=3` and no retries, that's 41 LLM calls — fast enough to run interactively, cheap on `gpt-4o-mini`-class models.

`suggest_schema_extensions` has the same shape: it runs the full discovery pipeline on the new sources plus the diff at the end (zero LLM cost). The diff against the committed ontology is purely structural.

---

## What's Not in the API (and Why)

| Removed / not added | Why |
|---------------------|-----|
| Auto-application of `SchemaExtensionProposal` | The evolution API enforces the data/ontology consistency invariant; auto-applying would re-introduce the drift the invariant is designed to prevent. Always proposal → review → mutation API. |
| Per-item evidence (proposal-id → chunk-ids) | Requires threading chunk identifiers through the pipeline. Coarse `sources_scanned` is the v1 compromise — enough to know what informed a proposal, not enough to spot-check individual claims. Planned upgrade. |
| Closed controlled vocabulary | The SDK is domain-agnostic. The dynamic equivalent — the `existing` ontology when supplied — gives you the same benefit (constrains the LLM, prevents drift) without locking you to a specific external ontology. |
| Relation-attribute discovery | Mirrors the v1 limitation on `add_attribute` for relation owners. Will lift when the evolution API does. |

---

## See Also

- [Ontology Evolution](ontology-evolution.md) — what to do with the schema once you've discovered it
- [Graph Schema](graph-schema.md) — the structural reference for `Ontology`, `Entity`, `Relation`, `Attribute`
