---
title: "Auto-discover a schema"
nav_order: 2
parent: "Guides"
grand_parent: "GraphRAG-SDK"
description: "Bootstrap an ontology from a corpus with Ontology.from_sources — LLM-driven or grounded against DBpedia. New in v1.2."
---

# Auto-discover a schema
{: .label .label-green }
New in v1.2
{: .fs-3 }

When you don't know the schema yet, let the SDK draft one from your documents. The result is a `Ontology` you inspect, edit, version-control, and then pass to `GraphRAG` like a hand-authored one.

## LLM-driven discovery (the default)

```python
import asyncio
from graphrag_sdk import (
    GraphRAG, ConnectionConfig, LiteLLM, LiteLLMEmbedder, Ontology,
)


async def main():
    llm = LiteLLM(model="openai/gpt-4o-mini")

    # 1. Draft an ontology from the corpus.
    draft = await Ontology.from_sources(
        ["docs/intro.md", "docs/history.pdf"],
        llm,
        boundaries="company history and biographies",
        sample_chunks_per_doc=3,
    )
    draft.save_to_file("ontology.json")
    print(f"Discovered {len(draft.entities)} entity types, "
          f"{len(draft.relations)} relation types.")

    # 2. (Manual step.) Open ontology.json, prune anything off-topic,
    # commit it to your repo. The drafted schema is a starting point,
    # not a final answer.

    # 3. Ingest with the curated schema.
    async with GraphRAG(
        connection=ConnectionConfig(host="localhost", graph_name="discover_demo"),
        llm=llm,
        embedder=LiteLLMEmbedder(model="openai/text-embedding-3-large"),
        ontology=Ontology.from_file("ontology.json"),
    ) as rag:
        await rag.ingest(source=["docs/intro.md", "docs/history.pdf"])
        await rag.finalize()


asyncio.run(main())
```

## Grounded discovery — no LLM cost

If your documents cover general topics (people, organizations, places, creative works), the live `DBpediaCatalog` can hand you a schema without any LLM calls — types come from DBpedia, attributes come from Schema.org:

```python
from graphrag_sdk.discovery.catalog import DBpediaCatalog

draft = await Ontology.from_sources(
    ["docs/news/*.md"],
    method="grounded",
    catalog=DBpediaCatalog(),
)
draft.save_to_file("ontology.json")
```

The pipeline runs NER on sampled chunks, asks DBpedia what types each detected entity belongs to, then asks Schema.org for the schema of each detected type. The corpus chooses *which* types appear; the catalog defines *what* they look like.

Trade-off: limited to what the catalog knows. Domain-specific types (`Polymer`, `RiskFactor`) won't be in DBpedia, so a grounded-only run will miss them.

### Combining grounded with a per-type trim

The catalog hands back every property Schema.org declares for a type — `Person` has 28, most of which probably aren't in your corpus. Pass an `llm` argument and the pipeline runs one extra LLM call per detected type to trim each type's properties down to the subset actually present in your text:

```python
draft = await Ontology.from_sources(
    ["docs/news/*.md"],
    llm=llm,                          # triggers the trim
    method="grounded",
    catalog=DBpediaCatalog(),
)
```

Cost is one call per *type*, not per chunk — much cheaper than `method="llm"`. On LLM failure the trim soft-fails to the catalog's full list, so a flaky model never silently loses schema information.

## Two-pass discovery for domain corpora

When the catalog covers some of your types but not all, run grounded first to anchor on canonical Schema.org types, then a second LLM pass with `existing=` to add the domain-specific ones:

```python
base = await Ontology.from_sources(
    sources, method="grounded", catalog=DBpediaCatalog(),
)
draft = await Ontology.from_sources(
    sources, llm, existing=base,    # LLM extends the catalog-derived base
    boundaries="materials science papers",
)
draft.save_to_file("ontology.json")
```

`existing=` is treated as a soft controlled vocabulary in the LLM prompts and merged into the final result — labels you already have are preferred.

## Always inspect before ingesting

The drafted schema is a *suggestion*. Open `ontology.json` and:

- **Prune off-topic types.** A 5-page intro might mention "Animal" once; you probably don't want a `:Animal` label crowding the graph.
- **Verify relation directions.** Patterns like `("Company", "Person")` for `WORKS_AT` are backwards — fix them.
- **Add descriptions.** The drafted schema rarely fills these; a one-line description per type helps the extractor on edge cases.
- **Remove duplicates the normalization pass missed.** Rare, but possible on small models.

Then commit the curated file to your repo and load it with `Ontology.from_file("ontology.json")`.

## See also

- [Concepts → Ontology discovery](../concepts/ontology-discovery) — algorithm details, cost model, how the LLM is kept honest.
- [API Reference → Ontology](../api-reference/ontology) — `from_sources` full signature.
- [API Reference → Discovery](../api-reference/discovery) — `Catalog`, `DBpediaCatalog`, exceptions.
- [Guides → Suggest extensions from new docs](./suggest-extensions-from-new-docs) — the next step once you've ingested.
