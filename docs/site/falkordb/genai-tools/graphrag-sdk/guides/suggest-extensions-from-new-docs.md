---
title: "Suggest extensions from new docs"
nav_order: 3
parent: "Guides"
grand_parent: "GraphRAG-SDK"
description: "Use GraphRAG.suggest_schema_extensions to find new entity types, relations, and attributes in newly arrived documents. New in v1.2."
---

# Suggest extensions from new docs
{: .label .label-green }
New in v1.2
{: .fs-3 }

Once an ontology is committed and you've ingested some documents, the question shifts from "what's the schema?" to "what's missing?" When new documents arrive, they may mention types or relations the committed ontology doesn't cover. `suggest_schema_extensions` finds those additions — without touching the graph.

## Runnable example

```python
import asyncio
from graphrag_sdk import (
    GraphRAG, ConnectionConfig, LiteLLM, LiteLLMEmbedder,
)


async def main():
    async with GraphRAG(
        connection=ConnectionConfig(host="localhost", graph_name="evolution_demo"),
        llm=LiteLLM(model="openai/gpt-4o-mini"),
        embedder=LiteLLMEmbedder(model="openai/text-embedding-3-large"),
    ) as rag:
        # 1. Propose schema additions from a newly arrived document.
        proposal = await rag.suggest_schema_extensions(
            "docs/acquisition_news.md",
            boundaries="company news",
        )

        print(proposal.summary())
        # SchemaExtensionProposal(entities=+2, relations=+1, patterns=+0, attributes=+3, sources_scanned=1)

        if proposal.is_empty:
            print("Nothing new to add.")
            return

        # 2. Review (programmatically, or print and ask a human).
        for entity in proposal.new_entities:
            print(f"+ Entity {entity.label}: {entity.description}")
        for relation in proposal.new_relations:
            print(f"+ Relation {relation.label}: {relation.patterns}")
        for rel_label, src, tgt in proposal.new_patterns:
            print(f"+ Pattern {rel_label}: ({src}) -> ({tgt})")
        for owner, attribute in proposal.new_attributes:
            print(f"+ Attribute {owner}.{attribute.name} ({attribute.type})")

        # 3. Apply the accepted parts via the existing evolution API.
        for entity in proposal.new_entities:
            await rag.add_entity(entity)

        for relation in proposal.new_relations:
            for src, tgt in relation.patterns:
                await rag.add_relation_pattern(relation.label, src, tgt)
            if relation.description:
                await rag.set_relation_description(relation.label, relation.description)

        for rel_label, src, tgt in proposal.new_patterns:
            await rag.add_relation_pattern(rel_label, src, tgt)

        for owner, attribute in proposal.new_attributes:
            # add_attribute is atomic and LLM-backfilled — it re-scans every
            # chunk that mentions an entity of `owner` and fills the value.
            try:
                await rag.add_attribute(owner, attribute)
            except NotImplementedError:
                # Relation-attribute evolution isn't in v1.2 — see
                # API Reference -> GraphRAG -> add_attribute for the
                # workaround. Skip those entries for now.
                continue

        # 4. Now ingest the new document, which the (extended) ontology covers.
        await rag.ingest("docs/acquisition_news.md")
        await rag.finalize()


asyncio.run(main())
```

## What `suggest_schema_extensions` returns

A `SchemaExtensionProposal` — additions only, never modifications or deletions:

| Field | Type | Description |
|---|---|---|
| `new_entities` | `list[Entity]` | Entity types not in the committed ontology. |
| `new_relations` | `list[Relation]` | Relation types not in the committed ontology. |
| `new_patterns` | `list[tuple[str, str, str]]` | Additional `(rel_label, src, tgt)` patterns for relation types that already exist. |
| `new_attributes` | `list[tuple[str, Attribute]]` | Additional `(owner_label, attribute)` for existing entity or relation types. |
| `sources_scanned` | `list[str]` | Source identifiers the proposal was derived from. |
| `is_empty` (property) | `bool` | `True` when there's nothing to apply. |
| `summary()` | `-> str` | One-line summary for logs. |

`sources_scanned` is coarse — it tells you what informed the proposal, not which input motivated which specific addition. Per-item evidence is a planned upgrade.

## Why nothing is auto-applied

`suggest_schema_extensions` never writes to the graph. Three reasons:

1. **Human review.** Discovered types are sometimes wrong (`Mention` instead of `Topic`, off-topic catch-alls, reversed relation directions). A review step is the cheapest way to catch them.
2. **Cost control.** `add_attribute` for entity owners runs an LLM call per chunk that mentions an entity of that label. Applying every proposed attribute blindly can be expensive.
3. **The evolution API is the single commit surface.** Auto-applying would re-introduce the data/ontology drift the evolution API is designed to prevent.

## Caveat — relation-owner attributes

The v1.2 evolution API doesn't support attributes on relation owners (`add_attribute(owner_label="WORKS_AT", ...)` raises `NotImplementedError`). Proposals may include them so you can see what discovery found, but applying them needs either:

- Waiting for the relation-attribute path to land in a future release; or
- Hand-managing via `delete_all()` + re-ingest with the updated schema.

The example above catches `NotImplementedError` and skips. Add a log line or a TODO so they don't get forgotten.

## See also

- [Concepts → Ontology discovery](../concepts/ontology-discovery) — algorithm details.
- [API Reference → GraphRAG](../api-reference/graphrag#suggest_schema_extensions) — full signature.
- [API Reference → Discovery](../api-reference/discovery#schemaextensionproposal) — full `SchemaExtensionProposal` reference.
