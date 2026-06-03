---
title: "Define a custom schema"
nav_order: 1
parent: "Guides"
grand_parent: "GraphRAG-SDK"
description: "Hand-author an Ontology with entities, relations, attributes, and patterns. The fastest path when the domain is well-understood."
---

# Define a custom schema

When you know your domain — entity types, relation types, the patterns that connect them — declare the schema by hand. The extractor will be constrained to those types, producing a tighter and more queryable graph than open-world extraction.

## Runnable example

```python
import asyncio
from graphrag_sdk import (
    GraphRAG, ConnectionConfig, LiteLLM, LiteLLMEmbedder,
    Ontology, Entity, Relation, Attribute,
)


ontology = Ontology(
    entities=[
        Entity(
            label="Person",
            description="A human being mentioned in the corpus.",
            properties=[
                Attribute(name="email", type="STRING"),
                Attribute(name="title", type="STRING", description="Job title."),
                Attribute(name="years_experience", type="INTEGER"),
            ],
        ),
        Entity(
            label="Organization",
            description="A company, university, or other formal institution.",
            properties=[
                Attribute(name="founded", type="DATE"),
                Attribute(name="hq_location", type="STRING"),
            ],
        ),
        Entity(label="Project", description="A delivered work item or initiative."),
    ],
    relations=[
        Relation(
            label="WORKS_AT",
            description="Person is employed by Organization.",
            patterns=[("Person", "Organization")],
        ),
        Relation(
            label="LEADS",
            description="Person leads Project.",
            patterns=[("Person", "Project")],
        ),
        Relation(
            label="SPONSORS",
            description="Organization funds Project.",
            patterns=[("Organization", "Project")],
        ),
    ],
)


async def main():
    async with GraphRAG(
        connection=ConnectionConfig(host="localhost", graph_name="hand_schema_demo"),
        llm=LiteLLM(model="openai/gpt-4o-mini"),
        embedder=LiteLLMEmbedder(model="openai/text-embedding-3-large"),
        ontology=ontology,
    ) as rag:
        await rag.ingest(text=(
            "Alice Johnson, a senior engineer at Acme Corp (founded 2005, "
            "headquartered in London), leads Project Orion. Acme Corp sponsors "
            "Project Orion alongside two partner firms."
        ))
        await rag.finalize()

        answer = await rag.completion("Who leads Project Orion and what's their experience level?")
        print(answer.answer)


asyncio.run(main())
```

## Anatomy

### Entities — what kinds of nodes exist

Every node in the resulting graph carries one of your declared labels. `description` is forwarded into the extraction prompt — short, intent-shaping descriptions ("a paid employee, contractor, or volunteer") nudge the LLM in tricky cases.

### Attributes — typed properties

`type` must be one of `STRING`, `INTEGER`, `FLOAT`, `BOOLEAN`, `DATE`, `LIST`. The SDK coerces LLM-emitted values at write time and drops coercion failures (logged at WARNING). Add a `description` if the attribute name is ambiguous ("title" — job title or honorific?).

A few names are reserved by the SDK (`name`, `description`, `source_chunk_ids`, …) — see [Concepts → Ontology](../concepts/ontology#reserved-attribute-names).

### Relations and patterns — directed type constraints

`patterns=[("A", "B")]` means `(A)-[REL]->(B)` — direction matters. The extractor silently prunes pattern mismatches and logs a structured warning naming the offending `(src, tgt)` pairs, so a swapped direction is easy to spot in logs.

Empty `patterns=[]` means "any source, any target" — open mode for that specific relation.

## Persisting the schema

The ontology is auto-persisted into a paired graph (`<graph_name>__ontology`). For version control, also save the JSON:

```python
ontology.save_to_file("ontology.json")
```

And on the next run:

```python
ontology = Ontology.from_file("ontology.json")
```

## Evolving the schema later

Once you've ingested with an ontology, evolve it through methods on `GraphRAG`:

```python
await rag.add_entity(Entity(label="Customer", description="A buyer of our products."))
await rag.add_attribute("Person", Attribute(name="department", type="STRING"))
await rag.rename_entity("Org", "Organization")
await rag.drop_attribute("Person", "deprecated_field")
```

`add_attribute` for entity owners is **atomic and LLM-backfilled** — it re-scans every chunk that mentions an entity of that label, asks the LLM to extract the new attribute's value, and fills it. Re-runs are idempotent (chunk markers skip completed work).

## See also

- [Concepts → Ontology](../concepts/ontology) — full mental model.
- [API Reference → Ontology](../api-reference/ontology) — every method's signature.
- [Guides → Auto-discover a schema](./auto-discover-schema) — if you don't know what schema you need yet.
