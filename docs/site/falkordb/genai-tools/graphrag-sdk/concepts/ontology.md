---
title: "Ontology"
nav_order: 2
parent: "Concepts"
grand_parent: "GraphRAG-SDK"
description: "The Ontology data model — entities, relations, attributes, and patterns — and how it shapes extraction and retrieval."
---

# Ontology

The **ontology** is the schema of your knowledge graph: which entity types are allowed, which relation types connect them, and which typed attributes hang off each.

```python
from graphrag_sdk import Ontology, Entity, Relation, Attribute

ontology = Ontology(
    entities=[
        Entity(label="Person", description="A human being", properties=[
            Attribute(name="email", type="STRING"),
            Attribute(name="age", type="INTEGER"),
        ]),
        Entity(label="Organization", description="A company or institution"),
    ],
    relations=[
        Relation(
            label="WORKS_AT",
            description="Is employed by",
            patterns=[("Person", "Organization")],
        ),
    ],
)
```

That object becomes the contract between you and the SDK:

- **Extraction** is constrained by it — the LLM sees the declared labels in its prompt and is asked to produce only those types.
- **Cypher generation** at retrieval time uses it to surface the available labels, properties, and patterns to the LLM.
- **Validation** rejects relation patterns that reference undeclared entity labels (warning at construction time, hard error during discovery).

## The four building blocks

### `Entity`

A node type. `label` is the only required field; `description` helps the LLM understand intent, `properties` declares typed attributes.

### `Relation`

An edge type. `patterns` is a list of `(source_label, target_label)` tuples — **direction matters**. An empty `patterns` list means the relation is open: any entity types can be connected. Patterns are a soft constraint at extraction time: mismatches are silently pruned and logged.

### `Attribute`

A typed property on an entity or relation. `type` must be one of `STRING`, `INTEGER`, `FLOAT`, `BOOLEAN`, `DATE`, `LIST`. Values from the LLM are coerced into that type at write time; failures are dropped (logged at WARNING).

### `Ontology`

The container. Carries `entities` and `relations`. Methods:

- `from_file(path)` / `save_to_file(path)` — JSON round-trip for version control.
- `from_sources(sources, llm, ...)` — discover an ontology from a corpus. See [Concepts → Ontology discovery](./ontology-discovery).
- `merge(other)` — union semantics. Last-write-wins on conflicting property types, order-preserving dedup on relation patterns.

## Reserved attribute names

The SDK writes a fixed set of properties on every node and edge during ingestion. Declaring an `Attribute` with one of these names shadows the system-written value:

`name`, `description`, `source_chunk_ids`, `spans`, `rel_type`, `fact`, `src_name`, `tgt_name`, `id`, `label`

`name` is special — every discovered ontology carries `name: STRING` on every entity type, because every entity in the graph has a name. The SDK fills it from the entity span detected during extraction, never from an LLM "what's this entity's name?" call. Treat `name` as documentation, not a knob. The other reserved names are SDK plumbing and will be rejected by the discovery validator.

## Persistence and evolution

When you construct `GraphRAG(ontology=...)`, the ontology is persisted into a paired graph named `<graph_name>__ontology`. That graph is the single source of truth — on subsequent runs you can re-construct `GraphRAG` without `ontology=` and the persisted version is used.

Evolving an existing ontology — adding entities, renaming labels, dropping attributes — happens through the methods on `GraphRAG`:

```python
await rag.add_entity(Entity(label="Project", description="A delivered work item"))
await rag.add_attribute("Person", Attribute(name="department", type="STRING"))
await rag.rename_entity("Org", "Organization")
await rag.drop_attribute("Person", "deprecated_field")
```

These methods maintain the invariant that the data graph and the ontology graph stay consistent — data migrations run first so a crash mid-call always leaves a state that can be recovered by re-running the same call.

See [API Reference → GraphRAG](../api-reference/graphrag) for every evolution method.

## Migrating from v1.1

The v1.1 vocabulary still works but emits `DeprecationWarning` on import:

| Old (v1.1) | New (v1.2+) |
|---|---|
| `GraphSchema` | `Ontology` |
| `EntityType` | `Entity` |
| `RelationType` | `Relation` |
| `PropertyType` | `Attribute` |
| `SchemaModificationNotAllowedError` | `OntologyModificationNotAllowedError` |
| `GraphRAG(schema=...)` | `GraphRAG(ontology=...)` |
| `rag.schema` | `rag.ontology` |

Behaviour is identical — only the names changed.

## See also

- [Concepts → Ontology discovery](./ontology-discovery) — bootstrap an ontology from a corpus.
- [API Reference → Ontology](../api-reference/ontology) — full parameter/return reference for every method.
- [Guides → Define a custom schema](../guides/define-custom-schema) — runnable example.
