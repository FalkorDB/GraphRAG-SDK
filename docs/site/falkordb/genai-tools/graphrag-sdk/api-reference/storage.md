---
title: "Storage"
nav_order: 9
parent: "API Reference"
grand_parent: "GraphRAG-SDK"
description: "Lower-level storage classes — GraphStore, OntologyStore, VectorStore — and ontology-store exceptions."
---

# Storage

Module: `graphrag_sdk.storage`  ·  Most users don't construct these directly — `GraphRAG` wires them.

The low-level storage layer. Three coordinated stores backed by one `FalkorDBConnection`. Documented for advanced users writing custom strategies or operating on the graph directly.

---

## `GraphStore`

Data-graph operations. Backs every entity/relationship write the ingestion pipeline performs, plus all of the evolution machinery (rename, drop, backfill).

```python
class GraphStore:
    def __init__(self, conn: FalkorDBConnection) -> None
```

Notable methods (non-exhaustive):

| Method | Purpose |
|---|---|
| `upsert_nodes(nodes)` | Idempotent MERGE on `(label, id)`. |
| `upsert_relationships(rels)` | Idempotent MERGE on edge type + endpoints. For `RELATES` edges, unions `source_chunk_ids` lists across merges. |
| `delete_all()` | `GRAPH.DELETE` the entire graph. |
| `get_statistics()` | Node and edge counts, type lists, density. |
| `query_raw(cypher, params=None, timeout=None)` | Pass-through to the connection. |

Most other methods are internal contracts between `GraphRAG` and the pipeline. Inspect the source if you're writing a custom strategy.

---

## `OntologyStore`

Owner of the paired `<graph_name>__ontology` graph. Every ontology mutation goes through here.

```python
class OntologyStore:
    def __init__(self, conn: FalkorDBConnection, graph_name: str) -> None
```

| Method | Purpose |
|---|---|
| `load()` | Read the persisted ontology into an `Ontology` instance. |
| `register(ontology)` | Additive registration — declares new entity/relation/attribute types. Refuses conflicting redeclarations. |
| `set_description(kind, label, description, owner_label=None)` | Update a description on an entity, relation, or property. |
| `rename_entity_label`, `rename_relation_label`, `rename_property_label` | Ontology side of the matching `GraphRAG.rename_*` methods. |
| `drop_entity_label`, `drop_relation_label`, `drop_relation_pattern_node`, `drop_entity_property` | Ontology side of the matching `GraphRAG.drop_*` methods. |
| `add_entity_property(label, attribute)` | Atomic property declaration (called by `add_attribute` as the final commit step). |
| `add_relation_pattern_node(rel_label, source, target)` | Declare a new pattern on an existing or new relation. |
| `clear()` | Drop the entire ontology graph. |

### `OntologyContradictionError`

```python
class OntologyContradictionError(ValueError):
    pass
```

Raised by `OntologyStore.register` when the user-supplied ontology redeclares an existing property with a conflicting type.

### `OntologyModificationNotAllowedError`

```python
class OntologyModificationNotAllowedError(RuntimeError):
    pass
```

Raised when an operation attempts to modify the ontology in a way the strict-mode invariants forbid (legacy from pre-v1.2; rarely surfaces in current code).

Deprecated alias `SchemaModificationNotAllowedError` is kept importable but emits `DeprecationWarning`.

---

## `VectorStore`

Vector and full-text index operations.

```python
class VectorStore:
    def __init__(
        self,
        conn: FalkorDBConnection,
        embedder: Embedder,
        embedding_dimension: int,
    ) -> None
```

| Method | Purpose |
|---|---|
| `ensure_indices()` | Create vector / full-text / range indexes as needed. Idempotent. |
| `backfill_entity_embeddings()` | Compute and write name-only embeddings on every entity. |
| `embed_relationships()` | Compute and write fact-text embeddings on `RELATES` edges. |
| `search_chunks_by_vector(question_embedding, top_k)` | Vector kNN over chunks. |
| `search_entities_by_vector(question_embedding, top_k)` | Vector kNN over entities. |
| `search_entities_by_fulltext(query, top_k)` | FT search on entity names. |

The bundled `MultiPathRetrieval` uses every method here.

## See also

- [API Reference → GraphRAG](./graphrag) — `finalize` orchestrates `VectorStore.backfill_*` and `VectorStore.ensure_indices`.
- [API Reference → Exceptions](./exceptions) — full exception hierarchy.
