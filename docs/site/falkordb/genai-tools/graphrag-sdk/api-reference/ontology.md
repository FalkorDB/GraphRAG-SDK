---
title: "Ontology"
nav_order: 2
parent: "API Reference"
grand_parent: "GraphRAG-SDK"
description: "Ontology, Entity, Relation, Attribute — the schema data model. Includes from_sources for v1.2 ontology discovery."
---

# Ontology

Module: `graphrag_sdk`  ·  Import: `from graphrag_sdk import Ontology, Entity, Relation, Attribute`

The schema of a knowledge graph. Pydantic v2 models — all four classes are JSON-serialisable round-trip.

---

## `Attribute`

A typed property on an entity or relation.

```python
class Attribute(DataModel):
    name: str
    type: str = "STRING"
    description: str | None = None
```

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | — required — | Property name. Must not be in the reserved set (see below). |
| `type` | `str` | `"STRING"` | One of `STRING`, `INTEGER`, `FLOAT`, `BOOLEAN`, `DATE`, `LIST`. Case-insensitive, normalised to uppercase. |
| `description` | `str \| None` | `None` | Human-readable description. Forwarded into LLM prompts. |

#### Raises (at construction)

- `ValidationError` (Pydantic) — `type` is not one of the allowed values.

**Reserved attribute names** (declaring an `Attribute` with these names shadows SDK-written values):

`name`, `description`, `source_chunk_ids`, `spans`, `rel_type`, `fact`, `src_name`, `tgt_name`, `id`, `label`.

`name` is special — every discovered ontology carries it. See [Concepts → Ontology](../concepts/ontology#reserved-attribute-names).

---

## `Entity`

A node type.

```python
class Entity(DataModel):
    label: str
    description: str | None = None
    properties: list[Attribute] = []
```

| Field | Type | Default | Description |
|---|---|---|---|
| `label` | `str` | — required — | Node label (`:Person`, `:Organization`, …). |
| `description` | `str \| None` | `None` | Forwarded into extraction prompts. |
| `properties` | `list[Attribute]` | `[]` | Typed attributes for this entity type. |

Identity is by `label` only. Two `Entity` instances with the same label compare and hash equal — schemas should declare each label once.

---

## `Relation`

An edge type.

```python
class Relation(DataModel):
    label: str
    description: str | None = None
    patterns: list[tuple[str, str]] = []
    properties: list[Attribute] = []
```

| Field | Type | Default | Description |
|---|---|---|---|
| `label` | `str` | — required — | Relation type. |
| `description` | `str \| None` | `None` | Forwarded into extraction prompts. |
| `patterns` | `list[tuple[str, str]]` | `[]` | Allowed `(source_label, target_label)` pairs. Direction is `source → target`. Empty list = open (any types). |
| `properties` | `list[Attribute]` | `[]` | Typed attributes on this edge type. |

Identity is by `label` only.

#### Pattern semantics

`Relation(label="WORKS_AT", patterns=[("Person", "Company")])` means `(Person)-[:WORKS_AT]->(Company)`. The extractor silently prunes mismatches and logs a structured warning naming the offending `(src, tgt)` pairs.

---

## `Ontology`

The container.

```python
class Ontology(DataModel):
    entities: list[Entity] = []
    relations: list[Relation] = []
```

#### Construction-time validation

A warning (not an error) is logged when a `Relation.patterns` entry references an entity label not declared in `entities` — catches typos at config time. Discovery has a stricter validator that errors out.

---

### `Ontology.from_file`

Load from a JSON file.

```python
@classmethod
def from_file(cls, path: str) -> Ontology
```

#### Parameters

| Name | Type | Description |
|---|---|---|
| `path` | `str` | Path to a JSON file written by `save_to_file`. |

#### Returns

`Ontology`.

---

### `Ontology.from_sources` (new in v1.2)

Auto-discover an ontology from a corpus. Two algorithms; pure function — no DB connection required.

```python
@classmethod
async def from_sources(
    cls,
    sources: str | list[str],
    llm: Any | None = None,
    *,
    method: Literal["llm", "grounded"] = "llm",
    # method="llm"
    boundaries: str | None = None,
    max_retries: int = 3,
    # method="grounded"
    catalog: Any | None = None,
    entity_extractor: Any | None = None,
    # shared
    existing: Ontology | None = None,
    sample_chunks_per_doc: int = 3,
    concurrency: int = 4,
    chunker: Any | None = None,
    loader: Any | None = None,
    ctx: Any | None = None,
    seed: int | None = None,
) -> Ontology
```

#### Parameters

| Name | Type | Default | Description |
|---|---|---|---|
| `sources` | `str \| list[str]` | — required — | Same union `GraphRAG.ingest` accepts. |
| `llm` | `LLMInterface \| None` | `None` | Required for `method="llm"`. Optional for `method="grounded"` — triggers per-type property trim. |
| `method` | `Literal["llm", "grounded"]` | `"llm"` | Discovery algorithm. See [Concepts → Ontology discovery](../concepts/ontology-discovery). |
| `boundaries` | `str \| None` | `None` | Free-text scope hint. `method="llm"` only. |
| `max_retries` | `int` | `3` | Retry budget per LLM call inside the validation-retry wrapper. `method="llm"` only. |
| `catalog` | `Catalog \| None` | `None` | Required for `method="grounded"`. Typically `DBpediaCatalog()`. |
| `entity_extractor` | `EntityExtractor \| None` | `GLiNERExtractor()` | NER backend for `method="grounded"`. |
| `existing` | `Ontology \| None` | `None` | Optional structured prior. Treated as soft controlled vocabulary by `method="llm"`; merged into the catalog result by `method="grounded"`. |
| `sample_chunks_per_doc` | `int` | `3` | Chunks sampled per document. |
| `concurrency` | `int` | `4` | Max in-flight LLM / NER calls. |
| `chunker` / `loader` | overrides for the bundled defaults. |
| `ctx` | `Context \| None` | `None` | Optional execution context. |
| `seed` | `int \| None` | `None` | RNG seed for deterministic chunk sampling. |

#### Returns

A new `Ontology` (a draft). Inspect, edit, persist with `save_to_file`, then pass to `GraphRAG(ontology=...)`.

#### Raises

- `ValueError` — `method="llm"` without `llm`, or `method="grounded"` without `catalog`, or an unknown `method` value.
- `OntologyDiscoveryError` — raised by individual LLM calls only on hard failure; the pipeline above catches these as soft-fail. You only see this exception if you call `extract_with_retry` directly.

---

### `Ontology.save_to_file`

```python
def save_to_file(self, path: str, *, indent: int = 2) -> None
```

Write the schema to `path` as JSON. Overwrites existing files.

---

### `Ontology.merge`

```python
def merge(self, other: Ontology) -> Ontology
```

Return a new `Ontology` that is the union of `self` and `other`.

- Entity / relation types are unioned by `label`.
- Properties are unioned by `name`; on conflict, `other`'s type/description wins (last-write-wins).
- Relation patterns are unioned with order-preserving dedup.

---

## Deprecated aliases

Importing these names emits `DeprecationWarning`:

| Old (≤ v1.1) | New (v1.2+) |
|---|---|
| `graphrag_sdk.GraphSchema` | `Ontology` |
| `graphrag_sdk.EntityType` | `Entity` |
| `graphrag_sdk.RelationType` | `Relation` |
| `graphrag_sdk.PropertyType` | `Attribute` |

## See also

- [Concepts → Ontology](../concepts/ontology)
- [Concepts → Ontology discovery](../concepts/ontology-discovery)
- [Discovery API reference](./discovery) — `Catalog`, `DBpediaCatalog`, `SchemaExtensionProposal`.
