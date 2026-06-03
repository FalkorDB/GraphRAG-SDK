---
title: "Discovery"
nav_order: 3
parent: "API Reference"
grand_parent: "GraphRAG-SDK"
description: "Catalog interface, DBpediaCatalog, SchemaExtensionProposal, OntologyDiscoveryError, and the lower-level discovery pipeline functions. New in v1.2."
---

# Discovery
{: .label .label-green }
New in v1.2
{: .fs-3 }

Module: `graphrag_sdk.discovery`  ·  Many symbols also importable from `graphrag_sdk` directly.

Catalog-backed and LLM-backed ontology discovery, plus the result and error types.

---

## `Catalog` (ABC)

Abstract base class for ontology vocabularies. Implement these three methods to plug in a custom catalog.

```python
from graphrag_sdk.discovery import Catalog


class Catalog(ABC):
    @abstractmethod
    def link_entity(self, name: str) -> list[str]: ...
    @abstractmethod
    def lookup(self, type_name: str) -> Entity | None: ...
    @abstractmethod
    def relations_among(self, type_names: Iterable[str]) -> list[Relation]: ...
```

| Method | Purpose |
|---|---|
| `link_entity(name)` | Resolve an entity mention string (`"Albert Einstein"`) to a list of type names (`["Person", "Scientist"]`). Empty list when no match. |
| `lookup(type_name)` | Return the schema for a type as an `Entity` (attributes + canonical URI in `description`). `None` if unknown. |
| `relations_among(type_names)` | Return every `Relation` whose source and target are both in the given set. |

The pipeline uses NER (default `GLiNERExtractor`) to find mention strings, then asks the catalog the rest. Adding a Wikidata catalog or a domain-specific catalog is a focused subclass.

---

## `DBpediaCatalog`

The bundled catalog. Fully live, no bundled data.

```python
from graphrag_sdk.discovery.catalog import DBpediaCatalog


catalog = DBpediaCatalog(
    sparql_endpoint=None,        # default: https://dbpedia.org/sparql
    schema_org_url=None,         # default: schema.org's live JSON-LD
    cache_path=None,             # default: $XDG_CACHE_HOME/graphrag-sdk/schema_org.json
    cache_ttl_days=30,           # 30 days; None disables expiry, 0 forces refetch every construction
)
```

### Constructor parameters

| Name | Type | Default | Description |
|---|---|---|---|
| `sparql_endpoint` | `str \| None` | `https://dbpedia.org/sparql` | DBpedia SPARQL endpoint. |
| `schema_org_url` | `str \| None` | Schema.org's full JSON-LD URL | Source for the type vocabulary. |
| `cache_path` | `str \| Path \| None` | `$XDG_CACHE_HOME/graphrag-sdk/schema_org.json` | On-disk cache for the processed Schema.org vocabulary. |
| `cache_ttl_days` | `int \| None` | `30` | Cache freshness. `None` disables expiry; `0` forces refetch on construction. |

### Per-call behaviour

- `link_entity(name)` — SPARQL for `rdfs:label` matches filtered to `http://dbpedia.org/ontology/` types. Results are cached in-process for the catalog instance's lifetime.
- `lookup(type)` / `relations_among(types)` — on first call, downloads Schema.org, applies `rdfs:subClassOf` inheritance, caches to disk.

### No offline fallback

If DBpedia or Schema.org is unreachable and the cache is missing or stale, `DBpediaFetchError` is raised. Pre-warm by running once with connectivity, or supply a pre-populated `cache_path`.

---

## `SchemaExtensionProposal`

Returned by `GraphRAG.suggest_schema_extensions`. Carries only additions.

```python
class SchemaExtensionProposal(BaseModel):
    new_entities: list[Entity]
    new_relations: list[Relation]
    new_patterns: list[tuple[str, str, str]]      # (rel_label, src, tgt)
    new_attributes: list[tuple[str, Attribute]]   # (owner_label, attribute)
    sources_scanned: list[str]
```

| Field | Type | Description |
|---|---|---|
| `new_entities` | `list[Entity]` | Entity types not present in the committed ontology. Apply with `rag.add_entity(entity)`. |
| `new_relations` | `list[Relation]` | Relation types not present in the committed ontology. Apply with `rag.add_relation_pattern(...)` once per pattern. |
| `new_patterns` | `list[tuple[str, str, str]]` | Additional `(rel_label, src, tgt)` patterns for relations that already exist. Apply with `rag.add_relation_pattern(rel_label, src, tgt)`. |
| `new_attributes` | `list[tuple[str, Attribute]]` | Additional `(owner_label, attribute)` pairs. Apply entity-owner pairs with `rag.add_attribute(...)`. Relation owners raise `NotImplementedError` in v1.2. |
| `sources_scanned` | `list[str]` | Source identifiers the proposal was derived from. Coarse — not per-item evidence. |

### Properties / methods

| Member | Type | Description |
|---|---|---|
| `is_empty` | `bool` (property) | `True` when there's nothing to apply. |
| `summary()` | `-> str` | One-line summary for logs and CLI output. |

---

## `OntologyDiscoveryError`

Raised by the validation-retry wrapper inside the discovery pipeline when an LLM call exhausts its retry budget.

```python
class OntologyDiscoveryError(RuntimeError):
    chunk_id: str | None
    attempts: int
    last_error: Exception | None
```

| Attribute | Type | Description |
|---|---|---|
| `chunk_id` | `str \| None` | Unit being processed — chunk uid, `"summary:<src>"`, `"normalize"`, or `None`. |
| `attempts` | `int` | LLM calls made before giving up. |
| `last_error` | `Exception \| None` | Last validation / parse error. |

The pipeline above catches these as soft-fail — most users never see this exception. You hit it if you call `extract_with_retry` directly.

---

## `DBpediaFetchError`

Raised by `DBpediaCatalog` when DBpedia or Schema.org is unreachable and the local cache is missing or stale.

```python
class DBpediaFetchError(RuntimeError):
    pass
```

---

## Lower-level pipeline functions

Internal to `Ontology.from_sources` / `GraphRAG.suggest_schema_extensions`. Documented for completeness — most users don't call them directly.

### `discover_ontology`

```python
async def discover_ontology(
    sources: str | list[str],
    llm: LLMInterface,
    *,
    boundaries: str | None = None,
    existing: Ontology | None = None,
    sample_chunks_per_doc: int = 3,
    max_retries: int = 3,
    concurrency: int = 4,
    chunker: ChunkingStrategy | None = None,
    loader: LoaderStrategy | None = None,
    ctx: Context | None = None,
    seed: int | None = None,
) -> Ontology
```

The implementation behind `Ontology.from_sources(method="llm", ...)`.

### `discover_grounded`

```python
async def discover_grounded(
    sources: str | list[str],
    *,
    catalog: Catalog,
    llm: LLMInterface | None = None,
    entity_extractor: EntityExtractor | None = None,
    existing: Ontology | None = None,
    sample_chunks_per_doc: int = 3,
    max_retries: int = 3,
    concurrency: int = 4,
    chunker: ChunkingStrategy | None = None,
    loader: LoaderStrategy | None = None,
    ctx: Context | None = None,
    seed: int | None = None,
) -> Ontology
```

The implementation behind `Ontology.from_sources(method="grounded", ...)`.

### `suggest_extensions`

```python
async def suggest_extensions(
    existing: Ontology,
    sources: str | list[str],
    llm: LLMInterface,
    *,
    boundaries: str | None = None,
    sample_chunks_per_doc: int = 3,
    max_retries: int = 3,
    concurrency: int = 4,
    chunker: ChunkingStrategy | None = None,
    loader: LoaderStrategy | None = None,
    seed: int | None = None,
) -> SchemaExtensionProposal
```

The implementation behind `GraphRAG.suggest_schema_extensions`.

### `extract_with_retry`

```python
async def extract_with_retry(
    llm: LLMInterface,
    prompt: str,
    response_model: type[BaseModel],
    *,
    validator: Callable[[BaseModel], None] | None = None,
    chunk_id: str | None = None,
    max_retries: int = 3,
) -> BaseModel
```

The validation-retry wrapper. Sends LLM responses through Pydantic validation and an optional semantic validator, retries with feedback in-conversation on failure, raises `OntologyDiscoveryError` on exhaustion.

---

## See also

- [Concepts → Ontology discovery](../concepts/ontology-discovery) — algorithm details, cost model, end-to-end example.
- [Ontology API reference](./ontology) — the `from_sources` entry point.
- [Guides → Auto-discover a schema](../guides/auto-discover-schema), [Suggest extensions from new docs](../guides/suggest-extensions-from-new-docs).
