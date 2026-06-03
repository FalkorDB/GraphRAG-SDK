---
title: "Changelog"
nav_order: 6
parent: "GraphRAG-SDK"
grand_parent: "GenAI Tools"
description: "Notable changes per release. Full changelog with diffs lives in the GitHub repository."
---

# Changelog

Condensed highlights per release. The [full CHANGELOG.md](https://github.com/FalkorDB/GraphRAG-SDK/blob/main/CHANGELOG.md) in the repository carries every entry plus migration notes and rationale.

The project follows [Semantic Versioning](https://semver.org/) and the [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) format.

---

## [1.2.0] — 2026-06-01

Two large, related additions and a comprehensive ontology vocabulary rename. **No breaking changes** — the rename ships with backwards-compatible aliases that emit `DeprecationWarning`.

### Added

- **Ontology discovery** (PR #271).
  - `Ontology.from_sources(sources, llm, *, method="llm" | "grounded", ...)` — bootstrap an ontology from a corpus. LLM-driven or catalog-grounded. See [Concepts → Ontology discovery](./concepts/ontology-discovery).
  - `GraphRAG.suggest_schema_extensions(sources, *, ...)` — propose additions to the committed ontology from new documents. Returns a `SchemaExtensionProposal` for review.
  - `DBpediaCatalog` — live SPARQL to DBpedia for entity→type, live JSON-LD from Schema.org for type→schema. Bundled implementation of the `Catalog` ABC.
  - `SchemaExtensionProposal`, `OntologyDiscoveryError`, `DBpediaFetchError` data and error types.
- **Persistent ontology graph and typed attributes** (PR #256).
  - `Ontology`, `Entity`, `Relation`, `Attribute` Pydantic models.
  - Allowed attribute types: `STRING`, `INTEGER`, `FLOAT`, `BOOLEAN`, `DATE`, `LIST`.
  - Paired `<data_graph>__ontology` graph for persistent schema; always-on per `GraphRAG` instance.
  - `GraphRAG.get_ontology()`, `refresh_ontology()`, `set_ontology()`, `save_ontology()`.
- **Mutating evolution API** (PR #268). Fifteen methods on `GraphRAG`:
  - Pure declarations: `set_entity_description`, `set_relation_description`, `set_attribute_description`, `add_entity`, `add_relation_pattern`.
  - Data migrations: `rename_entity`, `rename_attribute`, `rename_relation`, `drop_entity`, `drop_relation`, `drop_relation_pattern`.
  - Atomic evolution: `add_attribute`, `drop_attribute`.
  - Opportunistic backfills: `backfill_entity`, `backfill_relation_pattern`.
- **Supporting types:** `EvolutionResult`, `OntologyEvolutionError`, `BackfillResult`, `BackfillExecutor`, `ChunkContext`, `BackfillMergeStats`.

### Changed

- **Vocabulary rename** (non-breaking, aliased):

  | Old | New |
  |---|---|
  | `GraphSchema` | `Ontology` |
  | `EntityType` | `Entity` |
  | `RelationType` | `Relation` |
  | `PropertyType` | `Attribute` |
  | `SchemaModificationNotAllowedError` | `OntologyModificationNotAllowedError` |
  | `GraphRAG(schema=...)` | `GraphRAG(ontology=...)` |
  | `rag.schema` | `rag.ontology` |

  Old names still importable; each emits `DeprecationWarning`.

- **`set_ontology()` and `get_ontology()` propagate to retrieval** — concurrent retrieval calls always see the same ontology snapshot as the most recent evolution write.

### Deprecated

- Legacy ontology vocabulary (see Changed). Will be removed in a future major release.

### Invariants and rules established

- **Alignment invariant.** `add_attribute` is atomic and LLM-backfilled — runs the per-chunk backfill **before** committing the ontology graph. Schema never claims a property that the data hasn't been asked to populate.
- **Concurrency rule.** Evolution calls are **not safe** to run concurrently with `ingest()` or with each other on the same graph. Gate behind an application-level lock; treat evolution as a maintenance operation (pause ingest, run, resume).
- **Idempotent retries.** Per-chunk `extracted_ops` markers on `:Chunk` nodes make every evolution and backfill safe to retry.

---

## [1.1.1] — 2026-05-13

Patch release with crash-safety hardening for `update()` and `delete_document()`.

---

## [1.1.0] — 2026-05-05

- **Incremental updates.** `update(source, document_id=..., if_missing=...)`, `delete_document(document_id, if_missing=...)`, `apply_changes(added=, modified=, deleted=)`. SHA-256 content-hash short-circuits no-op updates. See [Concepts → Incremental updates](./concepts/incremental-updates).
- **Crash-safe cutover** for `update()` (state-machine rollforward) and `delete_document()` (single atomic commit marker).
- **`BatchEntry` / `ApplyChangesResult`** wrap per-file outcomes so `apply_changes` never raises.
- **`finalize()` cost model** documented as **O(graph size)** — call once per batch, never per file.

---

## [1.0.x] — 2026-04 → 2026-05

- 1.0.0 — first stable release. `GraphRAG` facade. `ingest`, `retrieve`, `completion`. LiteLLM providers. `MultiPathRetrieval`. PDF / Markdown / text loaders. `SentenceTokenCapChunking`, `FixedSizeChunking`. `ExactMatchResolution`, `SemanticResolution`, `LLMVerifiedResolution`.

---

## Migrating between versions

The legacy ontology vocabulary still works under v1.2 with deprecation warnings. To migrate cleanly:

```diff
-from graphrag_sdk import GraphSchema, EntityType, RelationType, PropertyType
+from graphrag_sdk import Ontology, Entity, Relation, Attribute

-schema = GraphSchema(
-    entities=[EntityType(label="Person")],
-    relations=[RelationType(label="WORKS_AT", patterns=[("Person", "Organization")])],
-)
+ontology = Ontology(
+    entities=[Entity(label="Person")],
+    relations=[Relation(label="WORKS_AT", patterns=[("Person", "Organization")])],
+)

-rag = GraphRAG(connection=..., llm=..., embedder=..., schema=schema)
+rag = GraphRAG(connection=..., llm=..., embedder=..., ontology=ontology)
```

No behavioural change — only the names.
