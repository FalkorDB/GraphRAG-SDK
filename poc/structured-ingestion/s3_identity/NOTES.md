# s3 — entity identity · DECIDED (design correction)

**Question.** Proposal #3 defaults `Entity.identity` to `["name"]` and frames proposal #4's
`alias_ids` as an *optional* bridge for the minority of types whose identity is a business key.
Measured against a real FalkorDB, that is backwards.

**Run:** `python s3_identity/spike.py` (needs FalkorDB; no keys). All checks pass.

Three policies x two ingest orders over the #82 acceptance corpus:

| Policy | Acme nodes | #82 traversal | order-independent |
| --- | --- | --- | --- |
| P1 name-first (`identity=["name"]`, the design's default) | **2** | **0 people reachable** | yes |
| P2 key-only | **2** | **0 people reachable** | yes |
| P3 key + `alias_ids` + resolve pass | **1** | **2 people reachable** | yes |

## Why name-first loses: normalised FKs carry keys, not names

`employees.csv` is a perfectly ordinary normalised table:

```
employee_id,full_name,age,job_title,org_id,start_date
E-1,Alice Smith,34,Engineer,ORG-42,2019-04-01
```

It references an organisation by `org_id=ORG-42`. It does **not** contain `org_name`. So under
`identity=["name"]` the employees mapping *cannot compute the identity of the entity it points at* —
there is no name in the record to compute it from. Proposal #2's rule "each mapping must supply the
type's identity attributes" is unsatisfiable for any normalised foreign key, which is the single
most common structured-data shape there is.

The result is not an error. It is a stub node `org-42__organization` sitting next to the real
`acme_corp__organization`, with `WORKS_AT` attached to the stub — so the prose about Acme's Q3
revenue miss and the engineers who work there are in the same graph and **not connected**. That is
precisely the acceptance criterion #82 exists to test, silently failing.

P2 fails the mirror image: structured sources converge on `org-42__organization`, but LLM
extraction can only ever produce a name, so the prose entity is stranded.

## Decision

1. **Structured writes are key-identified.** `NodeMapping.key` produces the node id via
   `compute_entity_id(key_value, label)`. This is what makes FK stubs land on the right node
   regardless of ingest order — confirmed: both orders converge for all three policies.
2. **`alias_ids` is on the critical path, not optional.** Any mapping that has both the key and a
   name emits `alias_ids=[compute_entity_id(name, label)]`, and the resolve pass merges the
   name-identified node from unstructured extraction into it. This is the *only* configuration
   tested that yields one Acme node and a working traversal.
3. Resolution stays deterministic and index-backed — the spike's implementation is four Cypher
   statements per merged pair, no LLM and no embeddings, exactly as proposal #4 claims.

So proposal #3's headline should be inverted: identity is declared on the type, but its **default
for structured sources is the record key**, and #4 is what makes the graph connected rather than a
nicety for SKU-shaped types.

## The caveat, measured

P3 is not magic. With prose + `employees.csv` and **no** `orgs.csv`, the result degrades to
2 Acme nodes and 0 reachable people — the alias bridge has nothing to be built from, because no
source carried both `ORG-42` and `"Acme Corp"` in the same record.

The precise contract is therefore: **ingest order does not matter, but presence does.** An entity
type needs at least one source that declares it (key *and* name) for its FK stubs and its
unstructured mentions to converge. That is a reasonable requirement — it is just the dimension
table — but it must be stated in the design and surfaced at ingest time: if a mapping references a
label that no ingested source has ever declared, the run should report the count of unbridged
stubs rather than leave the user with a quietly disconnected graph.
