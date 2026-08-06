# s2 — mapping DSL shape · DECIDED

**Question.** Proposal #2's `EdgeMapping(type=..., source="Person", target="Organization")`
addresses the record's nodes **by label**. Does that survive real record shapes?

**Run:** `python s2_mapping_dsl/spike.py` (no DB, no keys). All checks pass.

Four shapes from the fixture corpus, executed rather than argued about:

| | Record shape | Label-addressed (design as written) |
| --- | --- | --- |
| R1 | `orgs.csv` — one row, one entity | works |
| R2 | `employees.csv` — two nodes + FK edge | works |
| R3 | `transactions.csv` — reified event, **two `Organization`s in one record** | **broken** |
| R4 | `catalog.json` — nested object | works after dotted flattening |

## The defect: label is not a unique handle inside a record

A transaction has a buyer and a seller, both `Organization`. With only a label to resolve by, the
edge resolver has no way to pick — and produces a **self-loop**:

```
want:     BOUGHT_FROM  ORG-7 -> ORG-42
produced: BOUGHT_FROM  ORG-7 -> ORG-7
```

Silently wrong, not an error. And this is not an exotic case — buyer/seller, manager/report,
parent/subsidiary, origin/destination are the standard shape of any transactional or hierarchical
table, which is most of what "structured data" means in practice.

## Decision: nodes get an `alias`; edges address aliases

```python
RecordMapping(
    nodes=[
        NodeMapping(alias="txn",    label="Transaction",  key="txn_id"),
        NodeMapping(alias="buyer",  label="Organization", key="buyer_org_id", reference=True),
        NodeMapping(alias="seller", label="Organization", key="seller_org_id", reference=True),
    ],
    edges=[EdgeMapping(type="BOUGHT_FROM", source="buyer", target="seller")],
)
```

Verified: produces `BOUGHT_FROM ORG-7 -> ORG-42` and `INVOLVES_BUYER TXN-100 -> ORG-7` correctly.
`alias` defaults to the label, so the 80% single-node case in proposal #2 is unchanged and nobody
writing `orgs.csv` ever types an alias.

R4 also confirms nested JSON needs **no new DSL concept** — flatten to `sold_by.org_id` and the
same alias machinery applies. Nested containment does not need to be its own third edge kind.

## Two more traps found in `to_ontology()`

**Reserved attribute names.** A mapping declaring `properties={"description": ..., "id": ...}`
generates an ontology that shadows SDK-written values on every node. `_RESERVED_ATTRIBUTE_NAMES`
in `core/models.py` lists ten such keys. `to_ontology()` must **reject**
`_RESERVED_ATTRIBUTE_NAMES - _SDK_MANAGED_ATTRIBUTE_NAMES`, naming the offending `Label.attribute` —
this is the same "reject before any write" rule the design already applies to contradictions.

**Reference-only labels warn.** `Ontology`'s own `_warn_on_undeclared_pattern_labels` validator
fires when a relation pattern names a label not in `entities` — which is exactly what an FK
reference produces. The spike captured 2 warnings from one mapping. `to_ontology()` must emit bare
`Entity` stubs for reference labels, or be merged into the live ontology before validation, or
every structured ingest logs noise that trains users to ignore real warnings.

## Free win for proposal #3

`Entity(label="Product", identity=["sku"])` **already works** — `DataModel.Config.extra = "allow"`
carries it, and it survives `model_dump()`, so it persists to `ontology.json`. Two consequences:
`identity` is prototypable with zero `src` changes, and adding it as a real field later will not
break ontologies persisted in the meantime. It must still become a *declared* field defaulting to
`["name"]` so it is validated rather than a silent typo sink.
