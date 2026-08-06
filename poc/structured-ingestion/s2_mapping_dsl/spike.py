"""s2 — which mapping DSL shape survives all four real record shapes?

Proposal #2 proposes:

    RecordMapping(nodes=[NodeMapping(...)], edges=[EdgeMapping(type=..., source="Person",
                                                               target="Organization")])

`source`/`target` address nodes **by label**. This spike executes candidate DSLs
against four record shapes taken from the fixture corpus and checks the edges
they actually produce, rather than arguing about readability:

  R1  orgs.csv          one row  -> one entity                  (the 80% case)
  R2  employees.csv     one row  -> two nodes + FK edge
  R3  transactions.csv  one row  -> a reified event with TWO edges to the SAME label
  R4  catalog.json      nested object -> child node + containment edge

R3 is the case that decides it: a transaction has a buyer *and* a seller, both
`Organization`.

It also checks the two things a generated ontology can silently get wrong:
reserved attribute names, and whether `Entity.identity` (proposal #3) can even
be carried by today's model.
"""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _harness.env import FIXTURES, Report  # noqa: E402

from graphrag_sdk.core.models import (  # noqa: E402
    _RESERVED_ATTRIBUTE_NAMES,
    Attribute,
    Entity,
    Ontology,
    Relation,
)

# ── Candidate A: proposal #2 verbatim — edges address nodes by LABEL ──


@dataclass
class NodeA:
    label: str
    key: str
    name: str | None = None
    properties: dict[str, str] = field(default_factory=dict)
    reference: bool = False


@dataclass
class EdgeA:
    type: str
    source: str  # a LABEL
    target: str  # a LABEL
    properties: dict[str, str] = field(default_factory=dict)


@dataclass
class MappingA:
    nodes: list[NodeA]
    edges: list[EdgeA] = field(default_factory=list)

    def apply(self, record: dict[str, Any]) -> tuple[list[tuple[str, str]], list[tuple]]:
        """-> ([(label, key_value)], [(rel_type, src_key, tgt_key)])"""
        built: list[tuple[str, str]] = []
        by_label: dict[str, list[str]] = {}
        for n in self.nodes:
            if n.key not in record:
                continue
            kv = str(record[n.key])
            built.append((n.label, kv))
            by_label.setdefault(n.label, []).append(kv)
        edges = []
        for e in self.edges:
            # The design gives us only a label to resolve with.
            src = by_label.get(e.source, [])
            tgt = by_label.get(e.target, [])
            if not src or not tgt:
                continue
            # Ambiguity is unresolvable here — take the first, which is the
            # bug this spike is looking for.
            edges.append((e.type, src[0], tgt[0]))
        return built, edges


# ── Candidate A': same, but nodes carry an ALIAS ─────────────────


@dataclass
class NodeB:
    alias: str
    label: str
    key: str
    name: str | None = None
    properties: dict[str, str] = field(default_factory=dict)
    reference: bool = False


@dataclass
class EdgeB:
    type: str
    source: str  # an ALIAS
    target: str  # an ALIAS
    properties: dict[str, str] = field(default_factory=dict)


@dataclass
class MappingB:
    nodes: list[NodeB]
    edges: list[EdgeB] = field(default_factory=list)

    def apply(self, record: dict[str, Any]) -> tuple[list[tuple[str, str]], list[tuple]]:
        built: list[tuple[str, str]] = []
        by_alias: dict[str, str] = {}
        for n in self.nodes:
            if n.key not in record:
                continue
            kv = str(record[n.key])
            built.append((n.label, kv))
            by_alias[n.alias] = kv
        edges = [
            (e.type, by_alias[e.source], by_alias[e.target])
            for e in self.edges
            if e.source in by_alias and e.target in by_alias
        ]
        return built, edges

    def to_ontology(self) -> Ontology:
        label_props: dict[str, dict[str, str]] = {}
        identity: dict[str, list[str]] = {}
        for n in self.nodes:
            if n.reference:
                continue
            props = label_props.setdefault(n.label, {})
            props.update(dict.fromkeys(n.properties, "STRING"))
            if n.name:
                props["name"] = "STRING"
            identity[n.label] = ["name"] if n.name else [n.key]
        entities = [
            Entity(
                label=lbl,
                properties=[Attribute(name=p, type=t) for p, t in props.items()],
                identity=identity[lbl],  # proposal #3 — does extra="allow" carry it?
            )
            for lbl, props in label_props.items()
        ]
        alias_label = {n.alias: n.label for n in self.nodes}
        relations = [
            Relation(
                label=e.type,
                patterns=[(alias_label[e.source], alias_label[e.target])],
                properties=[Attribute(name=p) for p in e.properties],
            )
            for e in self.edges
        ]
        return Ontology(entities=entities, relations=relations)


def read_csv(name: str) -> list[dict[str, Any]]:
    with open(FIXTURES / name, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def main() -> int:
    r = Report("s2 — mapping DSL shape")

    orgs = read_csv("orgs.csv")
    employees = read_csv("employees.csv")
    transactions = read_csv("transactions.csv")
    catalog = json.loads((FIXTURES / "catalog.json").read_text())["products"]

    # R1 — one row, one entity. Both shapes handle it.
    a1 = MappingA(nodes=[NodeA(label="Organization", key="org_id", name="org_name")])
    nodes, edges = a1.apply(orgs[0])
    r.check(nodes == [("Organization", "ORG-42")] and edges == [], "R1 orgs.csv — trivial for A")

    # R2 — two nodes + FK edge, all distinct labels.
    a2 = MappingA(
        nodes=[
            NodeA(label="Person", key="employee_id", name="full_name"),
            NodeA(label="Organization", key="org_id", reference=True),
        ],
        edges=[EdgeA(type="WORKS_AT", source="Person", target="Organization")],
    )
    _, edges = a2.apply(employees[0])
    r.check(edges == [("WORKS_AT", "E-1", "ORG-42")], "R2 employees.csv — A resolves the FK edge")

    # R3 — the decider. Two Organizations in one record.
    a3 = MappingA(
        nodes=[
            NodeA(label="Transaction", key="txn_id"),
            NodeA(label="Organization", key="buyer_org_id", reference=True),
            NodeA(label="Organization", key="seller_org_id", reference=True),
        ],
        edges=[
            EdgeA(type="BOUGHT_FROM", source="Organization", target="Organization"),
            EdgeA(type="INVOLVES_BUYER", source="Transaction", target="Organization"),
        ],
    )
    _, edges = a3.apply(transactions[0])
    want_buyer, want_seller = "ORG-7", "ORG-42"
    self_loop = any(e[1] == e[2] for e in edges)
    r.check(
        self_loop,
        "R3 transactions.csv — label-addressed edges COLLAPSE to a self-loop",
        f"produced {edges} · buyer={want_buyer} seller={want_seller}",
    )
    r.note("label is not a unique handle inside a record; the design has no way to say which one")

    b3 = MappingB(
        nodes=[
            NodeB(alias="txn", label="Transaction", key="txn_id"),
            NodeB(alias="buyer", label="Organization", key="buyer_org_id", reference=True),
            NodeB(alias="seller", label="Organization", key="seller_org_id", reference=True),
        ],
        edges=[
            EdgeB(type="BOUGHT_FROM", source="buyer", target="seller"),
            EdgeB(type="INVOLVES_BUYER", source="txn", target="buyer"),
        ],
    )
    _, edges = b3.apply(transactions[0])
    r.check(
        edges == [("BOUGHT_FROM", "ORG-7", "ORG-42"), ("INVOLVES_BUYER", "TXN-100", "ORG-7")],
        "R3 transactions.csv — alias-addressed edges resolve correctly",
        str(edges),
    )

    # R4 — nested JSON, after flattening. Same alias machinery, no new concept.
    flat = [
        {
            "sku": p["sku"],
            "name": p["name"],
            "sold_by.org_id": p["sold_by"]["org_id"],
            "tags": p["tags"],
        }
        for p in catalog
    ]
    b4 = MappingB(
        nodes=[
            NodeB(alias="prod", label="Product", key="sku", name="name"),
            NodeB(alias="vendor", label="Organization", key="sold_by.org_id", reference=True),
        ],
        edges=[EdgeB(type="SOLD_BY", source="prod", target="vendor")],
    )
    _, edges = b4.apply(flat[0])
    r.check(
        edges == [("SOLD_BY", "SKU-1", "ORG-42")],
        "R4 catalog.json — dotted flattening needs no new DSL concept",
    )

    # ── the generated ontology ───────────────────────────────────
    import logging

    class _Capture(logging.Handler):
        def __init__(self) -> None:
            super().__init__()
            self.messages: list[str] = []

        def emit(self, record: logging.LogRecord) -> None:
            self.messages.append(record.getMessage())

    cap = _Capture()
    logging.getLogger("graphrag_sdk.core.models").addHandler(cap)
    onto = b3.to_ontology()
    logging.getLogger("graphrag_sdk.core.models").removeHandler(cap)
    r.check(
        {e.label for e in onto.entities} == {"Transaction"},
        "to_ontology() emits only non-reference nodes",
        f"entities={[e.label for e in onto.entities]} "
        "(Organization is a reference, declared by orgs.csv)",
    )
    r.check(
        [rel.label for rel in onto.relations] == ["BOUGHT_FROM", "INVOLVES_BUYER"],
        "to_ontology() round-trips into a real Ontology with directional patterns",
    )
    r.check(
        any("not declared in ontology.entities" in m for m in cap.messages),
        "a reference-only label makes Ontology's own validator warn on every ingest",
        f"{len(cap.messages)} warning(s), e.g. "
        "'Organization ... not declared in ontology.entities'",
    )
    r.note(
        "=> to_ontology() must emit bare Entity stubs for reference labels, or be merged into "
        "the live ontology before validation runs — otherwise every structured ingest logs noise"
    )

    # proposal #3's new field, on today's model.
    ent = Entity(label="Product", identity=["sku"])
    carried = getattr(ent, "identity", None)
    r.check(
        carried == ["sku"],
        "Entity accepts an `identity` field today via Config.extra='allow'",
        "so #3 is prototypable with zero src changes, and old ontology.json still loads",
    )
    r.check(
        "identity" in Entity(label="P", identity=["sku"]).model_dump(),
        "`identity` survives model_dump(), so it persists to ontology.json",
    )
    r.note(
        "but it is untyped/unvalidated as an extra — the real change must add it as a "
        "declared field defaulting to ['name']"
    )

    # reserved-name collision: the trap a generated ontology walks straight into.
    bad = MappingB(
        nodes=[
            NodeB(
                alias="o",
                label="Organization",
                key="org_id",
                name="org_name",
                properties={"description": "hq_country", "id": "org_id"},
            )
        ]
    )
    declared = {a.name for e in bad.to_ontology().entities for a in e.properties}
    collisions = (declared & _RESERVED_ATTRIBUTE_NAMES) - {"name"}
    r.check(
        bool(collisions),
        "a mapping can silently declare SDK-reserved attributes",
        f"collides on {sorted(collisions)} — these shadow SDK-written values on every node",
    )
    r.note("=> to_ontology() must reject _RESERVED_ATTRIBUTE_NAMES minus _SDK_MANAGED ones")

    return r.verdict()


if __name__ == "__main__":
    raise SystemExit(main())
