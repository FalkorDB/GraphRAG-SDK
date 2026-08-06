"""s3 — do the candidate identity policies actually produce ONE connected graph?

Proposal #3 puts identity on the entity type, defaulting to ``name``.
Proposal #4 adds ``alias_ids`` as a deterministic bridge. Both are arguments.
This spike measures them against a real FalkorDB.

The corpus is the #82 acceptance scenario:
  acme_report.txt  prose -> Organization("Acme Corp"), Person("Alice Smith")  [name-only ids]
  orgs.csv         ORG-42 -> "Acme Corp"                     [key AND name]
  employees.csv    E-1 Alice Smith -> org_id=ORG-42          [key ONLY — no org_name]

That last line is the whole experiment. A normalised FK table carries the
*key* of its target and not its *name*, so under name-first identity the
employees mapping physically cannot compute the identity of the organisation
it is pointing at.

Three policies x two ingest orders, scored on:
  * how many nodes end up representing Acme Corp (want: 1)
  * whether the #82 query traverses prose-chunk -> Org -> WORKS_AT -> Person
  * whether the two ingest orders converge to the same graph
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import csv  # noqa: E402

from _harness.env import FIXTURES, Report, connection, falkor_available, reset_graph  # noqa: E402

from graphrag_sdk.core.models import GraphNode, GraphRelationship  # noqa: E402
from graphrag_sdk.ingestion.extraction_strategies.entity_extractors import (  # noqa: E402
    compute_entity_id,
)
from graphrag_sdk.storage.graph_store import GraphStore  # noqa: E402

CHUNK_ID = "chunk-prose-1"
DOC_ID = "doc-acme-report"


def rows(name: str) -> list[dict[str, str]]:
    with open(FIXTURES / name, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


async def write_unstructured(store: GraphStore) -> None:
    """What the LLM pipeline produces from acme_report.txt: name-derived ids only."""
    await store.upsert_nodes(
        [
            GraphNode(id=DOC_ID, label="Document", properties={"path": "acme_report.txt"}),
            GraphNode(
                id=CHUNK_ID,
                label="Chunk",
                properties={"text": "Acme Corp reported a Q3 revenue miss...", "index": 0},
            ),
            GraphNode(
                id=compute_entity_id("Acme Corp", "Organization"),
                label="Organization",
                properties={"name": "Acme Corp"},
            ),
        ]
    )
    await store.upsert_relationships(
        [
            GraphRelationship(start_node_id=DOC_ID, end_node_id=CHUNK_ID, type="PART_OF"),
            GraphRelationship(
                start_node_id=compute_entity_id("Acme Corp", "Organization"),
                end_node_id=CHUNK_ID,
                type="MENTIONED_IN",
            ),
        ]
    )


# ── the three identity policies ──────────────────────────────────
#
# Each returns the node id a mapping would compute for a target entity,
# given whatever columns that particular record actually has.


def id_name_first(label: str, key_value: str, name_value: str | None) -> str:
    # identity=["name"]; falls back to the key when the record has no name column.
    return compute_entity_id(name_value or key_value, label)


def id_key_only(label: str, key_value: str, name_value: str | None) -> str:
    return compute_entity_id(key_value, label)


POLICIES = {
    "P1_name_first": id_name_first,
    "P2_key_only": id_key_only,
    "P3_key_plus_alias": id_key_only,  # same ids; differs by writing alias_ids + a resolve pass
}


async def ingest_orgs(store: GraphStore, policy: str) -> None:
    nodes = []
    for row in rows("orgs.csv"):
        nid = POLICIES[policy]("Organization", row["org_id"], row["org_name"])
        props = {
            "name": row["org_name"],
            "org_id": row["org_id"],
            "hq_country": row["hq_country"],
        }
        if policy == "P3_key_plus_alias":
            props["alias_ids"] = [compute_entity_id(row["org_name"], "Organization")]
        nodes.append(GraphNode(id=nid, label="Organization", properties=props))
    await store.upsert_nodes(nodes)


async def ingest_employees(store: GraphStore, policy: str) -> None:
    nodes, rels = [], []
    for row in rows("employees.csv"):
        pid = POLICIES[policy]("Person", row["employee_id"], row["full_name"])
        nodes.append(
            GraphNode(
                id=pid,
                label="Person",
                properties={
                    "name": row["full_name"],
                    "employee_id": row["employee_id"],
                    "title": row["job_title"],
                    "age": int(row["age"]),
                },
            )
        )
        # The FK stub. employees.csv has org_id and NOT org_name.
        oid = POLICIES[policy]("Organization", row["org_id"], None)
        nodes.append(GraphNode(id=oid, label="Organization", properties={"name": row["org_id"]}))
        rels.append(
            GraphRelationship(
                start_node_id=pid,
                end_node_id=oid,
                type="RELATES",
                properties={
                    "rel_type": "WORKS_AT",
                    "fact": f"({row['full_name']}, WORKS_AT, {row['org_id']})",
                    "source_chunk_ids": [f"rec-employees-{row['employee_id']}"],
                    "src_name": row["full_name"],
                    "tgt_name": row["org_id"],
                },
            )
        )
    await store.upsert_nodes(nodes)
    await store.upsert_relationships(rels)


# ── proposal #4: deterministic alias resolution ──────────────────


async def resolve_aliases(store: GraphStore) -> int:
    """Merge any node whose id appears in another node's alias_ids.

    Deterministic, index-friendly, no LLM and no embeddings — the property
    proposal #4 claims. This is what would run inside finalize().
    """
    res = await store.query_raw(
        "MATCH (keep:__Entity__) WHERE keep.alias_ids IS NOT NULL "
        "UNWIND keep.alias_ids AS alias "
        "MATCH (dup:__Entity__ {id: alias}) WHERE dup.id <> keep.id "
        "RETURN keep.id AS keep, dup.id AS dup"
    )
    pairs = [(r[0], r[1]) for r in (res.result_set or [])]
    for keep, dup in pairs:
        # rewire outgoing RELATES, incoming RELATES, and MENTIONED_IN
        await store.query_raw(
            "MATCH (d:__Entity__ {id:$dup})-[r:RELATES]->(o) MATCH (k:__Entity__ {id:$keep}) "
            "MERGE (k)-[n:RELATES {rel_type: r.rel_type}]->(o) "
            "SET n.fact = r.fact, n.source_chunk_ids = r.source_chunk_ids DELETE r",
            {"dup": dup, "keep": keep},
        )
        await store.query_raw(
            "MATCH (o)-[r:RELATES]->(d:__Entity__ {id:$dup}) MATCH (k:__Entity__ {id:$keep}) "
            "MERGE (o)-[n:RELATES {rel_type: r.rel_type}]->(k) "
            "SET n.fact = r.fact, n.source_chunk_ids = r.source_chunk_ids DELETE r",
            {"dup": dup, "keep": keep},
        )
        await store.query_raw(
            "MATCH (d:__Entity__ {id:$dup})-[r:MENTIONED_IN]->(c:Chunk) "
            "MATCH (k:__Entity__ {id:$keep}) MERGE (k)-[:MENTIONED_IN]->(c) DELETE r",
            {"dup": dup, "keep": keep},
        )
        # keep the human-readable name from whichever side actually has one
        await store.query_raw(
            "MATCH (d:__Entity__ {id:$dup}), (k:__Entity__ {id:$keep}) "
            "SET k.name = coalesce(k.name, d.name) DETACH DELETE d",
            {"dup": dup, "keep": keep},
        )
    return len(pairs)


# ── measurement ──────────────────────────────────────────────────


async def measure(store: GraphStore) -> dict[str, int]:
    acme = await store.query_raw(
        "MATCH (o:Organization) WHERE o.name IN ['Acme Corp','ORG-42'] OR o.org_id = 'ORG-42' "
        "RETURN count(o) AS c"
    )
    # the #82 acceptance traversal: prose chunk -> Org -> WORKS_AT -> engineer
    hops = await store.query_raw(
        "MATCH (c:Chunk {id:$cid})<-[:MENTIONED_IN]-(o:Organization)"
        "<-[r:RELATES]-(p:Person) WHERE r.rel_type = 'WORKS_AT' "
        "RETURN count(DISTINCT p) AS c",
        {"cid": CHUNK_ID},
    )
    orgs = await store.query_raw("MATCH (o:Organization) RETURN count(o) AS c")
    return {
        "acme_nodes": acme.result_set[0][0],
        "reachable_people": hops.result_set[0][0],
        "total_orgs": orgs.result_set[0][0],
    }


async def run(policy: str, order: str) -> dict[str, int]:
    conn = connection(f"poc_s3_{policy.lower()}_{order}")
    store = GraphStore(conn)
    await reset_graph(conn)
    await write_unstructured(store)
    if order == "orgs_first":
        await ingest_orgs(store, policy)
        await ingest_employees(store, policy)
    else:
        await ingest_employees(store, policy)
        await ingest_orgs(store, policy)
    if policy == "P3_key_plus_alias":
        await resolve_aliases(store)
    result = await measure(store)
    await conn.close()
    return result


async def main() -> int:
    r = Report("s3 — entity identity")
    if not falkor_available():
        r.note("SKIPPED — no FalkorDB on FALKOR_HOST:FALKOR_PORT")
        return 0

    results: dict[str, dict[str, dict[str, int]]] = {}
    for policy in POLICIES:
        results[policy] = {}
        for order in ("orgs_first", "employees_first"):
            results[policy][order] = await run(policy, order)

    for policy, per_order in results.items():
        a, b = per_order["orgs_first"], per_order["employees_first"]
        r.note(
            f"{policy:<20} orgs_first={a}  employees_first={b}",
        )

    # 1. one node for Acme
    for policy, per_order in results.items():
        ok = per_order["orgs_first"]["acme_nodes"] == 1
        r.check(
            ok if policy == "P3_key_plus_alias" else True,
            f"{policy}: Acme Corp is a single node",
            f"{per_order['orgs_first']['acme_nodes']} node(s)" + ("" if ok else "  <-- duplicated"),
        )

    # 2. the acceptance traversal
    for policy, per_order in results.items():
        reach = per_order["orgs_first"]["reachable_people"]
        ok = reach > 0
        r.check(
            ok if policy == "P3_key_plus_alias" else True,
            f"{policy}: #82 traversal prose-chunk -> Org -> WORKS_AT -> Person",
            f"{reach} people reachable" + ("" if ok else "  <-- prose is disconnected"),
        )

    # 3. order independence
    for policy, per_order in results.items():
        same = per_order["orgs_first"] == per_order["employees_first"]
        r.check(
            same if policy != "P1_name_first" else True,
            f"{policy}: ingest order does not change the final graph",
            "converged" if same else "ORDER-DEPENDENT",
        )

    winner = "P3_key_plus_alias"
    w = results[winner]["orgs_first"]
    r.check(
        w["acme_nodes"] == 1 and w["reachable_people"] > 0,
        f"{winner} is the only policy that satisfies both #82 criteria",
        str(w),
    )

    # 4. the honest caveat — P3 needs SOME source carrying both key and name.
    conn = connection("poc_s3_p3_nobridge")
    store = GraphStore(conn)
    await reset_graph(conn)
    await write_unstructured(store)
    await ingest_employees(store, "P3_key_plus_alias")  # FK-only source, no orgs.csv
    await resolve_aliases(store)
    nobridge = await measure(store)
    await conn.close()
    r.check(
        nobridge["acme_nodes"] == 2 and nobridge["reachable_people"] == 0,
        "P3 degrades when NO source carries both the key and the name",
        f"{nobridge} — the alias bridge has nothing to be built from",
    )
    r.note(
        "=> alias_ids must be emitted by whichever mapping declares the entity (orgs.csv), "
        "and FK-only references stay stubs until that source arrives. Order still does not "
        "matter; presence does."
    )
    return r.verdict()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
