"""s4 — is record-as-chunk really free, and is the cutover trap real?

Proposal #5 persists each structured record as a `Chunk` with a **deterministic**
uid (replacing today's `uuid4()`), and claims that this makes `update()` and
`delete_document()` work unchanged.

The design review predicted a data-loss trap in that claim, from reading
`api/main.py:2104` + `graph_store.rollforward_cutover()`. A prediction from
reading code is a hypothesis. This spike runs it against a real FalkorDB using
the **real `GraphStore`**, and either confirms it or kills it.

  A  chunk uid keyed on the CANONICAL document id  -> predicted: data loss
  B  chunk uid keyed on the run's EFFECTIVE document id -> predicted: safe

It then checks the three cleanup behaviours proposal #5 depends on:
`get_document_entity_candidates`, `delete_orphan_entities`,
`delete_stale_relationships`.
"""

from __future__ import annotations

import asyncio
import csv
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _harness.env import FIXTURES, Report, connection, falkor_available, reset_graph  # noqa: E402

from graphrag_sdk.core.models import GraphNode, GraphRelationship  # noqa: E402
from graphrag_sdk.ingestion.extraction_strategies.entity_extractors import (  # noqa: E402
    compute_entity_id,
)
from graphrag_sdk.storage.graph_store import GraphStore  # noqa: E402

DOC_ID = "employees.csv"


def record_chunk_uid(document_id: str, record_key: str) -> str:
    """Proposal #5's deterministic chunk uid."""
    return hashlib.sha256(f"{document_id}::{record_key}".encode()).hexdigest()[:32]


def rows(name: str = "employees.csv") -> list[dict[str, str]]:
    with open(FIXTURES / name, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


async def write_run(
    store: GraphStore,
    *,
    effective_doc_id: str,
    chunk_key_doc_id: str,
    records: list[dict[str, str]],
) -> list[str]:
    """One structured ingestion run, exactly as proposal #6 step 3+8+9 would.

    ``effective_doc_id`` is the Document node actually written (the pending id
    during an update). ``chunk_key_doc_id`` is what the record chunk uid is
    derived from — the whole point of the experiment.
    """
    await store.upsert_nodes(
        [GraphNode(id=effective_doc_id, label="Document", properties={"path": DOC_ID})]
    )
    chunk_ids: list[str] = []
    nodes: list[GraphNode] = []
    rels: list[GraphRelationship] = []
    for row in records:
        uid = record_chunk_uid(chunk_key_doc_id, row["employee_id"])
        chunk_ids.append(uid)
        text = f"{row['full_name']} · age {row['age']} · {row['job_title']} at {row['org_id']}"
        nodes.append(GraphNode(id=uid, label="Chunk", properties={"text": text, "kind": "record"}))
        pid = compute_entity_id(row["employee_id"], "Person")
        oid = compute_entity_id(row["org_id"], "Organization")
        nodes += [
            GraphNode(
                id=pid,
                label="Person",
                properties={"name": row["full_name"], "title": row["job_title"]},
            ),
            GraphNode(id=oid, label="Organization", properties={"name": row["org_id"]}),
        ]
        rels += [
            GraphRelationship(start_node_id=effective_doc_id, end_node_id=uid, type="PART_OF"),
            GraphRelationship(start_node_id=pid, end_node_id=uid, type="MENTIONED_IN"),
            GraphRelationship(start_node_id=oid, end_node_id=uid, type="MENTIONED_IN"),
            GraphRelationship(
                start_node_id=pid,
                end_node_id=oid,
                type="RELATES",
                properties={
                    "rel_type": "WORKS_AT",
                    "fact": f"({row['full_name']}, WORKS_AT, {row['org_id']})",
                    "source_chunk_ids": [uid],
                    "src_name": row["full_name"],
                    "tgt_name": row["org_id"],
                },
            ),
        ]
    await store.upsert_nodes(nodes)
    await store.upsert_relationships(rels)
    return chunk_ids


async def count_chunks(store: GraphStore, document_id: str) -> int:
    res = await store.query_raw(
        "MATCH (:Document {id:$id})-[:PART_OF]->(c:Chunk) RETURN count(c) AS n",
        {"id": document_id},
    )
    return res.result_set[0][0] if res.result_set else 0


async def cutover_scenario(*, key_on_canonical: bool) -> dict[str, int]:
    """v1 ingest -> update() writes a pending -> rollforward_cutover -> measure."""
    tag = "canonical" if key_on_canonical else "effective"
    conn = connection(f"poc_s4_cutover_{tag}")
    store = GraphStore(conn)
    await reset_graph(conn)

    v1 = rows()
    await write_run(store, effective_doc_id=DOC_ID, chunk_key_doc_id=DOC_ID, records=v1)
    before = await count_chunks(store, DOC_ID)

    # update(): api/main.py builds pending_id = f"{resolved_id}__pending__{uuid4().hex[:8]}"
    pending_id = f"{DOC_ID}__pending__ab12cd34"
    v2 = [dict(r) for r in v1]
    v2[0]["job_title"] = "Staff Engineer"  # a real edit
    await write_run(
        store,
        effective_doc_id=pending_id,
        chunk_key_doc_id=DOC_ID if key_on_canonical else pending_id,
        records=v2,
    )
    shared = await store.query_raw(
        "MATCH (:Document {id:$a})-[:PART_OF]->(c:Chunk)<-[:PART_OF]-(:Document {id:$b}) "
        "RETURN count(c) AS n",
        {"a": DOC_ID, "b": pending_id},
    )
    shared_chunks = shared.result_set[0][0] if shared.result_set else 0

    await store.rollforward_cutover(pending_id, DOC_ID, DOC_ID, "hash-v2")
    after = await count_chunks(store, DOC_ID)
    await conn.close()
    return {"before": before, "shared_with_pending": shared_chunks, "after_cutover": after}


async def cleanup_scenario() -> dict[str, int]:
    """Do the three cleanup primitives behave as proposal #5 assumes?"""
    conn = connection("poc_s4_cleanup")
    store = GraphStore(conn)
    await reset_graph(conn)

    v1 = rows()
    old_chunks = await write_run(
        store, effective_doc_id=DOC_ID, chunk_key_doc_id=DOC_ID, records=v1
    )
    candidates = await store.get_document_entity_candidates(DOC_ID)

    # Carol (E-3) is deleted from the source file; her org ORG-7 loses its only mention.
    survivors = [r for r in v1 if r["employee_id"] != "E-3"]
    removed_chunk = record_chunk_uid(DOC_ID, "E-3")
    await store.query_raw("MATCH (c:Chunk {id:$id}) DETACH DELETE c", {"id": removed_chunk})
    await write_run(store, effective_doc_id=DOC_ID, chunk_key_doc_id=DOC_ID, records=survivors)

    stale = await store.delete_stale_relationships(candidates, [removed_chunk])
    orphans = await store.delete_orphan_entities(candidates)
    remaining_people = await store.query_raw("MATCH (p:Person) RETURN count(p) AS n")
    remaining_edges = await store.query_raw("MATCH ()-[r:RELATES]->() RETURN count(r) AS n")
    await conn.close()
    return {
        "candidates": len(candidates),
        "old_chunks": len(old_chunks),
        "stale_edges_deleted": stale,
        "orphans_deleted": orphans,
        "people_left": remaining_people.result_set[0][0],
        "relates_left": remaining_edges.result_set[0][0],
    }


async def main() -> int:
    r = Report("s4 — record-as-chunk & the update() cutover")
    if not falkor_available():
        r.note("SKIPPED — no FalkorDB on FALKOR_HOST:FALKOR_PORT")
        return 0

    canonical = await cutover_scenario(key_on_canonical=True)
    effective = await cutover_scenario(key_on_canonical=False)
    r.note(f"chunk uid keyed on CANONICAL doc id: {canonical}")
    r.note(f"chunk uid keyed on EFFECTIVE doc id: {effective}")

    r.check(
        canonical["shared_with_pending"] == canonical["before"] > 0,
        "canonical keying makes the pending run MERGE onto the LIVE document's chunks",
        f"{canonical['shared_with_pending']} chunk nodes shared by both Documents",
    )
    r.check(
        canonical["after_cutover"] == 0,
        "CONFIRMED: the predicted data-loss trap is real",
        f"{canonical['before']} chunks before update, {canonical['after_cutover']} after cutover"
        " — every record chunk destroyed, no error raised",
    )
    r.check(
        effective["shared_with_pending"] == 0,
        "effective keying keeps the two runs' chunk sets disjoint",
    )
    r.check(
        effective["after_cutover"] == effective["before"] > 0,
        "effective keying survives the cutover intact",
        f"{effective['after_cutover']} chunks promoted",
    )

    cleanup = await cleanup_scenario()
    r.note(f"cleanup: {cleanup}")
    r.check(
        cleanup["candidates"] > 0,
        "get_document_entity_candidates() sees record-chunk entities",
        f"{cleanup['candidates']} candidates via "
        "(:__Entity__)-[:MENTIONED_IN]->(:Chunk)<-[:PART_OF]-(:Document)",
    )
    r.check(
        cleanup["stale_edges_deleted"] == 1,
        "delete_stale_relationships() GCs the removed row's fact",
        f"{cleanup['stale_edges_deleted']} edge(s) deleted via source_chunk_ids",
    )
    r.check(
        cleanup["orphans_deleted"] == 2 and cleanup["people_left"] == 2,
        "delete_orphan_entities() removes exactly the vanished row's entities",
        f"{cleanup['orphans_deleted']} orphans (Carol + ORG-7), "
        f"{cleanup['people_left']} people left",
    )
    r.note(
        "=> proposal #5's claim holds — record-as-chunk inherits orphan cleanup unchanged, "
        "PROVIDED chunk uids are keyed on the run's effective document id"
    )
    return r.verdict()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
