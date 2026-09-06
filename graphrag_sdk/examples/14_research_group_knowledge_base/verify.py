"""Check the graph against the CSVs — not against a model's opinion.

    python verify.py
    python verify.py --skip-pdfs     # after ingest.py --skip-pdfs

Every expected value is recomputed from ``data/tables`` here, then compared
with a Cypher query on the graph. What the checks establish:

  * every row is a node, no two rows were merged, blanks were omitted
  * typed columns are numbers and dates, so the graph can aggregate them
  * foreign keys became edges, including the drift-tolerant ones
  * a row and its prose mentions share one node
  * things that should stay apart stayed apart
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import statistics
from typing import Any

from common import TABLES, add_connection_args, build_rag, load_env

# (file, label, key column) for every declared table
ROW_TABLES = [
    ("papers.csv", "Paper", "paper_id"),
    ("mitigation_practices.csv", "MitigationPractice", "practice_id"),
    ("trade_shock_metrics.csv", "TradeMetric", "metric_id"),
    ("electricity_carbon_scenarios.csv", "PolicyScenario", "scenario_id"),
    ("institutions.csv", "Institution", "inst_id"),
    ("people.csv", "Person", "person_id"),
    ("paper_authors.csv", "Authorship", "authorship_id"),
    ("citations.csv", "Citation", "citation_id"),
    ("experiments.csv", "Experiment", "exp_id"),
    ("datasets.csv", "Dataset", "dataset_id"),
    ("lab_equipment.csv", "Equipment", "equipment_id"),
]


def rows(name: str) -> list[dict[str, str]]:
    with (TABLES / name).open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def mean(values: list[str]) -> float:
    return statistics.mean(float(v) for v in values if v.strip())


def truth() -> dict[str, Any]:
    """What the CSVs say. The graph has to agree with this."""
    practices, experiments = rows("mitigation_practices.csv"), rows("experiments.csv")
    people, authors = rows("people.csv"), rows("paper_authors.csv")
    baseline = [r for r in experiments if r["name"] == "baseline"]
    awd = [r for r in experiments if r["practice_id"] == "PR-AWD"]
    return {
        "row_counts": {label: len(rows(name)) for name, label, _ in ROW_TABLES},
        "rice_avg_ch4_max": mean(
            [r["ch4_reduction_max_pct"] for r in practices if r["crop_system"] == "rice"]
        ),
        "baseline_count": len(baseline),
        "baseline_avg_flux": mean([r["ch4_flux_kg_ha"] for r in baseline]),
        "awd_count": len(awd),
        "awd_avg_flux": mean([r["ch4_flux_kg_ha"] for r in awd]),
        "equipment_over_50k": sorted(
            r["equipment_id"] for r in rows("lab_equipment.csv") if float(r["cost_usd"]) > 50000
        ),
        "corresponding_true": sum(
            1 for r in authors if r["corresponding"].strip().lower() in ("true", "yes", "1")
        ),
        "h_index_blank": sum(1 for r in people if not r["h_index"].strip()),
        "survey_authors": sum(1 for r in authors if r["paper_id"].startswith("2603.20674")),
        "institutions": len(rows("institutions.csv")),
        "austria_institutions": sum(
            1 for r in rows("institutions.csv") if r["country"].strip().lower() == "austria"
        ),
    }


class Verifier:
    def __init__(self, rag) -> None:
        self.rag = rag
        self.passed = self.failed = 0

    async def value(self, cypher: str) -> Any:
        result = await self.rag.query(cypher)
        return result[0][0] if result and result[0] else None

    async def check(self, name: str, cypher: str, expected: Any, *, approx: bool = False) -> None:
        actual = await self.value(cypher)
        ok = (
            (actual is not None and abs(float(actual) - float(expected)) < 1e-6)
            if approx
            else actual == expected
        )
        self.passed += ok
        self.failed += not ok
        print(
            f"  {'PASS' if ok else 'FAIL'}  {name}: {actual!r}"
            + ("" if ok else f"  (expected {expected!r})")
        )

    async def show(self, name: str, cypher: str) -> None:
        result = await self.rag.query(cypher)
        print(f"  info  {name}: {[list(r) for r in result]}")


async def verify(rag, t: dict[str, Any], *, pdfs: bool) -> tuple[int, int]:
    v = Verifier(rag)

    print("\n-- rows are rows")
    for file, label, _ in ROW_TABLES:
        # Counted through the file's record chunks, not by label alone: a label can
        # also be picked for an undeclared table by the model (funding.csv, below).
        await v.check(
            f"{label} rows from {file}",
            f"MATCH (:Document {{id: '{file}'}})-[:PART_OF]->(:Chunk {{kind: 'record'}})"
            f"<-[:MENTIONED_IN]-(n:{label}) WHERE n.is_stub = false RETURN count(DISTINCT n)",
            t["row_counts"][label],
        )
    await v.check(
        "one record chunk per row",
        "MATCH (c:Chunk {kind: 'record'}) RETURN count(c)",
        sum(t["row_counts"].values()) + 5,
    )  # + the 5 undeclared funding.csv rows
    await v.check(
        "'baseline' x51 stayed separate nodes",
        "MATCH (e:Experiment) WHERE e.name = 'baseline' AND e.is_stub = false RETURN count(e)",
        t["baseline_count"],
    )
    await v.check(
        "no DISTINCT_FROM edge between two rows",
        "MATCH (a)-[:DISTINCT_FROM]-(b) WHERE a.is_stub = false AND b.is_stub = false "
        "RETURN count(*)",
        0,
    )

    print("\n-- typed columns aggregate")
    await v.check(
        "avg CH4 max over rice practices",
        "MATCH (n:MitigationPractice) WHERE n.mitigation_practices__crop_system = 'rice' "
        "RETURN avg(n.mitigation_practices__ch4_reduction_max_pct)",
        t["rice_avg_ch4_max"],
        approx=True,
    )
    await v.check(
        "avg flux of the 'baseline' experiments",
        "MATCH (e:Experiment) WHERE e.name = 'baseline' RETURN avg(e.experiments__ch4_flux_kg_ha)",
        t["baseline_avg_flux"],
        approx=True,
    )
    await v.check(
        "equipment over 50k USD (FLOAT filter)",
        "MATCH (e:Equipment) WHERE e.lab_equipment__cost_usd > 50000 "
        "WITH e ORDER BY e.entity_key RETURN collect(e.entity_key)",
        t["equipment_over_50k"],
    )
    await v.check(
        "start_date cells are ISO dates",
        "MATCH (e:Experiment) WHERE e.is_stub = false AND (size(e.experiments__start_date) <> 10 "
        "OR substring(e.experiments__start_date, 4, 1) <> '-') RETURN count(e)",
        0,
    )
    await v.check(
        "'yes' / 'true' became BOOLEAN true",
        "MATCH (a:Authorship) WHERE a.paper_authors__corresponding = true RETURN count(a)",
        t["corresponding_true"],
    )
    await v.check(
        "blank h_index cells left no property",
        "MATCH (p:Person) WHERE p.is_stub = false AND p.people__h_index IS NULL RETURN count(p)",
        t["h_index_blank"],
    )

    print("\n-- foreign keys became edges")
    await v.check(
        "experiments -TESTS-> the AWD row",
        "MATCH (e:Experiment)-[:RELATES {rel_type: 'TESTS'}]->"
        "(p:MitigationPractice {entity_key: 'PR-AWD'}) "
        "RETURN count(e)",
        t["awd_count"],
    )
    await v.check(
        "avg flux of those experiments",
        "MATCH (e:Experiment)-[:RELATES {rel_type: 'TESTS'}]->"
        "(:MitigationPractice {entity_key: 'PR-AWD'}) "
        "RETURN avg(e.experiments__ch4_flux_kg_ha)",
        t["awd_avg_flux"],
        approx=True,
    )
    await v.check(
        "survey authors reachable in two hops (junction table)",
        "MATCH (p:Paper {entity_key: '2603.20674v1.pdf'})"
        "<-[:RELATES {rel_type: 'OF'}]-(:Authorship)"
        "-[:RELATES {rel_type: 'BY'}]->(person:Person) RETURN count(DISTINCT person)",
        t["survey_authors"],
    )
    await v.check(
        "'austria' / 'Austria ' / 'Austria' are one Country node",
        "MATCH (c:Country) WHERE toLower(trim(c.name)) = 'austria' RETURN count(c)",
        1,
    )
    await v.check(
        "both Austrian institutions reach it",
        "MATCH (i:Institution)-[:RELATES {rel_type: 'LOCATED_IN'}]->(c:Country) "
        "WHERE toLower(trim(c.name)) = 'austria' RETURN count(i)",
        t["austria_institutions"],
    )
    await v.check(
        "every institution reaches a country",
        "MATCH (i:Institution)-[:RELATES {rel_type: 'LOCATED_IN'}]->(:Country) "
        "WHERE i.is_stub = false "
        "RETURN count(DISTINCT i)",
        t["institutions"],
    )
    await v.check(
        "dangling FK INST-99 is one placeholder with two people",
        "MATCH (p:Person)-[:RELATES {rel_type: 'AFFILIATED_WITH'}]->"
        "(i:Institution {entity_key: 'INST-99'}) "
        "WHERE i.is_stub = true RETURN count(p)",
        2,
    )

    print("\n-- rows and prose share nodes")
    await v.check(
        "'V, Priyanka' (P-001) is one node mentioned outside her own row",
        "MATCH (p:Person {entity_key: 'P-001'})-[:MENTIONED_IN]->(c:Chunk) "
        "WHERE c.kind IS NULL OR c.kind <> 'record' RETURN count(c) >= 1",
        True,
    )
    await v.check(
        "no second Person node is called Priyanka",
        "MATCH (p:Person) WHERE toLower(p.name) CONTAINS 'priyanka' RETURN count(p)",
        1,
    )
    await v.show(
        "the equipment rows the notes mention",
        "MATCH (e:Equipment)-[:MENTIONED_IN]->(c:Chunk) WHERE e.is_stub = false "
        "AND (c.kind IS NULL OR c.kind <> 'record') RETURN DISTINCT e.name",
    )
    if pdfs:
        await v.check(
            "AWD row is mentioned in the survey's prose",
            "MATCH (p:MitigationPractice {entity_key: 'PR-AWD'})-[:MENTIONED_IN]->(c:Chunk) "
            "WHERE c.kind IS NULL RETURN count(c) >= 5",
            True,
        )
        await v.check(
            "many entities are on both sides",
            "MATCH (e:__Entity__)-[:MENTIONED_IN]->(r:Chunk {kind: 'record'}) "
            "WITH DISTINCT e MATCH (e)-[:MENTIONED_IN]->(c:Chunk) WHERE c.kind IS NULL "
            "RETURN count(DISTINCT e) >= 40",
            True,
        )
        await v.check(
            "the unrelated paper mentions no practice / experiment / scenario / metric row",
            "MATCH (d:Document)-[:PART_OF]->(c:Chunk)<-[:MENTIONED_IN]-(e) "
            "WHERE d.id ENDS WITH '2606.01394v1.pdf' AND e.is_stub = false "
            "AND (e:MitigationPractice OR e:Experiment OR e:PolicyScenario OR e:TradeMetric) "
            "RETURN count(DISTINCT e)",
            0,
        )

    print("\n-- what the data could not join (expected)")
    await v.show(
        "Paper placeholders (arXiv ids, out-of-corpus targets, the ';' cell, the stale row)",
        "MATCH (p:Paper) WHERE p.is_stub = true RETURN p.entity_key ORDER BY p.entity_key",
    )
    await v.show(
        "what funding.csv became (undeclared: label and links chosen by the model, varies by run)",
        "MATCH (:Document {id: 'funding.csv'})-[:PART_OF]->(:Chunk {kind: 'record'})"
        "<-[:MENTIONED_IN]-(g {entity_key: 'GR-2023-01'}) MATCH (g)-[r:RELATES]->(x) "
        "RETURN [l IN labels(g) WHERE l <> '__Entity__'][0], g.name, r.rel_type, "
        "[l IN labels(x) WHERE l <> '__Entity__'][0], x.name ORDER BY r.rel_type",
    )
    await v.show(
        "what compute_usage.json became (not a tabular suffix)",
        "MATCH (d:Document)-[:PART_OF]->(c:Chunk)<-[:MENTIONED_IN]-(e) "
        "WHERE d.id ENDS WITH 'compute_usage.json' "
        "RETURN [l IN labels(e) WHERE l <> '__Entity__'][0], e.name ORDER BY e.name",
    )
    return v.passed, v.failed


async def main(args: argparse.Namespace) -> int:
    rag = build_rag(args)
    try:
        passed, failed = await verify(rag, truth(), pdfs=not args.skip_pdfs)
    finally:
        await rag.close()
    print(f"\n== {passed}/{passed + failed} checks passed")
    return 1 if failed else 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_connection_args(p)
    p.add_argument("--skip-pdfs", action="store_true", help="the graph was built without the PDFs")
    return p.parse_args()


if __name__ == "__main__":
    load_env()
    raise SystemExit(asyncio.run(main(parse_args())))
