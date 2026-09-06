"""Files that should not load cleanly — what does the SDK do with each?

    python dirty_files.py

Each file in ``data/dirty`` is a copy of one of the real tables with a single
fault a real export would have: a duplicate key, "13%" in a numeric column, a
day-first date, an empty key, a renamed column. Each is loaded alone on a
throwaway graph (``<graph-name>_dirty``) under the same declaration as its
clean twin, and the script records what happened: refused with a message that
names the cell, loaded with a warning, or loaded silently.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from typing import Any

from common import DIRTY, add_connection_args, build_rag, load_env
from ontology import ENTITIES, TABLES

from graphrag_sdk import Ontology, TableMapping

# dirty file -> (the clean table it was copied from, the fault, how to see what reached the graph)
CASES: list[dict[str, str]] = [
    {
        "file": "practices_duplicate_key.csv",
        "twin": "mitigation_practices.csv",
        "fault": "two rows share practice_id PR-AWD (CH4 max 50.0, then 55.0)",
        "after": "MATCH (n:MitigationPractice) WHERE n.is_stub = false "
        "RETURN n.entity_key, n.practices_duplicate_key__ch4_reduction_max_pct "
        "ORDER BY n.entity_key",
    },
    {
        "file": "metrics_percent_sign.csv",
        "twin": "trade_shock_metrics.csv",
        "fault": "value_num holds '13%' in a FLOAT column",
        "after": "MATCH (n:TradeMetric) WHERE n.is_stub = false RETURN count(n)",
    },
    {
        "file": "experiments_na_cell.csv",
        "twin": "experiments.csv",
        "fault": "ch4_flux_kg_ha holds 'n/a' in a FLOAT column",
        "after": "MATCH (n:Experiment) WHERE n.is_stub = false RETURN count(n)",
    },
    {
        "file": "experiments_bad_date.csv",
        "twin": "experiments.csv",
        "fault": "start_date holds '21/03/2024' in a DATE column",
        "after": "MATCH (n:Experiment) WHERE n.is_stub = false RETURN count(n)",
    },
    {
        "file": "papers_empty_key.csv",
        "twin": "papers.csv",
        "fault": "the first row has an empty paper_id",
        "after": "MATCH (n:Paper) WHERE n.is_stub = false RETURN n.name",
    },
    {
        "file": "people_schema_drift.csv",
        "twin": "people.csv",
        "fault": "the export renamed full_name to name; the mapping still says full_name",
        "after": "MATCH (n:Person) WHERE n.is_stub = false RETURN count(n)",
    },
]


def dirty_ontology() -> Ontology:
    """The clean declarations, pointed at the dirty file names."""
    by_source = {t.source: t for t in TABLES}
    tables = []
    for case in CASES:
        twin = by_source[case["twin"]]
        tables.append(
            TableMapping(
                source=case["file"],
                label=twin.label,
                key=twin.key,
                name=twin.name,
                properties=dict(twin.properties),
                links=list(twin.links),
                standalone=twin.standalone,
            )
        )
    return Ontology(entities=ENTITIES, tables=tables)


class Warnings(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.records: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        if record.name.startswith("graphrag_sdk.ingestion"):
            self.records.append(record.getMessage())


async def run_case(rag, case: dict[str, str]) -> dict[str, Any]:
    warnings = Warnings()
    logging.getLogger("graphrag_sdk").addHandler(warnings)
    outcome: dict[str, Any] = {"file": case["file"], "fault": case["fault"]}
    try:
        result = await rag.ingest(str(DIRTY / case["file"]))
        outcome["loaded"] = True
        outcome["rows"] = result.records
    except Exception as exc:  # the exception is the result here
        outcome["loaded"] = False
        outcome["refused"] = f"{type(exc).__name__}: {exc}"
    finally:
        logging.getLogger("graphrag_sdk").removeHandler(warnings)
    outcome["warnings"] = warnings.records
    outcome["graph"] = [list(r) for r in await rag.query(case["after"])]
    return outcome


async def main(args: argparse.Namespace) -> None:
    rag = build_rag(args, graph_name=f"{args.graph_name}_dirty", ontology=dirty_ontology())
    try:
        for case in CASES:
            await rag.delete_all()
            outcome = await run_case(rag, case)
            print(f"\n{case['file']}  —  {case['fault']}")
            if outcome["loaded"]:
                print(f"  loaded {outcome['rows']} rows")
            else:
                print(f"  REFUSED  {outcome['refused'][:400]}")
            for warning in outcome["warnings"]:
                print(f"  warning  {warning[:400]}")
            print(f"  in the graph: {outcome['graph']}")
    finally:
        await rag.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_connection_args(p)
    return p.parse_args()


if __name__ == "__main__":
    load_env()
    asyncio.run(main(parse_args()))
