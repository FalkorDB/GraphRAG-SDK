"""Build the graph: six papers, two notes, twelve tables and a JSON file.

    python ingest.py --reset                 # everything, ~20 min (PDF extraction dominates)
    python ingest.py --reset --skip-pdfs     # tables, JSON and notes only, ~4 min

The prose goes through the extractor; the tables are loaded from ``ontology.py``
without a model; ``finalize()`` embeds what is new and lets the LLM resolver
judge the cross-source pairs no spelling rule joins (a row called "V, Priyanka"
and a title page that says "Priyanka V"). Rows of one table are never asked
about each other — a key is an identity.

Order does not matter for the result; the default here is the order the files
would plausibly arrive in: the papers first, the partners' tables, the group's
own tables, the papers table late (so its placeholders exist before it does),
and the undeclared funding.csv last.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

from common import DOCS, JSON, NOTES, RESULTS, TABLES, add_connection_args, build_rag, load_env

TABLE_ORDER = (
    "institutions.csv",
    "people.csv",
    "paper_authors.csv",
    "mitigation_practices.csv",
    "experiments.csv",
    "trade_shock_metrics.csv",
    "electricity_carbon_scenarios.csv",
    "citations.csv",
    "papers.csv",
    "datasets.csv",
    "lab_equipment.csv",
    "funding.csv",  # not declared: the model proposes a mapping
)


class WarningLog(logging.Handler):
    """Every WARNING the SDK logs, kept for the report (they are findings too)."""

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.records: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(f"{record.name}: {record.getMessage()}"[:500])


async def ingest_prose(rag, files: list[Path]) -> list[dict[str, Any]]:
    """PDFs and markdown notes: chunked, extracted, written under the same labels the tables use."""
    from graphrag_sdk.ingestion.chunking_strategies.fixed_size import FixedSizeChunking

    chunker = FixedSizeChunking(chunk_size=1500, chunk_overlap=200)
    rows = []
    for path in files:
        t0 = time.perf_counter()
        result = await rag.ingest(str(path), chunker=chunker)
        row = {
            "source": path.name,
            "chunks": result.chunks_indexed,
            "nodes": result.nodes_created,
            "relationships": result.relationships_created,
            "seconds": round(time.perf_counter() - t0, 1),
        }
        print(f"  prose  {row}")
        rows.append(row)
    return rows


async def ingest_tables(rag) -> list[dict[str, Any]]:
    """One call per CSV. A refused file is recorded, not fatal."""
    rows = []
    for name in TABLE_ORDER:
        t0 = time.perf_counter()
        try:
            result = await rag.ingest(str(TABLES / name))
            row = {
                "source": name,
                "records": result.records,
                "entities": result.entities,
                "references": result.references,
                "edges": result.edges,
            }
        except Exception as exc:  # e.g. MappingError when a typed column holds text
            row = {"source": name, "error": f"{type(exc).__name__}: {str(exc)[:300]}"}
        row["seconds"] = round(time.perf_counter() - t0, 1)
        print(f"  table  {row}")
        rows.append(row)
    return rows


async def ingest_json(rag, path: Path) -> dict[str, Any]:
    """``.json`` is not a tabular suffix, so it takes the prose path. The outcome is the finding."""
    t0 = time.perf_counter()
    result = await rag.ingest(str(path))
    row = {
        "source": path.name,
        "chunks": result.chunks_indexed,
        "nodes": result.nodes_created,
        "seconds": round(time.perf_counter() - t0, 1),
    }
    print(f"  json   {row}")
    return row


def finalize_report(summary, before: int, after: int) -> dict[str, Any]:
    fields = (
        "entities_deduplicated",
        "entities_embedded",
        "relationships_embedded",
        "resolved_duplicates",
        "rejected_duplicates",
        "probable_duplicates",
        "proposed_mappings",
        "unresolved_references",
        "entities_without_a_name",
        "unmerged_name_collisions",
        "property_conflicts",
    )
    report = {"entities_before": before, "entities_after": after}
    report.update({f: getattr(summary, f) for f in fields})
    return report


def print_finalize(report: dict[str, Any]) -> None:
    print(
        f"\n[finalize] entities {report['entities_before']} -> {report['entities_after']}; "
        f"resolver accepted {len(report['resolved_duplicates'])} merges, "
        f"rejected {len(report['rejected_duplicates'])}, "
        f"left {len(report['probable_duplicates'])} for a human"
    )
    for merge in report["resolved_duplicates"][:12]:
        print(f"    merged   {merge}")
    if len(report["resolved_duplicates"]) > 12:
        print(f"    ... {len(report['resolved_duplicates']) - 12} more")
    for pair in report["probable_duplicates"]:
        print(f"    undecided {pair}")
    print(f"  proposed mappings (undeclared tables): {report['proposed_mappings']}")
    print(f"  unresolved references (placeholders):  {report['unresolved_references']}")
    print(f"  rows without a name (fact tables):     {report['entities_without_a_name']}")
    print(
        f"  same name, different labels:           {len(report['unmerged_name_collisions'])} "
        f"(e.g. {list(report['unmerged_name_collisions'].items())[:2]})"
    )


async def main(args: argparse.Namespace) -> None:
    warnings = WarningLog()
    logging.getLogger("graphrag_sdk").addHandler(warnings)
    stages: dict[str, float] = {}
    report: dict[str, Any] = {"graph_name": args.graph_name, "model": args.model}

    async def timed(name: str, coro):
        t0 = time.perf_counter()
        try:
            return await coro
        finally:
            stages[name] = round(time.perf_counter() - t0, 1)
            print(f"[{name}] {stages[name]}s")

    rag = build_rag(args)
    try:
        if args.reset:
            await rag.delete_all()
            print("[reset] graph cleared")

        prose = ([] if args.skip_pdfs else sorted(DOCS.glob("*.pdf"))) + sorted(NOTES.glob("*.md"))
        if not args.skip_pdfs and not list(DOCS.glob("*.pdf")):
            raise SystemExit(
                "no PDFs in data/docs — run download_pdfs.py first (or pass --skip-pdfs)"
            )

        if args.tables_first:
            report["tables"] = await timed("tables", ingest_tables(rag))
            report["prose"] = await timed("prose", ingest_prose(rag, prose))
        else:
            report["prose"] = await timed("prose", ingest_prose(rag, prose))
            report["tables"] = await timed("tables", ingest_tables(rag))
        report["json"] = await timed("json", ingest_json(rag, JSON / "compute_usage.json"))

        before = (await rag.query("MATCH (e:__Entity__) RETURN count(e)"))[0][0]
        summary = await timed("finalize", rag.finalize(resolve=not args.no_resolve))
        after = (await rag.query("MATCH (e:__Entity__) RETURN count(e)"))[0][0]
        report["finalize"] = finalize_report(summary, before, after)
        print_finalize(report["finalize"])
    finally:
        await rag.close()

    report["warnings"] = warnings.records
    report["stages_seconds"] = stages
    print(f"\n[done] {len(warnings.records)} SDK warnings; stages {stages}")
    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=1, default=str))
        print(f"[written] {out}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_connection_args(p)
    p.add_argument("--reset", action="store_true", help="delete the graph first")
    p.add_argument("--skip-pdfs", action="store_true", help="tables, JSON and notes only")
    p.add_argument(
        "--tables-first",
        action="store_true",
        help="load the CSVs before the prose; same graph either way",
    )
    p.add_argument(
        "--no-resolve",
        action="store_true",
        help="finalize(resolve=False): report cross-source pairs instead of asking the model",
    )
    p.add_argument("--output-json", default=str(RESULTS / "ingest.json"))
    return p.parse_args()


if __name__ == "__main__":
    load_env()
    asyncio.run(main(parse_args()))
