"""Ask the graph. Each question targets one thing the data does — or refuses to do.

    python ask.py                 # all 18
    python ask.py --only A3 H2    # a few
    python ask.py --kind aggregate

For every answer the script prints where the context came from: ``cypher_rows``
means text-to-Cypher ran over the declared columns; ``passages:csv`` / ``:pdf``
/ ``:md`` are the record and prose chunks that were retrieved. ``expect`` says
what a truthful answer looks like given how the data really is — several are
*meant* to come back partial, because the data does not join.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any

from common import RESULTS, add_connection_args, build_rag, load_env

QUESTIONS: list[dict[str, str]] = [
    # ── aggregation over declared columns: the answer is a number the graph computes ──
    {
        "id": "A1",
        "kind": "aggregate",
        "question": "Across all rice mitigation practices in the structured table, what is the "
        "average maximum CH4-reduction percentage?",
        "expect": "avg(ch4_reduction_max_pct) over the 58 rice rows (3 blank): 38.80",
    },
    {
        "id": "A2",
        "kind": "aggregate",
        "question": "Across all TradeMetric rows linked to policy_event 'China WTO accession', "
        "what is the average value_num?",
        "expect": "124 rows, mean 19.38",
    },
    {
        "id": "A3",
        "kind": "aggregate",
        "question": "How many experiments are named 'baseline', and what is their average CH4 flux "
        "in kg/ha?",
        "expect": "51 rows (each its own node although the name repeats); mean 159.05 over the "
        "non-blank cells",
    },
    {
        "id": "A4",
        "kind": "aggregate",
        "question": "How many experiments tested Alternate Wetting and Drying (AWD), what is their "
        "average CH4 flux in kg/ha, and what CH4-reduction range does the practices table give for "
        "AWD?",
        "expect": "23 experiments, mean 171.16; the AWD row says 30-50 %",
    },
    {
        "id": "A5",
        "kind": "aggregate",
        "question": "Which lab equipment costs more than 50,000 USD?",
        "expect": "Picarro G2508 (98 500), LI-COR LI-7810 (61 200), eddy covariance tower (142 "
        "000), GC-FID (72 000)",
    },
    {
        "id": "A6",
        "kind": "aggregate",
        "question": "From the structured scenarios table only, report the exact mechanism and "
        "bunching_risk values for scenario_id SC-HARD-THRESHOLD versus SC-LINEAR-RAMP.",
        "expect": "hard_threshold / high versus linear_ramp / low — field values, nothing invented",
    },
    # ── hybrid: half the answer is in a table, half in a paper or a note ──
    {
        "id": "H1",
        "kind": "hybrid",
        "question": "The carbon-farming survey discusses Alternate Wetting and Drying (AWD). Using "
        "the structured table, what CH4-reduction range is stored for AWD, and how does the paper "
        "describe the water-management practice?",
        "expect": "30-50 % from the row; intermittent flooding / drying cycles from the survey",
    },
    {
        "id": "H2",
        "kind": "hybrid",
        "question": "What policy event does the trade paper use as its liberalization shock, and "
        "what baseline iceberg trade-cost reduction percentage is stored for that event in the "
        "structured metrics table?",
        "expect": "China's WTO accession (2001) from the paper; TM-ICEBERG 13.0 from the table",
    },
    {
        "id": "H3",
        "kind": "hybrid",
        "question": "Who are the authors of the carbon-farming survey according to the "
        "paper_authors table, and which institution is each affiliated with?",
        "expect": "seven people via Authorship rows — IISc, U Chicago, Boomitra, Piramal "
        "Foundation. The row is 'V, Priyanka'; the paper says 'Priyanka V' — one node after "
        "finalize()",
    },
    {
        "id": "H4",
        "kind": "hybrid",
        "question": "Which institution is Sizhong Sun affiliated with according to the people "
        "table, and which paper in the corpus did he write?",
        "expect": "James Cook University (row 'Sun, Sizhong'); 'Trade Liberalization, Export and "
        "Product Innovation' — needs the resolver to have joined the two name forms",
    },
    {
        "id": "H5",
        "kind": "hybrid",
        "question": "Who attended the April 2026 lab meeting, and which piece of equipment was "
        "reported as down?",
        "expect": "Priyanka V, Simon Rütten, Noa Levi, Amir Cohen, Gal Shubeli; the eddy "
        "covariance tower (EQ-03 in lab_equipment.csv, an island joined only by this note)",
    },
    {
        "id": "H6",
        "kind": "hybrid",
        "question": "Which institutions in the institutions table are located in Austria, and what "
        "expenditure reduction does the 2025 Austria baseline scenario record?",
        "expect": "TU Wien and AIT (typed 'austria' and 'Austria ' in the CSV, one Country node); "
        "the paper says 8.5 % but the scaled table has seven 2025 Austria rows, so the number is "
        "ambiguous",
    },
    {
        "id": "H7",
        "kind": "hybrid",
        "question": "What does the UniD3 paper propose, and is it connected to any mitigation "
        "practice or experiment in the graph?",
        "expect": "a KG-enhanced RAG framework for drug-disease reasoning; no connection — the "
        "paper is an island",
    },
    # ── lookups where the data itself is the obstacle ──
    {
        "id": "L1",
        "kind": "lookup",
        "question": "According to the citations table, which papers cite the trade-liberalization "
        "paper (arXiv 2603.23825), and what context is recorded for each citation?",
        "expect": "CIT-004, citing '2603.25874' with context 'pass-through analogy'. citations.csv "
        "keys papers by arXiv id, papers.csv by file name — the citing side is a placeholder named "
        "by its id",
    },
    {
        "id": "L2",
        "kind": "lookup",
        "question": "Which dataset in the datasets table was used in more than one paper, and "
        "which papers?",
        "expect": "DS-02, but its used_in cell is '2603.23825v1.pdf;1801.02681v2.pdf' in one "
        "string — the graph has one placeholder with that whole key; a truthful answer names it or "
        "says it is unresolved",
    },
    {
        "id": "L3",
        "kind": "lookup",
        "question": "Which grants in the funding table have Priyanka V as PI, and what is the "
        "amount and currency?",
        "expect": "GR-2023-01, 1 250 000 USD (Gates Foundation) — funding.csv is undeclared, so "
        "this depends entirely on the mapping the model proposed",
    },
    {
        "id": "L4",
        "kind": "lookup",
        "question": "Is the Wageningen paper on reducing greenhouse gas emissions in organically "
        "produced rice listed in the papers table, and who are its authors according to "
        "paper_authors?",
        "expect": "not in papers.csv (the PDF is in the corpus, the table is stale); paper_authors "
        "rows point at a placeholder '2006.02840v1.pdf' -> Syed Faiz-ul Islam and Jan Willem van "
        "Groenigen",
    },
    {
        "id": "L5",
        "kind": "lookup",
        "question": "How many MitigationPractice entities are in the graph?",
        "expect": "400 rows plus whatever the survey named that the resolver did not merge into a "
        "row (469 and 502 in two runs). Rows alone are WHERE is_stub = false",
    },
]


def sources(result) -> dict[str, int]:
    """Which retrieval sections fed the answer, and where the passages came from."""
    counts: dict[str, int] = {}
    retriever = result.retriever_result
    for item in retriever.items if retriever else []:
        section = (item.metadata or {}).get("section", "")
        if section == "cypher_results":
            counts["cypher_rows"] = counts.get("cypher_rows", 0) + item.content.count("\n- ")
        elif section == "passages":
            for passage in item.content.split("\n---\n"):
                tag = (
                    passage.split("[Source: ", 1)[1].split("]", 1)[0]
                    if "[Source: " in passage
                    else ""
                )
                suffix = Path(tag).suffix.lstrip(".") or "untagged"
                counts[f"passages:{suffix}"] = counts.get(f"passages:{suffix}", 0) + 1
    return counts


async def ask(rag, questions: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows = []
    for q in questions:
        t0 = time.perf_counter()
        result = await rag.completion(q["question"], return_context=True)
        row = {
            **q,
            "answer": result.answer,
            "context": sources(result),
            "seconds": round(time.perf_counter() - t0, 1),
        }
        rows.append(row)
        print(f"\n[{q['id']}/{q['kind']}] {q['question']}")
        print(f"  expect  {q['expect']}")
        print(f"  answer  {result.answer}")
        print(f"  context {row['context']}  {row['seconds']}s")
    return rows


async def main(args: argparse.Namespace) -> None:
    questions = QUESTIONS
    if args.only:
        questions = [q for q in questions if q["id"] in set(args.only)]
    if args.kind:
        questions = [q for q in questions if q["kind"] == args.kind]

    rag = build_rag(args)
    try:
        rows = await ask(rag, questions)
    finally:
        await rag.close()

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rows, indent=1, ensure_ascii=False))
        print(f"\n[written] {out}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_connection_args(p)
    p.add_argument("--only", nargs="*", metavar="ID", help="question ids to run")
    p.add_argument("--kind", choices=["aggregate", "hybrid", "lookup"])
    p.add_argument("--output-json", default=str(RESULTS / "answers.json"))
    return p.parse_args()


if __name__ == "__main__":
    load_env()
    asyncio.run(main(parse_args()))
