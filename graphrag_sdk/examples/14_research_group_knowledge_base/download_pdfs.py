"""Fetch the six arXiv papers the example reads into ``data/docs/``.

The versions are pinned, so a re-run downloads the exact bytes the results in
README.md were produced from. Existing files are left alone.

    python download_pdfs.py
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

DOCS = Path(__file__).resolve().parent / "data" / "docs"

# arXiv id (with version)  ->  why it is in the corpus
PAPERS = {
    "2603.20674v1": "carbon-farming survey — the source of mitigation_practices.csv",
    "2603.23825v1": "trade liberalization and innovation — the source of trade_shock_metrics.csv",
    "2603.25874v1": "electricity/carbon market design — source of electricity_carbon_scenarios.csv",
    "2006.02840v1": "AWD in organically produced rice — on topic, authorship rows, no Paper row",
    "1801.02681v2": "Chinese exporters' diversification — cited by a row, connected only by prose",
    "2606.01394v1": "UniD3 drug–disease knowledge graph — unrelated; must stay an island",
}


def main() -> int:
    DOCS.mkdir(parents=True, exist_ok=True)
    for arxiv_id, why in PAPERS.items():
        target = DOCS / f"{arxiv_id}.pdf"
        if target.is_file():
            print(f"  have   {target.name}")
            continue
        url = f"https://arxiv.org/pdf/{arxiv_id}"
        print(f"  fetch  {target.name}  ({why})")
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                target.write_bytes(response.read())
        except OSError as exc:
            print(f"  FAILED {url}: {exc}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
