"""Run every spike. DB-backed spikes skip cleanly when FalkorDB is unreachable."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPIKES = [
    "s1_record_stream",
    "s2_mapping_dsl",
    "s3_identity",
    "s4_record_as_chunk",
    "s5_pipeline_seam",
]


def main() -> int:
    failures = []
    for spike in SPIKES:
        rc = subprocess.call([sys.executable, str(HERE / spike / "spike.py")])
        if rc != 0:
            failures.append(spike)
    print("\n" + "=" * 60)
    if failures:
        print(f"FAILED: {', '.join(failures)}")
        return 1
    print(f"All {len(SPIKES)} spikes passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
