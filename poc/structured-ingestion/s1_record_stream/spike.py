"""s1 — can ``RecordBatch`` actually carry a *lazy* record stream?

Proposal #1 says:

    class RecordBatch(DataModel):
        records: Iterable[dict[str, Any]]   # streamed, never fully materialised

That is a claim about pydantic behaviour, and the whole streaming story rests on
it. The failure mode that would matter in production is not "it raises" — it is
"something innocuous silently consumes the stream and ingestion writes zero
rows". So this spike attacks exactly that.

Questions:
  Q1  Does constructing the model consume the generator?
  Q2  Is the field re-iterable, or one-shot?
  Q3  Does an incidental ``model_dump()`` / ``repr()`` / ``model_copy()`` — the
      kind of thing logging and result objects do — eat the records?
  Q4  What does streaming actually buy in peak memory at POC scale?
"""

from __future__ import annotations

import sys
import tracemalloc
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _harness.env import Report  # noqa: E402

from graphrag_sdk.core.models import DataModel  # noqa: E402

N = 200_000


def rows(n: int = N) -> Iterator[dict[str, Any]]:
    for i in range(n):
        yield {"employee_id": f"E-{i}", "full_name": f"Person {i}", "age": 20 + i % 50}


# ── Candidate A: the design's literal shape ──────────────────────


class BatchIterable(DataModel):
    records: Iterable[dict[str, Any]]
    source: str


# ── Candidate B: a factory, so the stream is re-iterable ─────────


class BatchFactory(DataModel):
    open_records: Callable[[], Iterator[dict[str, Any]]]
    source: str

    def __iter__(self):  # type: ignore[override]
        return self.open_records()


def main() -> int:
    r = Report("s1 — record stream shape")

    # Q1 — construction must not drain the generator.
    gen = rows(10)
    batch = BatchIterable(records=gen, source="employees.csv")
    consumed_at_construction = sum(1 for _ in gen)
    r.check(
        consumed_at_construction in (0, 10),
        "construction does not silently drop records",
        f"generator still yields {consumed_at_construction} after model init",
    )
    r.note(f"field type after validation: {type(batch.records).__name__}")
    lazy = consumed_at_construction == 10 or type(batch.records).__name__ != "list"
    r.check(lazy, "records field stays lazy (not materialised to a list)")

    # Q2 — one-shot?
    b2 = BatchIterable(records=rows(10), source="x")
    first = list(b2.records)
    second = list(b2.records)
    r.check(
        len(first) == 10,
        "first iteration yields every record",
        f"{len(first)} records",
    )
    one_shot = len(second) == 0
    r.check(
        True,
        "second iteration behaviour recorded",
        f"re-iteration yields {len(second)} records -> {'ONE-SHOT' if one_shot else 're-iterable'}",
    )

    # Q3 — the dangerous one. Does incidental inspection eat the stream?
    for label, poke in (
        ("repr()", lambda b: repr(b)),
        ("model_dump()", lambda b: b.model_dump()),
        ("model_copy()", lambda b: b.model_copy()),
    ):
        b = BatchIterable(records=rows(10), source="x")
        try:
            poke(b)
            survived = len(list(b.records))
            r.check(
                survived == 10,
                f"{label} leaves the stream intact",
                f"{survived}/10 records survive",
            )
        except Exception as exc:  # noqa: BLE001
            r.check(False, f"{label} raised", f"{type(exc).__name__}: {exc}")

    # Candidate B under the same abuse.
    fb = BatchFactory(open_records=lambda: rows(10), source="x")
    fb.model_dump()
    r.check(
        len(list(fb)) == 10 and len(list(fb)) == 10,
        "factory shape survives model_dump() AND is re-iterable",
    )

    # Q2b — does the annotation erase list-ness even for an eager caller?
    eager = BatchIterable(records=[{"a": 1}, {"a": 2}], source="x")
    try:
        length: int | None = len(eager.records)  # type: ignore[arg-type]
    except TypeError:
        length = None
    r.check(
        length is None,
        "Iterable[dict] erases list-ness: len() fails even when a list was passed",
        f"type is {type(eager.records).__name__}; downstream code can never cheaply count records",
    )

    # Q5 — the consequence that actually bites. Proposal #6 iterates records
    # twice: step 3 builds record chunks, step 4 maps records -> GraphData.
    def two_pass(batch) -> tuple[int, int]:
        recs = batch.records if isinstance(batch, BatchIterable) else batch
        chunks = sum(1 for _ in recs)  # step 3
        nodes = sum(1 for _ in recs)  # step 4
        return chunks, nodes

    chunks, nodes = two_pass(BatchIterable(records=rows(10), source="employees.csv"))
    r.check(
        nodes == 0,
        "two-pass pipeline over a one-shot stream silently writes ZERO nodes",
        f"step 3 saw {chunks} records, step 4 saw {nodes} — no error raised",
    )
    fb2 = BatchFactory(open_records=lambda: rows(10), source="employees.csv")
    r.check(
        (sum(1 for _ in fb2), sum(1 for _ in fb2)) == (10, 10),
        "factory shape survives the same two-pass pipeline",
    )

    # Q4 — peak memory, streamed vs materialised.
    def peak(fn) -> float:
        tracemalloc.start()
        fn()
        peak_bytes = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        return peak_bytes / 1e6

    streamed = peak(lambda: sum(1 for _ in BatchIterable(records=rows(), source="s").records))
    materialised = peak(
        lambda: sum(1 for _ in BatchIterable(records=list(rows()), source="s").records)
    )
    r.note(
        f"{N:,} rows — streamed peak {streamed:.1f} MB · materialised peak {materialised:.1f} MB"
    )
    r.check(
        streamed < materialised / 10,
        "streaming keeps peak memory an order of magnitude below materialising",
        f"{materialised / max(streamed, 1e-6):.0f}x reduction",
    )

    return r.verdict()


if __name__ == "__main__":
    raise SystemExit(main())
