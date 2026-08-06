# Structured-ingestion spikes

Throwaway experiments backing [`docs/design/structured-ingestion.md`](../../docs/design/structured-ingestion.md)
(tracking: FalkorDB/research#82).

**This folder is disposable.** Nothing here ships. It is outside the wheel
(`[tool.hatch.build.targets.wheel] packages = ["src/graphrag_sdk"]`), outside pytest
(`testpaths = ["tests"]`), and outside CI (every job sets `working-directory: graphrag_sdk`
and lints only `src/`). Delete the whole directory once the findings are folded into the design
and the real implementation lands.

## Why it exists

The design doc makes claims. These spikes exist to **find out where the claims are wrong** before
we build on them — each folder answers *one specific open question* and records a decision, rather
than re-implementing a proposal. Five spikes, not twelve; a proposal without a genuine open
question does not get a folder.

| Spike | Proposal | The question it answers |
| --- | --- | --- |
| `s1_record_stream` | #1 | Can `RecordBatch` actually hold a lazy record stream, or does the model layer consume it? What does streaming buy in memory? |
| `s2_mapping_dsl` | #2 | Which DSL shape expresses all four real-world record shapes without special cases — and does `to_ontology()` round-trip into a valid `Ontology`? |
| `s3_identity` | #3, #4 | Do the three candidate identity policies actually produce one connected graph? Measured, not argued. |
| `s4_record_as_chunk` | #5 | Does record-as-chunk really inherit orphan cleanup unchanged — and is the predicted `update()` cutover data-loss trap real? |
| `s5_pipeline_seam` | #6 | Can `StructuredIngestionPipeline` reuse the existing pipeline's steps as-is, or do the signatures need to change? |

## Running

```bash
cd poc/structured-ingestion
python run_all.py             # everything; DB spikes skip if FalkorDB is unreachable
python s3_identity/spike.py   # one spike
```

Spikes needing a graph use `FALKOR_HOST` / `FALKOR_PORT` (default `localhost:6379`) and write to
throwaway graph names prefixed `poc_`. `docker compose up -d falkordb` from the repo root starts one.

No API keys are needed anywhere — the harness supplies a deterministic fake embedder, and the
spikes never call an LLM (which is the point of the whole design).

Each spike prints its findings and writes them to its own `NOTES.md`. [`FINDINGS.md`](FINDINGS.md)
rolls up the decisions that feed back into the design doc.
