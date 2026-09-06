# 14 · A research group's knowledge base — tables, PDFs, notes and JSON in one graph

A small research group works on methane from rice, trade shocks and electricity
markets. Its knowledge lives in the places it always does: twelve CSV exports,
six arXiv papers, two markdown notes and a JSON dump from a cluster scheduler.
This example loads all of it into **one FalkorDB graph** with GraphRAG-SDK, checks
the graph against the source files, and asks it questions that need both the
tables and the prose.

It is deliberately *not* a clean demo. The tables were written by different
people at different times: keys are inconsistent, one file has a percent sign in a
numeric column, one is a bare junction table, one is undeclared, two PDFs have no
row anywhere and one is about drug discovery and should stay an island. The point
is to show what structured ingestion does with data that fits the design about 80%.

## What is in `data/`

| Path | Rows | What it is | The awkward part |
|---|---|---|---|
| `tables/papers.csv` | 3 | the three papers the wave-1 tables came from | keyed by arXiv id, but other files reference papers by **file name** |
| `tables/mitigation_practices.csv` | 400 | crop practices with CH4-reduction and water-saving ranges | `source_doc` is a file name, not a paper id; 25 rows have a blank `ch4_reduction_max_pct` |
| `tables/trade_shock_metrics.csv` | 600 | trade metrics per country and policy event | `policy_event` is both a property and a link; 124 rows share the event `China WTO accession` |
| `tables/electricity_carbon_scenarios.csv` | 500 | electricity-market policy scenarios | `country` is both a property and a link |
| `tables/institutions.csv` | 14 | universities and institutes | `country` holds `austria`, `Austria ` and `Austria` (one node in the end) but also `USA` and `United States` (two); `INST-99` is referenced by `people.csv` but missing here |
| `tables/people.csv` | 48 | researchers | 18 blank `h_index`; names are `Last, First` inside quotes; `P-001` is also named in the notes, `funding.csv` and a paper |
| `tables/paper_authors.csv` | 13 | who authored which paper | pure junction table with no natural name; `corresponding` mixes `true`, `yes` and `false` |
| `tables/citations.csv` | 8 | paper → paper citations | cites arXiv ids that have no `papers.csv` row (placeholders) |
| `tables/experiments.csv` | 180 | field experiments with CH4 flux and dates | 51 rows are all named `baseline` — the name is not the identity, the key is; some flux cells are blank |
| `tables/datasets.csv` | 6 | datasets used by the papers | one `used_in` cell holds two file names joined by `;`; one is blank |
| `tables/lab_equipment.csv` | 8 | instruments and their cost | equipment also appears by name in `notes/` |
| `tables/funding.csv` | 5 | grants | **no mapping declared** — shows what an undeclared table does |
| `notes/2026-04-lab-meeting.md`, `notes/onboarding.md` | – | prose that names people, equipment and institutions from the tables | the only thing that connects the two gas analyzers |
| `json/compute_usage.json` | – | scheduler dump with five researchers' compute hours | `.json` is not a tabular suffix — it goes down the prose path |
| `docs/*.pdf` (6, downloaded) | – | the arXiv papers | two are connected only by prose; one must stay disconnected |
| `dirty/*.csv` | 6 | copies of real tables, each with one realistic fault | see `dirty_files.py` |

The PDFs are 15 MB and are fetched from arXiv rather than committed:

```
python download_pdfs.py
```

## Files

| File | Purpose |
|---|---|
| `ontology.py` | the declared half of the graph: 14 entity types and 11 `TableMapping`s, each with a comment on why it is shaped that way |
| `common.py` | connection flags, `.env` loading and `build_rag()` shared by every script |
| `download_pdfs.py` | fetches the six pinned arXiv PDFs into `data/docs/` |
| `ingest.py` | builds the graph: PDFs and notes → tables → JSON → `finalize(resolve=True)`; writes `results/ingest.json` |
| `verify.py` | recomputes 32 expectations from the CSVs and checks them against the graph with Cypher; exit code 1 on any failure |
| `ask.py` | 18 questions (aggregate / hybrid / lookup) through `rag.completion()`; prints which sources each answer used; writes `results/answers.json` |
| `dirty_files.py` | loads each faulty CSV alone into `<graph>_dirty` and shows what was loaded, refused or warned |
| `results/` | where the scripts write their JSON reports (gitignored) |

## Running it

Prerequisites: FalkorDB on `localhost:6379` (`docker run -p 6379:6379 -it falkordb/falkordb`),
`pip install graphrag-sdk[litellm]`, and an `OPENAI_API_KEY` either exported or in a
`.env` file in this folder. All scripts accept `--graph-name`, `--host`, `--port`,
`--model` (default `openai/gpt-4o-mini`).

```bash
cd examples/14_research_group_knowledge_base
python download_pdfs.py                     # once, ~15 MB
python ingest.py --reset                    # ~20 min, mostly PDF extraction
python verify.py                            # a few seconds
python ask.py                               # ~2 min for 18 questions
python dirty_files.py                       # ~5 s
```

To try the structured half without waiting for PDF extraction:

```bash
python ingest.py --reset --skip-pdfs --graph-name rg_quick
python verify.py --skip-pdfs --graph-name rg_quick
python ask.py --graph-name rg_quick --kind aggregate
```

Other `ingest.py` flags: `--tables-first` (load tables before prose — order does not
change the result, which is the point), `--no-resolve` (skip cross-source
de-duplication in `finalize`).

## What to expect

**Ingest.** Tables load in about 6 seconds total for 1,785 rows — no LLM call per
row. The PDFs take 12–20 minutes to extract and `finalize()` about 4 minutes; with
`--skip-pdfs` the whole build is under 4 minutes. Every row becomes one node with its typed columns as `<table>__<column>`
properties, one `Chunk {kind:'record'}` for provenance, and one `RELATES {rel_type}`
edge per foreign key. Foreign keys that point at a row that does not exist yet
(`INST-99`, cited arXiv ids, an experiment id in `datasets.csv`) become placeholder
nodes with `is_stub = true`; when the real row or a prose mention arrives, it fills
the same node. `funding.csv`, which has no mapping, still loads: the model proposes a
mapping from the column profile and the ontology — `Funding`, keyed on `grant_id`,
`LED_BY -> Person` and `FUNDED_BY -> Institution` in our runs — and `finalize()`
lists it under `proposed_mappings` so it can be declared properly. `finalize()` then embeds entities and edges and reports what it merged
across sources.

**Verify.** `verify.py` (32 checks, all passing on our runs) checks that rows are
rows (counts and record chunks per table), typed columns aggregate (e.g. rice practices average CH4 reduction, baseline
experiment flux, equipment over 50k), foreign keys became edges (authorship,
citations, `INST-99` shared by two people), rows and prose share nodes (`P-001` is
one `Person` mentioned in five places, the gas analyzers are joined by
`onboarding.md`), and expected non-joins hold (the drug-discovery paper has no edge
into the research-group data).

**Ask.** Aggregate questions (`A*`) are answered from Cypher over the typed
properties — e.g. *"How many baseline experiments and what is their average CH4
flux?"* generates `MATCH (e:Experiment) WHERE e.name CONTAINS 'baseline' RETURN
count(e), avg(e.experiments__ch4_flux_kg_ha)` and matches a recomputation from the
CSV. Hybrid questions (`H*`) need a table row and a passage, e.g. which instrument
the lab note says is broken and what it cost. Lookup questions (`L*`) are single-hop
facts. Each answer prints the sources it drew on (`cypher_rows`, `passages:<file>`).

In two full runs with `gpt-4o-mini`: all six aggregates were exact (all via Cypher);
five of seven hybrids were right, `H2` quoted the 13.5 % robustness row instead of
the 13.0 % baseline, and `H3` listed one or two of the seven authors because the
`Authorship` rows were not retrieved as passages and no Cypher was generated;
`L3` and `L5` were right, `L1` over-included a reversed citation in one run, `L2`
declined (the model wrote SQL `HAVING`, the Cypher failed and only passages were
left), and `L4` claimed a paper is in `papers.csv` when only its PDF is.

**Dirty files.** Six one-fault loads, each on its own empty graph. A duplicate key
loads every row as its own chunk but keeps the *first* row's values on the node, and
warns. `13%` or `n/a` in a FLOAT column and `21/03/2024` in a DATE column each make
the loader **refuse the whole file** — a `MappingError` names the column, the declared
type and the offending value; nothing is half-written. A row with an empty key is
skipped and counted in a warning; the rest of the file loads. A renamed column
(`full_name` → `name`) is refused before any write with
`name column 'full_name' is not in the source`.

## Known limitations this example exposes

These are real findings from running this corpus, kept here because the example is
meant to show the edges of the design, not hide them:

1. `.json` is not treated as tabular, so `compute_usage.json` is extracted as prose
   — the five researchers come out as `Person` nodes named by their id (`P-0xx`)
   rather than joined to `people.csv`.
2. Extraction noise from PDF title pages can attach a wrong description or edge to a
   row's node (a person from `people.csv` picking up a market-design paper's title).
3. A multi-valued FK cell (`2603.23825v1.pdf;1801.02681v2.pdf`) becomes one bogus
   placeholder instead of two links.
4. Junction tables with no natural name are reported by `finalize()` as
   `entities_without_a_name` every run.
5. Questions the model answers from record passages instead of Cypher can
   over-include — `L1` sometimes lists a citing paper the table does not have. The
   `context` line under each answer shows which happened; the generated Cypher is
   logged at DEBUG (`cypher_generation`).
6. The resolver leaves `Carbon Farming: An Expository Inter-Disciplinary Survey` and
   `Carbon farming: an expository, inter-disciplinary survey` (a placeholder from
   `citations.csv` and the `papers.csv` row) as a "probable duplicate" for a human
   rather than merging on a case- and punctuation-insensitive match.
7. Lookups that need a junction table (`H3`, who authored what) are only answered
   well when text-to-Cypher fires; when retrieval falls back to passages, the
   nameless `Authorship` rows are rarely among them.

Found and fixed while building this example: loading a table used to overwrite the
descriptions declared on the ontology's entity types with `Declared by a structured
source, keyed on …`, which made the model file the undeclared `funding.csv` under
`Experiment`. An existing label now keeps its description, and the mapping proposal
is shown each label's description.

## Cleaning up

```bash
redis-cli GRAPH.DELETE research_group
redis-cli GRAPH.DELETE research_group_dirty
```
