# Onboarding notes for new members

Welcome. This page is maintained by hand and is usually a little out of date.

## Who we are

The group is hosted by FalkorDB in Tel Aviv and works with three external partners:
the Indian Institute of Science in Bangalore (Y. Narahari's group), TU Wien, and the
International Rice Research Institute (IRRI) in the Philippines. Day-to-day contacts
are Noa Levi (field campaigns), Amir Cohen (data pipeline) and Gal Shubeli (graph
tooling).

## Data you will meet

* `mitigation_practices.csv` — one row per agricultural practice, keyed `PR-…`.
  The `ch4_reduction_*` columns are percentages taken from the survey literature.
* `experiments.csv` — one row per plot-season. Several rows are simply called
  "baseline" or "control plot"; tell them apart by `exp_id`, never by name.
* `people.csv` — exported from the HR system. Some names arrive surname-first
  ("Sun, Sizhong"); nobody has cleaned that up.
* `citations.csv` — from the reference manager. It identifies papers by arXiv id,
  not by the PDF file name the papers table uses.
* `funding.csv` — grants. Amounts are in the grant's own currency; do not sum them.
* `compute_usage.json` — nightly dump from the scheduler. Not a table, strictly.

## Equipment

The gas analyzers (Picarro G2508, LI-COR LI-7810) live in the field container at
Los Baños. The GPU workstation is in Tel Aviv; book it through Amir. The DJI M300
drone needs a licensed pilot — currently only Noa.

## Reading list

Start with the carbon-farming survey (Priyanka V et al.), then the Wageningen AWD
paper. If you work on the EU side, read the market-design proposal by Finster,
Kasberger and Rütten. The UniD3 paper is on the list because Gal likes the graph
construction, not because it is about agriculture.
