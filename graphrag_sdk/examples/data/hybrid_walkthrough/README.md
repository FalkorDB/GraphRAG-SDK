# hybrid_walkthrough fixtures

Synthetic data for `../../12_hybrid_walkthrough.ipynb`. No real company or person
appears here. Small enough to read in full before running anything.

| file | role |
|---|---|
| `board_review.pdf` | prose: a shortfall, two employees, a counterparty |
| `market_note.pdf` | prose: an acquisition, a third organization |
| `organizations.csv` | table: `org_id`, headcount, revenue |
| `employees.csv` | table: ages and titles, joined to orgs by `org_id` |
| `employees_v2.csv` | the same export a quarter later — one promotion, one leaver, one new hire |
| `contracts.csv` | table: two links out of one row (`BUYER`, `SELLER`) |
| `tickets.csv` | a table of free text, ingested deliberately as prose |

## The join

`Northwind Energy`, `Kestrel Grid`, `Meridian Fuels`, `Maya Ellison`,
`Tomas Reyes` and `Priya Raman` are spelled **identically** in the PDFs and in
the CSVs. That is what makes the two halves merge into one entity: the match is
exact string equality on the name, so changing a spelling on one side only will
leave you with two nodes where the notebook expects one.

Swapping in your own files is the point — keep the column names, or edit the
`Table(...)` declarations in the notebook to match yours.
