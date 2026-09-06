# Lab meeting — 14 April 2026

Present: Priyanka V (remote), Simon Rütten, Noa Levi, Amir Cohen, Gal Shubeli.
Apologies: Reiner Wassmann.

## Methane MRV project

Priyanka walked through the revised numbers for the carbon-farming survey. The
AWD row in the practices table still carries the 30–50 % CH4 range from the survey;
the Wageningen paper on organically produced rice (Faiz-ul Islam et al.) reports a
wider reduction, so we agreed to keep the survey figure as the table value and cite
the Wageningen result in prose only.

Noa has re-run the 2024 wet-season baseline plots at the IRRI site in Los Baños.
Fluxes from the Picarro G2508 and the static chambers now agree within 8 %. The
eddy covariance tower is still down (see equipment sheet, EQ-03).

Action: Priyanka to send the updated `experiments.csv` export by Friday. Amir to
check whether EXP-017 and EXP-052 are the same plot under two ids.

## EU markets

Simon reported that the Austrian Institute of Technology (AIT) will share hourly
ENTSO-E data for 2025; TU Wien remains the formal partner on the FWF grant led by
Simon Finster. The market-design paper's 8.5 % Austria and 4.7 % Germany
expenditure-reduction estimates stand.

## GraphRAG for agronomy

Gal demoed structured ingestion: the practices, scenarios and trade-metric tables
now sit in one FalkorDB graph with the three papers. Open question from Amir: the
citations export keys papers by arXiv id while the papers table keys them by file
name — do the two halves actually join? Gal to check after the next finalize().

Next meeting 12 May. Gal to invite Sizhong Sun (James Cook University) for a talk on
the WTO accession shock.
