"""The declared half of the graph: entity types and the twelve table mappings.

Everything a table contributes to the graph is stated here — which column is the
identity, which is the display name, which columns are typed values and which are
foreign keys to another label. Rows are loaded from these declarations without a
model; the PDFs and notes go through the ordinary extraction path and share the
same labels, so a row and a prose mention of the same thing land on one node.

The mappings are written the way a member of the group would write them, and
that is where the interesting behaviour comes from. Each awkward one says why.
"""

from __future__ import annotations

from graphrag_sdk import Column, Entity, Link, Ontology, TableMapping

# ── Entity types ─────────────────────────────────────────────────────────────
# The descriptions steer the extractor when it reads the PDFs and notes. The
# labels are the same ones the tables load into.

ENTITIES = [
    Entity(label="Paper", description="A research paper or article, identified by its title"),
    Entity(
        label="Person",
        description="A researcher, author or lab member, e.g. Priyanka V, Sizhong Sun, Noa Levi",
    ),
    Entity(
        label="Institution",
        description="A university, research institute, company or foundation, "
        "e.g. Indian Institute of Science, IRRI, Wageningen University, FalkorDB",
    ),
    Entity(label="Country", description="A country or region such as Austria, Germany, China, EU"),
    Entity(
        label="MitigationPractice",
        description="An agricultural practice that reduces emissions or sequesters carbon, "
        "e.g. Alternate Wetting and Drying (AWD), System of Rice Intensification",
    ),
    Entity(label="Crop", description="A crop or cropping system: rice, sugarcane, wheat, maize"),
    Entity(
        label="Experiment",
        description="A field or plot experiment measuring methane flux or yield for a "
        "mitigation practice",
    ),
    Entity(
        label="TradeMetric",
        description="A quantity estimated in a trade study: a trade-cost reduction, an "
        "elasticity, a probability",
    ),
    Entity(
        label="PolicyEvent",
        description="A policy change used as a shock or treatment, e.g. China WTO accession",
    ),
    Entity(
        label="PolicyScenario",
        description="A simulated electricity/carbon market scenario or settlement rule",
    ),
    Entity(label="Dataset", description="A dataset used in a study, e.g. ENTSO-E day-ahead prices"),
    Entity(
        label="Equipment",
        description="A lab instrument, e.g. Picarro G2508 gas analyzer, eddy covariance tower",
    ),
    Entity(
        label="Authorship",
        description="One author's position on one paper (a row of paper_authors.csv)",
    ),
    Entity(label="Citation", description="One paper citing another (a row of citations.csv)"),
]

# ── Tables ───────────────────────────────────────────────────────────────────
# ``source`` is matched by file name, so the mapping applies to any path ending
# in it. A plain string in ``properties`` is a STRING column; ``Column(...)``
# gives it a type. ``Link`` makes an edge to another label using the cell as
# that label's key; a missing target becomes a placeholder node (``is_stub``)
# that a later table or a prose mention can fill.

TABLES = [
    # The three papers the wave-1 tables came from. Keyed by file name.
    TableMapping(
        source="papers.csv",
        label="Paper",
        key="paper_id",
        name="title",
        properties={
            "arxiv_id": "arxiv_id",
            "year": Column("year", "INTEGER"),
            "first_author": "first_author",
            "affiliation": "affiliation",
            "topic": "topic",
        },
    ),
    TableMapping(
        source="mitigation_practices.csv",
        label="MitigationPractice",
        key="practice_id",
        name="practice_name",
        properties={
            "crop_system": "crop_system",
            "ch4_reduction_min_pct": Column("ch4_reduction_min_pct", "FLOAT"),
            "ch4_reduction_max_pct": Column("ch4_reduction_max_pct", "FLOAT"),
            "water_saving_min_pct": Column("water_saving_min_pct", "FLOAT"),
            "water_saving_max_pct": Column("water_saving_max_pct", "FLOAT"),
            "soil_carbon_benefit": "soil_carbon_benefit",
            "mrv_intensity": "mrv_intensity",
        },
        links=[
            Link("APPLIES_TO", to="Crop", by="crop_system"),
            Link("REPORTED_IN", to="Paper", by="source_doc"),
        ],
    ),
    TableMapping(
        source="trade_shock_metrics.csv",
        label="TradeMetric",
        key="metric_id",
        name="metric_name",
        properties={
            "value_num": Column("value_num", "FLOAT"),
            "value_unit": "value_unit",
            "sample_note": "sample_note",
            "policy_event": "policy_event",
        },
        links=[
            Link("MEASURES", to="PolicyEvent", by="policy_event"),
            Link("OBSERVED_IN", to="Country", by="geography"),
            Link("REPORTED_IN", to="Paper", by="source_doc"),
        ],
    ),
    TableMapping(
        source="electricity_carbon_scenarios.csv",
        label="PolicyScenario",
        key="scenario_id",
        name="scenario_name",
        properties={
            "year": Column("year", "INTEGER"),
            "expenditure_reduction_pct": Column("expenditure_reduction_pct", "FLOAT"),
            "mechanism": "mechanism",
            "bunching_risk": "bunching_risk",
            "country": "country",
        },
        links=[
            Link("APPLIES_TO", to="Country", by="country"),
            Link("REPORTED_IN", to="Paper", by="source_doc"),
        ],
    ),
    # The group's own tables.
    # ``country`` is typed by hand: "austria", "Austria ", "USA", "United States".
    # Link values are normalised (strip + lower) before they become a key, so the
    # first two land on one Country node; USA / United States is left to finalize().
    TableMapping(
        source="institutions.csv",
        label="Institution",
        key="inst_id",
        name="name",
        properties={"city": "city", "kind": "kind"},
        links=[Link("LOCATED_IN", to="Country", by="country")],
    ),
    # Names in two formats ("Priyanka V" and "Sun, Sizhong"), one FK spelled
    # "inst-10 ", one pointing at an institution that has no row (INST-99), and
    # blank h_index cells. Blanks are omitted; a typed blank is not a zero.
    TableMapping(
        source="people.csv",
        label="Person",
        key="person_id",
        name="full_name",
        properties={"role": "role", "orcid": "orcid", "h_index": Column("h_index", "INTEGER")},
        links=[Link("AFFILIATED_WITH", to="Institution", by="institution_id")],
    ),
    # A junction table. There is no edge-only table in the design, so each row is
    # an Authorship node with two links and a synthetic key the export had to add.
    # It has no ``name`` — nothing in prose is called "AU-003".
    TableMapping(
        source="paper_authors.csv",
        label="Authorship",
        key="authorship_id",
        properties={
            "author_position": Column("author_position", "INTEGER"),
            "corresponding": Column("corresponding", "BOOLEAN"),
        },
        links=[Link("OF", to="Paper", by="paper_id"), Link("BY", to="Person", by="person_id")],
    ),
    # Keys Paper by arXiv id while papers.csv keys it by file name: one label, two
    # id domains. ``name="cited_title"`` lets a row that carries the exact title
    # join the paper node anyway; the bare arXiv ids stay placeholders.
    TableMapping(
        source="citations.csv",
        label="Citation",
        key="citation_id",
        properties={"context": "context"},
        links=[
            Link("CITING", to="Paper", by="citing_arxiv_id"),
            Link("CITES", to="Paper", by="cited_arxiv_id", name="cited_title"),
        ],
    ),
    # 180 plots; 51 of them are just called "baseline". A repeated display name
    # is not an identity, so those rows are keyed on exp_id and a prose mention
    # of "baseline" joins none of them. start_date is a DATE.
    TableMapping(
        source="experiments.csv",
        label="Experiment",
        key="exp_id",
        name="name",
        properties={
            "season": "season",
            "site_country": "site_country",
            "ch4_flux_kg_ha": Column("ch4_flux_kg_ha", "FLOAT"),
            "yield_t_ha": Column("yield_t_ha", "FLOAT"),
            "start_date": Column("start_date", "DATE"),
            "notes": "notes",
        },
        links=[
            Link("TESTS", to="MitigationPractice", by="practice_id"),
            Link("SITED_IN", to="Country", by="site_country"),
        ],
    ),
    # One ``used_in`` cell holds two file names joined by ";". Links do not split
    # cells, so that row points at a placeholder whose key is the whole string.
    TableMapping(
        source="datasets.csv",
        label="Dataset",
        key="dataset_id",
        name="name",
        properties={"url": "url", "license": "license", "size_gb": Column("size_gb", "FLOAT")},
        links=[Link("USED_IN", to="Paper", by="used_in")],
    ),
    # Relates to nothing in the other tables. ``standalone=True`` says so; only
    # the markdown notes connect these rows to the rest of the graph.
    TableMapping(
        source="lab_equipment.csv",
        label="Equipment",
        key="equipment_id",
        name="name",
        properties={
            "vendor": "vendor",
            "purchase_year": Column("purchase_year", "INTEGER"),
            "cost_usd": Column("cost_usd", "FLOAT"),
        },
        standalone=True,
    ),
    # funding.csv is deliberately NOT declared. When it is ingested the model
    # proposes a label, key and links for it; finalize() lists the file under
    # ``proposed_mappings`` so someone can promote (or correct) the guess.
]

ONTOLOGY = Ontology(entities=ENTITIES, tables=TABLES)
