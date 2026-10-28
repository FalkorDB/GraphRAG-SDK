"""Unit tests for the agent ingestion tools: root-confined file helpers, the
rule planner, the pipeline's drop report and the toolkit's sample_docs /
set_ontology / ingest(files=...) over a fake GraphRAG. The same flow runs
against a real FalkorDB in test_tools_integration.py."""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from graphrag_sdk.core.exceptions import ReadOnlyViolation
from graphrag_sdk.core.models import (
    DocumentInfo,
    Entity,
    FinalizeResult,
    GraphData,
    GraphNode,
    GraphRelationship,
    IngestionResult,
    Ontology,
    Relation,
    UpdateResult,
)
from graphrag_sdk.ingestion import ingestion_planner as planner_mod
from graphrag_sdk.ingestion.ingestion_planner import (
    CHEAP_PLAN_LIMITS,
    PlanRule,
    RuleBasedIngestionPlanner,
    match_path,
)
from graphrag_sdk.ingestion.pipeline import IngestionPipeline
from graphrag_sdk.mcp.server import GUIDE, INGEST_GUIDE, TEXT_INGEST_GUIDE, guide_for
from graphrag_sdk.tools import GraphRAGToolkit, ToolError
from graphrag_sdk.tools import files as fs
from graphrag_sdk.tools import graph_ops as ops
from graphrag_sdk.tools.models import IngestResult, PrunedTypes

# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def corpus(tmp_path):
    """A small folder: two Markdown notes, a text file, a hidden file,
    an unsupported file and a nested folder."""
    root = tmp_path / "docs"
    (root / "notes").mkdir(parents=True)
    (root / "reports" / "2024").mkdir(parents=True)
    (root / ".git").mkdir()
    (root / "notes" / "a.md").write_text("# Alpha\n\n" + "alpha words " * 300)
    (root / "notes" / "b.md").write_text("# Beta\n\nBeta body.")
    (root / "reports" / "2024" / "q1.txt").write_text("Q1 report text. " * 50)
    (root / "readme.rst").write_text("Readme")
    (root / ".secret.md").write_text("hidden")
    (root / ".git" / "config.md").write_text("hidden folder")
    (root / "data.csv").write_text("a,b\n1,2\n")
    (tmp_path / "outside.md").write_text("outside the root")
    return root


# ── files.py ─────────────────────────────────────────────────────────


class TestFiles:
    def test_root_and_home_are_refused(self, monkeypatch, tmp_path):
        with pytest.raises(ValueError, match="filesystem root or the home folder"):
            fs.resolve_root("/")
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("USERPROFILE", str(tmp_path))
        with pytest.raises(ValueError, match="filesystem root or the home folder"):
            fs.resolve_root("~")
        (tmp_path / "docs").mkdir()
        assert fs.resolve_root(tmp_path / "docs") == (tmp_path / "docs").resolve()
        home = tmp_path / "users" / "me"
        home.mkdir(parents=True)
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.setenv("USERPROFILE", str(home))
        with pytest.raises(ValueError, match="filesystem root or the home folder"):
            fs.resolve_root(tmp_path / "users")

    def test_listing_is_capped(self, corpus):
        root = fs.resolve_root(corpus)
        with pytest.raises(ToolError, match="more than 3 entries"):
            fs.list_files(root, max_entries=3)

    def test_folder_patterns_match_at_any_depth_with_disk_casing(self, tmp_path):
        root = tmp_path / "root"
        (root / "reports").mkdir(parents=True)
        (root / "archive" / "reports").mkdir(parents=True)
        (root / "reports" / "a.md").write_text("A.")
        (root / "archive" / "reports" / "b.md").write_text("B.")
        root = fs.resolve_root(root)
        expected = ["archive/reports/b.md", "reports/a.md"]
        assert fs.list_files(root, "reports/*.md") == expected
        assert fs.list_files(root, "Reports/*.md") == expected

    def test_literal_names_with_glob_characters(self, corpus):
        root = fs.resolve_root(corpus)
        (corpus / "notes" / "draft [v2].md").write_text("Draft.")
        assert fs.resolve_files(root, ["notes/draft [v2].md"]) == ["notes/draft [v2].md"]

    def test_resolve_root(self, corpus, tmp_path):
        assert fs.resolve_root(corpus) == corpus.resolve()
        with pytest.raises(ValueError, match="not a directory"):
            fs.resolve_root(corpus / "readme.rst")
        with pytest.raises(ValueError):
            fs.resolve_root(tmp_path / "missing")

    def test_list_files_skips_hidden_unsupported_and_sorts(self, corpus):
        root = fs.resolve_root(corpus)
        assert fs.list_files(root) == [
            "notes/a.md",
            "notes/b.md",
            "readme.rst",
            "reports/2024/q1.txt",
        ]
        assert fs.list_files(root, "*.md") == ["notes/a.md", "notes/b.md"]
        assert fs.list_files(root, "reports/*") == ["reports/2024/q1.txt"]
        assert fs.list_files(root, "./notes/b.md") == ["notes/b.md"]
        assert fs.list_files(root, "*.pdf") == []

    @pytest.mark.parametrize("bad", ["/etc/passwd", "../outside.md", "notes/../../x", "C:/x.md"])
    def test_paths_outside_the_root_are_refused(self, corpus, bad):
        root = fs.resolve_root(corpus)
        with pytest.raises(ToolError, match="outside the files root"):
            fs.list_files(root, bad)
        with pytest.raises(ToolError, match="outside the files root"):
            fs.resolve_files(root, [bad])

    @pytest.mark.skipif(os.name == "nt", reason="symlinks need privileges on Windows")
    def test_symlinks_leaving_the_root_are_ignored(self, corpus, tmp_path):
        (corpus / "link.md").symlink_to(tmp_path / "outside.md")
        (corpus / "linkdir").symlink_to(tmp_path, target_is_directory=True)
        root = fs.resolve_root(corpus)
        listed = fs.list_files(root)
        assert "link.md" not in listed
        assert not any(p.startswith("linkdir/") for p in listed)
        with pytest.raises(ToolError, match="No file"):
            fs.resolve_files(root, ["link.md"])

    def test_resolve_files_expands_globs_and_dedups(self, corpus):
        root = fs.resolve_root(corpus)
        got = fs.resolve_files(root, ["notes/b.md", "*.md", "readme.rst"])
        assert got == ["notes/b.md", "notes/a.md", "readme.rst"]

    def test_resolve_files_errors(self, corpus):
        root = fs.resolve_root(corpus)
        with pytest.raises(ToolError, match="No supported file matches"):
            fs.resolve_files(root, ["*.pdf"])
        with pytest.raises(ToolError, match="No file"):
            fs.resolve_files(root, ["notes/missing.md"])
        with pytest.raises(ToolError, match="hidden"):
            fs.resolve_files(root, [".secret.md"])
        with pytest.raises(ToolError, match="hidden"):
            fs.resolve_files(root, [".git/config.md"])
        with pytest.raises(ToolError, match="not a supported file type"):
            fs.resolve_files(root, ["data.csv"])
        with pytest.raises(ToolError, match="empty path"):
            fs.resolve_files(root, ["  "])
        with pytest.raises(ToolError, match="at most 2"):
            fs.resolve_files(root, ["**/*"], max_files=2)

    def test_middle_sample(self):
        assert fs.middle_sample("  a \n b  ", 100) == "a b"
        text = " ".join(f"w{i}" for i in range(1000))
        sample = fs.middle_sample(text, 200)
        assert len(sample) <= 200
        assert "w0 " not in sample and "w999" not in sample
        assert sample.split()[0] in text.split() and sample.split()[-1] in text.split()


# ── Rule planner ─────────────────────────────────────────────────────


class TestRulePlanner:
    @pytest.mark.parametrize(
        ("pattern", "path", "expected"),
        [
            ("*.md", "a.md", True),
            ("*.md", "notes/a.md", True),
            ("*.MD", "Notes/A.md", True),
            ("**/*.md", "a.md", True),
            ("notes/*", "notes/deep/a.md", True),
            ("notes/*", "other/a.md", False),
            ("*.pdf", "a.md", False),
            ("", "a.md", False),
            ("a.md", "notes/a.md", True),
        ],
    )
    def test_match_path(self, pattern, path, expected):
        assert match_path(pattern, path) is expected

    def test_plan_rule_from_dict(self):
        rule = PlanRule.from_dict({"match": "*.md", "chunker": "structural", "reason": "headings"})
        assert rule.match == "*.md" and rule.plan.chunker == "structural"
        assert rule.plan.extractor == planner_mod.DEFAULT_EXTRACTOR
        assert PlanRule.from_dict({}).match == "*"
        with pytest.raises(ValueError, match="unknown plan rule key"):
            PlanRule.from_dict({"match": "*", "chunk": "fixed"})
        with pytest.raises(ValueError, match="unknown chunker"):
            PlanRule.from_dict({"chunker": "semantic"})

    async def test_first_matching_rule_wins_and_default_otherwise(self):
        planner = RuleBasedIngestionPlanner(
            [
                {"match": "*.md", "chunker": "structural"},
                {"match": "*", "chunker": "fixed", "chunker_params": {"chunk_size": 1500}},
            ]
        )
        md = await planner.plan(None, source="x/a.md")
        assert md.chunker == "structural" and md.reason == "rule '*.md'"
        txt = await planner.plan(None, source="x/a.txt")
        assert txt.chunker == "fixed" and txt.chunker_params["chunk_size"] == 1500
        none = await RuleBasedIngestionPlanner([{"match": "*.md"}]).plan(None, source="a.txt")
        assert none.reason == "default: no rule matched"

    async def test_root_relative_matching(self, tmp_path):
        root = tmp_path / "notes" / "corpus"
        root.mkdir(parents=True)
        planner = RuleBasedIngestionPlanner([{"match": "notes/*", "chunker": "fixed"}], root=root)
        # "notes" is above the root, so the rule must not match.
        assert (await planner.plan(None, source=str(root / "a.md"))).chunker != "fixed"
        (root / "notes").mkdir()
        assert (await planner.plan(None, source=str(root / "notes" / "a.md"))).chunker == "fixed"
        # Outside the root: matched as given.
        assert planner.rule_for("notes/x.md") is not None

    async def test_cheap_limits_restrict_the_plan(self):
        planner = RuleBasedIngestionPlanner(
            [
                {
                    "match": "*",
                    "chunker": "contextual",
                    "extractor": "llm",
                    "resolver": "llm_verified",
                }
            ],
            **CHEAP_PLAN_LIMITS,
        )
        plan = await planner.plan(None, source="a.md")
        assert plan.chunker in CHEAP_PLAN_LIMITS["chunkers"]
        assert plan.extractor == "gliner" and plan.resolver == "exact"
        assert "restricted" in plan.reason

    def test_invalid_limits_raise(self):
        with pytest.raises(ValueError):
            RuleBasedIngestionPlanner(chunkers=("semantic",))
        with pytest.raises(ValueError, match="llm_verified"):
            RuleBasedIngestionPlanner(resolvers=("llm_verified",), max_llm_pairs=0)


# ── Pipeline drop report ──────────────────────────────────────────────


class TestDropReport:
    def _pipeline(self) -> IngestionPipeline:
        return IngestionPipeline(
            loader=MagicMock(),
            chunker=MagicMock(),
            extractor=MagicMock(),
            resolver=MagicMock(),
            graph_store=MagicMock(),
            vector_store=MagicMock(),
        )

    def test_report_counts_each_kind_of_drop(self):
        ontology = Ontology(
            entities=[Entity(label="Drug"), Entity(label="Disease")],
            relations=[Relation(label="TREATS", patterns=[("Drug", "Disease")])],
        )
        data = GraphData(
            nodes=[
                GraphNode(id="d", label="Drug"),
                GraphNode(id="x", label="Disease"),
                GraphNode(id="g", label="Gene"),
                GraphNode(id="g2", label="Gene"),
            ],
            relationships=[
                GraphRelationship(start_node_id="d", end_node_id="x", type="TREATS"),
                GraphRelationship(start_node_id="x", end_node_id="d", type="TREATS"),
                GraphRelationship(start_node_id="d", end_node_id="x", type="CAUSES"),
                GraphRelationship(start_node_id="d", end_node_id="g", type="TARGETS"),
            ],
        )
        pruned, report = self._pipeline()._prune_with_report(data, ontology)
        assert {n.id for n in pruned.nodes} == {"d", "x"}
        assert len(pruned.relationships) == 1
        assert report == {
            "entities": {"Gene": 2},
            "relations": {"CAUSES": 1},
            "pattern_mismatches": {"TREATS": {"Disease->Drug": 1}},
            "dangling": 1,
        }

    def test_unknown_entities_are_kept_and_counted(self):
        ontology = Ontology(entities=[Entity(label="Drug")])
        data = GraphData(
            nodes=[GraphNode(id="d", label="Drug"), GraphNode(id="u", label="Unknown")],
            relationships=[],
        )
        pruned, report = self._pipeline()._prune_with_report(data, ontology)
        assert len(pruned.nodes) == 2 and report == {"unknown": 1}
        # Open ontology: nothing is checked, so nothing is reported.
        assert self._pipeline()._prune_with_report(data, Ontology())[1] == {}

    def test_nothing_dropped_is_an_empty_report(self):
        ontology = Ontology(entities=[Entity(label="Drug")])
        data = GraphData(nodes=[GraphNode(id="d", label="Drug")], relationships=[])
        assert self._pipeline()._prune_with_report(data, ontology)[1] == {}
        assert self._pipeline()._prune_with_report(data, Ontology())[1] == {}


class TestPrunedTypes:
    def test_from_report_add_and_summary(self):
        assert PrunedTypes.from_report({}) is None
        assert PrunedTypes.from_report(None) is None
        a = PrunedTypes.from_report({"entities": {"Gene": 2}, "dangling": 1})
        b = PrunedTypes.from_report(
            {"entities": {"Gene": 1}, "pattern_mismatches": {"TREATS": {"Disease->Drug": 3}}}
        )
        a.add(b)
        assert a.entities == {"Gene": 3} and a.dangling == 1
        assert a.pattern_mismatches == {"TREATS": {"Disease->Drug": 3}}
        assert a.missing_types
        assert "Gene (3)" in a.summary() and "TREATS Disease->Drug (3)" in a.summary()
        assert not PrunedTypes(dangling=2).missing_types
        unknown = PrunedTypes.from_report({"unknown": 2})
        unknown.add(PrunedTypes(unknown=1))
        assert unknown.unknown == 3 and unknown.missing_types
        assert "3 entities of no declared type" in unknown.summary()

    def test_ingest_result_text(self):
        result = IngestResult(
            chunks_indexed=3,
            documents=[],
            pruned=PrunedTypes(entities={"Gene": 2}),
            hint="Call finalize.",
        )
        text = result.to_llm_text()
        assert "Gene (2)" in text and "Call finalize." in text


# ── Toolkit over a fake GraphRAG ──────────────────────────────────────


class FakeOntologyStore:
    def __init__(self, ontology: Ontology | None = None):
        self.ontology = ontology or Ontology()

    async def load(self) -> Ontology:
        return self.ontology


class FakeRAG:
    """Just enough of GraphRAG for the ingestion tools."""

    def __init__(self, stored: Ontology | None = None, ontology: Ontology | None = None):
        self._graph_store = MagicMock()
        self._conn = SimpleNamespace(topology_version=1)
        self._ontology_store = FakeOntologyStore(stored)
        self.ontology = ontology or Ontology()
        self.llm = None
        self.embedder = None
        self.calls: list[tuple[str, Any]] = []
        self.finalized = 0
        self.pruned: dict[str, dict] = {}  # file name -> drop report
        self.fail: set[str] = set()
        self.unchanged: set[str] = set()
        self.new: set[str] = set()  # update() finds no stored document
        self.reject_ontology: str | None = None

    # ontology
    async def set_ontology(self, ontology: Ontology) -> Ontology:
        if self.reject_ontology:
            raise ValueError(self.reject_ontology)
        self.calls.append(("set_ontology", ontology))
        self.ontology = ontology
        self._ontology_store.ontology = ontology
        return ontology

    async def set_entity_description(self, label: str, description: str) -> Ontology:
        self.calls.append(("entity_description", (label, description)))
        ents = [
            e.model_copy(update={"description": description}) if e.label == label else e
            for e in self.ontology.entities
        ]
        self.ontology = self.ontology.model_copy(update={"entities": ents})
        self._ontology_store.ontology = self.ontology
        return self.ontology

    async def set_relation_description(self, label: str, description: str) -> Ontology:
        self.calls.append(("relation_description", (label, description)))
        return self.ontology

    async def add_relation_pattern(self, rel: str, src: str, tgt: str) -> Ontology:
        self.calls.append(("add_pattern", (rel, src, tgt)))
        rels = [
            r.model_copy(update={"patterns": [*r.patterns, (src, tgt)]}) if r.label == rel else r
            for r in self.ontology.relations
        ]
        self.ontology = self.ontology.model_copy(update={"relations": rels})
        self._ontology_store.ontology = self.ontology
        return self.ontology

    def _ontology_entity_types(self) -> list[str] | None:
        return [e.label for e in self.ontology.entities] or None

    # ingest
    def _result(self, path: str, plan: dict | None) -> IngestionResult:
        name = os.path.basename(path)
        meta: dict[str, Any] = {"pruned": self.pruned.get(name, {})}
        if plan is not None:
            meta["plan"] = {"planner": "RuleBasedIngestionPlanner", "plan": plan}
        if name in self.unchanged:
            meta["skipped_unchanged"] = True
        return IngestionResult(
            document_info=DocumentInfo(uid=path),
            nodes_created=0 if name in self.unchanged else 2,
            relationships_created=0 if name in self.unchanged else 1,
            chunks_indexed=0 if name in self.unchanged else 1,
            metadata=meta,
        )

    async def ingest(self, source=None, *, text=None, document_id=None, planner=None, **kw):
        if text is not None:
            self.calls.append(("ingest_text", (text, document_id)))
            res = self._result(document_id or "doc", None)
            res.metadata["pruned"] = self.pruned.get("text", {})
            return res
        self.calls.append(("ingest_files", list(source)))
        out: list[Any] = []
        for path in source:
            if os.path.basename(path) in self.fail:
                out.append(RuntimeError(f"cannot read {path}"))
                continue
            p = await planner.plan(None, source=path)
            out.append(
                self._result(
                    path,
                    {
                        "chunker": p.chunker,
                        "extractor": p.extractor,
                        "resolver": p.resolver,
                        "reason": p.reason,
                    },
                )
            )
        return out

    async def update(
        self,
        source=None,
        *,
        text=None,
        document_id=None,
        force,
        if_missing,
        chunker=None,
        extractor=None,
        resolver=None,
        ctx,
    ):
        key = source if source is not None else document_id
        self.calls.append(("update", (key, force, if_missing)))
        if os.path.basename(key) in self.fail:
            raise RuntimeError("update failed")
        base = self._result(key, None)
        if text is not None:
            base.metadata["pruned"] = self.pruned.get("text", {})
        return UpdateResult(
            **base.model_dump(exclude={"metadata"}),
            metadata=base.metadata,
            replaced_existing=os.path.basename(key) not in self.new,
        )

    async def finalize(self) -> FinalizeResult:
        self.finalized += 1
        return FinalizeResult()


@pytest.fixture(autouse=True)
def _no_live_counts(monkeypatch):
    async def counts(store):
        return {}, {}

    monkeypatch.setattr(ops, "schema_counts", counts)


@pytest.fixture
def fake_strategies(monkeypatch):
    built: list[Any] = []

    def build(plan, *, llm, embedder=None, entity_types=None):
        built.append((plan, entity_types))
        return ("chunker", "extractor", "resolver")

    monkeypatch.setattr(planner_mod, "build_ingestion_strategies", build)
    return built


def toolkit(rag: FakeRAG, root=None, **kw) -> GraphRAGToolkit:
    return GraphRAGToolkit(rag, files_root=root, **kw)  # type: ignore[arg-type]


class TestToolkitConfig:
    def test_files_root_enables_sample_docs(self, corpus):
        rag = FakeRAG()
        assert "sample_docs" not in [s.name for s in toolkit(rag).tool_specs()]
        tk = toolkit(rag, corpus)
        assert tk.files_root == corpus.resolve()
        names = [s.name for s in tk.tool_specs()]
        assert names[-4:] == ["sample_docs", "set_ontology", "ingest", "finalize"]

    def test_bad_root_and_limits_fail_at_construction(self, corpus):
        with pytest.raises(ValueError, match="not a directory"):
            toolkit(FakeRAG(), corpus / "readme.rst")
        with pytest.raises(ValueError):
            toolkit(FakeRAG(), plan_limits={"chunkers": ("semantic",)})

    async def test_read_only_refuses_every_ingestion_tool(self, corpus):
        tk = toolkit(FakeRAG(), corpus, read_only=True)
        assert {"sample_docs", "set_ontology", "ingest"}.isdisjoint(s.name for s in tk.tool_specs())
        for name, args in (
            ("sample_docs", {}),
            ("set_ontology", {"entities": [{"label": "Drug"}]}),
            ("ingest", {"files": ["*.md"]}),
        ):
            with pytest.raises(ReadOnlyViolation):
                await tk.call(name, args)
        with pytest.raises(ReadOnlyViolation):
            await tk.sample_docs()

    async def test_sample_docs_without_a_root_says_so(self):
        tk = toolkit(FakeRAG())
        with pytest.raises(ToolError, match="not enabled") as exc:
            await tk.call("sample_docs", {})
        assert "No files root" in (exc.value.hint or "")
        with pytest.raises(ToolError, match="needs a files root"):
            await tk.ingest(files=["a.md"])


class TestSampleDocs:
    async def test_pages_with_middle_samples(self, corpus):
        tk = toolkit(FakeRAG(), corpus)
        first = await tk.call("sample_docs", {"limit": 3, "max_chars": 200})
        assert first.total == 4 and first.next_offset == 3
        assert first.types == {"md": 2, "rst": 1, "txt": 1}
        assert [f.path for f in first.files] == ["notes/a.md", "notes/b.md", "readme.rst"]
        a = first.files[0]
        assert a.type == "md" and a.chars > 3000 and 0 < len(a.sample) <= 200
        assert a.sample.startswith("alpha") or a.sample.startswith("words")
        assert "Keep paging" in (first.hint or "")
        rest = await tk.sample_docs(offset=first.next_offset, limit=3)
        assert [f.path for f in rest.files] == ["reports/2024/q1.txt"]
        assert rest.next_offset is None and rest.hint is None
        assert len(tk.ledger) == 0  # samples are not evidence

    async def test_budget_is_split_across_the_page(self, corpus, monkeypatch):
        import graphrag_sdk.tools.toolkit as tk_mod

        monkeypatch.setattr(tk_mod, "SAMPLE_BUDGET_CHARS", 400)
        tk = toolkit(FakeRAG(), corpus)
        result = await tk.sample_docs(limit=4, max_chars=4000)
        assert all(len(f.sample) <= 100 for f in result.files)

    async def test_pattern_and_errors(self, corpus):
        tk = toolkit(FakeRAG(), corpus)
        none = await tk.sample_docs("*.pdf")
        assert none.total == 0 and "No supported files" in (none.hint or "")
        (corpus / "broken.pdf").write_bytes(b"not a pdf")
        bad = await tk.sample_docs("*.pdf")
        assert bad.total == 1 and bad.files[0].error
        with pytest.raises(ToolError, match="outside"):
            await tk.sample_docs("../*")


class TestSetOntology:
    async def test_first_declaration_writes_once_without_defaults(self):
        rag = FakeRAG()
        tk = toolkit(rag)
        result = await tk.call(
            "set_ontology",
            {
                "entities": [
                    {"label": "Drug", "description": "A medicine"},
                    {"label": "Disease"},
                ],
                "relations": [{"label": "TREATS", "patterns": [["Drug", "Disease"]]}],
            },
        )
        assert [c[0] for c in rag.calls] == ["set_ontology"]
        declared = rag.calls[0][1]
        assert [e.label for e in declared.entities] == ["Drug", "Disease"]
        assert declared.relations[0].patterns == [("Drug", "Disease")]
        assert {t.label for t in result.entity_types} == {"Drug", "Disease"}

    async def test_unpersisted_constructor_ontology_is_merged(self):
        rag = FakeRAG(ontology=Ontology(entities=[Entity(label="Person")]))
        await toolkit(rag).set_ontology(entities=[{"label": "Company"}])
        assert [e.label for e in rag.calls[0][1].entities] == ["Person", "Company"]

    async def test_existing_types_get_descriptions_and_patterns(self):
        stored = Ontology(
            entities=[Entity(label="Drug"), Entity(label="Disease")],
            relations=[Relation(label="TREATS", patterns=[("Drug", "Disease")])],
        )
        rag = FakeRAG(stored=stored, ontology=stored)
        await toolkit(rag).set_ontology(
            entities=[{"label": "Drug", "description": "A medicine"}, {"label": "Gene"}],
            relations=[
                {"label": "TREATS", "patterns": [["Drug", "Disease"], ["Drug", "Gene"]]},
                {"label": "TARGETS", "patterns": [["Drug", "Gene"]]},
            ],
        )
        kinds = [c[0] for c in rag.calls]
        assert kinds == ["set_ontology", "entity_description", "add_pattern"]
        merged = rag.calls[0][1]
        assert [e.label for e in merged.entities] == ["Drug", "Disease", "Gene"]
        # The existing relation keeps its stored patterns in the declaration;
        # the new pattern is added on its own.
        treats = next(r for r in merged.relations if r.label == "TREATS")
        assert treats.patterns == [("Drug", "Disease")]
        assert rag.calls[2][1] == ("TREATS", "Drug", "Gene")

    async def test_nothing_new_writes_nothing(self):
        stored = Ontology(entities=[Entity(label="Drug", description="x")])
        rag = FakeRAG(stored=stored, ontology=stored)
        await toolkit(rag).set_ontology(entities=[{"label": "Drug"}])
        assert rag.calls == []

    async def test_rejections(self):
        rag = FakeRAG()
        tk = toolkit(rag)
        with pytest.raises(ToolError, match="at least one"):
            await tk.set_ontology()
        with pytest.raises(ToolError, match="undeclared entity types: Gene"):
            await tk.set_ontology(
                entities=[{"label": "Drug"}],
                relations=[{"label": "R", "patterns": [["Drug", "Gene"]]}],
            )
        for label in ("A|B", "Bad`Label"):
            with pytest.raises(ToolError, match="not a usable type name"):
                await tk.set_ontology(entities=[{"label": label}])
        rag.reject_ontology = "contradiction"
        with pytest.raises(ToolError, match="not accepted: contradiction"):
            await tk.set_ontology(entities=[{"label": "Drug"}])
        assert rag.calls == []

    async def test_invalidates_the_schema_cache(self):
        tk = toolkit(FakeRAG())
        tk._shared["schema"] = object()
        await tk.set_ontology(entities=[{"label": "Drug"}])
        assert "schema" not in tk._shared


class TestIngestFiles:
    async def test_text_mode_reports_pruned(self):
        rag = FakeRAG()
        rag.pruned["text"] = {"entities": {"Gene": 2}}
        result = await toolkit(rag).ingest("Aspirin treats headache.", document_id="d1")
        assert result.document_id == "d1" and result.pruned.entities == {"Gene": 2}
        assert "set_ontology" in result.hint and "finalize" in result.hint

    async def test_files_with_plan_rules(self, corpus):
        rag = FakeRAG()
        rag.pruned["a.md"] = {"entities": {"Gene": 2}, "dangling": 1}
        tk = toolkit(rag, corpus)
        result = await tk.call(
            "ingest",
            {
                "files": ["notes/*.md", "reports/2024/q1.txt"],
                "plan": [
                    {"match": "notes/*", "chunker": "structural"},
                    {"match": "*.txt", "chunker": "fixed", "extractor": "llm"},
                ],
            },
        )
        wanted = ("notes/a.md", "notes/b.md", "reports/2024/q1.txt")
        assert rag.calls[0] == ("ingest_files", [str(corpus.resolve() / p) for p in wanted])
        docs = {d.path: d for d in result.documents}
        assert docs["notes/a.md"].plan["chunker"] == "structural"
        # extractor "llm" is outside the cheap limits, so it was restricted.
        assert docs["reports/2024/q1.txt"].plan["extractor"] == "gliner"
        assert "restricted" in docs["reports/2024/q1.txt"].plan["reason"]
        assert all(d.status == "ingested" for d in result.documents)
        assert result.nodes_created == 6 and result.chunks_indexed == 3
        assert result.pruned.entities == {"Gene": 2} and result.pruned.dangling == 1
        assert "ingest(files=['notes/a.md'], reextract=true)" in result.hint
        assert rag.finalized == 0 and "finalize" in result.hint

    async def test_unknown_entities_suggest_a_query(self, corpus):
        rag = FakeRAG()
        rag.pruned["b.md"] = {"unknown": 3}
        result = await toolkit(rag, corpus).ingest(files=["notes/*"])
        assert result.pruned.unknown == 3
        assert "MATCH (n:Unknown)" in result.hint
        assert "ingest(files=['notes/b.md'], reextract=true)" in result.hint

    async def test_dropped_patterns_say_how_to_keep_them(self, corpus):
        rag = FakeRAG()
        rag.pruned["a.md"] = {
            "pattern_mismatches": {
                "MANAGES": {"Person->Company": 2},
                "TREATS": {"Disease->Drug": 3, "Drug->Symptom": 1},
            }
        }
        result = await toolkit(rag, corpus).ingest(files=["notes/*"])
        hint = result.hint
        assert hint.startswith("6 extracted relation(s) were dropped only because")
        assert (
            'set_ontology(relations=[{"label": "TREATS", "patterns": [["Disease", "Drug"], '
            '["Drug", "Symptom"]]}, {"label": "MANAGES", "patterns": [["Person", "Company"]]}])'
        ) in hint
        assert "ingest(files=['notes/a.md'], reextract=true), then finalize" in hint
        assert "types the ontology does not declare" not in hint

    async def test_failed_and_unchanged_files(self, corpus):
        rag = FakeRAG()
        rag.fail.add("b.md")
        rag.unchanged.add("a.md")
        result = await toolkit(rag, corpus).ingest(files=["notes/a.md", "notes/b.md"])
        statuses = {d.path: d.status for d in result.documents}
        assert statuses == {"notes/a.md": "unchanged", "notes/b.md": "failed"}
        assert result.failed == 1 and "cannot read" in result.documents[1].error
        assert "1 file(s) failed" in result.hint

        rag.fail.clear()
        again = await toolkit(rag, corpus).ingest(files=["notes/a.md"])
        assert "reextract=true" in again.hint

    async def test_reextract_updates_in_place(self, corpus, fake_strategies):
        rag = FakeRAG(ontology=Ontology(entities=[Entity(label="Drug")]))
        tk = toolkit(rag, corpus)
        result = await tk.ingest(
            files=["notes/a.md"], plan=[{"match": "*.md", "chunker": "structural"}], reextract=True
        )
        assert rag.calls == [("update", (str(corpus.resolve() / "notes/a.md"), True, "ingest"))]
        doc = result.documents[0]
        assert doc.status == "reextracted" and doc.plan["chunker"] == "structural"
        assert fake_strategies[0][1] == ["Drug"]  # extraction told the ontology's types

    async def test_reextract_failure_is_per_file(self, corpus, fake_strategies):
        rag = FakeRAG()
        rag.fail.add("a.md")
        result = await toolkit(rag, corpus).ingest(files=["notes/*"], reextract=True)
        assert [d.status for d in result.documents] == ["failed", "reextracted"]

    async def test_reextract_of_a_new_file_is_an_ingest(self, corpus, fake_strategies):
        rag = FakeRAG()
        rag.new.add("b.md")
        result = await toolkit(rag, corpus).ingest(files=["notes/*"], reextract=True)
        assert [d.status for d in result.documents] == ["reextracted", "ingested"]

    async def test_unchanged_text_says_so(self):
        rag = FakeRAG()
        rag.unchanged.add("d1")
        result = await toolkit(rag).ingest("Aspirin treats headache.", document_id="d1")
        assert result.status == "unchanged"
        assert "reextract=true" in result.hint and "'d1'" in result.hint
        assert "already stored unchanged" in result.to_llm_text()

    async def test_text_reextract_updates_in_place(self):
        rag = FakeRAG()
        rag.pruned["text"] = {"entities": {"Gene": 1}}
        tk = toolkit(rag)
        result = await tk.call(
            "ingest", {"text": "Aspirin treats headache.", "document_id": "d1", "reextract": True}
        )
        assert rag.calls == [("update", ("d1", True, "ingest"))]
        assert result.status == "reextracted" and result.document_id == "d1"
        assert "ingest(text=..., document_id='d1', reextract=true)" in result.hint
        with pytest.raises(ToolError, match="document_id"):
            await tk.call("ingest", {"text": "Aspirin.", "reextract": True})
        with pytest.raises(ToolError, match="plan applies to files only"):
            await tk.call("ingest", {"text": "Aspirin.", "plan": [{"chunker": "fixed"}]})

    async def test_on_write_policy_finalizes_once_per_call(self, corpus):
        rag = FakeRAG()
        result = await toolkit(rag, corpus, finalize_policy="on_write").ingest(files=["**/*"])
        assert result.finalized and rag.finalized == 1 and len(result.documents) == 4

    async def test_bad_file_arguments(self, corpus):
        tk = toolkit(FakeRAG(), corpus)
        with pytest.raises(ToolError, match="outside"):
            await tk.ingest(files=["../outside.md"])
        with pytest.raises(ToolError, match="Invalid arguments"):
            await tk.call("ingest", {"files": ["a.md"], "text": "x"})
        with pytest.raises(ToolError, match="Invalid arguments"):
            await tk.call("ingest", {"files": ["*.md"], "plan": [{"chunker": "semantic"}]})


# ── MCP instructions ──────────────────────────────────────────────────


class TestGuide:
    def test_guide_follows_the_served_tools(self, corpus):
        rag = FakeRAG()
        assert guide_for(toolkit(rag, corpus)) == GUIDE + INGEST_GUIDE
        assert guide_for(toolkit(rag)) == GUIDE + TEXT_INGEST_GUIDE
        assert guide_for(toolkit(rag, corpus, read_only=True)) == GUIDE

    def test_ingest_guide_asks_for_the_users_ontology_first(self):
        assert "Ask the user whether they have an ontology" in INGEST_GUIDE
        assert INGEST_GUIDE.index("set_ontology") < INGEST_GUIDE.index("sample_docs")
        for step in ("next_offset", "ONE ontology", "at most 50", "finalize once", "reextract"):
            assert step in INGEST_GUIDE

    def test_cli_reads_files_only_from_an_explicit_root(self, monkeypatch, corpus):
        pytest.importorskip("mcp")
        import sys

        import graphrag_sdk.mcp.server as server_mod
        from graphrag_sdk.mcp import __main__ as cli

        seen: list[Any] = []

        class Server:
            def __init__(self, rag, **kw):
                seen.append(kw["files_root"])

        monkeypatch.setattr(server_mod, "GraphRAGMCPServer", Server)
        monkeypatch.setattr(cli, "_build_rag", lambda: None)
        monkeypatch.setattr(cli.asyncio, "run", lambda coro: coro.close())
        monkeypatch.delenv("GRAPHRAG_MCP_FILES_ROOT", raising=False)
        monkeypatch.chdir(corpus)
        monkeypatch.setattr(sys, "argv", ["graphrag-mcp"])
        cli.main()
        monkeypatch.setattr(sys, "argv", ["graphrag-mcp", "--files-root", str(corpus)])
        cli.main()
        assert seen == [None, str(corpus)]
        monkeypatch.setattr(sys, "argv", ["graphrag-mcp", "--files-root", "/"])
        with pytest.raises(SystemExit, match="filesystem root"):
            cli.main()

    def test_server_passes_files_root_and_picks_the_guide(self, corpus):
        pytest.importorskip("mcp")
        from graphrag_sdk.mcp.server import GraphRAGMCPServer

        server = GraphRAGMCPServer(FakeRAG(), files_root=corpus)
        assert server.toolkit.files_root == corpus.resolve()
        assert server.instructions == GUIDE + INGEST_GUIDE
        assert "sample_docs" in [t["name"] for t in server.tool_dicts()]
        assert GraphRAGMCPServer(FakeRAG(), instructions="x").instructions == "x"
