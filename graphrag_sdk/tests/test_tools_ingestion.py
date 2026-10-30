"""Unit tests for the agent ingestion tools: root-confined file helpers, the
rule planner, the pipeline's drop report and the toolkit's preview_files /
set_ontology / ingest(files=...) over a fake GraphRAG. The same flow runs
against a real FalkorDB in test_tools_integration.py."""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from graphrag_sdk.api import main as main_mod
from graphrag_sdk.api.main import GraphRAG
from graphrag_sdk.core.exceptions import LLMError, LoaderError, ReadOnlyViolation
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

    def test_case_variants_of_one_file_are_one_path(self, corpus):
        if not (corpus / "NOTES" / "A.MD").exists():
            pytest.skip("case-sensitive filesystem")
        root = fs.resolve_root(corpus)
        got = fs.resolve_files(root, ["NOTES/A.MD", "notes/a.md", "Notes/a.MD"])
        assert got == ["notes/a.md"]
        assert fs.resolve_root(str(corpus).upper()) == root  # the root is respelled too

    def test_folder_listings_are_cached_as_sets(self, corpus):
        # A set: a big flat folder costs one lookup per name, not a scan.
        root = fs.resolve_root(corpus)
        listings: dict = {}
        assert fs._respell(root, ["notes", "a.md"], listings) == ["notes", "a.md"]
        assert listings and all(isinstance(v, set) for v in listings.values())
        assert "a.md" in listings[root / "notes"]

    @pytest.mark.skipif(os.name == "nt", reason="symlinks need privileges on Windows")
    def test_links_to_one_file_are_one_path(self, corpus):
        os.link(corpus / "notes" / "a.md", corpus / "hard.md")
        (corpus / "soft.md").symlink_to(corpus / "notes" / "b.md")
        root = fs.resolve_root(corpus)
        assert fs.resolve_files(root, ["notes/a.md", "hard.md"]) == ["notes/a.md"]
        # A symlink is stored under its target's path, so both spellings are one id.
        assert fs.resolve_files(root, ["soft.md", "notes/b.md"]) == ["notes/b.md"]
        assert fs.resolve_files(root, ["*.md"]) == ["hard.md", "notes/b.md"]

    @pytest.mark.skipif(os.name == "nt", reason="symlinks need privileges on Windows")
    def test_links_to_hidden_or_unsupported_files_are_refused(self, corpus):
        (corpus / "peek.md").symlink_to(corpus / ".secret.md")
        (corpus / "table.md").symlink_to(corpus / "data.csv")
        root = fs.resolve_root(corpus)
        with pytest.raises(ToolError, match="hidden or outside"):
            fs.resolve_files(root, ["peek.md"])
        with pytest.raises(ToolError, match="unsupported file type"):
            fs.resolve_files(root, ["table.md"])

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

    @pytest.mark.skipif(os.name == "nt", reason="symlinks need privileges on Windows")
    def test_links_to_hidden_or_unsupported_files_are_not_listed(self, corpus):
        root = fs.resolve_root(corpus)
        (root / ".private").mkdir()
        (root / ".private" / "secret.txt").write_text("hidden text")
        (root / "public.md").symlink_to(".private/secret.txt")
        (root / "table.md").symlink_to("data.csv")
        (root / "alias.md").symlink_to("notes/b.md")
        listed = fs.list_files(root)
        assert "public.md" not in listed and "table.md" not in listed
        assert "alias.md" in listed
        # One such link no longer fails the whole glob.
        assert sorted(fs.resolve_files(root, ["*.md"])) == ["notes/a.md", "notes/b.md"]
        with pytest.raises(ToolError, match="hidden"):
            fs.resolve_files(root, ["public.md"])

    def test_sample_of_text_without_whitespace(self, tmp_path):
        p = tmp_path / "big.txt"
        p.write_text("a" * 20000)
        sample, chars, estimated = fs.sample_file(p, 1000)
        assert sample == "a" * 1000 and estimated and chars == 20000

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
            "pattern_examples": {"TREATS": {"Disease->Drug": "x -> d"}},
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
        ends = PrunedTypes.from_report({"unknown_endpoints": {"WORKS_AT": 2}})
        ends.add(PrunedTypes(unknown_endpoints={"WORKS_AT": 1, "OWNS": 1}))
        assert ends.unknown_endpoints == {"WORKS_AT": 3, "OWNS": 1} and ends.missing_types
        assert "relations to Unknown entities WORKS_AT (3), OWNS (1)" in ends.summary()

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

    # Re-extraction plans with the real GraphRAG helper, like a first ingest.
    _plan_ingestion_strategies = GraphRAG._plan_ingestion_strategies
    _markdown_elements = staticmethod(GraphRAG._markdown_elements)
    _chunk_budget = staticmethod(GraphRAG._chunk_budget)

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
                out.append(LoaderError(f"cannot read {path}"))
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
        loader=None,
        chunker=None,
        extractor=None,
        resolver=None,
        ctx,
    ):
        key = source if source is not None else document_id
        self.calls.append(("update", (key, force, if_missing)))
        self.update_strategies = (loader, chunker, extractor, resolver)
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
    """Records ``[chunker name, extraction entity types]`` per planned file."""
    built: list[list[Any]] = []

    def chunker(name, *, llm=None, params=None):
        built.append([name, None])
        return f"{name} chunker"

    def extractor(name, *, llm, entity_types=None, params=None):
        built[-1][1] = entity_types  # built right after its chunker, with no await between
        return f"{name} extractor"

    monkeypatch.setattr(main_mod, "build_chunker", chunker)
    monkeypatch.setattr(main_mod, "build_extractor", extractor)
    monkeypatch.setattr(main_mod, "build_resolver", lambda name, **kw: f"{name} resolver")
    return built


def toolkit(rag: FakeRAG, root=None, **kw) -> GraphRAGToolkit:
    return GraphRAGToolkit(rag, files_root=root, **kw)  # type: ignore[arg-type]


class TestToolkitConfig:
    def test_files_root_enables_preview_files(self, corpus):
        rag = FakeRAG()
        assert "preview_files" not in [s.name for s in toolkit(rag).tool_specs()]
        tk = toolkit(rag, corpus)
        assert tk.files_root == corpus.resolve()
        names = [s.name for s in tk.tool_specs()]
        assert names[-4:] == ["preview_files", "set_ontology", "ingest", "finalize"]

    def test_bad_root_and_limits_fail_at_construction(self, corpus):
        with pytest.raises(ValueError, match="not a directory"):
            toolkit(FakeRAG(), corpus / "readme.rst")
        with pytest.raises(ValueError):
            toolkit(FakeRAG(), plan_limits={"chunkers": ("semantic",)})

    async def test_read_only_refuses_every_ingestion_tool(self, corpus):
        tk = toolkit(FakeRAG(), corpus, read_only=True)
        assert {"preview_files", "set_ontology", "ingest"}.isdisjoint(
            s.name for s in tk.tool_specs()
        )
        for name, args in (
            ("preview_files", {}),
            ("set_ontology", {"entities": [{"label": "Drug"}]}),
            ("ingest", {"files": ["*.md"]}),
        ):
            with pytest.raises(ReadOnlyViolation):
                await tk.call(name, args)
        with pytest.raises(ReadOnlyViolation):
            await tk.preview_files()

    async def test_preview_files_without_a_root_says_so(self):
        tk = toolkit(FakeRAG())
        with pytest.raises(ToolError, match="not enabled") as exc:
            await tk.call("preview_files", {})
        assert "No files root" in (exc.value.hint or "")
        with pytest.raises(ToolError, match="needs a files root"):
            await tk.ingest(files=["a.md"])


class TestPreviewFiles:
    async def test_pages_with_middle_samples(self, corpus):
        tk = toolkit(FakeRAG(), corpus)
        first = await tk.call("preview_files", {"limit": 3, "max_chars": 200})
        assert first.total == 4 and first.next_offset == 3
        assert first.types == {"md": 2, "rst": 1, "txt": 1}
        assert [f.path for f in first.files] == ["notes/a.md", "notes/b.md", "readme.rst"]
        a = first.files[0]
        assert a.type == "md" and a.chars > 3000 and 0 < len(a.sample) <= 200
        assert a.sample.startswith("alpha") or a.sample.startswith("words")
        assert "Keep paging" in (first.hint or "")
        rest = await tk.preview_files(offset=first.next_offset, limit=3)
        assert [f.path for f in rest.files] == ["reports/2024/q1.txt"]
        assert rest.next_offset is None and rest.hint is None
        assert len(tk.ledger) == 0  # samples are not evidence

    async def test_budget_is_split_across_the_page(self, corpus, monkeypatch):
        import graphrag_sdk.tools.toolkit as tk_mod

        monkeypatch.setattr(tk_mod, "SAMPLE_BUDGET_CHARS", 400)
        tk = toolkit(FakeRAG(), corpus)
        result = await tk.preview_files(limit=4, max_chars=4000)
        assert all(len(f.sample) <= 100 for f in result.files)

    async def test_pattern_and_errors(self, corpus):
        tk = toolkit(FakeRAG(), corpus)
        none = await tk.preview_files("*.pdf")
        assert none.total == 0 and "No supported files" in (none.hint or "")
        (corpus / "broken.pdf").write_bytes(b"not a pdf")
        bad = await tk.preview_files("*.pdf")
        assert bad.total == 1 and bad.files[0].error
        with pytest.raises(ToolError, match="outside"):
            await tk.preview_files("../*")


def _text_pdf(path, pages: int) -> None:
    """A PDF whose page i reads 'pageI words here'."""
    from pypdf import PdfWriter
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

    writer = PdfWriter()
    font = writer._add_object(
        DictionaryObject(
            {
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            }
        )
    )
    for i in range(pages):
        page = writer.add_blank_page(612, 792)
        content = DecodedStreamObject()
        content.set_data(f"BT /F1 12 Tf 72 712 Td (page{i} words here) Tj ET".encode())
        page[NameObject("/Contents")] = writer._add_object(content)
        page[NameObject("/Resources")] = DictionaryObject(
            {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})}
        )
    with open(path, "wb") as fh:
        writer.write(fh)


class _ReadCounter:
    """Counts the bytes read through every Path.open of the test."""

    def __init__(self, monkeypatch):
        from pathlib import Path

        self.bytes = 0
        original = Path.open
        counter = self

        class Handle:
            def __init__(self, fh):
                self._fh = fh

            def read(self, *args):
                data = self._fh.read(*args)
                counter.bytes += len(data)
                return data

            def __getattr__(self, name):
                return getattr(self._fh, name)

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                self._fh.close()

        monkeypatch.setattr(Path, "open", lambda p, *a, **k: Handle(original(p, *a, **k)))


class TestBoundedPreview:
    async def test_a_large_text_file_is_read_only_around_its_middle(self, tmp_path, monkeypatch):
        root = tmp_path / "big"
        root.mkdir()
        half = "début " * 400_000  # ~2.8 MB of two-byte characters each side
        (root / "big.txt").write_text(half + "MIDDLE_MARK " + "fin " * 700_000)
        size = (root / "big.txt").stat().st_size
        counter = _ReadCounter(monkeypatch)
        result = await toolkit(FakeRAG(), root).preview_files(max_chars=200)
        f = result.files[0]
        assert counter.bytes <= 4096  # the window, not the ~5.6 MB file
        assert f.size_bytes == size and f.chars_estimated
        assert abs(f.chars - (size - 400_000)) < size * 0.05  # 'é' is 2 bytes, 1 char
        assert 0 < len(f.sample) <= 200 and "\ufffd" not in f.sample
        assert f"(~{f.chars} chars)" in result.to_llm_text()

    def test_middle_window_is_cut_at_words(self, tmp_path):
        path = tmp_path / "a.txt"
        path.write_text(" ".join(f"w{i}" for i in range(20_000)))
        sample, chars, estimated = fs.sample_file(path, 100)
        words = sample.split()
        assert estimated and chars == path.stat().st_size
        assert all(w.startswith("w") and w[1:].isdigit() for w in words)
        middle = [int(w[1:]) for w in words]
        assert middle == list(range(middle[0], middle[0] + len(middle)))  # contiguous
        assert 9_000 < middle[0] < 11_000

    def test_a_small_file_is_read_whole_and_counted_exactly(self, tmp_path):
        path = tmp_path / "a.md"
        path.write_text("# Title\n\nshort body")
        assert fs.sample_file(path, 1000) == ("# Title short body", 19, False)

    async def test_a_pdf_parses_only_its_middle_pages(self, tmp_path, monkeypatch):
        pytest.importorskip("pypdf")
        root = tmp_path / "pdfs"
        root.mkdir()
        _text_pdf(root / "r.pdf", 9)
        result = await toolkit(FakeRAG(), root).preview_files()
        f = result.files[0]
        assert f.error is None and f.chars_estimated
        assert f.sample.startswith("page4") and "page0" not in f.sample
        assert "page8" not in f.sample  # stopped after a few pages

    def test_middle_pages_stop_once_the_sample_is_full(self):
        read: list[int] = []

        def page(i):
            read.append(i)
            return "x" * 400

        _, chars, estimated = fs._middle_pages(100, page, 500)
        assert read == [50, 51] and estimated and chars == 40_100  # (800 + 2) / 2 pages * 100


class TestFileSizeCap:
    async def test_an_oversize_file_is_refused_with_a_hint(self, corpus):
        rag = FakeRAG()
        tk = toolkit(rag, corpus, max_file_bytes=1000)
        with pytest.raises(ToolError, match="larger than max_file_bytes") as exc:
            await tk.ingest(files=["notes/*.md"])
        assert "notes/a.md" in str(exc.value) and "notes/b.md" not in str(exc.value)
        assert "Split the file or raise max_file_bytes" in (exc.value.hint or "")
        assert rag.calls == []  # nothing ingested
        await tk.ingest(files=["notes/b.md"])  # a small one still goes through

    def test_default_and_validation(self, corpus):
        assert toolkit(FakeRAG(), corpus)._max_file_bytes == 50 * 1024 * 1024
        assert toolkit(FakeRAG(), corpus, max_file_bytes=None)._max_file_bytes is None
        for bad in (0, -1, 1.5, True):
            with pytest.raises(ValueError, match="max_file_bytes"):
                toolkit(FakeRAG(), corpus, max_file_bytes=bad)

    async def test_files_are_ingested_through_the_confined_loader(self, corpus):
        rag = FakeRAG()
        seen = []
        original = rag.ingest

        async def ingest(source=None, **kw):
            seen.append(kw.get("loader"))
            return await original(source, **kw)

        rag.ingest = ingest
        await toolkit(rag, corpus).ingest(files=["notes/b.md"])
        assert type(seen[0]).__name__ == "_ConfinedLoader"


class TestConfinedLoader:
    async def test_loads_the_file_by_its_real_path(self, corpus):
        from graphrag_sdk.core.context import Context
        from graphrag_sdk.tools.toolkit import _ConfinedLoader

        loader = _ConfinedLoader(corpus, None)
        doc = await loader.load(str(corpus / "notes" / "b.md"), Context())
        assert doc.text.startswith("# Beta") and doc.elements  # Markdown loader
        assert doc.document_info.path == str(corpus / "notes" / "b.md")

    @pytest.mark.skipif(os.name == "nt", reason="symlinks need privileges on Windows")
    async def test_a_symlink_swapped_in_after_validation_is_refused(self, corpus, tmp_path):
        from graphrag_sdk.core.context import Context
        from graphrag_sdk.tools.toolkit import _ConfinedLoader

        target = corpus / "notes" / "b.md"
        target.unlink()
        target.symlink_to(tmp_path / "outside.md")  # after resolve_files accepted it
        with pytest.raises(ToolError, match="no longer a file under the files root"):
            await _ConfinedLoader(corpus, None).load(str(target), Context())

    async def test_a_file_that_grew_past_the_cap_is_refused(self, corpus):
        from graphrag_sdk.core.context import Context
        from graphrag_sdk.tools.toolkit import _ConfinedLoader

        with pytest.raises(ToolError, match="larger than max_file_bytes"):
            await _ConfinedLoader(corpus, 100).load(str(corpus / "notes" / "a.md"), Context())


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

    async def test_a_blank_label_is_a_tool_error_with_a_hint(self):
        tk = toolkit(FakeRAG())
        for args in ({"entities": [{"label": "   "}]}, {"relations": [{"label": "\t"}]}):
            with pytest.raises(ToolError, match="not a usable type name") as err:
                await tk.call("set_ontology", args)
            assert "letters, digits, spaces" in err.value.hint

    @pytest.mark.parametrize(
        "label", ["Drug:Name", "a\nb", "tab\there", "zero\u200bwidth", "__Entity__", "__X__"]
    )
    async def test_labels_that_break_or_shadow_the_graph_are_refused(self, label):
        rag = FakeRAG()
        with pytest.raises(ToolError, match="not a usable type name") as err:
            await toolkit(rag).set_ontology(entities=[{"label": label}])
        assert "':'" in err.value.hint and "__X__" in err.value.hint
        with pytest.raises(ToolError, match="not a usable type name"):
            await toolkit(rag).set_ontology(relations=[{"label": label}])
        assert rag.calls == []

    async def test_spaces_hyphens_and_unicode_letters_are_fine(self):
        rag = FakeRAG()
        labels = ["Drug Brand", "Side-effect", "Médicament", "製品", "_private", "Gene__X"]
        await toolkit(rag).set_ontology(entities=[{"label": label} for label in labels])
        assert [e.label for e in rag.ontology.entities] == labels

    @pytest.mark.parametrize("label", ["Chunk", "Document", "Unknown"])
    async def test_reserved_labels_get_their_own_hint(self, label):
        rag = FakeRAG()
        with pytest.raises(ToolError, match=f"'{label}' is reserved by the SDK") as err:
            await toolkit(rag).set_ontology(entities=[{"label": label}])
        assert "choose another name" in err.value.hint
        assert "Existing types cannot change" not in err.value.hint
        assert rag.calls == []

    async def test_invalidates_the_schema_cache(self):
        tk = toolkit(FakeRAG())
        tk._shared["schema"] = object()
        await tk.set_ontology(entities=[{"label": "Drug"}])
        assert "schema" not in tk._shared


class TestFileErrorsAreRedacted:
    SECRET = "Incorrect API key provided: sk-SECRET123456 (api_key=hunter2)"

    async def test_ingest(self, corpus):
        rag = FakeRAG()
        secret = self.SECRET

        async def ingest(source=None, **kw):
            return [RuntimeError(f"Pipeline failed: {secret}") for _ in source]

        rag.ingest = ingest
        result = await toolkit(rag, corpus).ingest(files=["notes/b.md"])
        dumped = result.model_dump_json()
        assert "SECRET" not in dumped and "hunter2" not in dumped
        assert result.documents[0].status == "failed"
        assert "RuntimeError" in result.documents[0].error

    @pytest.mark.parametrize(
        "error",
        [LLMError("Incorrect API key provided"), RuntimeError("Incorrect API key provided")],
    )
    async def test_a_message_redact_cannot_mask_is_withheld(self, corpus, error):
        # redact() masks key-shaped text only; the MCP server withholds such messages.
        rag = FakeRAG()

        async def ingest(source=None, **kw):
            return [error for _ in source]

        rag.ingest = ingest
        result = await toolkit(rag, corpus).ingest(files=["notes/b.md"])
        assert "Incorrect API key" not in result.model_dump_json()
        assert type(error).__name__ in result.documents[0].error
        assert "withheld" in result.documents[0].error

    async def test_the_sdk_messages_about_the_file_are_kept_masked(self, corpus):
        rag = FakeRAG()
        secret = self.SECRET
        budget = planner_mod.ChunkBudgetExceededError("The document makes 900 chunks")

        async def ingest(source=None, **kw):
            return [LoaderError(f"Failed to read {secret}"), budget]

        rag.ingest = ingest
        result = await toolkit(rag, corpus).ingest(files=["notes/a.md", "notes/b.md"])
        errors = [d.error for d in result.documents]
        assert "SECRET" not in result.model_dump_json()
        assert errors[0].startswith("Failed to read") and "sk-***" in errors[0]
        assert errors[1] == "The document makes 900 chunks"

    async def test_reextract(self, corpus, fake_strategies):
        rag = FakeRAG()
        secret = self.SECRET

        async def update(*a, **kw):
            raise RuntimeError(secret)

        rag.update = update
        result = await toolkit(rag, corpus).ingest(files=["notes/b.md"], reextract=True)
        assert "SECRET" not in result.model_dump_json()
        assert result.documents[0].status == "failed"

    async def test_preview(self, corpus, monkeypatch):
        secret = self.SECRET

        def broken(path, max_chars):
            raise RuntimeError(secret)

        monkeypatch.setattr(fs, "sample_file", broken)
        result = await toolkit(FakeRAG(), corpus).preview_files("notes/b.md")
        assert result.files[0].error and "SECRET" not in result.model_dump_json()


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
        # Unknown may be a declared type classified with low confidence.
        assert "3 entities have an unresolved type" in result.hint
        assert "check them before adding types" in result.hint
        assert "types the ontology does not declare" not in result.hint
        assert "ingest(files=['notes/b.md'], reextract=true)" in result.hint

    async def test_relations_to_unknown_entities_hint_at_the_entity_type(self, corpus):
        rag = FakeRAG()
        rag.pruned["a.md"] = {"unknown": 2, "unknown_endpoints": {"WORKS_AT": 3}}
        result = await toolkit(rag, corpus).ingest(files=["notes/*"])
        assert result.pruned.unknown_endpoints == {"WORKS_AT": 3}
        assert result.pruned.pattern_mismatches == {}
        hint = result.hint
        assert "3 relation(s) (WORKS_AT) were dropped because an endpoint's type was not " in hint
        assert "resolved (stored as Unknown)" in hint and "is not declared" not in hint
        assert "classified with low confidence, so check those entities first" in hint
        assert "Only if they are of a type the ontology lacks, declare it with " in hint
        assert "set_ontology(entities=[...])" in hint
        assert 'Unknown"]' not in hint and "'Unknown']" not in hint
        assert "ingest(files=['notes/a.md'], reextract=true)" in hint

    async def test_dropped_patterns_say_how_to_keep_them(self, corpus):
        rag = FakeRAG()
        rag.pruned["a.md"] = {
            "pattern_mismatches": {
                "MANAGES": {"Person->Company": 2},
                "TREATS": {"Disease->Drug": 3, "Drug->Symptom": 1},
            },
            "pattern_examples": {"MANAGES": {"Person->Company": "Ann -> Acme: Ann runs Acme"}},
        }
        result = await toolkit(rag, corpus).ingest(files=["notes/*"])
        assert result.pruned.pattern_examples == {
            "MANAGES": {"Person->Company": "Ann -> Acme: Ann runs Acme"}
        }
        hint = result.hint
        assert hint.startswith("6 extracted relation(s) were dropped only because")
        assert "Dropped, e.g.: MANAGES Person->Company: Ann -> Acme: Ann runs Acme." in hint
        assert (
            'set_ontology(relations=[{"label": "TREATS", "patterns": [["Disease", "Drug"], '
            '["Drug", "Symptom"]]}, {"label": "MANAGES", "patterns": [["Person", "Company"]]}])'
        ) in hint
        assert "ingest(files=['notes/a.md'], reextract=true), then finalize" in hint
        assert "Keep only the pairs whose facts the relation's meaning fits" in hint
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

    async def test_reextract_keeps_the_last_plan(self, corpus, fake_strategies):
        rag = FakeRAG()
        tk = toolkit(rag, corpus)
        await tk.ingest(files=["notes/*"], plan=[{"match": "a.md", "chunker": "structural"}])
        result = await tk.new_session().ingest(files=["notes/*"], reextract=True)
        plans = {d.path: d.plan for d in result.documents}
        assert plans["notes/a.md"]["chunker"] == "structural"
        assert plans["notes/a.md"]["reason"] == "the plan of this file's last ingest"
        assert plans["notes/b.md"]["chunker"] != "structural"
        built = sorted(name for name, _ in fake_strategies)
        assert built == sorted(["structural", plans["notes/b.md"]["chunker"]])

        given = await tk.ingest(
            files=["notes/a.md"], plan=[{"match": "*", "chunker": "sentence"}], reextract=True
        )
        assert given.documents[0].plan["chunker"] == "sentence"  # a passed plan wins
        again = await tk.ingest(files=["notes/a.md"], reextract=True)
        assert again.documents[0].plan["chunker"] == "sentence"

    async def test_reextract_without_a_remembered_plan_says_so(self, corpus, fake_strategies):
        tk = toolkit(FakeRAG(), corpus)  # a fresh toolkit, as after a restart
        first = await tk.ingest(files=["notes/a.md"], reextract=True)
        reason = first.documents[0].plan["reason"]
        assert "kept in memory only" in reason and "plan=[...]" in reason
        again = await tk.ingest(files=["notes/a.md"], reextract=True)
        assert again.documents[0].plan["reason"] == "the plan of this file's last ingest"
        given = await tk.ingest(
            files=["notes/a.md"], plan=[{"match": "*", "chunker": "sentence"}], reextract=True
        )
        assert "memory" not in given.documents[0].plan["reason"]

    async def test_reextract_parses_markdown_in_a_txt_like_a_first_ingest(
        self, corpus, fake_strategies
    ):
        (corpus / "notes" / "md.txt").write_text("# Intro\n\nAlice works at Acme.\n\n# Later\n")
        rag = FakeRAG()
        structural = [{"match": "*", "chunker": "structural"}]
        result = await toolkit(rag, corpus).ingest(
            files=["notes/md.txt", "reports/2024/q1.txt"], plan=structural, reextract=True
        )
        plans = {d.path: d.plan for d in result.documents}
        assert plans["notes/md.txt"]["chunker"] == "structural"
        # Plain text has no structure: the plan says what really chunks it.
        assert plans["reports/2024/q1.txt"]["chunker"] == "sentence"
        assert sorted(name for name, _ in fake_strategies) == ["sentence", "structural"]

    async def test_reextract_hands_the_parsed_elements_to_update(self, corpus, fake_strategies):
        (corpus / "notes" / "md.txt").write_text("# Intro\n\nAlice works at Acme.\n\n# Later\n")
        rag = FakeRAG()
        await toolkit(rag, corpus).ingest(
            files=["notes/md.txt"], plan=[{"match": "*", "chunker": "structural"}], reextract=True
        )
        document = await rag.update_strategies[0].load("ignored", None)
        assert document.elements and document.document_info.path.endswith("md.txt")

    async def test_reextract_falls_back_slot_by_slot(self, corpus, fake_strategies, monkeypatch):
        def broken(name, **kw):
            raise RuntimeError("no gliner here")

        monkeypatch.setattr(main_mod, "build_extractor", broken)
        rag = FakeRAG()
        result = await toolkit(rag, corpus).ingest(
            files=["notes/a.md"], plan=[{"match": "*", "chunker": "fixed"}], reextract=True
        )
        doc = result.documents[0]
        assert doc.status == "reextracted" and doc.plan["chunker"] == "fixed"
        _, chunker, extractor, resolver = rag.update_strategies
        assert chunker.chunker == "fixed chunker"  # under the default max_chunks budget
        assert extractor is None  # the default extractor
        assert resolver.endswith("resolver")

    async def test_reextract_keeps_the_chunk_budget(self, corpus, fake_strategies):
        from graphrag_sdk.ingestion.ingestion_planner import ChunkBudget

        rag = FakeRAG()
        tk = toolkit(rag, corpus, plan_limits={"max_chunks": 7})
        await tk.ingest(files=["notes/a.md"])
        await tk.ingest(files=["notes/a.md"], reextract=True)  # with the remembered plan
        chunker = rag.update_strategies[1]
        assert isinstance(chunker, ChunkBudget) and chunker.max_chunks == 7

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

    async def test_on_write_policy_skips_finalize_when_nothing_changed(self, corpus):
        rag = FakeRAG()
        rag.unchanged.update({"a.md", "b.md", "d1"})
        tk = toolkit(rag, corpus, finalize_policy="on_write")
        files = await tk.ingest(files=["notes/*"])
        assert {d.status for d in files.documents} == {"unchanged"}
        text = await tk.ingest("Aspirin treats headache.", document_id="d1")
        assert text.status == "unchanged"
        assert rag.finalized == 0 and not files.finalized and not text.finalized
        assert "Not finalized" not in files.to_llm_text() + text.to_llm_text()
        rag.unchanged.discard("b.md")
        partly = await tk.ingest(files=["notes/*"])
        assert partly.finalized and rag.finalized == 1

    async def test_on_write_policy_skips_finalize_when_nothing_was_written(self, corpus):
        rag = FakeRAG()
        rag.fail.add("b.md")
        rag.unchanged.add("a.md")
        tk = toolkit(rag, corpus, finalize_policy="on_write")
        failed = await tk.ingest(files=["notes/b.md"])
        mixed = await tk.ingest(files=["notes/*"])
        assert failed.failed == 1 and mixed.failed == 1
        assert rag.finalized == 0 and not failed.finalized and not mixed.finalized

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

    @pytest.mark.parametrize("files", [True, False])
    @pytest.mark.parametrize("policy", ["on_write", "never"])
    def test_guide_leaves_out_finalize_when_it_is_not_served(self, corpus, policy, files):
        tk = toolkit(FakeRAG(), corpus if files else None, finalize_policy=policy)
        assert "finalize" not in {s.name for s in tk.tool_specs()}
        guide = guide_for(tk)
        assert "ingest(" in guide and "set_ontology" in guide
        assert "finalize" not in guide

    @pytest.mark.parametrize("files", [True, False])
    def test_guide_leaves_out_set_ontology_when_it_is_not_served(self, corpus, files):
        include = ["ingest", "finalize", "get_ontology"] + (["preview_files"] if files else [])
        tk = toolkit(FakeRAG(), corpus if files else None, include=include)
        guide = guide_for(tk)
        assert "set_ontology" not in guide
        assert "finalize" in guide and "server operator" in guide
        if files:
            assert "preview_files" in guide and "get_ontology" in guide

    def test_ingest_guide_asks_for_the_users_ontology_first(self):
        assert "Ask the user whether they have an ontology" in INGEST_GUIDE
        assert INGEST_GUIDE.index("set_ontology") < INGEST_GUIDE.index("preview_files")
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
        assert "preview_files" in [t["name"] for t in server.tool_dicts()]
        assert GraphRAGMCPServer(FakeRAG(), instructions="x").instructions == "x"
        assert server.toolkit._max_file_bytes == fs.MAX_FILE_BYTES
        capped = GraphRAGMCPServer(FakeRAG(), files_root=corpus, max_file_bytes=10)
        assert capped.toolkit._max_file_bytes == 10
