# Tests for the LLM-judged cross-document deduplication phase.

from __future__ import annotations

import asyncio
import re
from types import SimpleNamespace

import numpy as np
import pytest

from graphrag_sdk.core.providers.base import LLMBatchItem
from graphrag_sdk.storage.deduplicator import EntityDeduplicator
from graphrag_sdk.storage.judge_dedup import (
    MAX_DESC_CHARS,
    NAME_IN_DESC_CAP,
    PROMPT_TOKEN_BUDGET,
    LLMJudgeDeduplicator,
    knn_pairs,
    make_groups_dense,
    name_in_desc_pairs,
    pack_prompts,
    pairs_from_response,
    parse_groups,
    render_set,
)

# ── pure helpers ────────────────────────────────────────────────────────────


def test_knn_pairs_gate_and_symmetry():
    v = np.array([[1, 0], [0.99, 0.1], [0, 1], [0, 0]], dtype=np.float32)
    pairs = knn_pairs(v, k=3, gate=0.65)
    assert (0, 1) in pairs
    assert (0, 2) not in pairs
    assert all(a < b for a, b in pairs)
    assert not any(3 in p for p in pairs)  # zero vector never nominated


def test_name_in_desc_pairs_requires_all_name_tokens():
    ents = [
        {"name": "the museum", "description": "A cottage."},
        {"name": "Whitford Museum", "description": "The museum in Whitford founded 1902."},
        {"name": "Alice", "description": "Visited the Whitford Museum."},
    ]
    pairs = name_in_desc_pairs(ents)
    assert (1, 2) in pairs  # 'whitford museum' both in Alice's description
    assert (0, 1) in pairs  # 'museum' in Whitford Museum's description
    assert (0, 2) in pairs


def test_make_groups_dense_blocks_chains_and_caps():
    chain = {(i, i + 1): 0.9 for i in range(5)}
    sets = make_groups_dense(chain, cap=8)
    assert all(len(s) <= 4 for s in sets)  # density < .5 stops a long chain
    assert {tuple(e) for e in chain} <= {(s[0], s[1]) for s in sets if len(s) == 2} | {
        (a, b) for s in sets for a in s for b in s if a < b
    }  # every nominated edge is still judged somewhere
    clique = {(a, b): 0.9 for a in range(4) for b in range(a + 1, 4)}
    assert make_groups_dense(clique, cap=8) == [[0, 1, 2, 3]]
    assert all(len(s) <= 2 for s in make_groups_dense(clique, cap=2))


def test_parse_groups_and_pairs():
    text = "SET 1 GROUP: 1, 3\nSET 1 GROUP: 2\nSET 2 GROUP: 1, 2\nSET 2 GROUP: 9\nnoise"
    groups = parse_groups(text, [3, 2])
    assert groups == {0: [[0, 2], [1]], 1: [[0, 1]]}
    pairs = pairs_from_response(text, [[10, 11, 12], [20, 21]])
    assert pairs == {(10, 12), (20, 21)}


def test_pairs_from_response_empty_on_garbage():
    assert pairs_from_response("I cannot decide.", [[1, 2]]) == set()


def test_overlapping_groups_are_not_a_partition_and_merge_nothing():
    """``1, 2`` and ``1, 3`` would merge 2 and 3 through 1 though the model
    never grouped them; the whole set is discarded, the other set is kept."""
    text = "SET 1 GROUP: 1, 2\nSET 1 GROUP: 1, 3\nSET 2 GROUP: 1, 2"
    assert parse_groups(text, [3, 2]) == {1: [[0, 1]]}
    assert pairs_from_response(text, [[10, 11, 12], [20, 21]]) == {(20, 21)}


def test_name_in_desc_pairs_caps_common_words():
    """A one-word name found in more descriptions than the cap is a common
    word, not an identifying mention; it nominates nothing."""
    n = NAME_IN_DESC_CAP + 5
    ents = [{"name": "Company", "description": "An organisation."}] + [
        {"name": f"Widgets{i}", "description": f"Company number {i}."} for i in range(n)
    ]
    assert name_in_desc_pairs(ents) == {}
    ents = [{"name": "Company", "description": "An organisation."}] + [
        {"name": f"Widgets{i}", "description": f"Company number {i}."} for i in range(3)
    ]
    assert set(name_in_desc_pairs(ents)) == {(0, 1), (0, 2), (0, 3)}


def test_pack_prompts_lists_every_set():
    ents = [{"name": f"E{i}", "label": "Person", "description": "d" * 50} for i in range(6)]
    prompts, orders = pack_prompts([[0, 1], [2, 3, 4], [5, 0]], ents)
    assert sum(len(o) for o in orders) == 3
    assert "E3 [Person]" in "".join(prompts)


def test_render_set_shows_every_label_and_caps_the_description():
    ents = [
        {"name": "Acme", "labels": ["Organization", "Company"], "description": "x" * 5000},
        {"name": "Bob", "label": "Person", "description": ""},
    ]
    text = render_set([0, 1], ents)
    assert "1. Acme [Organization/Company]" in text
    assert "2. Bob [Person] — (no description)" in text
    assert len(text.splitlines()[0]) < MAX_DESC_CHARS + 50


def test_pack_prompts_keeps_an_oversized_set_within_budget():
    """Eight members with long merged descriptions used to be appended past
    the budget; the per-entity cap keeps one set inside it."""
    import tiktoken

    ents = [{"name": f"E{i}", "label": "Person", "description": "word " * 3000} for i in range(8)]
    prompts, orders = pack_prompts([list(range(8))], ents)
    assert len(prompts) == 1 and orders == [[list(range(8))]]
    enc = tiktoken.get_encoding("cl100k_base")
    assert len(enc.encode(prompts[0])) <= PROMPT_TOKEN_BUDGET


# ── deduplicator behaviour against a scripted graph ─────────────────────────


class ScriptedGraph:
    """Records every Cypher call; answers the entity fetch with fixed rows."""

    def __init__(self, rows):
        self.rows = rows
        self.calls: list[tuple[str, dict]] = []

    async def query_raw(self, cypher, params=None):
        self.calls.append((cypher, params or {}))
        if "RETURN e.id, e.name, e.description" in cypher:
            return SimpleNamespace(result_set=list(self.rows))
        return SimpleNamespace(result_set=[])


class FakeEmbedder:
    def __init__(self, table):
        self.table = table

    async def aembed_documents(self, texts, **kw):
        return [self.table.get(t, [0.0, 0.0, 1.0]) for t in texts]


class ScriptedLLM:
    """Oracle judge: answers by entity NAME so shuffled second passes stay
    consistent. ``per_pass`` lists, per pass, the name groups that are the
    same entity; ``None`` for a pass makes every call of that pass fail."""

    def __init__(self, per_pass):
        self.per_pass = list(per_pass)
        self.prompts: list[str] = []
        self.pass_no = 0

    async def abatch_invoke(self, prompts, **kw):
        self.prompts.extend(prompts)
        same = self.per_pass[self.pass_no] if self.pass_no < len(self.per_pass) else []
        self.pass_no += 1
        out = []
        for i, p in enumerate(prompts):
            if same is None:
                out.append(LLMBatchItem(index=i, error=RuntimeError("boom")))
                continue
            lines = []
            for si, block in enumerate(re.split(r"--- Set \d+ ---\n", p)[1:], start=1):
                block = block.split("\n\nDECISION PROCEDURE")[0]
                names = re.findall(r"^(\d+)\. (.+?) \[", block, flags=re.M)
                used = set()
                for num, name in names:
                    if num in used:
                        continue
                    grp = [n for n, nm in names if any(name in g and nm in g for g in same)] or [
                        num
                    ]
                    used.update(grp)
                    lines.append(f"SET {si} GROUP: {', '.join(grp)}")
            out.append(LLMBatchItem(index=i, response=SimpleNamespace(content="\n".join(lines))))
        return out


IBM = {"IBM", "International Business Machines"}
ROWS = [
    # id, name, description, labels, aliases, embedding
    ("e1", "IBM", "Computer company founded 1911.", ["Organization"], None, [1.0, 0.0, 0.0]),
    (
        "e2",
        "International Business Machines",
        "American computer firm; IBM.",
        ["Company"],
        None,
        [0.95, 0.3, 0.0],
    ),
    ("e3", "Thomas Watson", "Led IBM.", ["Person"], None, [0.0, 1.0, 0.0]),
]
DESC_EMB = {
    "Computer company founded 1911.": [1.0, 0.0, 0.0],
    "American computer firm; IBM.": [0.9, 0.4, 0.0],
    "Led IBM.": [0.6, 0.8, 0.0],
}


class RecordingMerge:
    """The ``merge_group`` hook: records each agreed group and "merges" it into
    its first member by id order, so the tests see exactly what identity the
    judge handed over. How a merge is written is ``EntityDeduplicator``'s job
    and is tested through it below."""

    def __init__(self):
        self.groups: list[list[str]] = []

    async def __call__(self, members):
        ids = sorted(m["id"] for m in members)
        self.groups.append(ids)
        return ids[0], set(ids[1:])


def _run(graph, llm, vote=True, **kw):
    merge = RecordingMerge()
    dd = LLMJudgeDeduplicator(graph, FakeEmbedder(DESC_EMB), llm, merge, vote=vote)
    return asyncio.run(dd.deduplicate(**kw)), merge


def test_agreed_pair_is_handed_to_the_merge_hook_once():
    graph = ScriptedGraph(ROWS)
    # both passes: IBM == International Business Machines, Watson separate
    llm = ScriptedLLM([[IBM], [IBM]])
    stats, merge = _run(graph, llm)
    # a single 3-member set was judged twice (vote)
    assert stats["llm_calls"] == 2 and stats["merged"] == 1 and stats["linked"] == 0
    assert merge.groups == [["e1", "e2"]]
    # the judge writes no merge of its own
    assert not any("DETACH DELETE" in c[0] or "SET s." in c[0] for c in graph.calls)


def test_disagreement_links_instead_of_merging():
    graph = ScriptedGraph(ROWS)
    llm = ScriptedLLM([[IBM], []])
    stats, merge = _run(graph, llm)
    assert stats["merged"] == 0 and stats["linked"] == 1 and stats["disagreed_pairs"] == 1
    assert merge.groups == []
    link = [c for c in graph.calls if "SAME_AS" in c[0]]
    assert len(link) == 1 and {link[0][1]["a"], link[0][1]["b"]} == {"e1", "e2"}


def test_no_vote_merges_on_single_pass():
    graph = ScriptedGraph(ROWS)
    llm = ScriptedLLM([[IBM]])
    stats, _ = _run(graph, llm, vote=False)
    assert stats["llm_calls"] == 1 and stats["merged"] == 1


def test_failed_llm_call_merges_nothing():
    graph = ScriptedGraph(ROWS)
    llm = ScriptedLLM([None, [IBM]])
    stats, merge = _run(graph, llm)
    assert stats["llm_failed"] == 1 and stats["merged"] == 0 and merge.groups == []
    assert stats["linked"] == 1  # the one-pass yes survives as a link


def test_missing_name_embedding_is_written_back():
    rows = [(*r[:5], None) for r in ROWS]
    graph = ScriptedGraph(rows)
    llm = ScriptedLLM([[], []])
    _run(graph, llm)
    writes = [c for c in graph.calls if "SET e.embedding = vecf32" in c[0]]
    assert len(writes) == 1 and {it["id"] for it in writes[0][1]["items"]} == {"e1", "e2", "e3"}


def test_nameless_entities_are_not_judged():
    """A fact row keyed on a reading id has no name; giving it its id as a
    name would let ``e2`` nominate against a prose ``E2``. It is left out."""
    rows = [ROWS[0], ("e2", None, "American computer firm; IBM.", ["Company"], None, None), ROWS[2]]
    graph = ScriptedGraph(rows)
    llm = ScriptedLLM([[IBM], [IBM]])
    stats, merge = _run(graph, llm)
    assert stats["entities"] == 2 and merge.groups == []
    assert not any("e2" in p for p in llm.prompts)


def test_distinct_ids_are_never_asked_about_each_other():
    """Two rows written from a declared key are two things whatever they are
    called: the pair is dropped before the model is called. Measured as
    ``finalize()`` spending two judge calls on E-1 / E-2 'Alice Smith' rows
    the merge rules would have refused anyway."""
    graph = ScriptedGraph(ROWS)
    llm = ScriptedLLM([[IBM], [IBM]])
    stats, merge = _run(graph, llm, distinct_ids={"e1", "e2", "e3"})
    assert stats["llm_calls"] == 0 and stats["candidates"] == 0 and merge.groups == []
    assert llm.prompts == []
    # Only the rows are protected: Watson is still judged against IBM, but the
    # two rows are never put in one set nor in one group.
    graph = ScriptedGraph(ROWS)
    llm = ScriptedLLM([[IBM], [IBM]])
    stats, merge = _run(graph, llm, distinct_ids={"e1", "e2"})
    assert merge.groups == [] and stats["merged"] == 0
    for prompt in llm.prompts:
        for block in re.split(r"--- Set \d+ ---", prompt)[1:]:
            names = re.findall(r"^\d+\. (.+?) \[", block, flags=re.M)
            assert not IBM <= set(names), block


def test_skip_pairs_are_not_nominated_linked_or_grouped_transitively():
    """A~B and A~C agreed while B|C is decided: grouping all three would fold
    B and C into one node and lose the remembered NO. B joins A; C stays out
    and is not linked to the survivor either."""
    rows = [
        ("a", "Acme", "Acme Corp, maker of anvils.", ["Organization"], None, [1.0, 0.0, 0.0]),
        ("b", "Acme Corp", "Anvil maker.", ["Organization"], None, [0.98, 0.1, 0.0]),
        ("c", "Acme Inc", "Anvils.", ["Organization"], None, [0.97, 0.15, 0.0]),
    ]
    desc = {
        "Acme Corp, maker of anvils.": [1.0, 0.0, 0.0],
        "Anvil maker.": [0.95, 0.2, 0.0],
        "Anvils.": [0.95, 0.25, 0.0],
    }
    same = {"Acme", "Acme Corp", "Acme Inc"}
    graph = ScriptedGraph(rows)
    merge = RecordingMerge()
    dd = LLMJudgeDeduplicator(graph, FakeEmbedder(desc), ScriptedLLM([[same], [same]]), merge)
    stats = asyncio.run(dd.deduplicate(skip_pairs={frozenset(("b", "c"))}))
    assert merge.groups == [["a", "b"]]
    assert stats["merged"] == 1
    assert not any("SAME_AS" in c[0] for c in graph.calls)


def test_links_are_one_per_surviving_pair_and_never_a_protected_pair():
    """A~B agreed, A~C and B~C disagreed: after B folds into A both disagreed
    pairs are the same edge, written and counted once. With A|C decided the
    edge is not written at all."""
    rows = [
        ("a", "Acme", "Acme Corp, maker of anvils.", ["Organization"], None, [1.0, 0.0, 0.0]),
        ("b", "Acme Corp", "Anvil maker.", ["Organization"], None, [0.98, 0.1, 0.0]),
        ("c", "Acme Inc", "Anvils.", ["Organization"], None, [0.97, 0.15, 0.0]),
    ]
    desc = {
        "Acme Corp, maker of anvils.": [1.0, 0.0, 0.0],
        "Anvil maker.": [0.95, 0.2, 0.0],
        "Anvils.": [0.95, 0.25, 0.0],
    }
    ab, everything = {"Acme", "Acme Corp"}, {"Acme", "Acme Corp", "Acme Inc"}
    graph = ScriptedGraph(rows)
    merge = RecordingMerge()
    dd = LLMJudgeDeduplicator(graph, FakeEmbedder(desc), ScriptedLLM([[everything], [ab]]), merge)
    stats = asyncio.run(dd.deduplicate())
    assert merge.groups == [["a", "b"]] and stats["disagreed_pairs"] == 2
    links = [c for c in graph.calls if "SAME_AS" in c[0]]
    assert len(links) == 1 and stats["linked"] == 1
    assert {links[0][1]["a"], links[0][1]["b"]} == {"a", "c"}

    graph = ScriptedGraph(rows)
    dd = LLMJudgeDeduplicator(
        graph, FakeEmbedder(desc), ScriptedLLM([[everything], [ab]]), RecordingMerge()
    )
    stats = asyncio.run(dd.deduplicate(skip_pairs={frozenset(("a", "c"))}))
    assert stats["linked"] == 0 and not any("SAME_AS" in c[0] for c in graph.calls)


def test_a_failed_link_write_is_counted_as_not_linked_and_does_not_abort():
    class FailingLinks(ScriptedGraph):
        async def query_raw(self, cypher, params=None):
            if "SAME_AS" in cypher:
                self.calls.append((cypher, params or {}))
                raise RuntimeError("write failed")
            return await super().query_raw(cypher, params)

    graph = FailingLinks(ROWS)
    stats, _ = _run(graph, ScriptedLLM([[IBM], []]))
    assert stats["disagreed_pairs"] == 1 and stats["linked"] == 0


def test_exact_phase_joins_descriptions_and_records_aliases():
    class G:
        def __init__(self):
            self.calls = []
            self.pages = [
                SimpleNamespace(
                    result_set=[
                        # id, name, description, label, aliases, is_stub, degree
                        ("a", "Globex Limited", "Maker of widgets.", "Organization", None, None, 0),
                        ("b", "Globex Ltd", "Founded 1990.", "Organization", None, None, 0),
                    ]
                ),
                SimpleNamespace(result_set=[]),
            ]

        async def query_raw(self, cypher, params=None):
            self.calls.append((cypher, params or {}))
            if "RETURN e.id" in cypher and "SKIP" in cypher:
                return self.pages.pop(0) if self.pages else SimpleNamespace(result_set=[])
            if "DETACH DELETE dup RETURN s.id" in cypher:
                return SimpleNamespace(result_set=[[params["survivor_id"]]])
            return SimpleNamespace(result_set=[])

    g = G()
    dd = EntityDeduplicator(g, FakeEmbedder({}))
    n = asyncio.run(dd.deduplicate())
    assert n == 1
    upd = [c for c in g.calls if "s.aliases = $aliases" in c[0]]
    assert len(upd) == 1
    assert upd[0][1]["desc"] == "Maker of widgets. | Founded 1990."
    assert upd[0][1]["descs"] == ["Maker of widgets.", "Founded 1990."]
    assert upd[0][1]["aliases"] == ["Globex Ltd"]


class _JudgeGraph:
    """Answers the exact phase's paged fetch and the judge's fetch with the
    same three entities, in each query's row shape; records every call."""

    def __init__(self, rows=ROWS, distinct=()):
        self.calls: list[tuple[str, dict]] = []
        self.rows = list(rows)
        self.distinct = list(distinct)
        # id, name, description, label, aliases, is_stub, degree
        self.page = [(r[0], r[1], r[2], r[3][0], r[4], None, 0) for r in self.rows]

    async def query_raw(self, cypher, params=None):
        self.calls.append((cypher, params or {}))
        if "SKIP" in cypher:
            # every phase re-reads the graph; one page, then the end
            rows = self.page if params.get("offset", 0) == 0 else []
            return SimpleNamespace(result_set=list(rows))
        if "RETURN e.id, e.name, e.description" in cypher:
            return SimpleNamespace(result_set=list(self.rows))
        if "DISTINCT_FROM" in cypher and "RETURN a.id, b.id" in cypher:
            return SimpleNamespace(result_set=list(self.distinct))
        if "DETACH DELETE dup RETURN s.id" in cypher:
            return SimpleNamespace(result_set=[[params["survivor_id"]]])
        return SimpleNamespace(result_set=[])


@pytest.mark.parametrize("judge", [False, True])
def test_entity_deduplicator_judge_toggle(judge):
    g = _JudgeGraph()
    llm = ScriptedLLM([[IBM], [IBM]])
    dd = EntityDeduplicator(g, FakeEmbedder(DESC_EMB))
    n = asyncio.run(dd.deduplicate(judge_llm=llm if judge else None))
    assert n == (1 if judge else 0)
    assert bool(dd.last_judge_stats) is judge


def test_judge_merges_through_absorb_and_unions_labels():
    """Inside EntityDeduplicator the judge decides identity and ``_absorb``
    performs the merge: one atomic absorb-and-delete write carrying the
    descriptions list, aliases and the loser's label."""
    g = _JudgeGraph()
    dd = EntityDeduplicator(g, FakeEmbedder(DESC_EMB))
    n = asyncio.run(dd.deduplicate(judge_llm=ScriptedLLM([[IBM], [IBM]])))
    assert n == 1 and dd.last_judge_stats["merged"] == 1
    # the judge's own direct write is not used; the shared absorb path is
    assert not any(
        c[0].startswith("MATCH (s:__Entity__ {id: $id}) SET s.description") for c in g.calls
    )
    absorb = [c for c in g.calls if "DETACH DELETE dup RETURN s.id" in c[0]]
    assert len(absorb) == 1
    # survivor rank: equal provenance and degree, so the longest description
    # ("American computer firm; IBM.") keeps its node
    assert absorb[0][1]["survivor_id"] == "e2" and absorb[0][1]["dup_id"] == "e1"
    assert absorb[0][1]["descs"] == [
        "American computer firm; IBM.",
        "Computer company founded 1911.",
    ]
    assert absorb[0][1]["aliases"] == ["IBM"]
    # the loser's label is set in the same statement that deletes it
    cypher = absorb[0][0]
    assert "SET s:`Organization`" in cypher
    assert "SET s:`Company`" not in cypher  # already the survivor's own
    assert cypher.index("SET s:`Organization`") < cypher.index("DETACH DELETE")
    assert not any("SET s:`" in c[0] and "DETACH DELETE" not in c[0] for c in g.calls)


def test_judge_never_merges_a_pair_a_resolver_judged_distinct():
    g = _JudgeGraph(distinct=[("e1", "e2")])
    dd = EntityDeduplicator(g, FakeEmbedder(DESC_EMB))
    n = asyncio.run(dd.deduplicate(judge_llm=ScriptedLLM([[IBM], [IBM]])))
    assert n == 0 and dd.last_judge_stats["merged"] == 0
    assert not any("DETACH DELETE" in c[0] for c in g.calls)
    assert not any("SAME_AS" in c[0] for c in g.calls)


def test_remap_queries_bind_survivor_before_merge():
    """An unbound ``(s {id})`` inside a path MERGE creates a stub node per
    edge (measured: 136 stubs from 24 merges). Every remap query must MATCH
    the survivor first and MERGE only the relationship."""
    from graphrag_sdk.storage.deduplicator import _REMAP_QUERIES

    for q in _REMAP_QUERIES:
        assert "(s:__Entity__ {id: $survivor_id})" in q.split("MERGE")[0]
        assert "MERGE (s:__Entity__" not in q and "->(s:__Entity__" not in q


# Three spellings the canonical-name phase does NOT unify, so what merges here
# is the judge's doing alone.
ACME_ROWS = [
    ("a", "Acme", "Acme Corp, maker of anvils.", ["Organization"], None, [1.0, 0.0, 0.0]),
    ("b", "Akme Corp", "Anvil maker.", ["Organization"], None, [0.98, 0.1, 0.0]),
    ("c", "Acme Industries", "Anvils.", ["Company"], None, [0.97, 0.15, 0.0]),
]
ACME_DESC = {
    "Acme Corp, maker of anvils.": [1.0, 0.0, 0.0],
    "Anvil maker.": [0.95, 0.2, 0.0],
    "Anvils.": [0.95, 0.25, 0.0],
}
ACME = {"Acme", "Akme Corp", "Acme Industries"}


def test_judged_group_honours_a_distinct_pair_between_two_losers():
    """The resolver said a | c. The judge's grouping never puts them in one
    set, but ``_merge_judged_group`` is the last line: handed all three with
    b as survivor (long forms over the 4-letter "Acme"), it folds c into b and
    must then not fold a onto a survivor that already holds c."""
    g = _JudgeGraph(rows=ACME_ROWS)
    dd = EntityDeduplicator(g, FakeEmbedder(ACME_DESC))
    members = [{"id": r[0], "labels": r[3]} for r in ACME_ROWS]
    entities = asyncio.run(dd._fetch_all_entities(500))
    by_id = {e["id"]: e for e in entities}
    surv, absorbed = asyncio.run(dd._merge_judged_group(members, by_id, {frozenset(("a", "c"))}))
    assert surv == "b" and absorbed == {"c"}
    deletes = [c[1]["dup_id"] for c in g.calls if "DETACH DELETE dup RETURN s.id" in c[0]]
    assert deletes == ["c"]


def test_two_keyed_rows_are_not_asked_and_a_mention_between_them_is_not_moved():
    """The CI contract from structured ingestion: two rows keyed E-1 / E-2
    named 'Alice Smith' and a passage mentioning her. The exact phase leaves
    all three; the judge must spend no LLM call on them."""
    rows = [
        ("alice__person", "Alice Smith", "Presented the plan", ["Person"], None, [1.0, 0.0, 0.0]),
        ("e-1__person", "Alice Smith", "", ["Person"], None, [1.0, 0.0, 0.0]),
        ("e-2__person", "Alice Smith", "", ["Person"], None, [1.0, 0.0, 0.0]),
    ]
    g = _JudgeGraph(rows=rows)
    # rows: is_stub False marks a keyed node
    g.page = [
        (r[0], r[1], r[2], r[3][0], r[4], None if i == 0 else False, 0) for i, r in enumerate(rows)
    ]
    llm = ScriptedLLM([[{"Alice Smith"}], [{"Alice Smith"}]])
    dd = EntityDeduplicator(g, FakeEmbedder({}))
    n = asyncio.run(dd.deduplicate(judge_llm=llm))
    assert n == 0 and llm.prompts == []
    assert dd.last_judge_stats["llm_calls"] == 0 and dd.last_judge_stats["candidates"] == 0
    assert not any("DETACH DELETE" in c[0] or "SAME_AS" in c[0] for c in g.calls)


def test_a_failed_link_write_keeps_the_committed_merge_count():
    """A merge committed before a SAME_AS write failed is a merge; the phase
    reports it, and the failed link is not counted."""

    class Boom(_JudgeGraph):
        async def query_raw(self, cypher, params=None):
            if "SAME_AS" in cypher:
                raise RuntimeError("link storage down")
            return await super().query_raw(cypher, params)

    g = Boom(rows=ACME_ROWS)
    ab = {"Acme", "Akme Corp"}
    dd = EntityDeduplicator(g, FakeEmbedder(ACME_DESC))
    n = asyncio.run(dd.deduplicate(judge_llm=ScriptedLLM([[ACME], [ab]])))
    assert n == 1 and dd.last_judge_stats["merged"] == 1 and dd.last_judge_stats["linked"] == 0


def test_absorb_refuses_a_label_that_is_not_an_identifier():
    g = _JudgeGraph(rows=ACME_ROWS)
    dd = EntityDeduplicator(g, FakeEmbedder(ACME_DESC))
    entities = asyncio.run(dd._fetch_all_entities(500))
    by_id = {e["id"]: e for e in entities}
    ok = asyncio.run(
        dd._absorb(by_id["a"], by_id["b"], add_labels=["Company", "X` DETACH DELETE (n) //"])
    )
    assert ok
    (absorb,) = [c for c in g.calls if "DETACH DELETE dup RETURN s.id" in c[0]]
    assert "SET s:`Company`" in absorb[0] and "X`" not in absorb[0]
