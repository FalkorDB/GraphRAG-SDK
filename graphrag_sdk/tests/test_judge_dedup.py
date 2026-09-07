# Tests for the LLM-judged cross-document deduplication phase.

from __future__ import annotations

import asyncio
import re
from types import SimpleNamespace

import numpy as np

from graphrag_sdk.core.providers.base import LLMBatchItem
from graphrag_sdk.storage.judge_dedup import (
    LLMJudgeDeduplicator,
    knn_pairs,
    make_groups_dense,
    merge_description,
    name_in_desc_pairs,
    pack_prompts,
    pairs_from_response,
    parse_groups,
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


def test_pack_prompts_lists_every_set():
    ents = [{"name": f"E{i}", "label": "Person", "description": "d" * 50} for i in range(6)]
    prompts, orders = pack_prompts([[0, 1], [2, 3, 4], [5, 0]], ents)
    assert sum(len(o) for o in orders) == 3
    assert "E3 [Person]" in "".join(prompts)


def test_merge_description_pipe_join_dedups_and_skips_empty():
    assert merge_description(["A.", "", "B.", "A.", None]) == "A. | B."


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


async def _remap_ok(dup, surv):
    return True


def _run(graph, llm, vote=True):
    dd = LLMJudgeDeduplicator(graph, FakeEmbedder(DESC_EMB), llm, _remap_ok, vote=vote)
    return asyncio.run(dd.deduplicate())


def _set_calls(graph):
    return [c for c in graph.calls if "SET s.description" in c[0]]


def test_agreed_pair_merges_with_pipe_description_labels_and_aliases():
    graph = ScriptedGraph(ROWS)
    # both passes: IBM == International Business Machines, Watson separate
    llm = ScriptedLLM([[IBM], [IBM]])
    stats = _run(graph, llm)
    # a single 3-member set was judged twice (vote)
    assert stats["llm_calls"] == 2 and stats["merged"] == 1 and stats["linked"] == 0
    deletes = [c for c in graph.calls if "DETACH DELETE" in c[0]]
    assert [c[1]["d"] for c in deletes] == ["e2"]  # survivor = longest description (e1)
    ((cypher, params),) = _set_calls(graph)
    assert params["id"] == "e1"
    assert params["desc"] == "Computer company founded 1911. | American computer firm; IBM."
    assert params["aliases"] == ["International Business Machines"]
    assert "SET s:`Company`" in cypher and "SET s:`Organization`" in cypher


def test_disagreement_links_instead_of_merging():
    graph = ScriptedGraph(ROWS)
    llm = ScriptedLLM([[IBM], []])
    stats = _run(graph, llm)
    assert stats["merged"] == 0 and stats["linked"] == 1 and stats["disagreed_pairs"] == 1
    assert not any("DETACH DELETE" in c[0] for c in graph.calls)
    link = [c for c in graph.calls if "SAME_AS" in c[0]]
    assert len(link) == 1 and {link[0][1]["a"], link[0][1]["b"]} == {"e1", "e2"}


def test_no_vote_merges_on_single_pass():
    graph = ScriptedGraph(ROWS)
    llm = ScriptedLLM([[IBM]])
    stats = _run(graph, llm, vote=False)
    assert stats["llm_calls"] == 1 and stats["merged"] == 1


def test_failed_llm_call_merges_nothing():
    graph = ScriptedGraph(ROWS)
    llm = ScriptedLLM([None, [IBM]])
    stats = _run(graph, llm)
    assert stats["llm_failed"] == 1 and stats["merged"] == 0
    assert stats["linked"] == 1  # the one-pass yes survives as a link


def test_missing_name_embedding_is_written_back():
    rows = [(*r[:5], None) for r in ROWS]
    graph = ScriptedGraph(rows)
    llm = ScriptedLLM([[], []])
    _run(graph, llm)
    writes = [c for c in graph.calls if "SET e.embedding = vecf32" in c[0]]
    assert len(writes) == 1 and {it["id"] for it in writes[0][1]["items"]} == {"e1", "e2", "e3"}


def test_remap_queries_bind_survivor_before_merge():
    """An unbound ``(s {id})`` inside a path MERGE creates a stub node per
    edge (measured: 136 stubs from 24 merges). Every remap query must MATCH
    the survivor first and MERGE only the relationship."""
    from graphrag_sdk.storage.deduplicator import _REMAP_QUERIES

    for q in _REMAP_QUERIES:
        assert "(s:__Entity__ {id: $survivor_id})" in q.split("MERGE")[0]
        assert "MERGE (s:__Entity__" not in q and "->(s:__Entity__" not in q
