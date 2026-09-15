# GraphRAG SDK — Storage: LLM-judged cross-document deduplication
#
# "Similarity nominates, the LLM decides." Runs once over the whole graph at
# finalize() time, after exact-name dedup:
#
#   1. embed every entity's name and description (the name vector is the
#      node's ``embedding`` property — written here if missing, so every
#      entity carries a name embedding for retrieval)
#   2. nominate candidate pairs: top-k name neighbours, top-k description
#      neighbours, and "A's name appears inside B's description"
#   3. group nominated pairs into small dense sets (cap 8)
#   4. ask the LLM to partition each set into same-referent groups
#   5. ask again with every set shuffled; a pair is a duplicate only if BOTH
#      passes agree. Pairs the passes disagree on become SAME_AS edges.
#   6. merge agreed groups through the caller's ``merge_group`` hook —
#      ``EntityDeduplicator._merge_judged_group``, which applies its ``_absorb``
#      rules (survivor rank, property carry, provenance union, descriptions
#      list, aliases) and adds every member's label to the survivor. This
#      class decides *identity* only; it never writes a merge itself.
#
# Pairs the caller has already settled — a resolver's ``DISTINCT_FROM``, a
# mention two keyed rows could equally own (``skip_pairs``), and any two rows
# written from a declared key (``distinct_ids``) — are dropped before the
# model is asked, never unioned into one group, and never linked.
#
# Measured on the benchmark corpus (RESULTS.md P3.119-P3.124): grouping loses
# 0 gold pairs; the two-pass vote cuts wrong merges ~65 % at 2x LLM cost; a
# wrong merge destroys a node, a wrong SAME_AS edge is a deletable edge.

from __future__ import annotations

import logging
import random
import re
from collections import defaultdict
from collections.abc import Awaitable, Callable
from typing import Any

import numpy as np

from graphrag_sdk.core.providers import Embedder, LLMInterface

logger = logging.getLogger(__name__)

NAME_GATE = 0.65
DESC_GATE = 0.55
TOP_K = 10
GROUP_CAP = 8
MIN_DENSITY = 0.5
NAME_IN_DESC_WEIGHT = 0.65
# A name found inside more descriptions than this is a common word ("Company",
# "Paris"), not an identifying mention; nominating it would pair one entity
# with most of the graph and bypass the top-k bound of the embedding doors.
NAME_IN_DESC_CAP = 50
PROMPT_TOKEN_BUDGET = 3000
# Longest description rendered per entity, so a set of GROUP_CAP members with
# merged descriptions still fits the prompt budget.
MAX_DESC_CHARS = 600
_STOP = frozenset(
    "the a an of and in on at to for de la le el les du des von van der al ibn bin y et".split()
)

JUDGE_PROMPT = (
    "You are the final judge in an entity-resolution pipeline for a knowledge "
    "graph.\n\n"
    "WHAT YOU ARE GIVEN\n"
    "Entities were extracted from documents by an NER model and an LLM. Each has "
    "a name, a type label in [brackets], and a description summarised from the "
    "text where it was mentioned. Descriptions are partial and may miss facts; "
    "type labels are sometimes wrong, so the same entity can appear under two "
    "labels. Below are {n} independent SETS. A set was formed automatically by "
    "embedding similarity of names and of descriptions, and by one entity's "
    "name appearing inside another's description. Because of how they were "
    "formed, most members of a set are merely RELATED — same document, same "
    "topic, same family, same firm, same place — and only some, or none, are "
    "the SAME real-world entity.\n\n"
    "YOUR TASK\n"
    "For each set, partition it: put two entities in one group only if they are "
    "the exact same real-world referent. Everything else stays in its own group. "
    "Sets are independent; never group across sets.\n\n"
    "{sets}\n\n"
    "DECISION PROCEDURE — apply to every pair in a set, in this order\n"
    "1. NAME-VARIANT TEST. Are the two names two spellings of ONE name? "
    "Translation (Munich / München), transliteration (Yusuf / Joseph, Mohammed / "
    "Muhammad), dropped or added middle name (John F. Kennedy / John Kennedy), "
    "initials, acronym or shortened company name (IBM / International Business "
    "Machines; Barbier et Cie / Barbier), diminutive (Bob / Robert), a title plus "
    "surname vs full name (Mrs Vega / Mirabel Vega). If yes -> SAME, unless the "
    "descriptions clearly contradict (different birth years, different cities of "
    "operation).\n"
    '2. IDENTITY-BY-DESCRIPTION TEST. One name may be a generic reference ("the '
    'museum", "the register", "the physician", "the Society"). It is '
    "SAME as a named entity only if its description names that entity, or states "
    "a specific identifying fact (a unique role + place + time, a unique work or "
    "event) that the named entity's description also states. A shared topic, "
    "document or vocabulary is NOT an identifying fact.\n"
    "3. RELATED-BUT-DISTINCT CHECK. These are DIFFERENT entities even when "
    "closely linked: a founder and the company; a person and a ship, building, "
    "prize or place named after them; parent and child; two siblings or two "
    "relatives with different given names; a firm and an unrelated firm sharing a "
    "surname; an event and its year; two different years, dates or numbers; a "
    "predecessor and successor organisation unless the text says it is a rename.\n"
    "4. DEFAULT. If tests 1 and 2 both fail, the entities are DIFFERENT. A wrong "
    "merge corrupts the graph permanently; a missed merge is recoverable. When in "
    "doubt, separate.\n\n"
    "OUTPUT\n"
    "One line per group, prefixed by its set number, listing the member numbers:\n"
    "  SET 1 GROUP: 1, 4\n"
    "  SET 1 GROUP: 2\n"
    "  SET 2 GROUP: 1, 2\n"
    "List every entity of every set exactly once. No commentary.\n\n"
    "Answer:"
)


# ── pure helpers (unit-testable without a graph) ────────────────────────────


def _toks(s: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", s.lower()) if t not in _STOP and len(t) > 1}


def knn_pairs(vecs: np.ndarray, k: int, gate: float) -> dict[tuple[int, int], float]:
    """Top-k cosine neighbours per row kept above ``gate`` -> {(i, j): sim}."""
    n = vecs.shape[0]
    if n < 2:
        return {}
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    keep = np.where(norms[:, 0] > 0)[0]
    if len(keep) < 2:
        return {}
    unit = np.zeros_like(vecs, dtype=np.float32)
    unit[keep] = vecs[keep] / norms[keep]
    kk = min(k + 1, len(keep))
    out: dict[tuple[int, int], float] = {}
    if n <= 2000:
        sims = unit[keep] @ unit[keep].T
        for row, i in enumerate(keep):
            top = np.argpartition(-sims[row], kk - 1)[:kk]
            for col in top:
                j = int(keep[col])
                if j == i:
                    continue
                sim = float(sims[row, col])
                if sim >= gate:
                    key = (min(i, j), max(i, j))
                    out[key] = max(sim, out.get(key, -1.0))
        return out
    import hnswlib

    idx = hnswlib.Index(space="ip", dim=vecs.shape[1])
    idx.init_index(max_elements=n, ef_construction=200, M=16)
    idx.add_items(unit[keep], keep)
    idx.set_ef(max(64, kk))
    labels, dists = idx.knn_query(unit[keep], k=kk)
    for row, i in enumerate(keep):
        for j, dist in zip(labels[row], dists[row]):
            j = int(j)
            if j == i:
                continue
            sim = float(1.0 - dist)
            if sim >= gate:
                key = (min(i, j), max(i, j))
                out[key] = max(sim, out.get(key, -1.0))
    return out


def name_in_desc_pairs(ents: list[dict]) -> dict[tuple[int, int], float]:
    """Every content token of A's name occurs in B's description (or vice versa)."""
    nt = [_toks(e["name"]) for e in ents]
    inv: dict[str, set[int]] = defaultdict(set)
    for j, e in enumerate(ents):
        for t in _toks(e["description"]):
            inv[t].add(j)
    out: dict[tuple[int, int], float] = {}
    for i, ts in enumerate(nt):
        if not ts or not all(t in inv for t in ts):
            continue
        hits = set.intersection(*(inv[t] for t in ts)) - {i}
        if len(hits) > NAME_IN_DESC_CAP:
            continue
        for j in hits:
            out[(min(i, j), max(i, j))] = NAME_IN_DESC_WEIGHT
    return out


class _UF:
    def __init__(self, items: list[int]) -> None:
        self.p = {x: x for x in items}

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        self.p[rb] = ra
        return True


def make_groups_dense(
    edges: dict[tuple[int, int], float],
    cap: int = GROUP_CAP,
    min_density: float = MIN_DENSITY,
    keep_apart: Callable[[int, int], bool] | None = None,
) -> list[list[int]]:
    """Connected components under a size cap that must stay dense: at least
    ``min_density`` of all member pairs are nominated edges. A chain A-B-C-D
    where only neighbours are similar cannot form; a real alias cluster can.
    Edges left straddling two groups are kept as 2-member sets.

    ``keep_apart(i, j)`` names pairs already decided distinct; two components
    are not joined when that would put such a pair in one set, so the model
    is never shown — and never asked about — a pair the graph has settled.
    """
    nodes = sorted({x for e in edges for x in e})
    uf = _UF(nodes)
    mem = {x: {x} for x in nodes}
    eset = set(edges)
    for (a, b), _ in sorted(edges.items(), key=lambda kv: -kv[1]):
        ra, rb = uf.find(a), uf.find(b)
        if ra == rb:
            continue
        m = mem[ra] | mem[rb]
        if len(m) > cap:
            continue
        if keep_apart is not None and any(keep_apart(x, y) for x in mem[ra] for y in mem[rb]):
            continue
        ml = sorted(m)
        inner = sum(
            1 for i in range(len(ml)) for j in range(i + 1, len(ml)) if (ml[i], ml[j]) in eset
        )
        if inner / (len(ml) * (len(ml) - 1) / 2) < min_density:
            continue
        uf.union(a, b)
        r = uf.find(a)
        mem[r] = m
        for x in (ra, rb):
            if x != r:
                mem.pop(x, None)
    groups: dict[int, list[int]] = defaultdict(list)
    for x in nodes:
        groups[uf.find(x)].append(x)
    sets = [sorted(g) for g in groups.values() if len(g) >= 2]
    sets += [sorted(e) for e in edges if uf.find(e[0]) != uf.find(e[1])]
    return sets


def render_set(members: list[int], ents: list[dict]) -> str:
    """One numbered line per member: name, every label, capped description."""
    lines = []
    for i, idx in enumerate(members, 1):
        ent = ents[idx]
        labels = "/".join(ent.get("labels") or []) or ent.get("label") or "Entity"
        desc = (ent.get("description") or "").strip()
        if len(desc) > MAX_DESC_CHARS:
            desc = desc[: MAX_DESC_CHARS - 1].rstrip() + "…"
        lines.append(f"{i}. {ent['name']} [{labels}] — {desc or '(no description)'}")
    return "\n".join(lines)


def pack_prompts(
    sets: list[list[int]], ents: list[dict], budget: int = PROMPT_TOKEN_BUDGET
) -> tuple[list[str], list[list[list[int]]]]:
    """Pack whole sets into prompts until the token budget is reached.
    Returns (prompts, orders) where orders[p][s] = member indices of set s in prompt p."""
    import tiktoken

    # cl100k_base, as everywhere else in the SDK: it exists in every tiktoken
    # the project allows (>=0.5), where o200k_base needs 0.7 and would raise.
    enc = tiktoken.get_encoding("cl100k_base")
    base = len(enc.encode(JUDGE_PROMPT.format(n=0, sets="")))
    prompts: list[str] = []
    orders: list[list[list[int]]] = []
    cur: list[list[int]] = []
    cur_tok = base

    def flush() -> None:
        prompts.append(
            JUDGE_PROMPT.format(
                n=len(cur),
                sets="\n\n".join(
                    f"--- Set {k + 1} ---\n{render_set(m, ents)}" for k, m in enumerate(cur)
                ),
            )
        )
        orders.append(list(cur))

    for s in sets:
        t = len(enc.encode(f"--- Set 99 ---\n{render_set(s, ents)}\n\n"))
        if cur and cur_tok + t > budget:
            flush()
            cur, cur_tok = [], base
        cur.append(s)
        cur_tok += t
    if cur:
        flush()
    return prompts, orders


def parse_groups(text: str, sizes: list[int]) -> dict[int, list[list[int]]]:
    """Parse ``SET 2 GROUP: 1, 4`` lines into {set index: [0-based member groups]}.

    Out-of-range numbers are dropped; an unmentioned set yields no groups (no
    merge). A set whose groups overlap — ``1, 2`` and ``1, 3`` — is not a
    partition, and taking it at face value would merge 2 and 3 through 1 though
    the model never put them together; such a set yields no groups either.
    """
    out: dict[int, list[list[int]]] = {}
    for line in text.splitlines():
        up = line.upper()
        if "GROUP" not in up:
            continue
        m = re.search(r"SET\s*(\d+)", up)
        si = int(m.group(1)) - 1 if m else 0
        if not 0 <= si < len(sizes):
            continue
        after = line[m.end() :] if m else line
        nums = [int(x) - 1 for x in re.findall(r"\d+", after.split(":", 1)[-1])]
        picked = sorted({x for x in nums if 0 <= x < sizes[si]})
        if picked:
            out.setdefault(si, []).append(picked)
    for si, groups in list(out.items()):
        listed = [x for grp in groups for x in grp]
        if len(listed) != len(set(listed)):
            logger.warning("judge dedup: set %d answered with overlapping groups; ignored", si + 1)
            del out[si]
    return out


def pairs_from_response(text: str, order: list[list[int]]) -> set[tuple[int, int]]:
    """Entity-index pairs the judge put in one group, for one prompt."""
    out: set[tuple[int, int]] = set()
    for si, groups in parse_groups(text, [len(m) for m in order]).items():
        members = order[si]
        for grp in groups:
            real = [members[k] for k in grp]
            for x in range(len(real)):
                for y in range(x + 1, len(real)):
                    out.add((min(real[x], real[y]), max(real[x], real[y])))
    return out


# ── the deduplicator ────────────────────────────────────────────────────────


class LLMJudgeDeduplicator:
    """Cross-document dedup: similarity nominates, the LLM decides, a second
    shuffled pass must agree before anything is merged.

    Args:
        graph_store: object with ``query_raw(cypher, params)``.
        embedder: embedding provider (names and descriptions).
        llm: judge model. gpt-4.1-class models measured 9 wrong merges where
            gpt-4o-mini made 51 on the same input; use the strongest you can.
        merge_group: ``async (members: list[dict]) -> (survivor_id,
            absorbed_ids)`` that performs the merge of one agreed group.
            ``EntityDeduplicator`` supplies :meth:`_merge_judged_group`, so a
            judge merge obeys the same rules as every other phase (a table row
            survives a mention, two keyed rows never merge, properties and
            provenance are carried, labels are unioned). This class decides
            identity only and never writes a merge itself.
        vote: run the second shuffled pass (default True). ``False`` halves
            LLM cost and roughly triples wrong merges.
    """

    def __init__(
        self,
        graph_store: Any,
        embedder: Embedder,
        llm: LLMInterface,
        merge_group: Callable[[list[dict]], Awaitable[tuple[str, set[str]]]],
        *,
        vote: bool = True,
        max_concurrency: int = 8,
        seed: int = 0,
    ) -> None:
        self._graph = graph_store
        self._embedder = embedder
        self._llm = llm
        self._merge = merge_group
        self._vote = vote
        self._conc = max_concurrency
        self._seed = seed
        # Stats of the run in progress, so a caller that catches a failure
        # part-way can still report the merges that were committed.
        self.last_stats: dict[str, int] = {}

    async def _fetch_entities(self, batch_size: int) -> list[dict]:
        """Every named entity. A node without a name — a fact row keyed on a
        reading id — cannot be judged to be anything, and manufacturing a name
        from its id would let ``e-1`` nominate against a prose ``E-1``."""
        out: list[dict] = []
        last = ""
        while True:
            r = await self._graph.query_raw(
                "MATCH (e:__Entity__) WHERE e.id > $last "
                "RETURN e.id, e.name, e.description, "
                "[l IN labels(e) WHERE l <> '__Entity__'], e.aliases, e.embedding, "
                "e.descriptions "
                "ORDER BY e.id LIMIT $limit",
                {"last": last, "limit": batch_size},
            )
            rows = r.result_set or []
            for row in rows:
                eid, name, desc, labels, aliases, emb = row[:6]
                if not name or not str(name).strip():
                    continue
                descs = row[6] if len(row) > 6 else None
                out.append(
                    {
                        "id": eid,
                        "name": name,
                        "description": desc or "",
                        "descriptions": list(descs) if descs else [],
                        "labels": list(labels or []),
                        "label": (labels or [""])[0] if labels else "",
                        "aliases": list(aliases or []),
                        "embedding": list(emb) if emb else None,
                    }
                )
            if len(rows) < batch_size:
                break
            last = rows[-1][0]
        return out

    async def _name_vectors(self, ents: list[dict]) -> np.ndarray:
        """Name embeddings: reuse ``e.embedding`` where present, embed the rest
        and persist them so every entity keeps a name vector for retrieval."""
        missing = [i for i, e in enumerate(ents) if not e["embedding"]]
        if missing:
            fresh = await self._embedder.aembed_documents([ents[i]["name"] for i in missing])
            for i, v in zip(missing, fresh):
                ents[i]["embedding"] = list(v)
            for k in range(0, len(missing), 200):
                items = [
                    {"id": ents[i]["id"], "v": ents[i]["embedding"]}
                    for i in missing[k : k + 200]
                    if ents[i]["embedding"]
                ]
                if items:
                    await self._graph.query_raw(
                        "UNWIND $items AS it MATCH (e:__Entity__ {id: it.id}) "
                        "SET e.embedding = vecf32(it.v)",
                        {"items": items},
                    )
        dim = max((len(e["embedding"]) for e in ents if e["embedding"]), default=0)
        vecs = np.zeros((len(ents), dim), dtype=np.float32)
        for i, e in enumerate(ents):
            if e["embedding"] and len(e["embedding"]) == dim:
                vecs[i] = e["embedding"]
        return vecs

    async def _judge(self, sets: list[list[int]], ents: list[dict]) -> tuple[set, int, int]:
        prompts, orders = pack_prompts(sets, ents)
        results = await self._llm.abatch_invoke(prompts, max_concurrency=self._conc)
        same: set[tuple[int, int]] = set()
        failed = 0
        for item in results:
            if not item.ok or item.response is None:
                failed += 1  # a failed call merges nothing — the safe default
                continue
            same |= pairs_from_response(item.response.content or "", orders[item.index])
        return same, len(prompts), failed

    async def deduplicate(
        self,
        batch_size: int = 500,
        *,
        skip_pairs: set[frozenset[str]] | None = None,
        distinct_ids: set[str] | None = None,
    ) -> dict[str, int]:
        """Run the judge over the whole graph and return its stats.

        ``skip_pairs`` are id pairs already decided elsewhere — a resolver's
        ``DISTINCT_FROM`` verdicts, a mention two keyed rows could equally own.
        ``distinct_ids`` are ids that are each a different thing from every
        other id in the set — rows written from a declared key, two of which
        are two rows whatever they are called. Neither kind of pair is put to
        the model, unioned into one group through a third member, or linked,
        so a NO the graph already holds is not re-litigated by a second model.
        """
        skip = skip_pairs or set()
        distinct = distinct_ids or set()

        def protected(a: str, b: str) -> bool:
            return frozenset((a, b)) in skip or (a in distinct and b in distinct)

        ents = await self._fetch_entities(batch_size)

        def protected_idx(i: int, j: int) -> bool:
            return protected(ents[i]["id"], ents[j]["id"])

        stats = self.last_stats = {
            "entities": len(ents),
            "candidates": 0,
            "sets": 0,
            "llm_calls": 0,
            "llm_failed": 0,
            "agreed_pairs": 0,
            "disagreed_pairs": 0,
            "merged": 0,
            "linked": 0,
        }
        if len(ents) < 2:
            return stats

        nv = await self._name_vectors(ents)
        descs = [e["description"] or e["name"] for e in ents]
        dv = np.asarray(await self._embedder.aembed_documents(descs), dtype=np.float32)

        edges = knn_pairs(dv, TOP_K, DESC_GATE)
        for k, v in knn_pairs(nv, TOP_K, NAME_GATE).items():
            edges[k] = max(v, edges.get(k, -1.0))
        for k, v in name_in_desc_pairs(ents).items():
            edges.setdefault(k, v)
        edges = {(a, b): v for (a, b), v in edges.items() if not protected_idx(a, b)}
        stats["candidates"] = len(edges)
        if not edges:
            return stats

        sets = make_groups_dense(edges, keep_apart=protected_idx if (skip or distinct) else None)
        stats["sets"] = len(sets)
        same1, calls, failed = await self._judge(sets, ents)
        stats["llm_calls"] += calls
        stats["llm_failed"] += failed
        if self._vote:
            rng = random.Random(self._seed)
            shuffled = [rng.sample(s, len(s)) for s in reversed(sets)]
            same2, calls, failed = await self._judge(shuffled, ents)
            stats["llm_calls"] += calls
            stats["llm_failed"] += failed
            agreed, disagreed = same1 & same2, same1 ^ same2
        else:
            agreed, disagreed = same1, set()
        # The model never saw a protected pair as a candidate, but a partition
        # of a set can still put the two in one group; a decided pair stays
        # decided.
        agreed = {(a, b) for a, b in agreed if not protected_idx(a, b)}
        disagreed = {(a, b) for a, b in disagreed if not protected_idx(a, b)}
        stats["agreed_pairs"], stats["disagreed_pairs"] = len(agreed), len(disagreed)

        # Union agreed pairs into groups — unless doing so would put a
        # protected pair into one group through a third member (A~B and A~C
        # agreed, B|C decided): the merge would then fold both into A and the
        # remembered NO would be gone with the node that held it.
        uf = _UF(list(range(len(ents))))
        members: dict[int, set[int]] = {i: {i} for i in range(len(ents))}
        for a, b in sorted(agreed):
            ra, rb = uf.find(a), uf.find(b)
            if ra == rb:
                continue
            if any(protected_idx(x, y) for x in members[ra] for y in members[rb]):
                logger.info(
                    "judge dedup: not grouping %s with %s — it would join a pair already "
                    "decided distinct",
                    ents[a]["id"],
                    ents[b]["id"],
                )
                continue
            merged_members = members.pop(ra) | members.pop(rb)
            uf.union(a, b)
            members[uf.find(a)] = merged_members
        groups: dict[int, list[int]] = defaultdict(list)
        for i in range(len(ents)):
            groups[uf.find(i)].append(i)
        byid = {e["id"]: e for e in ents}
        root_id: dict[int, str] = {}
        for r, idxs in groups.items():
            ids = [ents[i]["id"] for i in idxs]
            if len(ids) < 2:
                root_id[idxs[0]] = ids[0]
                continue
            surv, absorbed = await self._merge([byid[i] for i in ids])
            stats["merged"] += len(absorbed)
            for i in idxs:
                # A member the merge rules kept apart (two rows of one table)
                # is still its own node, so a link must point at it, not at
                # the survivor it was not folded into.
                eid = ents[i]["id"]
                root_id[i] = surv if (eid in absorbed or eid == surv) else eid

        # One SAME_AS edge per surviving pair: two disagreed pairs that merged
        # to the same survivors are one edge, and a pair the merges have made
        # protected (B~C disagreed, B folded into A, A|C decided) is not linked.
        links: set[tuple[str, str]] = set()
        for a, b in disagreed:
            ra, rb = root_id[a], root_id[b]
            if ra == rb or protected(ra, rb):
                continue
            links.add((min(ra, rb), max(ra, rb)))
        for ra, rb in sorted(links):
            try:
                await self._graph.query_raw(
                    "MATCH (a:__Entity__ {id: $a}), (b:__Entity__ {id: $b}) "
                    "MERGE (a)-[r:SAME_AS]->(b) SET r.source = 'llm_judge', r.agreement = 1",
                    {"a": ra, "b": rb},
                )
            except Exception as exc:
                logger.warning("judge dedup: failed to link %s ~ %s: %s", ra, rb, exc)
                continue
            stats["linked"] += 1
        logger.info("LLMJudgeDeduplicator: %s", stats)
        return stats
