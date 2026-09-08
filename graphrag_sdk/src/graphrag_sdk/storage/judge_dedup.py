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
#   6. merge agreed groups: survivor keeps every member's label, description
#      is the members' descriptions joined with " | ", names of the merged
#      members are kept in ``aliases``, RELATES/MENTIONED_IN edges remapped.
#
# Measured on the benchmark corpus (RESULTS.md P3.119-P3.124): grouping loses
# 0 gold pairs; the two-pass vote cuts wrong merges ~65 % at 2x LLM cost; a
# wrong merge destroys a node, a wrong SAME_AS edge is a deletable edge.

from __future__ import annotations

import logging
import random
import re
from collections import defaultdict
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
PROMPT_TOKEN_BUDGET = 3000
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
        for j in set.intersection(*(inv[t] for t in ts)):
            if j != i:
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
    edges: dict[tuple[int, int], float], cap: int = GROUP_CAP, min_density: float = MIN_DENSITY
) -> list[list[int]]:
    """Connected components under a size cap that must stay dense: at least
    ``min_density`` of all member pairs are nominated edges. A chain A-B-C-D
    where only neighbours are similar cannot form; a real alias cluster can.
    Edges left straddling two groups are kept as 2-member sets."""
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
    return "\n".join(
        f"{i}. {ents[idx]['name']} [{ents[idx]['label'] or 'Entity'}] — "
        f"{ents[idx]['description'] or '(no description)'}"
        for i, idx in enumerate(members, 1)
    )


def pack_prompts(
    sets: list[list[int]], ents: list[dict], budget: int = PROMPT_TOKEN_BUDGET
) -> tuple[list[str], list[list[list[int]]]]:
    """Pack whole sets into prompts until the token budget is reached.
    Returns (prompts, orders) where orders[p][s] = member indices of set s in prompt p."""
    import tiktoken

    enc = tiktoken.get_encoding("o200k_base")
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
    Out-of-range numbers are dropped; an unmentioned set yields no groups (no merge)."""
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


def merge_description_list(members: list[dict]) -> list[str]:
    """Rule for every merge: the survivor keeps **every** member's description.

    Each member may carry ``descriptions`` (a list, if it was merged before) or
    only ``description`` (possibly an older ``" | "``-joined string). Returns
    the flattened, deduplicated list in member order.
    """
    out: list[str] = []
    for m in members:
        raw = m.get("descriptions")
        items = (
            [str(d) for d in raw]
            if isinstance(raw, list) and raw
            else str(m.get("description") or "").split(" | ")
        )
        for d in items:
            d = d.strip()
            if d and d not in out:
                out.append(d)
    return out


def merge_description(descriptions: list[str]) -> str:
    """Joined form of the description list — what the fulltext index, search
    and LLM prompts read. Kept alongside ``descriptions`` (the list)."""
    seen: list[str] = []
    for d in descriptions:
        d = (d or "").strip()
        if d and d not in seen:
            seen.append(d)
    return " | ".join(seen)


# ── the deduplicator ────────────────────────────────────────────────────────


class LLMJudgeDeduplicator:
    """Cross-document dedup: similarity nominates, the LLM decides, a second
    shuffled pass must agree before anything is merged.

    Args:
        graph_store: object with ``query_raw(cypher, params)``.
        embedder: embedding provider (names and descriptions).
        llm: judge model. gpt-4.1-class models measured 9 wrong merges where
            gpt-4o-mini made 51 on the same input; use the strongest you can.
        remap_edges: ``async (dup_id, survivor_id) -> bool`` — the
            ``EntityDeduplicator`` edge-remap routine, reused so both phases
            move RELATES/MENTIONED_IN identically.
        vote: run the second shuffled pass (default True). ``False`` halves
            LLM cost and roughly triples wrong merges.
    """

    def __init__(
        self,
        graph_store: Any,
        embedder: Embedder,
        llm: LLMInterface,
        remap_edges: Any,
        *,
        vote: bool = True,
        max_concurrency: int = 8,
        seed: int = 0,
    ) -> None:
        self._graph = graph_store
        self._embedder = embedder
        self._llm = llm
        self._remap = remap_edges
        self._vote = vote
        self._conc = max_concurrency
        self._seed = seed

    async def _fetch_entities(self, batch_size: int) -> list[dict]:
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
                descs = row[6] if len(row) > 6 else None
                out.append(
                    {
                        "id": eid,
                        "name": name or str(eid),
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

    async def _merge_group(self, ids: list[str], byid: dict[str, dict]) -> int:
        """Merge ``ids`` into the member with the longest description. Survivor
        gains every member's labels, ``aliases`` gets the merged names,
        ``descriptions`` becomes the list of every member's descriptions and
        ``description`` their ' | ' join. Edges are moved to the survivor before
        each loser is deleted. Returns the number of nodes deleted."""
        ids = sorted(ids, key=lambda i: (-len(byid[i]["description"]), i))
        surv, dups = ids[0], ids[1:]
        s = byid[surv]
        labels = {lab for i in ids for lab in byid[i]["labels"] if lab}
        aliases: list[str] = list(s["aliases"])
        for i in ids:
            for nm in [byid[i]["name"], *byid[i]["aliases"]]:
                if nm and nm != s["name"] and nm not in aliases:
                    aliases.append(nm)
        deleted = 0
        for d in dups:
            if not await self._remap(d, surv):
                logger.warning("judge dedup: edge remap incomplete for %s -> %s, kept", d, surv)
                continue
            try:
                await self._graph.query_raw(
                    "MATCH (e:__Entity__ {id: $d}) DETACH DELETE e", {"d": d}
                )
                deleted += 1
            except Exception as exc:
                logger.warning("judge dedup: failed to delete %s: %s", d, exc)
        desc_list = merge_description_list([byid[i] for i in ids])
        new_desc = merge_description(desc_list)
        label_clause = "".join(f" SET s:`{lab}`" for lab in sorted(labels) if _safe_label(lab))
        await self._graph.query_raw(
            "MATCH (s:__Entity__ {id: $id}) "
            "SET s.description = $desc, s.descriptions = $descs, s.aliases = $aliases"
            + label_clause,
            {"id": surv, "desc": new_desc, "descs": desc_list, "aliases": aliases},
        )
        return deleted

    async def deduplicate(self, batch_size: int = 500) -> dict[str, int]:
        ents = await self._fetch_entities(batch_size)
        stats = {
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
        stats["candidates"] = len(edges)
        if not edges:
            return stats

        sets = make_groups_dense(edges)
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
        stats["agreed_pairs"], stats["disagreed_pairs"] = len(agreed), len(disagreed)

        uf = _UF(list(range(len(ents))))
        for a, b in sorted(agreed):
            uf.union(a, b)
        groups: dict[int, list[int]] = defaultdict(list)
        for i in range(len(ents)):
            groups[uf.find(i)].append(i)
        byid = {e["id"]: e for e in ents}
        root_id: dict[int, str] = {}
        for r, members in groups.items():
            ids = [ents[i]["id"] for i in members]
            if len(ids) >= 2:
                stats["merged"] += await self._merge_group(ids, byid)
            surv = sorted(ids, key=lambda i: (-len(byid[i]["description"]), i))[0]
            for i in members:
                root_id[i] = surv

        for a, b in sorted(disagreed):
            ra, rb = root_id[a], root_id[b]
            if ra == rb:
                continue
            await self._graph.query_raw(
                "MATCH (a:__Entity__ {id: $a}), (b:__Entity__ {id: $b}) "
                "MERGE (a)-[r:SAME_AS]->(b) SET r.source = 'llm_judge', r.agreement = 1",
                {"a": ra, "b": rb},
            )
            stats["linked"] += 1
        logger.info("LLMJudgeDeduplicator: %s", stats)
        return stats


def _safe_label(lab: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_ -]*", lab))
