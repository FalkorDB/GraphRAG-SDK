"""Incremental, graph-aware entity resolution.

Most resolvers deduplicate a document's entities *among themselves*. This one
resolves them **against the knowledge graph built so far**, so a mention in
today's document can be linked to an entity extracted from a document ingested
last week — the case ordinary batch resolution structurally cannot see.

The algorithm is a funnel: cheap, certain work first; the language model only
where cheap signals genuinely cannot decide.

    ┌ 1. collapse ─ merge a batch's own same-name duplicates            (free)
    │ 2. retrieve ─ for each survivor, fetch look-alike graph entities  (free)
    │ 3. pile     ─ group survivors that share a candidate              (free)
    └ 4. link     ─ ask the LLM, per pile: which are the same entity,
                    and which existing node (if any) is the target      (LLM)

Two duplication axes, each handled where it is cheapest and most reliable:

* **same name, different type** (``GraphRAG`` the *Concept* vs the *Technology*)
  is resolved in step 1 by description similarity — no model call. An LLM asked
  to be careful tends to *over-split* these; the embedding is both cheaper and
  more accurate.
* **different name, same entity** (``llama_index`` ≈ ``LlamaIndex``, or a new
  mention of an existing ``Gal Sh``) is resolved in step 4, where the model is
  asked to *partition* a small pile — the framing that keeps genuine
  look-alikes apart (``FALKORDB_USERNAME`` ≠ ``FALKORDB_PASSWORD``).

Design choices worth knowing:

* **Fail toward splitting.** When evidence is thin or a call errors, entities
  stay separate. A stray duplicate is recoverable later; a wrong merge is data
  loss.
* **Existing nodes win.** When a batch entity links to a graph node, the graph
  node's id survives, so the store's ``MERGE (n {id})`` write attaches the new
  mention with no special-casing and existing edges are untouched.
* **Facts are never invented.** The LLM owns free text (canonical name, type,
  merged description); provenance is unioned and immutable-property conflicts
  are *flagged*, never guessed.

The graph lookup is injected as ``candidate_retriever``, so the strategy is
fully testable without a live database; in production it wraps the graph
store's name/vector search.
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from graphrag_sdk.core.context import Context
from graphrag_sdk.core.models import GraphData, GraphNode, ResolutionResult
from graphrag_sdk.core.prompts import ENTITY_DESCRIPTION_RULE
from graphrag_sdk.core.providers import Embedder, LLMInterface
from graphrag_sdk.ingestion.resolution_strategies.base import (
    ResolutionStrategy,
    remap_relationships,
)

logger = logging.getLogger(__name__)

_SEPARATORS = re.compile(r"[\s\-_]+")

# Internal property keys the resolver writes; never treated as extracted facts.
_SOURCE_IDS = "source_chunk_ids"
_CONFLICTS = "_merge_conflicts"
_NEEDS_REVIEW = "_needs_review"


def normalize_name(name: str) -> str:
    """Fold case and separators so surface variants share one key.

    ``"GraphRAG-SDK"``, ``"graphrag_sdk"`` and ``"GraphRAG SDK"`` all normalize
    to ``"graphrag sdk"``.
    """
    return _SEPARATORS.sub(" ", str(name).strip().lower()).strip()


# Returns the existing graph entities that look like ``(name, description)``.
# ``GraphNode.id`` is the live graph id, used as the merge target.
CandidateRetriever = Callable[[str, str, int], Awaitable[Sequence[GraphNode]]]


_LINK_PROMPT = (
    "You are linking newly extracted entities to an existing knowledge graph.\n"
    "Each item has: ref, origin ('new' = just extracted, 'graph' = already in "
    "the graph), name, type, description.\n\n"
    "Items:\n{items}\n\n"
    "Group items that are the SAME real-world entity. Rules:\n"
    "- Group two items only if they truly denote the same entity (spelling "
    "variants, abbreviations). Similar names for different things (e.g. two "
    "people who share a first name) must stay in separate groups.\n"
    "- If a group contains a 'graph' item, the new items merge INTO it: set "
    "target to that item's ref.\n"
    "- If a group has only 'new' items, it is a new entity: target = \"new\".\n"
    "- A 'graph' item that matches nothing is left out of every group.\n\n"
    "Each group is ONE entity and gets its OWN 'description' (never shared "
    "across groups). Write that entity's description by merging the descriptions "
    "of the items IN THAT GROUP, following this rule: " + ENTITY_DESCRIPTION_RULE + "\n\n"
    "Respond with JSON only:\n"
    '{{"groups": [{{"members": [refs...], "target": <ref or "new">, '
    '"canonical": "<name>", "type": "<type>", '
    '"description": "<merged description per the rule above, at most {max_tokens} tokens>"}}]}}'
)


@dataclass(frozen=True)
class _PileItem:
    """One entry the LLM sees, tagged so its verdict can be mapped back."""

    ref: int  # 1-based label used in the prompt
    node: GraphNode
    survivor_idx: int | None  # index into ``survivors`` when freshly extracted
    graph_id: str | None  # live graph id when already in the graph

    @property
    def origin(self) -> str:
        return "new" if self.survivor_idx is not None else "graph"


@dataclass(frozen=True)
class _LinkDecision:
    """One parsed group from the LLM's partition of a pile."""

    members: list[int]  # refs the model judged to be the same entity
    target: int | str  # a graph item's ref, or "new"
    canonical: str | None
    type: str | None
    description: str | None


class IncrementalResolution(ResolutionStrategy):
    """Resolve a document's entities against the existing graph (see module doc).

    Args:
        llm: Partitions ambiguous piles and writes merged descriptions — one
            call per pile. Without it, resolution reduces to the free
            within-batch collapse (step 1 only).
        embedder: Scores description similarity for the within-batch collapse.
        candidate_retriever: ``async (name, description, k) -> [GraphNode]``
            returning look-alike existing graph nodes. Without it, this behaves
            as a within-batch resolver.
        top_k: Candidates fetched per survivor.
        pile_cap: Upper bound on items (survivors + candidates) sent to the LLM
            in a single call, so a crowded name cannot blow up a prompt.
        same_name_threshold: Minimum pairwise description cosine at which
            same-name entities auto-merge for free. Same name is already strong
            evidence, so this bar is deliberately low; genuine homographs
            (Paris the city vs the person) fall below it and stay separate.
        max_summary_tokens: Budget hint for LLM-written merged descriptions.
        immutable_props: Properties that must not change on merge. A conflict
            flags the survivor for review instead of silently picking a value —
            often the sign of a wrong merge.
    """

    def __init__(
        self,
        llm: LLMInterface | None = None,
        embedder: Embedder | None = None,
        candidate_retriever: CandidateRetriever | None = None,
        *,
        top_k: int = 3,
        pile_cap: int = 12,
        same_name_threshold: float = 0.80,
        max_summary_tokens: int = 300,
        immutable_props: Sequence[str] = (),
    ) -> None:
        self.llm = llm
        self.embedder = embedder
        self.candidate_retriever = candidate_retriever
        self.top_k = top_k
        self.pile_cap = pile_cap
        self.same_name_threshold = same_name_threshold
        self.max_summary_tokens = max_summary_tokens
        self.immutable_props = frozenset(immutable_props)

    async def resolve(self, graph_data: GraphData, ctx: Context) -> ResolutionResult:
        nodes = list(graph_data.nodes)
        if not nodes:
            return ResolutionResult(nodes=[], relationships=[], merged_count=0, remap={})
        ctx.log(f"IncrementalResolution: resolving {len(nodes)} extracted entities")

        # 1. Collapse the batch's own same-name duplicates (free).
        survivors, remap, merged = await self._collapse_within_batch(nodes)

        # 2. Fetch look-alike existing entities for each survivor (free).
        candidates = await self._fetch_candidates(survivors)

        # 3. Group survivors that share a candidate into piles (free).
        # 4. Ask the LLM to link each pile against the graph (LLM).
        # A pile with more survivors than the prompt budget is split into
        # chunks — otherwise the survivors alone would breach ``pile_cap`` and
        # evict every candidate, so the LLM could never link. Reserve room for
        # each chunk's candidates.
        survivor_budget = max(1, self.pile_cap - self.top_k)
        for pile in self._partition_into_piles(candidates):
            if self.llm is None or not any(candidates[i] for i in pile):
                continue  # nothing to link → these survivors are new; keep as-is
            for chunk in _chunked(pile, survivor_budget):
                merged += await self._link_pile(chunk, survivors, candidates, remap)

        _flatten(remap)
        absorbed = set(remap)
        # Collapse carriers that were independently linked onto the same graph
        # node — across pile chunks or across two LLM groups sharing a target —
        # so the output never contains two nodes with one id (which would make
        # the store MERGE the same (label, id) twice and clobber the earlier
        # node's provenance/conflicts). The first keeper wins; the rest fold in.
        keeper_by_id: dict[str, GraphNode] = {}
        surviving_nodes: list[GraphNode] = []
        for node in survivors:
            if node.id in absorbed:
                continue
            keeper = keeper_by_id.get(node.id)
            if keeper is None:
                keeper_by_id[node.id] = node
                surviving_nodes.append(node)
            else:
                self._absorb(keeper, node)
        relationships = remap_relationships(graph_data.relationships, remap)
        ctx.log(
            f"IncrementalResolution: {len(surviving_nodes)} entities ({merged} merged or linked)"
        )
        return ResolutionResult(
            nodes=surviving_nodes,
            relationships=relationships,
            merged_count=merged,
            remap=remap,
        )

    # ── step 1: within-batch collapse ──────────────────────────────────────

    async def _collapse_within_batch(
        self, nodes: list[GraphNode]
    ) -> tuple[list[GraphNode], dict[str, str], int]:
        """Merge same-name duplicates the extractor produced in this document.

        Within one name group: if the descriptions are coherent the whole group
        is one entity (type wobble); otherwise only exact same-type duplicates
        merge and any genuine homograph is preserved for the LLM to judge later.
        """
        remap: dict[str, str] = {}
        merged = 0
        absorbed: set[int] = set()  # tracked by object identity — ids may collide

        for group in _group_by(nodes, key=lambda n: normalize_name(_name_of(n))):
            if len(group) < 2:
                continue
            for survivor, losers in await self._same_name_merges(group):
                for loser in losers:
                    self._absorb(survivor, loser)
                    if loser.id != survivor.id:
                        remap[loser.id] = survivor.id
                    absorbed.add(id(loser))
                    merged += 1

        survivors = [n for n in nodes if id(n) not in absorbed]
        return survivors, remap, merged

    async def _same_name_merges(
        self, group: list[GraphNode]
    ) -> list[tuple[GraphNode, list[GraphNode]]]:
        """Yield ``(survivor, losers)`` pairs for one same-name group.

        A cross-type merge requires *real* descriptions on every member: without
        them description similarity degrades to name-vs-name (always cosine 1.0),
        which would merge genuine homographs (Paris the city vs the person) on
        name identity alone — against the fail-toward-splitting stance. When any
        description is missing, only exact same-type duplicates are folded.
        """
        descriptions = [str(n.properties.get("description") or "") for n in group]
        units = await self._embed(descriptions) if all(descriptions) else None
        if units is not None and _min_pairwise_cosine(units) >= self.same_name_threshold:
            return [(group[0], group[1:])]  # coherent → one entity (type wobble)
        # No evidence to cross types: only fold exact same-type duplicates.
        return [
            (same_type[0], same_type[1:])
            for same_type in _group_by(group, key=lambda n: n.label)
            if len(same_type) >= 2
        ]

    # ── step 2 & 3: candidate retrieval and piling ─────────────────────────

    async def _fetch_candidates(self, survivors: list[GraphNode]) -> list[list[GraphNode]]:
        """For each survivor, the existing graph entities that look like it.

        A retriever failure for one survivor degrades to "no candidates" (that
        entity is treated as new) rather than aborting the whole document —
        keeping with the strategy's fail-toward-splitting stance.
        """
        if self.candidate_retriever is None:
            return [[] for _ in survivors]
        found: list[list[GraphNode]] = []
        for survivor in survivors:
            try:
                candidates = await self.candidate_retriever(
                    _name_of(survivor), _description_of(survivor), self.top_k
                )
                found.append(list(candidates))
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(
                    "IncrementalResolution: candidate retrieval failed for %r: %s",
                    _name_of(survivor),
                    exc,
                )
                found.append([])
        return found

    @staticmethod
    def _partition_into_piles(candidates: list[list[GraphNode]]) -> list[list[int]]:
        """Group survivor indices that share at least one candidate.

        Survivors linked through a common graph node must be judged together, so
        the LLM can, say, route three spellings of one name onto the same
        existing entity in a single call.
        """
        shared: dict[str, list[int]] = defaultdict(list)
        for i, cands in enumerate(candidates):
            for c in cands:
                shared[c.id].append(i)
        return _connected_components(len(candidates), shared.values())

    # ── step 4: link a pile against the graph via the LLM ──────────────────

    async def _link_pile(
        self,
        pile: list[int],
        survivors: list[GraphNode],
        candidates: list[list[GraphNode]],
        remap: dict[str, str],
    ) -> int:
        """Partition one pile with the LLM and apply the resulting merges."""
        items = self._build_pile(pile, survivors, candidates)
        decisions = await self._ask(items)
        if decisions is None:
            return 0  # fail-safe: unparseable verdict → leave the pile untouched

        by_ref = {item.ref: item for item in items}
        merged = 0
        for group in decisions:
            members = [by_ref[r] for r in group.members if r in by_ref]
            survivor_members = [m.survivor_idx for m in members if m.survivor_idx is not None]
            if not survivor_members:
                continue  # a group of only existing nodes — nothing new to place

            carrier = survivors[survivor_members[0]]
            target = by_ref.get(group.target) if isinstance(group.target, int) else None
            if target is not None and target.graph_id is not None:
                # Link into an existing node: keep its id AND its label. Graph
                # writes are label-scoped (``MERGE (n:<label> {id})``), so a
                # different label would create a second same-id node instead of
                # updating the existing one.
                final_id, final_label = target.graph_id, target.node.label
            else:
                final_id, final_label = carrier.id, group.type or carrier.label

            for idx in survivor_members[1:]:
                self._absorb(carrier, survivors[idx])
                remap[survivors[idx].id] = final_id
                merged += 1
            if final_id != carrier.id:
                remap[carrier.id] = final_id  # link this batch entity onto a graph node
                merged += 1

            _apply_canonical(carrier, final_id, final_label, group)
        return merged

    def _build_pile(
        self,
        pile: list[int],
        survivors: list[GraphNode],
        candidates: list[list[GraphNode]],
    ) -> list[_PileItem]:
        """Assemble the labelled items for one LLM call: survivors then their
        deduplicated candidates, capped at ``pile_cap``."""
        items = [
            _PileItem(ref=n + 1, node=survivors[i], survivor_idx=i, graph_id=None)
            for n, i in enumerate(pile)
        ]
        seen: set[str] = set()
        for i in pile:
            if len(items) >= self.pile_cap:
                break
            for cand in candidates[i]:
                if cand.id in seen:
                    continue
                seen.add(cand.id)
                items.append(
                    _PileItem(ref=len(items) + 1, node=cand, survivor_idx=None, graph_id=cand.id)
                )
                if len(items) >= self.pile_cap:
                    break
        return items

    async def _ask(self, items: list[_PileItem]) -> list[_LinkDecision] | None:
        """Send one partition request and parse it; ``None`` on any failure.

        Descriptions are sent in FULL, not truncated: extraction and this step
        both write them under ``ENTITY_DESCRIPTION_RULE``, so they are already
        bounded. Truncating the input here is what eroded rich descriptions
        across repeated merges — the model would rewrite from a snippet and the
        result would overwrite the stored full text.
        """
        lines = "\n".join(
            f"[{it.ref}] origin={it.origin} name={_name_of(it.node)!r} "
            f"type={it.node.label} desc={_description_of(it.node)!r}"
            for it in items
        )
        prompt = _LINK_PROMPT.format(items=lines, max_tokens=self.max_summary_tokens)
        results = await self.llm.abatch_invoke([prompt])
        if results and results[0].ok:
            return _parse_decisions(results[0].response.content)
        return None

    # ── shared helpers ─────────────────────────────────────────────────────

    def _absorb(self, survivor: GraphNode, loser: GraphNode) -> None:
        """Fold ``loser`` into ``survivor``: union provenance, keep the
        survivor's values, and flag any immutable-property disagreement.

        Conflicts are recorded as ``"<field>: <a> vs <b>"`` strings, not dicts:
        the graph store persists lists of primitives but silently drops lists of
        dicts, so a structured record would never reach the graph. Both the
        loser's prior conflicts and its ``_needs_review`` flag are carried over,
        so a node flagged in an earlier stage keeps that flag when it is later
        absorbed.
        """
        sources = _unique(
            survivor.properties.get(_SOURCE_IDS) or [],
            loser.properties.get(_SOURCE_IDS) or [],
        )
        if sources:
            survivor.properties[_SOURCE_IDS] = sources

        conflicts = list(survivor.properties.get(_CONFLICTS, [])) + list(
            loser.properties.get(_CONFLICTS, [])
        )
        for key, value in loser.properties.items():
            if key in (_SOURCE_IDS, _CONFLICTS, _NEEDS_REVIEW):
                continue
            if key not in survivor.properties:
                survivor.properties[key] = value
            elif key in self.immutable_props and survivor.properties[key] != value:
                conflicts.append(f"{key}: {survivor.properties[key]} vs {value}")
        if conflicts or loser.properties.get(_NEEDS_REVIEW):
            survivor.properties[_CONFLICTS] = conflicts
            survivor.properties[_NEEDS_REVIEW] = True

    async def _embed(self, texts: list[str]) -> np.ndarray | None:
        """Return unit-normalized row vectors for ``texts``, or ``None`` if no
        embedder is configured or embedding fails."""
        if self.embedder is None or not texts:
            return None
        try:
            raw = await self.embedder.aembed_documents(texts)
            # np.array is inside the guard: a ragged/malformed embedder result
            # would otherwise raise here and crash the whole document.
            matrix = np.array([v or [0.0] for v in raw], dtype=np.float32)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("IncrementalResolution: embedding failed: %s", exc)
            return None
        if matrix.ndim != 2 or matrix.shape[1] <= 1:
            return None
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms


# ── module-level helpers ───────────────────────────────────────────────────


def _apply_canonical(
    node: GraphNode, final_id: str, final_label: str, decision: _LinkDecision
) -> None:
    """Retarget ``node`` to ``(final_id, final_label)`` and apply the LLM's
    canonical name and description. The label is resolved by the caller so that
    linking into an existing graph node preserves that node's label."""
    node.id = final_id
    node.label = final_label
    if decision.canonical:
        node.properties["name"] = decision.canonical
    if decision.description:
        node.properties["description"] = decision.description


def _connected_components(size: int, edges: Iterable[list[int]]) -> list[list[int]]:
    """Union-find components over ``size`` items linked by each group in ``edges``."""
    parent = list(range(size))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for members in edges:
        for other in members[1:]:
            parent[find(other)] = find(members[0])

    components: dict[int, list[int]] = defaultdict(list)
    for i in range(size):
        components[find(i)].append(i)
    return list(components.values())


def _group_by(items: Iterable, key: Callable) -> list[list]:
    """Stable group-by that preserves first-seen order of keys and members."""
    groups: dict = defaultdict(list)
    for item in items:
        groups[key(item)].append(item)
    return list(groups.values())


def _min_pairwise_cosine(units: np.ndarray) -> float:
    """Smallest cosine similarity among unit-normalized row vectors."""
    similarities = units @ units.T
    return float(similarities[np.triu_indices(len(units), k=1)].min())


def _unique(*sequences: Sequence[str]) -> list[str]:
    """Order-preserving union of string sequences."""
    seen: dict[str, None] = {}
    for seq in sequences:
        for value in seq:
            seen.setdefault(value, None)
    return list(seen)


def _name_of(node: GraphNode) -> str:
    return str(node.properties.get("name", node.id))


def _description_of(node: GraphNode) -> str:
    return str(node.properties.get("description") or node.properties.get("name") or "")


def _chunked(items: list[int], size: int) -> Iterable[list[int]]:
    """Split ``items`` into consecutive chunks of at most ``size``."""
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _parse_decisions(content: str) -> list[_LinkDecision] | None:
    """Parse the LLM's JSON partition; ``None`` on anything malformed."""
    if not content:
        return None
    start, end = content.find("{"), content.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        payload = json.loads(content[start : end + 1])
    except json.JSONDecodeError:
        return None
    groups = payload.get("groups")
    if not isinstance(groups, list):
        return None

    decisions = []
    for group in groups:
        if not isinstance(group, dict) or not isinstance(group.get("members"), list):
            continue
        members = [r for r in map(_as_ref, group["members"]) if r is not None]
        if members:
            target = _as_ref(group.get("target"))
            decisions.append(
                _LinkDecision(
                    members=members,
                    target=target if target is not None else "new",
                    canonical=group.get("canonical"),
                    type=group.get("type"),
                    description=group.get("description"),
                )
            )
    return decisions or None


def _as_ref(value: object) -> int | None:
    """Coerce an int-like value (``5``, ``5.0``, ``"5"``) to an int ref.

    LLMs are inconsistent about number formatting, so a group's members and its
    ``target`` may arrive as strings or floats. Anything not cleanly integer —
    including ``"new"``, booleans and fractional floats — returns ``None``.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else None
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value)
    return None


def _flatten(remap: dict[str, str]) -> None:
    """Collapse transitive chains in place so every key maps to its final id."""
    for key in list(remap):
        target, seen = remap[key], {key}
        while target in remap and target not in seen:
            seen.add(target)
            target = remap[target]
        remap[key] = target
