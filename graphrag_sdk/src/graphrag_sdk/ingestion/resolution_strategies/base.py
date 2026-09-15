# GraphRAG SDK — Ingestion: Resolution Strategy ABC
# Pattern: Strategy — different deduplication approaches implement this interface.

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable
from typing import TYPE_CHECKING

from graphrag_sdk.core.context import Context
from graphrag_sdk.core.models import GraphData, GraphNode, GraphRelationship, ResolutionResult

if TYPE_CHECKING:
    from graphrag_sdk.core.providers import LLMInterface


logger = logging.getLogger(__name__)

_SUMMARY_PROMPT = (
    "Summarise the following descriptions of the entity '{entity_name}' "
    "into a single concise description (max {max_tokens} tokens).\n\n"
    "Descriptions:\n{descriptions}\n\n"
    "Summary:"
)

_SUMMARY_WITH_TYPE_PROMPT = (
    "Entities named '{entity_name}' appear under different types: {types}.\n\n"
    "Descriptions:\n{descriptions}\n\n"
    "Do ALL of these descriptions refer to the SAME real-world entity?\n\n"
    "If YES, respond with:\n"
    "  Line 1: 'YES <canonical_type>' (pick the most accurate type from {types})\n"
    "  Line 2+: a single concise summary (max {max_tokens} tokens)\n\n"
    "If NO (these are distinct real-world entities that happen to share a name), "
    "respond with:\n"
    "  Line 1: 'NO'\n"
    "  Line 2: brief reason (max 20 words)\n\n"
    "Do not attempt partial merges. Answer YES only if all entries describe "
    "the same real-world entity.\n\n"
    "Answer:"
)


def description_list(props: dict) -> list[str]:
    """The member descriptions of a node as a list.

    A merged node carries ``descriptions`` (list) and ``description`` (the
    same texts joined with ``" | "``, which is what search and prompts read).
    An unmerged node has only ``description``. Either way this returns the
    list, deduplicated, empty strings dropped.

    A node holding only ``description`` contributes it as **one** member, and
    is deliberately not split on ``" | "``. Splitting would recover the members
    of a survivor written before ``descriptions`` existed, but nothing
    distinguishes that node from a fresh one whose single description happens
    to contain the separator ("CEO | founder of Acme"), and the two failures
    are not symmetric. Splitting a fresh node invents members that were never
    written: the count drives ``force_summary_threshold``, so it buys LLM
    summary calls nothing asked for, and the invented entries are rendered as
    separate bullets in the prompts below. Not splitting a legacy survivor
    costs only granularity — every character is still carried, and the
    re-joined ``description`` is byte-identical either way.
    """
    raw = props.get("descriptions")
    items: list[str] = []
    if isinstance(raw, list):
        items = [str(d).strip() for d in raw]
    else:
        single = str(props.get("description") or "").strip()
        items = [single] if single else []
    out: list[str] = []
    for d in items:
        if d and d not in out:
            out.append(d)
    return out


def set_merged_descriptions(survivor: GraphNode, members: list[GraphNode]) -> list[str]:
    """Rule for every merge: the survivor keeps **every** member's description.

    Writes ``descriptions`` (a list — ``["desc1", "desc2", ...]``, survivor's
    first, duplicates dropped) and ``description`` (the same list joined with
    ``" | "`` for the fulltext index, search and LLM prompts). Returns the list.
    """
    merged: list[str] = []
    for node in [survivor, *members]:
        for d in description_list(node.properties):
            if d not in merged:
                merged.append(d)
    if merged:
        survivor.properties["descriptions"] = merged
        survivor.properties["description"] = " | ".join(merged)
    return merged


def merge_source_ids(survivor: GraphNode, members: list[GraphNode]) -> list[str]:
    """Rule for every merge: the survivor keeps **every** member's provenance.

    ``source_chunk_ids`` is the link back to the chunks an entity was extracted
    from — it drives chunk retrieval and the ``MENTIONED_IN`` edges. Extraction
    sets it on every node, so a "copy the keys the survivor lacks" rule never
    copies it and the duplicate's chunks are lost. Stage 1 and Stage 5 in this
    module union it explicitly; strategies must do the same. Returns the union.
    """
    merged: list[str] = []
    for node in [survivor, *members]:
        raw = node.properties.get("source_chunk_ids") or []
        if not isinstance(raw, list):
            raw = [raw]
        for sid in raw:
            if sid and sid not in merged:
                merged.append(sid)
    if merged:
        survivor.properties["source_chunk_ids"] = merged
    return merged


def _pick_canonical_label(nodes: list[GraphNode]) -> str:
    """Heuristic label selection: most frequent non-Unknown label wins.

    Ties are broken lexicographically for deterministic results across runs.
    """
    counts: dict[str, int] = defaultdict(int)
    for n in nodes:
        counts[n.label] += 1
    # Prefer any specific type over "Unknown"
    candidates = {k: v for k, v in counts.items() if k != "Unknown"}
    if not candidates:
        return "Unknown"
    # Sort by (-count, label) for deterministic tie-breaking
    return sorted(candidates, key=lambda k: (-candidates[k], k))[0]


async def exact_match_merge(
    nodes: list[GraphNode],
    llm: LLMInterface | None,
    *,
    force_summary_threshold: int = 3,
    max_summary_tokens: int = 500,
    cross_label_merge: bool = False,
    cross_label_min_descriptions: int = 3,
    resolve_property: str = "name",
    label_gate: Callable[[str, str], bool] | None = None,
    cross_label_vote: bool = False,
    rejected_out: set[frozenset[str]] | None = None,
) -> tuple[list[GraphNode], dict[str, str], int]:
    """Phase 1: group nodes by normalized name and merge exact duplicates.

    *resolve_property* selects which property carries the entity name; it
    exists so ``ExactMatchResolution`` can expose the same knob while
    sharing this implementation. Nodes lacking that property fall back to
    ``node.id``, which for extracted entities is already
    ``name__type`` normalized.

    When *cross_label_merge* is False (default), groups by
    ``(normalized_name, label)`` — only same-type duplicates merge.

    When *cross_label_merge* is True, this runs a two-stage resolution:
    first the same-label pass (safe, unconditional), then a cross-label
    pass that asks the LLM to verify whether same-name entries under
    different labels refer to the same real-world entity. On YES the LLM
    also picks the canonical type. On NO — or if evidence is too sparse,
    or LLM is unavailable, or the call errors — homograph nodes are
    preserved under their original labels. Both stages share a single LLM
    batch invocation; no extra calls beyond what the same-label summary
    path already performs.

    *cross_label_min_descriptions* is the evidence floor for that second
    stage: a name/label group with fewer descriptions than this is left
    alone rather than put to the LLM. It is deliberately separate from
    *force_summary_threshold*, which answers a different question (when to
    have the LLM summarise a merged description).

    Returns:
        (deduplicated_nodes, id_remap, merged_count)
    """
    # ── Stage 1: always group by (name, label) for the safe same-label pass ──
    sl_groups: dict[tuple[str, str], list[GraphNode]] = defaultdict(list)
    for node in nodes:
        name = node.properties.get(resolve_property, node.id)
        norm = str(name).strip().lower()
        sl_groups[(norm, node.label)].append(node)

    # Collect per-group evidence. Each entry tracks nodes, descriptions,
    # source ids, and original-description count (used by cross-label
    # threshold check after same-label merging collapses the descriptions).
    sl_entries: list[dict] = []
    for _key, group_nodes in sl_groups.items():
        descriptions: list[str] = []
        all_source_ids: list[str] = []
        for n in group_nodes:
            desc = n.properties.get("description", "")
            if desc:
                descriptions.append(str(desc))
            src_ids = n.properties.get("source_chunk_ids", [])
            if isinstance(src_ids, list):
                for sid in src_ids:
                    if sid not in all_source_ids:
                        all_source_ids.append(sid)
        sl_entries.append(
            {
                "nodes": group_nodes,
                "descriptions": descriptions,
                "all_source_ids": all_source_ids,
                "orig_desc_count": len(descriptions),
                "is_merge": len(group_nodes) >= 2,
            }
        )

    # ── Stage 2: detect cross-label candidates on top of same-label groups ──
    cl_candidates: list[dict] = []
    if cross_label_merge and llm is not None:
        by_name: dict[str, list[int]] = defaultdict(list)
        for i, entry in enumerate(sl_entries):
            first = entry["nodes"][0]
            # Must use the same key and the same fallback as the same-label
            # grouping above. This read used to default to "", so every node
            # lacking `resolve_property` normalised to the empty string and
            # landed in one bucket -- unrelated nameless nodes under different
            # labels then looked like a cross-label homograph group and were
            # put to the LLM together.
            norm = str(first.properties.get(resolve_property, first.id)).strip().lower()
            by_name[norm].append(i)
        for _name_key, indices in by_name.items():
            if len(indices) < 2:
                continue
            labels = {sl_entries[i]["nodes"][0].label for i in indices}
            if len(labels) < 2:
                continue
            # Same safety policy as the embedding stage, applied here too.
            # This phase merges on the NAME alone, with no vector filter, so a
            # Person/Place homograph with enough descriptions reached the LLM
            # and could be merged before any family gate or second vote ran —
            # the caller's cross-family guarantees were simply absent from the
            # first phase. A group spanning two different known families is not
            # asked about at all; the members survive under their own labels.
            if label_gate is not None:
                ordered = sorted(labels)
                if any(
                    not label_gate(a, b) for k, a in enumerate(ordered) for b in ordered[k + 1 :]
                ):
                    logger.debug(
                        "Label family gate dropped cross-label group %s for '%s'",
                        ordered,
                        _name_key,
                    )
                    continue
            total_orig_descs = sum(sl_entries[i]["orig_desc_count"] for i in indices)
            if total_orig_descs < cross_label_min_descriptions:
                # Fail-safe: too little evidence to ask the LLM here, so
                # preserve both nodes under their original labels and let the
                # cross-label PASS 2 in LLMVerifiedResolution decide instead.
                #
                # The gate used to read `< force_summary_threshold`, which is a
                # different question -- that knob decides when the LLM should
                # *summarise* merged descriptions. Changing the summary policy
                # silently changed which entities merged. The two are now
                # separate; the default is unchanged at 3.
                #
                # Lowering it to 2 was tried and rejected on evidence. This
                # stage has no vector filter, so every same-name cross-label
                # group it accepts costs an LLM call and its precision rests
                # entirely on that one answer. PASS 2 screens the same pairs
                # against a cosine floor first and rejects the cheap cases for
                # free -- `Apple` the fruit vs `Apple` the company score 0.437
                # and never reach the model. Measured end to end (3 reps): what
                # this gate skips, PASS 2 merges correctly 3/3, and it still
                # rejects the homograph trap 3/3. So the gate costs no recall
                # downstream, and lowering it only moves work to the stage with
                # the weaker safety net.
                logger.debug(
                    "cross-label merge skipped for '%s' (%s): %d description(s), need %d",
                    sl_entries[indices[0]]["nodes"][0].properties.get(resolve_property, ""),
                    "/".join(sorted(labels)),
                    total_orig_descs,
                    cross_label_min_descriptions,
                )
                continue
            all_descs: list[str] = []
            for i in indices:
                for n in sl_entries[i]["nodes"]:
                    d = n.properties.get("description", "")
                    if d:
                        all_descs.append(str(d))
            _first = sl_entries[indices[0]]["nodes"][0]
            entity_name = str(_first.properties.get(resolve_property, _first.id))
            cl_candidates.append(
                {
                    "sl_indices": indices,
                    "name": entity_name,
                    "descriptions": all_descs,
                    "types": sorted(labels),
                    # The nodes that carry on from the same-label pass; a
                    # definitive NO about this group is a decision about them.
                    "ids": [sl_entries[i]["nodes"][0].id for i in indices],
                }
            )

    # ── Stage 3: build a single LLM batch (summaries + cross-label verifies) ──
    prompts: list[str] = []
    summary_prompt_refs: list[tuple[int, list[str]]] = []  # (sl_entry_idx, descs)
    for i, entry in enumerate(sl_entries):
        if (
            entry["is_merge"]
            and entry["descriptions"]
            and len(entry["descriptions"]) >= force_summary_threshold
            and llm is not None
        ):
            _n0 = entry["nodes"][0]
            name = str(_n0.properties.get(resolve_property, _n0.id))
            prompts.append(
                _SUMMARY_PROMPT.format(
                    entity_name=name,
                    max_tokens=max_summary_tokens,
                    descriptions="\n".join(f"- {d}" for d in entry["descriptions"]),
                )
            )
            summary_prompt_refs.append((i, entry["descriptions"]))

    cl_prompt_start = len(prompts)
    for cand in cl_candidates:
        types_str = ", ".join(cand["types"])
        prompts.append(
            _SUMMARY_WITH_TYPE_PROMPT.format(
                entity_name=cand["name"],
                max_tokens=max_summary_tokens,
                types=types_str,
                descriptions="\n".join(f"- {d}" for d in cand["descriptions"]),
            )
        )

    batch_results: list = []
    if prompts and llm is not None:
        batch_results = await llm.abatch_invoke(prompts)

    def _record_rejection(cl_idx: int) -> None:
        """Remember a DEFINITIVE refusal so a later run need not pay to re-ask.

        Only ever called for an explicit NO — never for a failed call or an
        unparseable reply, which carry no decision at all. Recording those
        would turn one transient error into a permanent "distinct" verdict.
        """
        if rejected_out is None:
            return
        ids = cl_candidates[cl_idx]["ids"]
        for x in range(len(ids)):
            for y in range(x + 1, len(ids)):
                rejected_out.add(frozenset((ids[x], ids[y])))

    # ── Stage 4: parse results ──
    sl_summaries: dict[int, str] = {}  # sl_entry_idx -> summary text
    cl_approvals: dict[int, tuple[str, str]] = {}  # cl_idx -> (chosen_type, summary)
    for item in batch_results:
        idx = item.index
        if idx < cl_prompt_start:
            sl_entry_idx, descs = summary_prompt_refs[idx]
            if item.ok:
                sl_summaries[sl_entry_idx] = item.response.content.strip()
            else:
                sl_summaries[sl_entry_idx] = " | ".join(descs)
        else:
            cl_idx = idx - cl_prompt_start
            if not item.ok:
                # LLM error on cross-label verification → fail-safe: no merge.
                continue
            content = item.response.content.strip()
            first_line = content.split("\n", 1)[0].strip()
            if not first_line.upper().startswith("YES"):
                # NO or malformed → fail-safe: preserve homographs. Only the
                # explicit NO is a decision; a malformed reply is not one, and
                # must stay askable on a later run.
                if first_line.upper().startswith("NO"):
                    _record_rejection(cl_idx)
                continue
            parts = first_line.split(None, 1)
            chosen = parts[1].strip() if len(parts) >= 2 else ""
            lines = content.split("\n", 1)
            cl_summary = (
                lines[1].strip()
                if len(lines) > 1
                else " | ".join(cl_candidates[cl_idx]["descriptions"])
            )
            cl_approvals[cl_idx] = (chosen, cl_summary)

    # ── Stage 4b: second opinion on cross-label approvals ──
    # The same rule the embedding stage applies to its cross-label YESes, and
    # for the same reason: this phase merges on the NAME alone, so a single
    # approval is the only thing standing between a homograph and a merge.
    # Asked again with the candidate types listed in the opposite order —
    # the analogue of swapping A and B — and only the approvals the model
    # repeats survive. A failed or malformed re-ask is not agreement.
    if cross_label_vote and cl_approvals and llm is not None:
        revote_idx = sorted(cl_approvals)
        revote_prompts = [
            _SUMMARY_WITH_TYPE_PROMPT.format(
                entity_name=cl_candidates[cl_idx]["name"],
                max_tokens=max_summary_tokens,
                types=", ".join(reversed(cl_candidates[cl_idx]["types"])),
                descriptions="\n".join(f"- {d}" for d in cl_candidates[cl_idx]["descriptions"]),
            )
            for cl_idx in revote_idx
        ]
        second = await llm.abatch_invoke(revote_prompts)
        agreed: dict[int, bool] = {}
        for item in second:
            if not item.ok:
                continue
            first_line = item.response.content.strip().split("\n", 1)[0].strip().upper()
            # Only a reply that actually says YES or NO is a verdict. Storing
            # `startswith("YES")` put "MAYBE" — or an empty string — in as
            # False, which the veto branch below then reads as an explicit NO
            # and persists as a permanent distinct decision. A parse failure
            # withdraws the approval without being remembered, the same rule
            # the first vote uses.
            if first_line.startswith("YES"):
                agreed[item.index] = True
            elif first_line.startswith("NO"):
                agreed[item.index] = False
        for k, cl_idx in enumerate(revote_idx):
            if not agreed.get(k):
                logger.debug(
                    "Cross-label vote %s the phase-1 merge of '%s' (%s)",
                    "vetoed" if k in agreed else "left unconfirmed",
                    cl_candidates[cl_idx]["name"],
                    cl_candidates[cl_idx]["types"],
                )
                # An explicit NO on the re-ask is a decision; an unanswered one
                # only withdraws the approval.
                if k in agreed:
                    _record_rejection(cl_idx)
                del cl_approvals[cl_idx]

    # ── Stage 5: apply same-label merges, producing one survivor per sl group ──
    sl_survivor_by_idx: dict[int, GraphNode] = {}
    id_remap: dict[str, str] = {}
    merged_count = 0
    for i, entry in enumerate(sl_entries):
        group_nodes = entry["nodes"]
        if not entry["is_merge"]:
            sl_survivor_by_idx[i] = group_nodes[0]
            continue
        survivor = group_nodes[0]
        if entry["descriptions"]:
            set_merged_descriptions(survivor, group_nodes[1:])
            if i in sl_summaries:
                # An LLM summary (force_summary_threshold reached) replaces the
                # joined string only; the member list is kept intact.
                survivor.properties["description"] = sl_summaries[i]
        if entry["all_source_ids"]:
            survivor.properties["source_chunk_ids"] = entry["all_source_ids"]
        for dup in group_nodes[1:]:
            for k, v in dup.properties.items():
                if k not in survivor.properties:
                    survivor.properties[k] = v
            id_remap[dup.id] = survivor.id
            merged_count += 1
        sl_survivor_by_idx[i] = survivor

    # ── Stage 6: apply approved cross-label merges ──
    absorbed_sl_ids: set[str] = set()
    for cl_idx, cand in enumerate(cl_candidates):
        if cl_idx not in cl_approvals:
            continue
        chosen_type, cl_summary = cl_approvals[cl_idx]
        sl_idx_list = cand["sl_indices"]
        sl_survivors_in_cand = [sl_survivor_by_idx[i] for i in sl_idx_list]
        cl_survivor = next(
            (s for s in sl_survivors_in_cand if s.label.lower() == chosen_type.lower()),
            None,
        )
        if cl_survivor is None:
            canonical = _pick_canonical_label(sl_survivors_in_cand)
            cl_survivor = next(
                (s for s in sl_survivors_in_cand if s.label == canonical),
                sl_survivors_in_cand[0],
            )
        merged_sources: list[str] = []
        existing_srcs = cl_survivor.properties.get("source_chunk_ids", [])
        if isinstance(existing_srcs, list):
            merged_sources.extend(existing_srcs)
        # The losers' labels are otherwise destroyed here: this loop copies only
        # keys the survivor lacks, and the survivor always has a label. Same
        # defect as the embedding stage (P3.29) and PASS 2 (P3.21). graph_store
        # promotes this property to real Cypher labels on write, so a merged
        # node stays reachable under every type it was extracted as.
        absorbed_labels: list[str] = []
        for s in sl_survivors_in_cand:
            if s.id == cl_survivor.id:
                continue
            if s.label and s.label != cl_survivor.label and s.label not in absorbed_labels:
                absorbed_labels.append(s.label)
            for k, v in s.properties.items():
                if k not in cl_survivor.properties:
                    cl_survivor.properties[k] = v
            s_srcs = s.properties.get("source_chunk_ids", [])
            if isinstance(s_srcs, list):
                for sid in s_srcs:
                    if sid not in merged_sources:
                        merged_sources.append(sid)
            absorbed_sl_ids.add(s.id)
            id_remap[s.id] = cl_survivor.id
            merged_count += 1
        if merged_sources:
            cl_survivor.properties["source_chunk_ids"] = merged_sources
        if absorbed_labels:
            existing = cl_survivor.properties.get("merged_labels")
            parts = [p.strip() for p in str(existing).split(" | ")] if existing else []
            for lab in absorbed_labels:
                if lab not in parts:
                    parts.append(lab)
            cl_survivor.properties["merged_labels"] = " | ".join(p for p in parts if p)
        set_merged_descriptions(
            cl_survivor, [n for n in sl_survivors_in_cand if n.id != cl_survivor.id]
        )
        # Only when the model actually wrote one. ``cl_summary`` is the second
        # line of the verdict, so a reply of "YES Person\n" — verdict line, no
        # body — yields "". Assigning that unconditionally would blank the
        # description ``set_merged_descriptions`` just built from every member,
        # and it is the field search, the fulltext index and the prompts read.
        if cl_summary:
            cl_survivor.properties["description"] = cl_summary

    # ── Stage 7: resolve transitive id_remap chains (sl-loser → sl-survivor →
    # cl-survivor becomes sl-loser → cl-survivor directly) ──
    for k in list(id_remap.keys()):
        target = id_remap[k]
        seen = {k}
        while target in id_remap and target not in seen:
            seen.add(target)
            target = id_remap[target]
        id_remap[k] = target

    # ── Stage 8: build final node list in original sl-group order ──
    deduplicated_nodes: list[GraphNode] = []
    for i in range(len(sl_entries)):
        survivor = sl_survivor_by_idx[i]
        if survivor.id in absorbed_sl_ids:
            continue
        deduplicated_nodes.append(survivor)

    return deduplicated_nodes, id_remap, merged_count


def flatten_remap(id_remap: dict[str, str]) -> dict[str, str]:
    """Collapse multi-hop remap chains so every key points at its final survivor.

    Resolution runs in successive passes, and each pass merges the survivors of
    the previous one. Phase 1 may record ``dup -> A``; a later pass then merges
    A itself and records ``A -> B``. The combined mapping contains both hops,
    but ``remap_relationships`` performs a single lookup, so a relationship
    pointing at ``dup`` would be rewritten to ``A`` — a node that was removed.
    The relationship is then left dangling.

    Cycles cannot arise from union-find output, but a defensive visit set is
    kept so a malformed mapping degrades to "stop early" rather than hanging.
    """
    flattened: dict[str, str] = {}
    for start in id_remap:
        seen = {start}
        target = id_remap[start]
        while target in id_remap and target not in seen:
            seen.add(target)
            target = id_remap[target]
        flattened[start] = target
    return flattened


def remap_relationships(
    relationships: list[GraphRelationship],
    id_remap: dict[str, str],
) -> list[GraphRelationship]:
    """Remap relationship endpoints using id_remap and deduplicate.

    The dedup key includes ``properties["rel_type"]``, not just
    ``rel.type``. Every data edge is written with ``rel.type ==
    "RELATES"`` and its *semantic* type in the ``rel_type`` property
    (see ``GraphExtraction._relations_to_relationships``), so a key of
    ``(start, rel.type, end)`` reads as "one edge per entity pair" and
    silently discards every fact after the first: ``Alice WORKS_AT
    Acme`` and ``Alice FOUNDED Acme`` collapse, and the second is
    dropped here — in Python, before anything reaches the graph.

    Structural edges (``MENTIONED_IN``, ``PART_OF``, ``NEXT_CHUNK``)
    carry no ``rel_type``; they fall back to ``""`` and keep their
    previous one-edge-per-pair behaviour.

    ``id_remap`` is flattened first. A multi-pass resolver records each
    hop separately (``dup -> A`` in one pass, ``A -> B`` in the next), and
    the single lookup below would otherwise re-point an edge at ``A``,
    which that later pass removed. Flattening here rather than at each
    call site means no caller can forget it; it is a no-op on a mapping
    that is already one hop deep.
    """
    id_remap = flatten_remap(id_remap)
    deduplicated_rels: list[GraphRelationship] = []
    seen_rels: set[tuple[str, str, str, str]] = set()
    for rel in relationships:
        start = id_remap.get(rel.start_node_id, rel.start_node_id)
        end = id_remap.get(rel.end_node_id, rel.end_node_id)
        rel_key = (start, rel.type, rel.properties.get("rel_type", ""), end)
        if rel_key not in seen_rels:
            seen_rels.add(rel_key)
            deduplicated_rels.append(
                GraphRelationship(
                    start_node_id=start,
                    end_node_id=end,
                    type=rel.type,
                    properties=rel.properties,
                )
            )
    return deduplicated_rels


#: ``ctx.metadata`` key: ``set[frozenset[str]]`` of node-id pairs already judged to
#: be two things. A strategy must not merge such a pair and should not spend a
#: call asking about it again. Set by the caller.
RESOLUTION_SKIP_PAIRS = "resolution_skip_pairs"

#: ``ctx.metadata`` key: ``set[frozenset[str]]`` of node-id pairs the caller wants
#: judged even where the strategy's own candidate search would not surface them —
#: a table's "M. Ellison" beside a document's "Maya Ellison", which a name rule
#: spotted and an embedding threshold did not. Set by the caller.
RESOLUTION_ASK_PAIRS = "resolution_ask_pairs"

#: ``ctx.metadata`` key: ``set[str]`` of node ids whose identity was declared, not
#: extracted — rows of a table, each keyed. No two of them are one thing, so a
#: strategy should not merge, or ask about, a pair drawn from this set. Set by
#: the caller; a set rather than pairs because a table has n rows and n² pairs.
RESOLUTION_DISTINCT_IDS = "resolution_distinct_ids"

#: ``ctx.metadata`` key: ``set[frozenset[str]]`` of node-id pairs the strategy
#: examined and judged to be two things. Written by the strategy, so the caller
#: can remember the answer and pass it back as ``RESOLUTION_SKIP_PAIRS`` next time.
RESOLUTION_REJECTED_PAIRS = "resolution_rejected_pairs"


class ResolutionStrategy(ABC):
    """Abstract base class for entity resolution (deduplication) strategies.

    A resolution strategy receives extracted graph data and merges
    duplicate entities, producing a deduplicated result.

    Example::

        class VectorFuzzyResolution(ResolutionStrategy):
            async def resolve(self, graph_data, ctx):
                # Use embeddings to find near-duplicate entities
                ...

    A strategy is run within one document by ``ingest`` and, when passed to
    ``finalize(resolver=...)``, over the whole graph. In the second setting the
    caller knows things the strategy does not, and says so through
    ``ctx.metadata``: :data:`RESOLUTION_SKIP_PAIRS` are pairs already decided
    against, :data:`RESOLUTION_DISTINCT_IDS` are ids that are pairwise distinct
    by declaration, :data:`RESOLUTION_ASK_PAIRS` are pairs it wants an answer
    on. A strategy that judges pairs reports the ones it rejected under
    :data:`RESOLUTION_REJECTED_PAIRS`. All four are optional; a strategy that
    ignores them is still correct, only less economical.
    """

    @abstractmethod
    async def resolve(
        self,
        graph_data: GraphData,
        ctx: Context,
    ) -> ResolutionResult:
        """Resolve duplicate entities in the extracted graph data.

        Args:
            graph_data: Extracted nodes and relationships.
            ctx: Execution context.

        Returns:
            ResolutionResult with deduplicated data and merge statistics.
        """
        ...
