# GraphRAG SDK — Ingestion: Resolution Strategy ABC
# Pattern: Strategy — different deduplication approaches implement this interface.

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import TYPE_CHECKING

from graphrag_sdk.core.context import Context
from graphrag_sdk.core.models import GraphData, GraphNode, GraphRelationship, ResolutionResult

if TYPE_CHECKING:
    from graphrag_sdk.core.providers import LLMInterface


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
) -> tuple[list[GraphNode], dict[str, str], int]:
    """Phase 1: group nodes by normalized name and merge exact duplicates.

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

    Returns:
        (deduplicated_nodes, id_remap, merged_count)
    """
    # ── Stage 1: always group by (name, label) for the safe same-label pass ──
    sl_groups: dict[tuple[str, str], list[GraphNode]] = defaultdict(list)
    for node in nodes:
        name = node.properties.get("name", node.id)
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
            norm = str(entry["nodes"][0].properties.get("name", "")).strip().lower()
            by_name[norm].append(i)
        for _name_key, indices in by_name.items():
            if len(indices) < 2:
                continue
            labels = {sl_entries[i]["nodes"][0].label for i in indices}
            if len(labels) < 2:
                continue
            total_orig_descs = sum(sl_entries[i]["orig_desc_count"] for i in indices)
            if total_orig_descs < force_summary_threshold:
                # Fail-safe: insufficient evidence to ask the LLM; preserve.
                continue
            all_descs: list[str] = []
            for i in indices:
                for n in sl_entries[i]["nodes"]:
                    d = n.properties.get("description", "")
                    if d:
                        all_descs.append(str(d))
            entity_name = str(sl_entries[indices[0]]["nodes"][0].properties.get("name", ""))
            cl_candidates.append(
                {
                    "sl_indices": indices,
                    "name": entity_name,
                    "descriptions": all_descs,
                    "types": sorted(labels),
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
            name = str(entry["nodes"][0].properties.get("name", ""))
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
                # NO or malformed → fail-safe: preserve homographs.
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
            survivor.properties["description"] = (
                sl_summaries[i] if i in sl_summaries else " | ".join(entry["descriptions"])
            )
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
        for s in sl_survivors_in_cand:
            if s.id == cl_survivor.id:
                continue
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


def remap_relationships(
    relationships: list[GraphRelationship],
    id_remap: dict[str, str],
) -> list[GraphRelationship]:
    """Remap relationship endpoints using id_remap and deduplicate."""
    deduplicated_rels: list[GraphRelationship] = []
    seen_rels: set[tuple[str, str, str]] = set()
    for rel in relationships:
        start = id_remap.get(rel.start_node_id, rel.start_node_id)
        end = id_remap.get(rel.end_node_id, rel.end_node_id)
        rel_key = (start, rel.type, end)
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


class ResolutionStrategy(ABC):
    """Abstract base class for entity resolution (deduplication) strategies.

    A resolution strategy receives extracted graph data and merges
    duplicate entities, producing a deduplicated result.

    Example::

        class VectorFuzzyResolution(ResolutionStrategy):
            async def resolve(self, graph_data, ctx):
                # Use embeddings to find near-duplicate entities
                ...
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
