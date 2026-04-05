# GraphRAG SDK 2.0 — Ingestion: LLM-Verified Resolution
# Three-tier deduplication:
#   Phase 1: normalized name exact-match merge (free)
#   Phase 2: embedding cosine similarity
#     >= hard_threshold  → hard merge (no LLM)
#     soft_threshold..hard_threshold → LLM YES/NO verification
#     < soft_threshold   → skip
#
# Inspired by: "semantic blocking + LLM pairwise verification" (SOTA 2024-25)

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from graphrag_sdk.core.context import Context
from graphrag_sdk.core.models import (
    GraphData,
    GraphNode,
    GraphRelationship,
    ResolutionResult,
)
from graphrag_sdk.core.providers import Embedder, LLMInterface
from graphrag_sdk.ingestion.resolution_strategies.base import (
    ResolutionStrategy,
    exact_match_merge,
    remap_relationships,
)

logger = logging.getLogger(__name__)

_VERIFY_PROMPT = (
    "You are an entity resolution assistant. Decide whether the two entities below "
    "refer to the exact same real-world entity.\n\n"
    "Entity A (type: {label}):\n"
    "  Name: {name_a}\n"
    "  Description: {desc_a}\n"
    "  Relationships: {neighbors_a}\n\n"
    "Entity B (type: {label}):\n"
    "  Name: {name_b}\n"
    "  Description: {desc_b}\n"
    "  Relationships: {neighbors_b}\n\n"
    "Embedding cosine similarity: {similarity:.3f}\n\n"
    "Answer with exactly one of:\n"
    "  YES — they are the same entity\n"
    "  NO  — they are different entities\n\n"
    "Then on a new line give a brief reason (one sentence, max 20 words).\n\n"
    "Answer:"
)


@dataclass
class _VerificationRequest:
    """One ambiguous pair waiting for LLM confirmation."""

    node_a: GraphNode
    node_b: GraphNode
    idx_a: int  # index within the label group's valid_nodes list
    idx_b: int
    similarity: float
    label: str
    prompt_index: int = field(default=-1)  # set just before batch submission


class LLMVerifiedResolution(ResolutionStrategy):
    """Three-tier entity resolution: exact-match → hard embedding merge →
    LLM-verified ambiguous zone → skip.

    Flow:
      1. Group by (normalized_name, label) — exact-match merge, same as
         DescriptionMergeResolution. No LLM or embedder needed here.
      2. Embed all surviving node names within each label group.
      3. For each pair (within same label only):
           similarity >= hard_threshold  → hard merge immediately
           soft_threshold <= sim < hard  → send to LLM YES/NO batch
           sim < soft_threshold          → skip
      4. LLM confirms or rejects each ambiguous pair.
      5. Apply Union-Find clusters from hard + LLM-confirmed merges.
      6. Remap and deduplicate relationships.

    Args:
        llm: LLM provider for YES/NO verification and description summaries.
        embedder: Embedder for pairwise cosine similarity.
        hard_threshold: Similarity at or above which entities are merged
            without LLM confirmation (default: 0.95).
        soft_threshold: Similarity below which pairs are skipped entirely
            (default: 0.80).
        max_llm_pairs: Maximum ambiguous pairs sent to LLM per call.
            Pairs are ranked by descending similarity before capping
            (default: 500).
        max_llm_concurrency: Override the LLM provider's concurrency limit
            for the verification batch (default: None → provider default).
        force_summary_threshold: Number of descriptions that triggers LLM
            summarisation in Phase 1 (default: 3).
        max_summary_tokens: Token budget hint for description summaries.
        ann_top_k: Number of nearest neighbours to retrieve per node from the
            hnswlib HNSW index (default: 50). Higher values improve recall at
            the cost of speed.
    """

    def __init__(
        self,
        llm: LLMInterface | None = None,
        embedder: Embedder | None = None,
        *,
        hard_threshold: float = 0.95,
        soft_threshold: float = 0.80,
        max_llm_pairs: int = 500,
        max_llm_concurrency: int | None = None,
        force_summary_threshold: int = 3,
        max_summary_tokens: int = 500,
        ann_top_k: int = 50,
    ) -> None:
        if hard_threshold <= soft_threshold:
            raise ValueError(
                f"hard_threshold ({hard_threshold}) must be > soft_threshold ({soft_threshold})"
            )
        self.llm = llm
        self.embedder = embedder
        self.hard_threshold = hard_threshold
        self.soft_threshold = soft_threshold
        self.max_llm_pairs = max_llm_pairs
        self.max_llm_concurrency = max_llm_concurrency
        self.force_summary_threshold = force_summary_threshold
        self.max_summary_tokens = max_summary_tokens
        self.ann_top_k = ann_top_k

    async def resolve(
        self,
        graph_data: GraphData,
        ctx: Context,
    ) -> ResolutionResult:
        ctx.log(
            f"LLMVerifiedResolution: {len(graph_data.nodes)} nodes, "
            f"{len(graph_data.relationships)} rels | "
            f"hard={self.hard_threshold}, soft={self.soft_threshold}"
        )

        # ── Phase 1: Normalized name exact-match merge ────────────────────────
        deduplicated_nodes, id_remap, merged_count = await exact_match_merge(
            graph_data.nodes,
            self.llm,
            force_summary_threshold=self.force_summary_threshold,
            max_summary_tokens=self.max_summary_tokens,
        )
        ctx.log(
            f"Phase 1 (exact-match): {merged_count} merged, {len(deduplicated_nodes)} surviving"
        )

        # ── Phase 2-5: Embedding + three-tier classification ──────────────────
        if self.embedder and len(deduplicated_nodes) >= 2:
            fuzzy_remap, hard_merges, llm_merges = await self._embedding_and_llm_merge(
                deduplicated_nodes, ctx, graph_data.relationships, id_remap
            )
            if fuzzy_remap:
                final_nodes: list[GraphNode] = []
                for node in deduplicated_nodes:
                    if node.id not in fuzzy_remap:
                        final_nodes.append(node)
                    else:
                        merged_count += 1
                id_remap.update(fuzzy_remap)
                deduplicated_nodes = final_nodes
                ctx.log(
                    f"Phase 2-5 (embedding+LLM): {hard_merges} hard merges, "
                    f"{llm_merges} LLM-confirmed merges"
                )

        # ── Phase 6: Remap relationships and deduplicate ──────────────────────
        deduplicated_rels = remap_relationships(graph_data.relationships, id_remap)

        ctx.log(
            f"LLMVerifiedResolution complete: {len(deduplicated_nodes)} nodes "
            f"({merged_count} merged), {len(deduplicated_rels)} rels"
        )
        return ResolutionResult(
            nodes=deduplicated_nodes,
            relationships=deduplicated_rels,
            merged_count=merged_count,
        )

    async def _embedding_and_llm_merge(
        self,
        nodes: list[GraphNode],
        ctx: Context,
        relationships: list[GraphRelationship],
        phase1_remap: dict[str, str],
    ) -> tuple[dict[str, str], int, int]:
        """Phases 2-5: embed → classify pairs → LLM verify ambiguous zone.

        Returns:
            (id_remap, hard_merge_count, llm_merge_count)
        """

        # Build adjacency using remapped IDs (Issue 3: use post-phase-1 IDs)
        # node_id → canonical survivor id after phase 1
        def _canonical(nid: str) -> str:
            return phase1_remap.get(nid, nid)

        node_id_to_name: dict[str, str] = {n.id: str(n.properties.get("name", n.id)) for n in nodes}
        adjacency: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for rel in relationships:
            src = _canonical(rel.start_node_id)
            tgt = _canonical(rel.end_node_id)
            src_name = node_id_to_name.get(src, src)
            tgt_name = node_id_to_name.get(tgt, tgt)
            adjacency[src].append((tgt_name, rel.type))
            adjacency[tgt].append((src_name, rel.type))

        def _fmt_neighbors(node_id: str, top_n: int = 5) -> str:
            nbrs = adjacency.get(node_id, [])[:top_n]
            if not nbrs:
                return "(none)"
            return "; ".join(f"{rel_type} -> {name}" for name, rel_type in nbrs)

        # Group by label — never merge across labels
        by_label: dict[str, list[GraphNode]] = defaultdict(list)
        for n in nodes:
            by_label[n.label].append(n)

        remap: dict[str, str] = {}
        total_hard = 0
        total_llm = 0

        emb_cache: dict[str, list[float]] = ctx.metadata.setdefault("embedding_cache", {})

        for label, label_nodes in by_label.items():
            if len(label_nodes) < 2:
                continue

            miss_nodes = [n for n in label_nodes if n.id not in emb_cache]
            miss_names = [str(n.properties.get("name", n.id)) for n in miss_nodes]
            try:
                if miss_names:
                    miss_vecs = await self.embedder.aembed_documents(miss_names)
                    for node, vec in zip(miss_nodes, miss_vecs):
                        if vec:
                            emb_cache[node.id] = vec
            except Exception as exc:
                ctx.log(f"Embedding failed for label '{label}': {exc}", logging.WARNING)
                continue

            vectors = [emb_cache.get(n.id, []) for n in label_nodes]

            # Filter failed embeddings
            valid = [
                (i, node, vec) for i, (node, vec) in enumerate(zip(label_nodes, vectors)) if vec
            ]
            if len(valid) < 2:
                continue

            _, valid_nodes, vecs = zip(*valid)
            valid_nodes = list(valid_nodes)
            mat = np.array(vecs, dtype=np.float32)
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            mat_normed = mat / norms

            n_nodes = len(valid_nodes)

            # Union-Find
            parent: dict[int, int] = {i: i for i in range(n_nodes)}

            def find(x: int) -> int:
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            def union(x: int, y: int) -> None:
                px, py = find(x), find(y)
                if px != py:
                    parent[py] = px

            hard_pairs: list[tuple[int, int]] = []
            ambiguous_pairs: list[tuple[int, int, float]] = []

            # ANN via hnswlib HNSW (O(N log N)) — no OpenMP, no macOS deadlock.
            import hnswlib

            top_k = min(self.ann_top_k, n_nodes - 1)
            dim = mat_normed.shape[1]
            hnsw_index = hnswlib.Index(space="ip", dim=dim)
            hnsw_index.init_index(max_elements=n_nodes, ef_construction=200, M=32)
            hnsw_index.set_ef(max(top_k + 1, 64))
            hnsw_index.add_items(mat_normed, list(range(n_nodes)))
            # "ip" space with unit-normed vectors: distance = 1 - cosine_similarity
            nbrs, dists = hnsw_index.knn_query(mat_normed, k=top_k + 1)
            for i in range(n_nodes):
                for rank in range(1, top_k + 1):
                    j = int(nbrs[i, rank])
                    if j <= i:
                        continue
                    sim_val = 1.0 - float(dists[i, rank])
                    if sim_val >= self.hard_threshold:
                        hard_pairs.append((i, j))
                    elif sim_val >= self.soft_threshold:
                        ambiguous_pairs.append((i, j, sim_val))

            # Hard merges — no LLM needed
            for gi, gj in hard_pairs:
                union(gi, gj)
            total_hard += len(hard_pairs)

            # Ambiguous zone — LLM verification
            if ambiguous_pairs and self.llm is not None:
                # Cluster ambiguous pairs using scipy agglomerative clustering.
                # Build a distance matrix (1 - sim) for nodes involved in ambiguous pairs,
                # then use average-linkage fcluster to find tight groups.
                # Intra-cluster pairs → hard merge; cross-cluster pairs → LLM.
                import scipy.cluster.hierarchy as sch
                import scipy.spatial.distance as ssd

                amb_set = {i for i, j, _ in ambiguous_pairs} | {j for i, j, _ in ambiguous_pairs}
                amb_indices = sorted(amb_set)
                idx_map = {v: k for k, v in enumerate(amb_indices)}
                n_amb = len(amb_indices)

                # Condensed distance matrix for scipy
                dist_matrix = np.ones((n_amb, n_amb), dtype=np.float32)
                np.fill_diagonal(dist_matrix, 0.0)
                for gi, gj, sim_val in ambiguous_pairs:
                    ai, aj = idx_map[gi], idx_map[gj]
                    dist_matrix[ai, aj] = 1.0 - sim_val
                    dist_matrix[aj, ai] = 1.0 - sim_val

                condensed = ssd.squareform(dist_matrix)
                linkage = sch.linkage(condensed, method="average")
                # Cut at distance = 1 - soft_threshold to cluster pairs above soft_threshold
                cut = 1.0 - self.soft_threshold
                cluster_labels = sch.fcluster(linkage, t=cut, criterion="distance")
                node_to_comm = {amb_indices[k]: int(cluster_labels[k]) for k in range(n_amb)}

                # Intra-cluster pairs → hard merge (no LLM needed)
                for gi, gj, _ in ambiguous_pairs:
                    if node_to_comm.get(gi, -1) == node_to_comm.get(gj, -2):
                        union(gi, gj)

                # Cross-cluster pairs → LLM (the genuinely ambiguous ones)
                boundary_pairs = [
                    (gi, gj, sim_val)
                    for gi, gj, sim_val in ambiguous_pairs
                    if node_to_comm.get(gi, -1) != node_to_comm.get(gj, -2)
                ]
                boundary_pairs.sort(key=lambda t: t[2], reverse=True)
                capped = boundary_pairs[: self.max_llm_pairs]
                ctx.log(
                    f"Label '{label}': {len(ambiguous_pairs)} ambiguous → "
                    f"{len(ambiguous_pairs) - len(boundary_pairs)} intra-community hard merges, "
                    f"{len(boundary_pairs)} boundary pairs → LLM"
                )

                requests: list[_VerificationRequest] = []
                prompts: list[str] = []

                for gi, gj, sim_val in capped:
                    node_a = valid_nodes[gi]
                    node_b = valid_nodes[gj]
                    req = _VerificationRequest(
                        node_a=node_a,
                        node_b=node_b,
                        idx_a=gi,
                        idx_b=gj,
                        similarity=sim_val,
                        label=label,
                        prompt_index=len(prompts),
                    )
                    prompts.append(
                        _VERIFY_PROMPT.format(
                            label=label,
                            name_a=str(node_a.properties.get("name", node_a.id)),
                            desc_a=str(node_a.properties.get("description", "(no description)")),
                            neighbors_a=_fmt_neighbors(node_a.id),
                            name_b=str(node_b.properties.get("name", node_b.id)),
                            desc_b=str(node_b.properties.get("description", "(no description)")),
                            neighbors_b=_fmt_neighbors(node_b.id),
                            similarity=sim_val,
                        )
                    )
                    requests.append(req)

                batch_results = await self.llm.abatch_invoke(
                    prompts,
                    max_concurrency=self.max_llm_concurrency,
                )
                result_map = {item.index: item for item in batch_results}

                llm_confirmed = 0
                for req in requests:
                    item = result_map.get(req.prompt_index)
                    if item is None or not item.ok:
                        logger.warning(
                            "LLM verification failed for '%s' vs '%s': %s",
                            req.node_a.properties.get("name", req.node_a.id),
                            req.node_b.properties.get("name", req.node_b.id),
                            getattr(item, "error", "missing result"),
                        )
                        continue
                    answer = item.response.content.strip().upper()
                    if answer.startswith("YES"):
                        union(req.idx_a, req.idx_b)
                        llm_confirmed += 1

                total_llm += llm_confirmed
                ctx.log(
                    f"Label '{label}': {len(capped)} ambiguous pairs → "
                    f"{llm_confirmed} LLM-confirmed merges"
                )

            # Build clusters from Union-Find
            clusters: dict[int, list[int]] = defaultdict(list)
            for i in range(n_nodes):
                clusters[find(i)].append(i)

            label_remap_before = len(remap)
            for root, members in clusters.items():
                if len(members) <= 1:
                    continue
                survivor = valid_nodes[members[0]]
                for mi in members[1:]:
                    dup = valid_nodes[mi]
                    remap[dup.id] = survivor.id
                    for key, value in dup.properties.items():
                        if key not in survivor.properties:
                            survivor.properties[key] = value

            label_merges = len(remap) - label_remap_before
            if label_merges:
                ctx.log(f"Label '{label}': {label_merges} total merges")

        return remap, total_hard, total_llm
