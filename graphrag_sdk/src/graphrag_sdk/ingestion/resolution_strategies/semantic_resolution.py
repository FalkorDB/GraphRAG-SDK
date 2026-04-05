# GraphRAG SDK 2.0 — Ingestion: Semantic Resolution
# Groups entities by (normalized_name, label) to prevent cross-type merges,
# then uses embeddings for near-duplicate detection within same-label groups.

from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np

try:
    import hnswlib as _hnswlib
except ImportError:  # pragma: no cover
    _hnswlib = None  # type: ignore[assignment]

from graphrag_sdk.core.context import Context
from graphrag_sdk.core.models import (
    GraphData,
    GraphNode,
    ResolutionResult,
)
from graphrag_sdk.core.providers import Embedder, LLMInterface
from graphrag_sdk.ingestion.resolution_strategies.base import (
    ResolutionStrategy,
    exact_match_merge,
    remap_relationships,
)

logger = logging.getLogger(__name__)


class SemanticResolution(ResolutionStrategy):
    """Deduplicate entities by (normalized_name, label) grouping with
    optional embedding-based near-duplicate detection.

    Key improvement over DescriptionMergeResolution: groups by
    ``(name.lower().strip(), label)`` so entities with the same name but
    different labels (e.g., Person "Paris" vs Location "Paris") are kept
    separate.

    Optionally uses an embedder to detect near-duplicates within the same
    label group (e.g., "J.R.R. Tolkien" ≈ "Tolkien") above a similarity
    threshold.

    Args:
        llm: LLM provider (used for description summarisation).
        embedder: Embedder for near-duplicate detection (optional).
        similarity_threshold: Cosine similarity threshold for fuzzy merge
            (default: 0.95 — very conservative).
        force_summary_threshold: Number of descriptions that triggers LLM
            summary instead of simple concatenation.
        max_summary_tokens: Token budget hint for the LLM prompt.
        ann_top_k: Number of nearest neighbours to retrieve per node from the
            hnswlib HNSW index (default: 50). Higher values improve recall at
            the cost of speed; 50 is sufficient for typical per-label group
            sizes.
    """

    def __init__(
        self,
        llm: LLMInterface | None = None,
        embedder: Embedder | None = None,
        *,
        similarity_threshold: float = 0.95,
        force_summary_threshold: int = 3,
        max_summary_tokens: int = 500,
        ann_top_k: int = 50,
    ) -> None:
        self.llm = llm
        self.embedder = embedder
        self.similarity_threshold = similarity_threshold
        self.force_summary_threshold = force_summary_threshold
        self.max_summary_tokens = max_summary_tokens
        self.ann_top_k = ann_top_k

    async def resolve(
        self,
        graph_data: GraphData,
        ctx: Context,
    ) -> ResolutionResult:
        ctx.log(
            f"SemanticResolution: resolving {len(graph_data.nodes)} nodes, "
            f"{len(graph_data.relationships)} rels"
        )

        # Phase 1: exact-match merge by (normalized_name, label)
        deduplicated_nodes, id_remap, merged_count = await exact_match_merge(
            graph_data.nodes,
            self.llm,
            force_summary_threshold=self.force_summary_threshold,
            max_summary_tokens=self.max_summary_tokens,
        )

        # Phase 3: Optional embedding-based fuzzy merge within same label
        if self.embedder and len(deduplicated_nodes) >= 2:
            fuzzy_remap = await self._fuzzy_merge(deduplicated_nodes, ctx)
            if fuzzy_remap:
                # Apply fuzzy merges
                final_nodes: list[GraphNode] = []
                for node in deduplicated_nodes:
                    if node.id not in fuzzy_remap:
                        final_nodes.append(node)
                    else:
                        merged_count += 1
                id_remap.update(fuzzy_remap)
                deduplicated_nodes = final_nodes

        # Remap relationship endpoints and deduplicate
        deduplicated_rels = remap_relationships(graph_data.relationships, id_remap)

        ctx.log(
            f"SemanticResolution complete: {len(deduplicated_nodes)} nodes "
            f"({merged_count} merged), {len(deduplicated_rels)} rels"
        )
        return ResolutionResult(
            nodes=deduplicated_nodes,
            relationships=deduplicated_rels,
            merged_count=merged_count,
        )

    async def _fuzzy_merge(
        self,
        nodes: list[GraphNode],
        ctx: Context,
    ) -> dict[str, str]:
        """Detect near-duplicate entities within same-label groups via embeddings.

        Returns a remap dict: duplicate_id → survivor_id.
        """
        # Group by label
        by_label: dict[str, list[GraphNode]] = defaultdict(list)
        for n in nodes:
            by_label[n.label].append(n)

        remap: dict[str, str] = {}

        for label, label_nodes in by_label.items():
            if len(label_nodes) < 2:
                continue

            remap_before = len(remap)

            emb_cache: dict[str, list[float]] = ctx.metadata.setdefault("embedding_cache", {})

            miss_nodes = [n for n in label_nodes if n.id not in emb_cache]
            miss_names = [str(n.properties.get("name", n.id)) for n in miss_nodes]
            try:
                if miss_names:
                    logger.info(
                        "SemanticResolution: embedding %d '%s' nodes",
                        len(miss_names),
                        label,
                    )
                    logger.debug(
                        "SemanticResolution: '%s' node sample (up to 5 of %d): %s",
                        label,
                        len(miss_names),
                        miss_names[:5],
                    )
                    miss_vecs = await self.embedder.aembed_documents(miss_names)
                    for node, vec in zip(miss_nodes, miss_vecs):
                        if vec:
                            emb_cache[node.id] = vec
                    logger.info(
                        "SemanticResolution: embeddings done for '%s' (%d/%d cached)",
                        label,
                        len([n for n in label_nodes if n.id in emb_cache]),
                        len(label_nodes),
                    )
            except Exception as _e:
                logger.warning("SemanticResolution: skipping '%s' embeddings: %s", label, _e)
                continue

            vectors = [emb_cache.get(n.id, []) for n in label_nodes]

            # Filter out failed embeddings
            valid = [
                (i, node, vec) for i, (node, vec) in enumerate(zip(label_nodes, vectors)) if vec
            ]
            if len(valid) < 2:
                continue

            _, valid_nodes, vecs = zip(*valid)
            mat = np.array(vecs, dtype=np.float32)
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            mat_normed = mat / norms

            n = len(valid_nodes)
            top_k = min(self.ann_top_k, n - 1)

            # Use Union-Find to cluster
            parent: dict[int, int] = {i: i for i in range(n)}

            def find(x: int) -> int:
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            def union(x: int, y: int) -> None:
                px, py = find(x), find(y)
                if px != py:
                    parent[py] = px

            logger.info(
                "SemanticResolution: hnswlib ANN for '%s' (n=%d, top_k=%d)",
                label,
                n,
                top_k,
            )
            dim = mat_normed.shape[1]
            hnsw_index = _hnswlib.Index(space="ip", dim=dim)
            hnsw_index.init_index(max_elements=n, ef_construction=200, M=32)
            hnsw_index.set_ef(max(top_k + 1, 64))
            hnsw_index.add_items(mat_normed, list(range(n)))
            # knn_query for "ip" space returns (1 - dot_product) as distance
            nbrs, dists = hnsw_index.knn_query(mat_normed, k=top_k + 1)
            for i in range(n):
                for rank in range(1, top_k + 1):
                    j = int(nbrs[i, rank])
                    if j <= i:
                        continue
                    sim_val = 1.0 - float(dists[i, rank])
                    if sim_val >= self.similarity_threshold:
                        union(i, j)
            logger.info("SemanticResolution: ANN done for '%s'", label)

            # Build clusters
            clusters: dict[int, list[int]] = defaultdict(list)
            for i in range(n):
                clusters[find(i)].append(i)

            for root, members in clusters.items():
                if len(members) <= 1:
                    continue
                survivor = valid_nodes[members[0]]
                for mi in members[1:]:
                    dup = valid_nodes[mi]
                    remap[dup.id] = survivor.id
                    # Merge properties
                    for key, value in dup.properties.items():
                        if key not in survivor.properties:
                            survivor.properties[key] = value

            label_merges = len(remap) - remap_before
            if label_merges:
                ctx.log(f"Fuzzy merge on label '{label}': {label_merges} duplicates found")

        return remap
