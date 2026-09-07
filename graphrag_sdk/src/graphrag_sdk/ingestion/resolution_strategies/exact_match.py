# GraphRAG SDK — Ingestion: Exact Match Resolution
# Origin: Neo4j SinglePropertyExactMatchResolver — simplified.

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from graphrag_sdk.core.context import Context
from graphrag_sdk.core.models import GraphData, ResolutionResult
from graphrag_sdk.ingestion.resolution_strategies.base import (
    ResolutionStrategy,
    exact_match_merge,
    remap_relationships,
)

if TYPE_CHECKING:
    from graphrag_sdk.core.providers import LLMInterface

logger = logging.getLogger(__name__)


class ExactMatchResolution(ResolutionStrategy):
    """Deduplicate entities by exact name match.

    This is Phase 1 of :class:`LLMVerifiedResolution` offered on its own,
    and it delegates to the same :func:`exact_match_merge` helper so the two
    cannot drift apart:

    ==============================  ==================================
    case                            what happens
    ==============================  ==================================
    same name, same label           merge, no LLM
    same name, different label      LLM verdict, and only when there
                                    are at least
                                    *cross_label_min_descriptions*
                                    descriptions as evidence; without
                                    an LLM the nodes are left alone
    ==============================  ==================================

    Names are normalised (lower-cased and trimmed) before grouping, so
    ``"Ada Lovelace"`` and ``"  ada lovelace "`` are the same entity.
    Descriptions are joined with ``" | "``, unless there are at least
    *force_summary_threshold* of them, in which case the LLM writes a
    single summary instead. ``source_chunk_ids`` are unioned across every
    duplicate. When a cross-label merge is approved, the absorbed labels
    are recorded in ``merged_labels`` and promoted to real Cypher labels
    on write, so the node stays reachable under every type it was
    extracted as.

    This class used to re-implement all of the above and got it wrong in
    three ways, which is why it now delegates instead. It grouped on the
    raw ``id`` property, so same-named entities did not merge at all
    unless they already shared an id; and it merged by copying only the
    keys the survivor lacked, so -- because the survivor always has one --
    every duplicate's ``description`` and ``source_chunk_ids`` were
    silently discarded. That is the same defect found in the embedding
    stage, in cross-label PASS 2, and in Stage 1 itself; the cause each
    time was a separate copy of the merge loop.

    Args:
        resolve_property: Property carrying the entity name
            (default: ``"name"``). Nodes without it fall back to
            ``node.id``, which for extracted entities is already the
            normalised ``name__type``.
        llm: Optional LLM. Required for cross-label verification and for
            description summarisation; without it both are skipped and
            same-label merging still runs.
        force_summary_threshold: Description count at which the LLM
            summarises rather than concatenating.
        max_summary_tokens: Token ceiling requested for that summary.
        cross_label_merge: Whether same-name/different-label groups are
            put to the LLM at all.
        cross_label_min_descriptions: Evidence floor for that check.
    """

    def __init__(
        self,
        resolve_property: str = "name",
        *,
        llm: LLMInterface | None = None,
        force_summary_threshold: int = 3,
        max_summary_tokens: int = 500,
        cross_label_merge: bool = True,
        cross_label_min_descriptions: int = 3,
    ) -> None:
        self.resolve_property = resolve_property
        self.llm = llm
        self.force_summary_threshold = force_summary_threshold
        self.max_summary_tokens = max_summary_tokens
        self.cross_label_merge = cross_label_merge
        self.cross_label_min_descriptions = cross_label_min_descriptions

    async def resolve(
        self,
        graph_data: GraphData,
        ctx: Context,
    ) -> ResolutionResult:
        ctx.log(
            f"Resolving duplicates by exact match on '{self.resolve_property}' "
            f"({len(graph_data.nodes)} nodes, {len(graph_data.relationships)} rels)"
        )

        deduplicated_nodes, id_remap, merged_count = await exact_match_merge(
            graph_data.nodes,
            self.llm,
            force_summary_threshold=self.force_summary_threshold,
            max_summary_tokens=self.max_summary_tokens,
            cross_label_merge=self.cross_label_merge,
            cross_label_min_descriptions=self.cross_label_min_descriptions,
            resolve_property=self.resolve_property,
        )

        deduplicated_rels = remap_relationships(graph_data.relationships, id_remap)

        ctx.log(
            f"Resolution complete: {len(deduplicated_nodes)} nodes "
            f"({merged_count} merged), {len(deduplicated_rels)} rels"
        )
        return ResolutionResult(
            nodes=deduplicated_nodes,
            relationships=deduplicated_rels,
            merged_count=merged_count,
            remap=id_remap,
        )
