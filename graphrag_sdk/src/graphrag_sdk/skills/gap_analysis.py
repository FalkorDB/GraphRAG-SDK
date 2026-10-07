# GraphRAG SDK — Skills: Gap Analysis (Phase 3.4)

from __future__ import annotations

from typing import Any

from graphrag_sdk.core.context import Context
from graphrag_sdk.core.models import SkillResult
from graphrag_sdk.skills.base import Skill


class GapAnalysisSkill(Skill):
    """Surface sparse or missing areas of the knowledge graph.

    Flags entities with no relationships (isolated nodes) and entity
    labels that are underrepresented, so users can target ingestion or
    backfill where the graph is thin.
    """

    name = "gap_analysis"
    description = (
        "Identify gaps in the knowledge graph: isolated entities and "
        "sparsely populated entity types."
    )

    async def run(self, ctx: Context | None = None, **params: Any) -> SkillResult:
        min_instances = int(params.get("min_instances", 2))
        limit = int(params.get("limit", 50))

        isolated_rows = await self._rows(
            f"MATCH (e:__Entity__) WHERE NOT (e)-[]-(:__Entity__) RETURN e.id AS id LIMIT {limit}"
        )
        isolated = [r[0] for r in isolated_rows if r and r[0] is not None]

        label_rows = await self._rows(
            "MATCH (e:__Entity__) "
            "WITH [l IN labels(e) WHERE l <> '__Entity__'][0] AS label "
            "RETURN label, count(*) AS n ORDER BY n ASC"
        )
        sparse_labels = [
            {"label": r[0], "count": int(r[1])}
            for r in label_rows
            if r and r[0] is not None and int(r[1]) < min_instances
        ]

        data = {
            "isolated_entities": isolated,
            "num_isolated": len(isolated),
            "sparse_labels": sparse_labels,
        }
        summary = await self._summarize(
            ctx,
            f"The graph has {len(isolated)} isolated entities and these "
            f"sparse types: {sparse_labels}. Suggest where to focus ingestion.",
        )
        return SkillResult(skill=self.name, summary=summary, data=data, sources=isolated[:10])
