# GraphRAG SDK — Skills: Gap Analysis

from __future__ import annotations

from typing import Any

from graphrag_sdk.core.context import Context
from graphrag_sdk.core.models import SkillResult
from graphrag_sdk.skills.base import Skill, skill_parameters

MAX_ISOLATED = 500


class GapAnalysisSkill(Skill):
    """Surface sparse or missing areas of the knowledge graph.

    Flags entities with no relationships to other entities (isolated nodes)
    and entity labels that are underrepresented, so users can target
    ingestion or backfill where the graph is thin.
    """

    name = "gap_analysis"
    description = (
        "Identify gaps in the knowledge graph: isolated entities and "
        "sparsely populated entity types."
    )
    parameters = skill_parameters(
        {
            "min_instances": {
                "type": "integer",
                "description": "Entity types with fewer instances are reported, default 2.",
            },
            "limit": {
                "type": "integer",
                "description": f"Isolated entities listed, 1-{MAX_ISOLATED}, default 50.",
            },
        }
    )

    async def run(self, ctx: Context | None = None, **params: Any) -> SkillResult:
        min_instances = self._int_param(params, "min_instances", 2, 1, 10**6)
        limit = self._int_param(params, "limit", 50, 1, MAX_ISOLATED)

        isolated_match = "MATCH (e:__Entity__) WHERE NOT (e)-[:RELATES]-(:__Entity__) "
        isolated_rows = await self._rows(
            isolated_match + "RETURN e.id AS id ORDER BY id LIMIT $limit",
            {"limit": limit},
        )
        isolated = [r[0] for r in isolated_rows if r and r[0] is not None]
        num_isolated = len(isolated)
        if len(isolated_rows) >= limit:
            # The list is capped: count them all.
            count_rows = await self._rows(isolated_match + "RETURN count(e) AS total")
            try:
                num_isolated = max(num_isolated, int(count_rows[0][0]))
            except (IndexError, TypeError, ValueError):
                pass  # no usable count: report the listed ones (num_isolated stays a floor)

        label_rows = await self._rows(
            "MATCH (e:__Entity__) "
            "UNWIND [l IN labels(e) WHERE l <> '__Entity__'] AS label "
            "RETURN label, count(*) AS n ORDER BY n ASC"
        )
        sparse_labels = [
            {"label": r[0], "count": int(r[1])}
            for r in label_rows
            if r and r[0] is not None and int(r[1]) < min_instances
        ]

        data = {
            "isolated_entities": isolated,
            "num_isolated": num_isolated,
            "truncated": num_isolated > len(isolated),
            "sparse_labels": sparse_labels,
        }
        summary = await self._summarize(
            ctx,
            f"The graph has {num_isolated} isolated entities and these "
            f"sparse types: {sparse_labels}. Suggest where to focus ingestion.",
        )
        return SkillResult(skill=self.name, summary=summary, data=data, sources=isolated[:10])
