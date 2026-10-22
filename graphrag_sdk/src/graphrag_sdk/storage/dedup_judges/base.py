# GraphRAG SDK — Storage: Dedup Judge ABC
# Pattern: Strategy — who decides that two entities are one.
# Origin: User design — absent in Neo4j.

from __future__ import annotations

from abc import ABC, abstractmethod

from graphrag_sdk.core.models import JudgeSet, SetVerdict


class DedupJudge(ABC):
    """Abstract base class for the judge of cross-document deduplication.

    Every dedup run has one; :meth:`GraphRAG.finalize` uses :class:`LLMJudge`
    with the GraphRAG LLM unless given another. The run nominates the
    candidates — each pending entity with its nearest names, in small dense
    sets — and the judge decides only which members of a set are the same
    real-world entity.

    The rest stays with the run, whatever the judge:

    - every answer is checked: members the set has, each in one group at
      most, a name or ``keep`` pick only on a group of two or more. An answer
      that fails is no answer;
    - a set without an answer is asked again, up to 2 more times, then stays
      pending for the next run; after 3 runs it is reported unresolved;
    - the merge — survivor, labels, properties, relationships — follows the
      run's own rules. A judge writes nothing.

    Example::

        from graphrag_sdk import DedupJudge, JudgeGroup, SetVerdict

        class SpellingJudge(DedupJudge):
            async def judge(self, sets, *, retry=False):
                verdicts = []
                for s in sets:
                    by_name: dict[str, list[int]] = {}
                    for i, ent in enumerate(s.entities):
                        by_name.setdefault(ent.name.casefold(), []).append(i)
                    verdicts.append(SetVerdict(groups=[
                        JudgeGroup(members=m) for m in by_name.values() if len(m) > 1
                    ]))
                return verdicts
    """

    @abstractmethod
    async def judge(self, sets: list[JudgeSet], *, retry: bool = False) -> list[SetVerdict | None]:
        """Partition each set into groups of the same real-world entity.

        Args:
            sets: Candidate sets of one run.
            retry: ``True`` when the sets are asked again because no usable
                answer came back for them. The LLM judge then asks one set per
                prompt, so one confusing set cannot spoil another.

        Returns:
            One verdict per set, in order; ``None`` for a set it could not
            decide. A call that raises, or returns a list of another length,
            answers none of its sets.
        """
        ...

    def usage(self) -> dict[str, int]:
        """Running totals of what this judge has spent, as whole numbers, e.g.
        ``{"llm_calls": 12}``. A run adds what it spent — how much the totals
        grew while it ran — to its ``judge_stats``, each key at 0 when it
        judged nothing; a key the run reports itself, and a value that is not
        a whole number, are left out. A judge shared by runs at the same time
        sees their spending mixed. Empty by default."""
        return {}
