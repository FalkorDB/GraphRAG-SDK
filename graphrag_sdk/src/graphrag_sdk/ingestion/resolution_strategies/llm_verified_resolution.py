# GraphRAG SDK — Ingestion: LLM-Verified Resolution
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
import re
from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd

try:
    import hnswlib as _hnswlib
except ImportError:  # pragma: no cover
    _hnswlib = None  # type: ignore[assignment]

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
    flatten_remap,
    remap_relationships,
    set_merged_descriptions,
)

logger = logging.getLogger(__name__)

# Group key used when the unified stage disables label bucketing entirely.
_UNIFIED_GROUP = "*"

_PAIR_BLOCK = (
    "Entity A (type: {label}):\n"
    "  Name: {name_a}\n"
    "  Description: {desc_a}\n"
    "  Relationships: {neighbors_a}\n\n"
    "Entity B (type: {label}):\n"
    "  Name: {name_b}\n"
    "  Description: {desc_b}\n"
    "  Relationships: {neighbors_b}\n\n"
    "Embedding cosine similarity: {similarity:.3f}"
)

# PASS 2 compares entities that carry DIFFERENT labels, so the type cannot be
# hoisted into a single header the way _PAIR_BLOCK does it. The differing types
# are stated explicitly because they are evidence the model should weigh, not
# noise to hide: "Person" vs "Engineer" is compatible, "City" vs "Airport" is
# not.
# Label families for the cross-label gate. Two entities whose extractor labels
# fall in DIFFERENT known families are never the same thing, whatever the
# embedding or the LLM says: a person is not the institution named after them,
# a town is not its church. Measured on the 11-document benchmark, 3 full runs
# of the unified stage: 54 of 55 merges across gold types were wrong (Person |
# Institution 12, Person | Organization 7, Institution | Place 6, Person | Place
# 5 ...) against 8 wrong of 105 same-type merges. Labels not listed here
# (custom ontologies: "Engineer", "Vessel") belong to no family and are left to
# the LLM, so Person | Engineer still resolves. Generic labels are wildcards.
LABEL_FAMILIES: dict[str, frozenset[str]] = {
    "person": frozenset({"person", "people", "individual", "human"}),
    "organization": frozenset(
        {
            "organization",
            "organisation",
            "company",
            "institution",
            "agency",
            "government",
            "team",
            "group",
            "university",
            "school",
            "corporation",
        }
    ),
    "place": frozenset(
        {
            "location",
            "place",
            "city",
            "country",
            "region",
            "state",
            "province",
            "building",
            "facility",
            "site",
            "address",
            "geo",
            "gpe",
        }
    ),
    "event": frozenset({"event", "incident", "conference", "war", "battle"}),
    "time": frozenset({"date", "time", "period", "year", "era"}),
    "work": frozenset(
        {
            "product",
            "technology",
            "concept",
            "method",
            "dataset",
            "law",
            "artifact",
            "document",
            "work",
            "tool",
            "material",
            "software",
            "algorithm",
            "theory",
        }
    ),
}
_GENERIC_LABELS = frozenset({"", "entity", "__entity__", "unknown", "thing", "other", "misc"})
_LABEL_TO_FAMILY: dict[str, str] = {
    member: family for family, members in LABEL_FAMILIES.items() for member in members
}


def labels_compatible(label_a: str, label_b: str) -> bool:
    """May two entities with these extractor labels be the same thing?

    ``True`` when the labels are equal, when either is generic, when either is
    unknown to :data:`LABEL_FAMILIES`, or when both map to the same family.
    ``False`` only when both map to *different* known families.
    """
    a = (label_a or "").strip().lower()
    b = (label_b or "").strip().lower()
    if a == b or a in _GENERIC_LABELS or b in _GENERIC_LABELS:
        return True
    fa, fb = _LABEL_TO_FAMILY.get(a), _LABEL_TO_FAMILY.get(b)
    if fa is None or fb is None:
        return True
    return fa == fb


_CROSS_LABEL_PAIR_BLOCK = (
    "Entity A:\n"
    "  Name: {name_a}\n"
    "  Type: {label_a}\n"
    "  Description: {desc_a}\n"
    "  Relationships: {neighbors_a}\n\n"
    "Entity B:\n"
    "  Name: {name_b}\n"
    "  Type: {label_b}\n"
    "  Description: {desc_b}\n"
    "  Relationships: {neighbors_b}\n\n"
    "Embedding cosine similarity: {similarity:.3f}"
)

# Rules 1-4 are a measured change, not a style edit. Without them the model
# scored precision 0.733 on the benchmark corpus and merged 4 pairs that are
# not the same entity, breaking 2 of 33 deliberate hard negatives ("Cape
# Morrow" with "Cape Morrow Lighthouse", "Elias Whitford, Jr." with "Elias
# Whitford"). With them: precision 1.000, zero wrong merges, zero traps
# broken, at the cost of 2 correct merges.
#
# Rule 1 is split into 1a/1b deliberately. An earlier version said only
# "a place is not the structure on it"; on four unrelated corpora that
# generalised into "longer name = different thing" and wrongly refused
# "Toyota Land Cruiser" = "Land Cruiser" and "Data Protection Directive
# 95/46/EC" = "Data Protection Directive", while correctly refusing
# "Haloferax volcanii" = "Haloferax". The distinction that matters is
# whether the extra words narrow the reference or merely label it.
#
# Defined once and shared by both the single-pair and the batched prompt: when
# the two were maintained separately, one sentence of rule 3 was lost in
# transcription and precision silently fell from 1.000 to 0.750.
#
# The diminutive sentence in rule 3 was added after measurement, not intuition.
# Rule 3 covered acronyms and shortened COMPANY names but said nothing about
# personal short forms, so the model rejected "Bob"/"Robert Mitchell" with the
# reason "different names" even though both descriptions said chief financial
# officer of the same group. Candidate generation was never the constraint —
# every one of those pairs already reached the LLM. Adding this one sentence
# moved recall from 1/6 to 4/6 on the alias corpus with hard negatives broken
# unchanged at 0/9, identical across 3 repeats. The remaining misses are bare
# truncations ("Na", "JR"), which are not diminutives and need document
# co-occurrence evidence this strategy does not yet have.
_RULES = (
    "Rules:\n"
    "1. When one name contains the other, decide WHY the extra words are there.\n"
    "   a. If they NARROW the reference to a smaller or more specific thing, the "
    "two are DIFFERENT: a structure standing on a place is not the place; a "
    "species is not its genus; a specific variant or model is not the base "
    "product; a part is not the whole system.\n"
    "   b. If they are only a fuller LABEL for the same thing, they are the "
    "SAME: a manufacturer or brand prefix, an official number or code, a "
    "marketing suffix, a legal form, or a spelled-out form of an abbreviation.\n"
    "2. Names differing by an explicit generational suffix (Jr., Sr., II, III, "
    "the Younger, the Elder) are DIFFERENT people. This rule applies ONLY when "
    "such a suffix is literally present in one of the names. Two relatives "
    "sharing a surname are not automatically different entities.\n"
    "3. The SAME entity may still be written differently: a translated name, "
    "initials instead of a full given name, an acronym, or a shortened company "
    "name. A dropped or added MIDDLE name is not a generational difference and "
    "does not by itself make two names refer to different people. A common short "
    "form or diminutive of a given name (Bob for Robert, Liz for Elizabeth, Kat "
    "for Katherine) is the same name, not a different person. These are the "
    "same entity.\n"
    "4. High similarity is evidence, not proof. Two entities of the same type "
    "from one document often read alike.\n"
    "5. A name is not the only way an entity is referred to. One side may be a "
    "DESCRIPTIVE or ROLE reference rather than a proper name \u2014 a job title, a "
    "role in an organisation, a translated or shortened form, or a definite "
    'phrase such as "the physicist" or "the ship\'s master". When the two '
    "names share little or no text, do NOT answer NO on that basis alone: read "
    "both descriptions and decide whether they describe one and the same "
    "real-world entity. If the descriptions agree on the specific identifying "
    "facts (same role, same place, same work, same events), they are the SAME "
    "entity even though the names look nothing alike. If the descriptions "
    "conflict or describe different individuals, they are DIFFERENT.\n"
)

_VERIFY_PROMPT_INTRO = (
    "You are an entity resolution assistant. Decide whether the two entities below "
    "refer to the exact same real-world entity.\n\n"
)

_VERIFY_PROMPT_ANSWER = (
    "\nAnswer with exactly one of:\n"
    "  YES \u2014 they are the same entity\n"
    "  NO  \u2014 they are different entities\n\n"
    "Then on a new line give a brief reason (one sentence, max 20 words).\n\n"
    "Answer:"
)


# "--- Pair 12 ---" plus the blank line that follows each block.
_PAIR_WRAPPER_TOKENS = 12

# ── PASS 2 cross-label candidate generation ───────────────────────────────────
#
# PASS 1 buckets nodes by label and only ever compares within a bucket, so an
# entity extracted once as Person and once as Engineer can never be considered.
# Measured on a 40-node corpus holding 10 such pairs: PASS 1 returned 40
# survivors of 40 and zero merges. The pairs are not rejected, they are never
# formed.
#
# Comparing every cross-label pair is O(n^2) and unaffordable, so candidates are
# admitted through three cheap "doors" OR-ed together, then filtered by a
# similarity floor. On that corpus the doors caught 6, 2 and 2 of the 10 pairs
# respectively — no door is redundant, and dropping any one costs recall:
#
#   token only              P=0.857 R=0.600 F1=0.706
#   token + acronym         P=0.889 R=0.800 F1=0.842
#   all three (shipped)     P=0.909 R=1.000 F1=0.952
#
# The vector door is what catches "Munich"/"Munchen" (0.609) — same city, no
# shared substring, not an acronym. Requiring a shared token AND a vector score
# instead of OR-ing them was tested and rejected: it drops recall to 0.800 and
# does not even save calls once the floor is applied.
_STOPWORDS = frozenset({"the", "of", "and", "a", "an", "inc", "ltd", "corp", "co"})
_TOKEN_SPLIT = re.compile(r"[^a-z0-9]+")
_ALPHA_ONLY = re.compile(r"[^a-z]")
_ALPHA_SPLIT = re.compile(r"[^a-z]+")


def _name_tokens(name: str) -> set[str]:
    """Significant lowercase word tokens of a name."""
    return {t for t in _TOKEN_SPLIT.split(name.lower()) if t and t not in _STOPWORDS}


def _pair_distance(similarity: float) -> float:
    """Cosine similarity as a clustering distance, clamped to [0, 1].

    The similarities reaching this come from hnswlib's float32 inner product,
    which for two identical unit vectors can return a distance of -1.19e-07
    instead of 0 — so the similarity is 1.0000001 and the raw ``1 - similarity``
    is negative. ``scipy.cluster.hierarchy.fcluster`` refuses a linkage matrix
    built from negative distances and raises, aborting the whole resolution
    stage. Clamping costs nothing and cannot change any ordering, because only
    values already outside [0, 1] move.
    """
    return min(1.0, max(0.0, 1.0 - float(similarity)))


def _is_acronym(a: str, b: str) -> bool:
    """Do one name's initials spell the other? Checked in both directions."""
    for short, long in ((a, b), (b, a)):
        s = _ALPHA_ONLY.sub("", short.lower())
        words = [w for w in _ALPHA_SPLIT.split(long.lower()) if w and w not in _STOPWORDS]
        if len(s) >= 2 and len(words) >= 2 and s == "".join(w[0] for w in words):
            return True
    return False


@lru_cache(maxsize=1)
def _encoding():
    import tiktoken

    return tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    return len(_encoding().encode(text))


# Asking about several pairs in one call is ~14x cheaper, but the single-pair
# wording above does not survive it: reused verbatim for a list it dropped
# precision to 0.769 and merged a headland with the lighthouse standing on it.
# Requiring each answer to restate the two names and name the rule it applied
# restores it. Measured head-to-head in the same time window, interleaved to
# control for provider drift: batched 4/5 runs with zero wrong merges, one call
# per pair 1/5.
_BATCH_HEADER = (
    "You are an entity resolution assistant. Below are numbered pairs of "
    "entities. Judge EACH pair completely independently, as if it were the "
    "only pair you were given. A decision about one pair tells you nothing "
    "about any other pair.\n\n"
    + _RULES
    + "\nFor every pair, output exactly one line in this format:\n"
    "  <number>. <name A> vs <name B> | rule: <which rule decided it> | "
    "<YES or NO>\n\n"
    "Work through the pairs in order. Do not skip, merge or reorder them. "
    "Output nothing else.\n\n"
    "Pairs:\n"
)


def _parse_batch_verdicts(text: str, count: int) -> dict[int, bool]:
    """Read one verdict per line out of a batched reply.

    Takes the leading number and the LAST YES/NO on the line, because the
    "rule:" segment can itself contain one. A line that cannot be read is
    simply absent from the result, and an absent verdict never merges.
    """
    verdicts: dict[int, bool] = {}
    for line in (text or "").splitlines():
        head = re.match(r"\s*(\d+)\s*[:.\)]", line)
        if not head:
            continue
        index = int(head.group(1))
        if not 1 <= index <= count:
            continue
        found = re.findall(r"\b(YES|NO)\b", line[head.end() :], re.IGNORECASE)
        if found:
            verdicts[index] = found[-1].upper() == "YES"

    if not verdicts and count == 1:
        # A lone pair does not need numbering to be unambiguous, and a model
        # answering a one-item list with a bare "YES" is answering correctly.
        bare = re.match(r"\s*(YES|NO)\b", text or "", re.IGNORECASE)
        if bare:
            verdicts[1] = bare.group(1).upper() == "YES"
    return verdicts


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
         the same-label description merge. No LLM or embedder needed here.
      2. Embed all surviving node names within each label group.
      3. For each pair (within same label only):
           similarity >= hard_threshold  → hard merge immediately
           soft_threshold <= sim < hard  → send to LLM YES/NO batch
           sim < soft_threshold          → skip
      4. LLM confirms or rejects each ambiguous pair.
      5. Apply Union-Find clusters from hard + LLM-confirmed merges.
      6. PASS 2: cross-label pairs only — three doors (shared token, acronym,
         vector) OR-ed, filtered by a rank floor, one LLM call per pair,
         Union-Find. Steps 2-5 group by label and can never form these pairs.
      7. Flatten the remap chain, then remap and deduplicate relationships.

    Args:
        llm: LLM provider for YES/NO verification and description summaries.
        embedder: Embedder for pairwise cosine similarity.
        hard_threshold: Similarity at or above which entities are merged
            without LLM confirmation (default: 0.95).
        soft_threshold: Similarity below which pairs are skipped entirely
            (default: 0.65).

            Was 0.80. That left a coverage hole: Stage 2 asked the LLM only
            about same-label pairs at or above 0.80, while the cross-label
            pass starts at 0.55 but never looks at same-label pairs. Same-label
            pairs between the two were put to no stage at all and silently
            never merged.

            Measured on the v2 benchmark gold (130 real clusters, 239 true
            duplicate pairs, 39 must-not-merge pairs), 3 reps against real
            ``gpt-4o-mini`` + ``text-embedding-3-small``:

            ==========  ========  =============  ===================
            floor       recall    wrong merges   Stage 2 LLM calls
            ==========  ========  =============  ===================
            0.80        0.100     0              111  (1.0x)
            0.70        0.170     0              253  (2.3x)
            **0.65**    **0.213** **0**          **344  (3.1x)**
            0.60        0.251     0              479  (4.3x)
            0.55        0.285     0              728  (6.6x)
            ==========  ========  =============  ===================

            0.65 is the knee. The marginal cost of a recovered duplicate is
            flat at ~8-9 LLM calls per duplicate all the way down to 0.65,
            then jumps to 15 at 0.60 and 31 at 0.55. Recall roughly doubles
            for 3.1x the Stage 2 calls; below 0.65 the price per duplicate
            rises sharply, which matters because LLM work dominates ingest
            wall time.

            No must-not-merge pair broke at any floor in the sweep, including
            hard ones such as ``Musa al-Farabi`` vs ``Abu Nasr al-Farabi``, so
            the limit here is cost rather than precision.

            See RESULTS.md P3.25.
        max_llm_pairs: Maximum ambiguous pairs sent to the LLM per label group
            (default: 500). Pairs are ranked by descending similarity and the
            remainder are discarded unverified, so this is a recall ceiling
            rather than a batching detail — a duplicate below the cut is never
            checked and stays unmerged.

            Candidate pairs grow as roughly the square of the node count
            (measured exponent 1.99 on the benchmark corpus at the default
            threshold: 755 nodes produce 513 candidates, so the cap is nearly
            inert there, while a projected 5,000 nodes produce ~22,000 and only
            2% are verified). The cap is therefore quiet on small graphs and
            binding on large ones; it emits a warning whenever it truncates.

            It is deliberately not raised by default because verification is
            LLM work, which dominates ingest wall time — a larger cap buys
            recall at a cost that grows with the same square. Raise it when
            recall matters more than ingest time, or raise
            ``unified_threshold`` to generate fewer candidates instead.

            See RESULTS.md P3.52.
        max_llm_concurrency: Override the LLM provider's concurrency limit
            for the verification batch (default: None → provider default).
        force_summary_threshold: Number of descriptions that triggers LLM
            summarisation in Phase 1 (default: 3).
        cross_label_min_descriptions: Evidence floor for the Phase 1
            cross-label pass — a same-name group spanning two labels is only
            put to the LLM once it has this many descriptions between them
            (default: 3). Previously this reused ``force_summary_threshold``,
            so changing the summary policy silently changed which entities
            merged; the two are now independent. Anything below the floor is
            left to the cross-label PASS 2, which screens on cosine
            similarity before spending an LLM call.
        max_summary_tokens: Token budget hint for description summaries.
        ann_top_k: Number of nearest neighbours to retrieve per node from the
            hnswlib HNSW index (default: 50). Higher values improve recall at
            the cost of speed.
        cross_label_merge: Run the legacy PASS 2, which compares entities
            carrying different labels (default: True). Only consulted when
            ``unified_stage=False``; the unified stage forms cross-label
            candidates itself and is not switched off by this flag.
        cross_label_vector_door: Cosine at or above which a pair is admitted to
            PASS 2 even with no shared token and no acronym match
            (default: 0.60, the measured knee).
        cross_label_rank_floor: Cosine below which a PASS 2 candidate is dropped
            regardless of which door admitted it (default: 0.55). This is what
            rejects same-name homographs such as Paris the city and Paris the
            prince (0.387) without spending an LLM call.
        cross_label_max_pairs: Cap on PASS 2 pairs sent to the LLM, ranked by
            descending similarity (default: 200).
        batch_verification: Ask about several pairs per LLM call (default: True).
        verification_token_budget: Token ceiling for one batched call
            (default: 4000). The per-call pair cap below is usually the binding
            limit; this only matters for unusually long descriptions.
        verification_max_pairs_per_call: Pairs packed into one batched call
            (default: 5). What degrades under batching is the number of
            questions in one reply, not the token count. Measured on 174 real
            pairs, 3 runs each: 30 per call gives 51.3 correct merges in 11.3s,
            5 per call gives 58.7 in 7.9s, and one per call gives 61.0 in 20.6s.
            Pass ``batch_verification=False`` for the one-pair-per-call path,
            which buys ~2 more merges for 5x the calls and 2.6x the wall time.
        label_family_gate: Drop cross-label candidate pairs whose labels fall
            in different known families (see :data:`LABEL_FAMILIES`) before
            they reach the LLM (default: True). Measured: 54 of 55
            cross-type merges on the benchmark were wrong; the gate removes
            them at zero cost. Labels outside the table are unaffected.
        cross_label_vote: Ask the LLM twice about each cross-label pair that
            survives the gate, with A and B swapped, and merge only when both
            answers are YES (default: True). Same-label pairs are asked once.
            Cross-label YES answers flipped between identical runs (8 / 26 /
            31 wrong merges); requiring agreement removes the coin-flips.
        unified_stage: Compare all entities in one pass, embedding
            ``"name: description"`` rather than the name alone, instead of
            bucketing by label first (default: True). Measured on 216 gold
            pairs, the description text admits 125 of them past a 0.55 cosine
            versus 87 for names alone, while admitting FEWER false pairs
            (24 of 39 must-not-merge versus 33). Descriptions pull aliases of
            one entity together and push same-type entities apart, so this is
            better on both axes, not a recall-for-precision trade.
        unified_threshold: Cosine at or above which a pair becomes a candidate
            for LLM verification in the unified stage (default: 0.65). This is
            the binding constraint on how much this strategy can merge at all:
            the LLM only ever sees pairs above it. Measured end to end on the
            v2 benchmark, 3 runs each, 216 trustworthy gold pairs and 39
            must-not-merge pairs:

            ==========  ==============  ========  =======  =====
            threshold   merges (mean)   recall    wrong    calls
            ==========  ==============  ========  =======  =====
            0.75        84.7            39.2%     2-3      42
            0.65        107.0           49.5%     2        100
            ==========  ==============  ========  =======  =====

            0.65 recovers 22 more true merges — 26% more recall — with no
            precision cost; ``wrong`` was exactly 2 in every 0.65 run while
            0.75 drifted to 3, and the gap far exceeds the run-to-run spread
            of 4-5. It costs ~2.4x the LLM calls. Lowering further does not
            help: below 0.65 the extra candidates are descriptive coreference
            pairs ("Adelaide" / "the tug") whose cosine is near zero, so no
            reachable gate recovers them. Raise it back toward 0.75 to halve
            the LLM cost of this stage at a fifth of the recall.
    """

    def __init__(
        self,
        llm: LLMInterface | None = None,
        embedder: Embedder | None = None,
        *,
        hard_threshold: float = 0.95,
        soft_threshold: float = 0.65,
        max_llm_pairs: int = 500,
        max_llm_concurrency: int | None = None,
        force_summary_threshold: int = 3,
        max_summary_tokens: int = 500,
        ann_top_k: int = 50,
        batch_verification: bool = True,
        verification_token_budget: int = 4000,
        verification_max_pairs_per_call: int = 5,
        cross_label_merge: bool = True,
        cross_label_vector_door: float = 0.60,
        cross_label_rank_floor: float = 0.55,
        cross_label_max_pairs: int = 200,
        cross_label_min_descriptions: int = 3,
        unified_stage: bool = True,
        unified_threshold: float = 0.65,
        label_family_gate: bool = True,
        cross_label_vote: bool = True,
    ) -> None:
        if hard_threshold <= soft_threshold:
            raise ValueError(
                f"hard_threshold ({hard_threshold}) must be > soft_threshold ({soft_threshold})"
            )
        if not 0.0 <= unified_threshold <= 1.0:
            raise ValueError(f"unified_threshold ({unified_threshold}) must be in [0.0, 1.0]")
        self.llm = llm
        self.embedder = embedder
        self.hard_threshold = hard_threshold
        self.soft_threshold = soft_threshold
        self.max_llm_pairs = max_llm_pairs
        self.max_llm_concurrency = max_llm_concurrency
        self.force_summary_threshold = force_summary_threshold
        self.max_summary_tokens = max_summary_tokens
        self.ann_top_k = ann_top_k
        self.batch_verification = batch_verification
        self.verification_token_budget = verification_token_budget
        self.verification_max_pairs_per_call = max(1, verification_max_pairs_per_call)
        self.cross_label_merge = cross_label_merge
        self.cross_label_vector_door = cross_label_vector_door
        self.cross_label_rank_floor = cross_label_rank_floor
        self.cross_label_max_pairs = cross_label_max_pairs
        self.cross_label_min_descriptions = cross_label_min_descriptions
        self.unified_stage = unified_stage
        self.label_family_gate = label_family_gate
        self.cross_label_vote = cross_label_vote
        self.unified_threshold = unified_threshold

    def _embed_text(self, node: GraphNode) -> str:
        """Text embedded for candidate generation.

        The unified stage embeds ``"name: description"``; the legacy label-bucket
        path embeds the name alone. See RESULTS.md P3.27 for the measurement
        that motivated the change.
        """
        name = str(node.properties.get("name", node.id))
        if not self.unified_stage:
            return name
        desc = str(node.properties.get("description", "") or "").strip()
        return f"{name}: {desc}" if desc else name

    def _pack(self, blocks: list[str]) -> list[tuple[int, int]]:
        """Split pairs into calls that each stay under the token budget.

        A single call per label group is unbounded: borderline pairs grow at
        roughly 0.19 per entity, so a 50K-entity graph would put ~9.6K pairs in
        one prompt. Packing greedily to a budget keeps every call a fixed
        maximum size regardless of graph size, and a pair that exceeds the
        budget on its own still gets its own call rather than being dropped.

        The budget is set well below the model's context window on purpose. At
        the previous 20000 it packed up to 155 pairs into a single call, and
        measured over 8 trials on 174 real pairs that result was **bimodal**:
        25, 26, 27, 28 correct merges on four runs and 51, 53, 54, 58 on the
        others -- a coin flip between working and collapsing, with nothing in
        the input changing. The reply always came back complete and correctly
        numbered, so no error surfaces; ingestion just silently loses half its
        merges. At 4000 (up to 29 pairs per call) four trials gave 51, 51, 52,
        54 -- the good mode every time, zero wrong merges. That costs 7 calls
        instead of 2, still ~25x cheaper than one call per pair.

        A token budget alone does not bound the thing that actually breaks. The
        measurement above rode on ~200-token name+description blocks; a corpus
        with no descriptions produces ~30-token blocks, and 4000 tokens of those
        is **119 pairs in one call** -- straight back into the range that went
        bimodal. What degrades is the number of questions in one reply, not the
        token count, so the count is capped directly as well and the tighter of
        the two wins.
        """
        header_cost = _count_tokens(_BATCH_HEADER)
        windows: list[tuple[int, int]] = []
        start, used = 0, header_cost
        for i, block in enumerate(blocks):
            cost = _count_tokens(block) + _PAIR_WRAPPER_TOKENS
            over_budget = used + cost > self.verification_token_budget
            over_count = i - start >= self.verification_max_pairs_per_call
            if i > start and (over_budget or over_count):
                windows.append((start, i))
                start, used = i, header_cost
            used += cost
        windows.append((start, len(blocks)))
        return windows

    async def _ask_windows(
        self, blocks: list[str], windows: list[tuple[int, int]]
    ) -> dict[int, bool]:
        """Run one batched round over the given windows of ``blocks``."""
        prompts = [
            _BATCH_HEADER
            + "".join(f"--- Pair {n + 1} ---\n{blocks[start + n]}\n\n" for n in range(stop - start))
            for start, stop in windows
        ]
        results = await self.llm.abatch_invoke(
            prompts,
            max_concurrency=self.max_llm_concurrency,
        )
        by_index = {item.index: item for item in results}

        confirmed: dict[int, bool] = {}
        for w, (start, stop) in enumerate(windows):
            item = by_index.get(w)
            if item is None or not item.ok:
                logger.warning(
                    "Batched verification failed for pairs %d-%d: %s",
                    start,
                    stop - 1,
                    getattr(item, "error", "missing result"),
                )
                continue
            verdicts = _parse_batch_verdicts(item.response.content, stop - start)
            for n, same in verdicts.items():
                confirmed[start + n - 1] = same
        return confirmed

    async def _verify_batched(self, blocks: list[str]) -> dict[int, bool]:
        """Ask about many pairs per call. Returns {block index: is_same}.

        A batched reply can come back short: the model answers four of five
        numbered questions, or a call fails outright. Those pairs carry no
        verdict, and treating "no verdict" as "not a duplicate" silently loses
        real merges with nothing but a log line to show for it. Every
        unanswered pair is therefore re-asked one pair per call, where a short
        reply is not possible, before any of them is allowed to stay unmerged.
        """
        confirmed = await self._ask_windows(blocks, self._pack(blocks))

        retry = [i for i in range(len(blocks)) if i not in confirmed]
        if retry:
            logger.info(
                "Batched verification left %d of %d pairs unanswered; re-asking individually",
                len(retry),
                len(blocks),
            )
            # Re-asked singly rather than repacked: a one-question prompt cannot
            # come back short for the same reason it did the first time.
            singles = [blocks[i] for i in retry]
            answers = await self._ask_windows(singles, [(n, n + 1) for n in range(len(singles))])
            for n, same in answers.items():
                confirmed[retry[n]] = same
            still = len(retry) - len(answers)
            if still:
                logger.warning(
                    "Verification could not answer %d pairs after retry; left unmerged", still
                )
        return confirmed

    async def _verify_one_by_one(
        self,
        blocks: list[str],
        requests: list[_VerificationRequest],
    ) -> dict[int, bool]:
        """Ask about a single pair per call — the pre-batching behaviour."""
        prompts = [
            _VERIFY_PROMPT_INTRO + block + "\n\n" + _RULES + _VERIFY_PROMPT_ANSWER
            for block in blocks
        ]
        results = await self.llm.abatch_invoke(
            prompts,
            max_concurrency=self.max_llm_concurrency,
        )
        by_index = {item.index: item for item in results}

        confirmed: dict[int, bool] = {}
        for req in requests:
            item = by_index.get(req.prompt_index)
            if item is None or not item.ok:
                logger.warning(
                    "LLM verification failed for '%s' vs '%s': %s",
                    req.node_a.properties.get("name", req.node_a.id),
                    req.node_b.properties.get("name", req.node_b.id),
                    getattr(item, "error", "missing result"),
                )
                continue
            confirmed[req.prompt_index] = item.response.content.strip().upper().startswith("YES")
        return confirmed

    async def _cross_label_merge(
        self,
        nodes: list[GraphNode],
        ctx: Context,
        relationships: list[GraphRelationship],
        prior_remap: dict[str, str],
    ) -> tuple[dict[str, str], int]:
        """PASS 2: merge entities that survived PASS 1 under different labels.

        PASS 1 buckets by label, so an entity extracted once as Person and once
        as Engineer is never even compared. This pass forms only cross-label
        pairs, admits them through three doors (shared token / acronym / vector),
        drops anything below the rank floor, asks the LLM one pair at a time,
        and joins the confirmed pairs with union-find.

        Every pair is asked independently and against the ORIGINAL entities;
        nothing is merged until all answers are back. That ordering is what lets
        the union step repair a wrong NO: in a measured 4-variant case the model
        rejected "Na"/"Naseem" directly, but both were confirmed against
        "Naseem Ali", so the chain grouped all four correctly. Merging greedily
        as answers arrive would instead make the outcome depend on pair order.

        Returns:
            (remap, llm_confirmed_merge_count)
        """
        if len(nodes) < 2 or self.llm is None or self.embedder is None:
            return {}, 0

        def _name(n: GraphNode) -> str:
            return str(n.properties.get("name", n.id))

        def _desc(n: GraphNode) -> str:
            d = n.properties.get("description")
            return str(d) if isinstance(d, str) and d else "(no description)"

        # PASS 1 embeds names only. That is not enough here: name-only vectors
        # put "A320"/"A321" (different) above "Naseem"/"Naseem Ali" (same), so no
        # threshold on them can separate the two. Embedding "name: description"
        # is what makes the vector door usable, and it is also what carries the
        # decision — with an identical description the model merged "Na" into
        # "Naseem Ali" 3/3 (cosine 0.911), and with a generic one it refused 3/3
        # (cosine 0.510). These embeddings are cached separately from the PASS 1
        # name embeddings because they are a different text.
        cache: dict[str, list[float]] = ctx.metadata.setdefault("cross_label_embedding_cache", {})
        missing = [n for n in nodes if n.id not in cache]
        if missing:
            try:
                vecs = await self.embedder.aembed_documents(
                    [f"{_name(n)}: {_desc(n)}" for n in missing]
                )
            except Exception as exc:
                ctx.log(
                    f"PASS 2 embedding failed, skipping cross-label merge: {exc}",
                    logging.WARNING,
                )
                return {}, 0
            for node, vec in zip(missing, vecs):
                if vec:
                    cache[node.id] = vec

        valid = [n for n in nodes if cache.get(n.id)]
        if len(valid) < 2:
            return {}, 0

        mat = np.array([cache[n.id] for n in valid], dtype=np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        mat_normed = mat / norms
        sims = mat_normed @ mat_normed.T

        names = [_name(n) for n in valid]
        token_sets = [_name_tokens(nm) for nm in names]

        candidates: list[tuple[int, int, float, str]] = []
        for i in range(len(valid)):
            for j in range(i + 1, len(valid)):
                if valid[i].label == valid[j].label:
                    continue  # PASS 1 already owns same-label pairs
                sim = float(sims[i, j])
                if sim < self.cross_label_rank_floor:
                    continue
                if token_sets[i] & token_sets[j]:
                    door = "token"
                elif _is_acronym(names[i], names[j]):
                    door = "acronym"
                elif sim >= self.cross_label_vector_door:
                    door = "vector"
                else:
                    continue
                candidates.append((i, j, sim, door))

        if not candidates:
            ctx.log("PASS 2 (cross-label): no candidate pairs")
            return {}, 0

        candidates.sort(key=lambda t: t[2], reverse=True)
        capped = candidates[: self.cross_label_max_pairs]
        door_counts: dict[str, int] = defaultdict(int)
        for _, _, _, door in capped:
            door_counts[door] += 1
        ctx.log(
            f"PASS 2 (cross-label): {len(candidates)} candidates "
            f"({dict(door_counts)}), {len(capped)} sent to LLM"
        )

        node_id_to_name = {n.id: _name(n) for n in nodes}
        adjacency: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for rel in relationships:
            src = prior_remap.get(rel.start_node_id, rel.start_node_id)
            tgt = prior_remap.get(rel.end_node_id, rel.end_node_id)
            adjacency[src].append((node_id_to_name.get(tgt, tgt), rel.type))
            adjacency[tgt].append((node_id_to_name.get(src, src), rel.type))

        def _fmt_neighbors(node_id: str, top_n: int = 5) -> str:
            nbrs = adjacency.get(node_id, [])[:top_n]
            return "; ".join(f"{t} -> {nm}" for nm, t in nbrs) if nbrs else "(none)"

        prompts = [
            _VERIFY_PROMPT_INTRO
            + _CROSS_LABEL_PAIR_BLOCK.format(
                name_a=names[i],
                label_a=valid[i].label,
                desc_a=_desc(valid[i]),
                neighbors_a=_fmt_neighbors(valid[i].id),
                name_b=names[j],
                label_b=valid[j].label,
                desc_b=_desc(valid[j]),
                neighbors_b=_fmt_neighbors(valid[j].id),
                similarity=sim,
            )
            + "\n\n"
            + _RULES
            + _VERIFY_PROMPT_ANSWER
            for i, j, sim, _ in capped
        ]

        # Deliberately one pair per call. Batched verification was measured on
        # this shape and loses merges: an unanswered index in a batch reply is
        # indistinguishable from a NO.
        results = await self.llm.abatch_invoke(prompts, max_concurrency=self.max_llm_concurrency)
        by_index = {item.index: item for item in results}

        parent = list(range(len(valid)))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        confirmed_pairs = 0
        for pos, (i, j, sim, door) in enumerate(capped):
            item = by_index.get(pos)
            if item is None or not item.ok:
                logger.warning(
                    "PASS 2 verification failed for '%s' vs '%s': %s",
                    names[i],
                    names[j],
                    getattr(item, "error", "missing result"),
                )
                continue
            if not item.response.content.strip().upper().startswith("YES"):
                continue
            confirmed_pairs += 1
            logger.debug(
                "PASS 2 merge '%s' (%s) + '%s' (%s) sim=%.3f door=%s",
                names[i],
                valid[i].label,
                names[j],
                valid[j].label,
                sim,
                door,
            )
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[rj] = ri

        clusters: dict[int, list[int]] = defaultdict(list)
        for i in range(len(valid)):
            clusters[find(i)].append(i)

        remap: dict[str, str] = {}
        for members in clusters.values():
            if len(members) <= 1:
                continue
            survivor = valid[members[0]]
            # Join the descriptions rather than letting the survivor's win.
            # The plain "copy keys the survivor lacks" rule silently discarded
            # the duplicate's description, which is the opposite of what the
            # same-label path does (base.py joins with " | ") and drops exactly
            # the text the LLM used to decide the two were the same entity.
            # Measured before the fix: Person/"DESC-PERSON" merged with
            # Engineer/"DESC-ENGINEER" survived as "DESC-PERSON" alone.
            # The duplicate's LABEL was also lost here. An earlier comment said
            # recording it as a node property had been "tried and rejected"
            # because graph_store only persists name/type/description/
            # source_chunk_ids. That is not true of the current write path:
            # `_clean_properties` whitelists no keys, it only drops None and
            # coerces non-primitives, and `upsert_nodes` issues
            # `SET n += item.properties`. Verified by resolving a
            # Person+Engineer duplicate and driving the survivor through
            # GraphStore.upsert_nodes with a recording connection — the
            # parameters sent to the database contained
            # `merged_labels='Engineer'`. GraphNode.label is a single str, so
            # widening it would ripple through storage and query building;
            # the property is additive and loses nothing.
            merged_labels: list[str] = []
            for mi in members[1:]:
                dup = valid[mi]
                remap[dup.id] = survivor.id
                if dup.label != survivor.label and dup.label not in merged_labels:
                    merged_labels.append(dup.label)
                # A dup that already absorbed labels in an earlier merge carries
                # them in its own merged_labels; those must travel too.
                for lbl in str(dup.properties.get("merged_labels", "") or "").split(" | "):
                    if lbl and lbl != survivor.label and lbl not in merged_labels:
                        merged_labels.append(lbl)
                for key, value in dup.properties.items():
                    if key not in survivor.properties:
                        survivor.properties[key] = value
            set_merged_descriptions(survivor, [valid[mi] for mi in members[1:]])
            if merged_labels:
                existing = str(survivor.properties.get("merged_labels", "") or "")
                kept = [p for p in existing.split(" | ") if p]
                for lbl in merged_labels:
                    if lbl not in kept:
                        kept.append(lbl)
                survivor.properties["merged_labels"] = " | ".join(kept)

        ctx.log(f"PASS 2 (cross-label): {confirmed_pairs} pairs confirmed → {len(remap)} merged")
        return remap, len(remap)

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
            cross_label_merge=True,
            cross_label_min_descriptions=self.cross_label_min_descriptions,
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

        # ── PASS 2: cross-label merge ─────────────────────────────────────────
        # Only needed when label bucketing is on. The unified stage already
        # compares every pair regardless of label, so running this afterwards
        # would re-ask questions the unified stage has already answered.
        if (
            not self.unified_stage
            and self.cross_label_merge
            and self.llm
            and self.embedder
            and len(deduplicated_nodes) >= 2
        ):
            cross_remap, cross_merges = await self._cross_label_merge(
                deduplicated_nodes, ctx, graph_data.relationships, id_remap
            )
            if cross_remap:
                deduplicated_nodes = [n for n in deduplicated_nodes if n.id not in cross_remap]
                merged_count += cross_merges
                id_remap.update(cross_remap)

        # ── Phase 6: Remap relationships and deduplicate ──────────────────────
        # Flatten first. Each pass merges the previous pass's survivors, so the
        # accumulated mapping holds chains (dup -> A from phase 1, A -> B from a
        # later pass). remap_relationships does one lookup, so without this an
        # edge into `dup` would be rewritten to `A`, which is no longer a node.
        id_remap = flatten_remap(id_remap)
        deduplicated_rels = remap_relationships(graph_data.relationships, id_remap)

        ctx.log(
            f"LLMVerifiedResolution complete: {len(deduplicated_nodes)} nodes "
            f"({merged_count} merged), {len(deduplicated_rels)} rels"
        )
        return ResolutionResult(
            nodes=deduplicated_nodes,
            relationships=deduplicated_rels,
            merged_count=merged_count,
            remap=id_remap,
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

        # Candidate grouping. The unified stage puts every surviving node in one
        # group; the legacy path buckets by label and never compares across
        # buckets. Measured on the v2 benchmark gold (239 real duplicate pairs,
        # 39 must-not-merge, real LLM-written descriptions for all 755 surface
        # forms, both shipped caps applied):
        #
        #   arm                          LLM calls  embeddings  reached  exposed
        #   label buckets + cross-label      548       1,510     67/239   23/39
        #   unified @ 0.75                   181         755     77/239    7/39
        #
        # Better on every axis at once. The label bucket was the defect: most
        # real duplicates are two surface forms of ONE entity, so they share a
        # label — the cross-label pass only looks at pairs whose labels DIFFER
        # and never sees them, while the same-label pass compares names only and
        # cannot see "the Ashford light" = "Cape Morrow Lighthouse". Between
        # them the two passes are capped at 28% reach at any threshold. See
        # RESULTS.md P3.27.
        by_label: dict[str, list[GraphNode]] = defaultdict(list)
        if self.unified_stage:
            by_label[_UNIFIED_GROUP] = list(nodes)
        else:
            for n in nodes:
                by_label[n.label].append(n)

        remap: dict[str, str] = {}
        total_hard = 0
        total_llm = 0

        emb_cache: dict[str, list[float]] = ctx.metadata.setdefault(
            "unified_embedding_cache" if self.unified_stage else "embedding_cache", {}
        )
        soft_floor = self.unified_threshold if self.unified_stage else self.soft_threshold

        for label, label_nodes in by_label.items():
            if len(label_nodes) < 2:
                continue

            miss_nodes = [n for n in label_nodes if n.id not in emb_cache]
            # Name alone cannot separate "the Ashford light" from "Cape Morrow
            # Lighthouse". Embedding "name: description" is the only signal
            # measured with a POSITIVE duplicate/negative separation gap
            # (+0.010, versus -0.197 for names alone) — on names, must-not-merge
            # pairs actually score HIGHER than true duplicates (0.687 vs 0.489).
            # Cosine is a candidate generator here, not a discriminator; the LLM
            # carries the precision burden. See RESULTS.md P3.27.
            miss_names = [self._embed_text(n) for n in miss_nodes]
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

            # In unified mode a pair can span two labels. The label-bucket
            # design never merged such a pair without an explicit LLM YES —
            # PASS 2 asked about every candidate it admitted. Keep that
            # guarantee: cosine alone may not fuse two different types, however
            # high it scores. This costs nothing measurable, because the 0.95
            # shortcut is already dormant for realistic near-duplicates (0/7
            # true duplicates reach it in either embedding space, DEAD_ENDS C10)
            # — Stage 1's lowercase+trim takes the near-identical strings first.
            def _needs_llm(gi: int, gj: int) -> bool:
                return self.unified_stage and valid_nodes[gi].label != valid_nodes[gj].label

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
            gated_pairs = 0

            # ANN via hnswlib HNSW (O(N log N)) — no OpenMP, no macOS deadlock.
            top_k = min(self.ann_top_k, n_nodes - 1)
            dim = mat_normed.shape[1]
            if _hnswlib is None:
                raise ImportError(
                    "hnswlib is required for LLMVerifiedResolution embedding merge. "
                    "Install with: pip install hnswlib"
                )
            hnsw_index = _hnswlib.Index(space="ip", dim=dim)
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
                    if sim_val < soft_floor:
                        continue
                    if (
                        self.unified_stage
                        and self.label_family_gate
                        and not labels_compatible(valid_nodes[i].label, valid_nodes[j].label)
                    ):
                        gated_pairs += 1
                        continue
                    if sim_val >= self.hard_threshold and not _needs_llm(i, j):
                        hard_pairs.append((i, j))
                    else:
                        ambiguous_pairs.append((i, j, sim_val))

            if gated_pairs:
                ctx.log(f"Label family gate dropped {gated_pairs} cross-family candidate pair(s)")

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
                amb_set = {i for i, j, _ in ambiguous_pairs} | {j for i, j, _ in ambiguous_pairs}
                amb_indices = sorted(amb_set)
                idx_map = {v: k for k, v in enumerate(amb_indices)}
                n_amb = len(amb_indices)

                # Condensed distance matrix for scipy
                dist_matrix = np.ones((n_amb, n_amb), dtype=np.float32)
                np.fill_diagonal(dist_matrix, 0.0)
                for gi, gj, sim_val in ambiguous_pairs:
                    ai, aj = idx_map[gi], idx_map[gj]
                    dist = _pair_distance(sim_val)
                    dist_matrix[ai, aj] = dist
                    dist_matrix[aj, ai] = dist

                condensed = ssd.squareform(dist_matrix)
                linkage = sch.linkage(condensed, method="average")
                # Cut at distance = 1 - hard_threshold: only cluster nodes that are
                # very similar (near the hard-merge boundary). Nodes in the wider
                # soft..hard ambiguous zone but spanning multiple tight groups go to LLM.
                cut = 1.0 - self.hard_threshold
                cluster_labels = sch.fcluster(linkage, t=cut, criterion="distance")
                node_to_comm = {amb_indices[k]: int(cluster_labels[k]) for k in range(n_amb)}

                def _same_cluster(gi: int, gj: int) -> bool:
                    return (
                        gi in node_to_comm
                        and gj in node_to_comm
                        and node_to_comm[gi] == node_to_comm[gj]
                        and not _needs_llm(gi, gj)
                    )

                # Intra-cluster pairs → hard merge (no LLM needed)
                for gi, gj, sim_val in ambiguous_pairs:
                    if _same_cluster(gi, gj):
                        union(gi, gj)
                        logger.debug(
                            "Intra-cluster auto-merge '%s' + '%s' (sim=%.3f, label=%s)",
                            valid_nodes[gi].properties.get("name", valid_nodes[gi].id),
                            valid_nodes[gj].properties.get("name", valid_nodes[gj].id),
                            sim_val,
                            label,
                        )

                # Cross-cluster pairs → LLM (the genuinely ambiguous ones)
                boundary_pairs = [
                    (gi, gj, sim_val)
                    for gi, gj, sim_val in ambiguous_pairs
                    if not _same_cluster(gi, gj)
                ]
                boundary_pairs.sort(key=lambda t: t[2], reverse=True)
                capped = boundary_pairs[: self.max_llm_pairs]
                dropped = len(boundary_pairs) - len(capped)
                ctx.log(
                    f"Label '{label}': {len(ambiguous_pairs)} ambiguous → "
                    f"{len(ambiguous_pairs) - len(boundary_pairs)} intra-community hard merges, "
                    f"{len(capped)} boundary pairs → LLM"
                    + (f" ({dropped} dropped by max_llm_pairs)" if dropped else "")
                )
                if dropped:
                    # Silent truncation here is a recall ceiling, not a slowdown:
                    # the dropped pairs are never verified, so any duplicate
                    # among them survives into the graph with no trace that a
                    # decision was skipped. Candidate count grows as roughly
                    # n^2 (measured 1.99 on the benchmark corpus), so a cap that
                    # is inert on a small graph binds hard on a large one —
                    # which is exactly when the misses matter most.
                    ctx.log(
                        f"max_llm_pairs={self.max_llm_pairs} truncated verification for "
                        f"label '{label}': {len(boundary_pairs)} candidate pairs were "
                        f"generated but only the {len(capped)} highest-similarity pairs "
                        f"were checked, so {dropped} pair(s) were left unverified and any "
                        f"duplicates among them remain unmerged. Raise max_llm_pairs to "
                        f"verify more (cost grows with it), or raise unified_threshold to "
                        f"generate fewer candidates.",
                        logging.WARNING,
                    )

                requests: list[_VerificationRequest] = []
                blocks: list[str] = []

                for gi, gj, sim_val in capped:
                    node_a = valid_nodes[gi]
                    node_b = valid_nodes[gj]
                    req = _VerificationRequest(
                        node_a=node_a,
                        node_b=node_b,
                        idx_a=gi,
                        idx_b=gj,
                        similarity=sim_val,
                        label=node_a.label if self.unified_stage else label,
                        prompt_index=len(blocks),
                    )
                    # In unified mode a candidate pair can span two labels, so
                    # the type cannot be hoisted into a shared header. The
                    # differing types are evidence the model should weigh
                    # ("Person" vs "Engineer" is compatible, "City" vs
                    # "Airport" is not), not noise to hide.
                    if self.unified_stage and node_a.label != node_b.label:
                        blocks.append(
                            _CROSS_LABEL_PAIR_BLOCK.format(
                                name_a=str(node_a.properties.get("name", node_a.id)),
                                label_a=node_a.label,
                                desc_a=str(
                                    node_a.properties.get("description", "(no description)")
                                ),
                                neighbors_a=_fmt_neighbors(node_a.id),
                                name_b=str(node_b.properties.get("name", node_b.id)),
                                label_b=node_b.label,
                                desc_b=str(
                                    node_b.properties.get("description", "(no description)")
                                ),
                                neighbors_b=_fmt_neighbors(node_b.id),
                                similarity=sim_val,
                            )
                        )
                    else:
                        blocks.append(
                            _PAIR_BLOCK.format(
                                label=node_a.label if self.unified_stage else label,
                                name_a=str(node_a.properties.get("name", node_a.id)),
                                desc_a=str(
                                    node_a.properties.get("description", "(no description)")
                                ),
                                neighbors_a=_fmt_neighbors(node_a.id),
                                name_b=str(node_b.properties.get("name", node_b.id)),
                                desc_b=str(
                                    node_b.properties.get("description", "(no description)")
                                ),
                                neighbors_b=_fmt_neighbors(node_b.id),
                                similarity=sim_val,
                            )
                        )
                    requests.append(req)

                if not blocks:
                    continue

                if self.batch_verification:
                    confirmed = await self._verify_batched(blocks)
                else:
                    confirmed = await self._verify_one_by_one(blocks, requests)

                # Second opinion on cross-label YESes. A YES on "Eleanor
                # Whitford (Person) = Whitford Archive (Organization)" flipped
                # between identical runs; asking again with A and B swapped and
                # requiring agreement keeps only the answers the model gives
                # consistently. Same-label pairs are not re-asked.
                vetoed = 0
                if self.unified_stage and self.cross_label_vote:
                    recheck = [
                        req
                        for req in requests
                        if confirmed.get(req.prompt_index) and req.node_a.label != req.node_b.label
                    ]
                    if recheck:
                        swapped_blocks = [
                            _CROSS_LABEL_PAIR_BLOCK.format(
                                name_a=str(req.node_b.properties.get("name", req.node_b.id)),
                                label_a=req.node_b.label,
                                desc_a=str(
                                    req.node_b.properties.get("description", "(no description)")
                                ),
                                neighbors_a=_fmt_neighbors(req.node_b.id),
                                name_b=str(req.node_a.properties.get("name", req.node_a.id)),
                                label_b=req.node_a.label,
                                desc_b=str(
                                    req.node_a.properties.get("description", "(no description)")
                                ),
                                neighbors_b=_fmt_neighbors(req.node_a.id),
                                similarity=req.similarity,
                            )
                            for req in recheck
                        ]
                        swapped_reqs = [
                            _VerificationRequest(
                                node_a=req.node_b,
                                node_b=req.node_a,
                                idx_a=req.idx_b,
                                idx_b=req.idx_a,
                                similarity=req.similarity,
                                label=req.label,
                                prompt_index=k,
                            )
                            for k, req in enumerate(recheck)
                        ]
                        if self.batch_verification:
                            second = await self._verify_batched(swapped_blocks)
                        else:
                            second = await self._verify_one_by_one(swapped_blocks, swapped_reqs)
                        for k, req in enumerate(recheck):
                            if not second.get(k):
                                confirmed[req.prompt_index] = False
                                vetoed += 1
                                logger.debug(
                                    "Cross-label vote vetoed '%s' (%s) + '%s' (%s)",
                                    req.node_a.properties.get("name", req.node_a.id),
                                    req.node_a.label,
                                    req.node_b.properties.get("name", req.node_b.id),
                                    req.node_b.label,
                                )

                llm_confirmed = 0
                for req in requests:
                    if confirmed.get(req.prompt_index):
                        union(req.idx_a, req.idx_b)
                        llm_confirmed += 1

                total_llm += llm_confirmed
                ctx.log(
                    f"Label '{label}': {len(capped)} ambiguous pairs → "
                    f"{llm_confirmed} LLM-confirmed merges"
                    + (f" ({vetoed} cross-label YES vetoed on second vote)" if vetoed else "")
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
                # Join descriptions rather than letting the survivor's win.
                # The plain "copy keys the survivor lacks" rule below never
                # copies `description`, because the survivor always has one, so
                # the duplicate's text was dropped. Measured: two Person nodes
                # at cosine 1.0 merged and "DESC-TWO" vanished, leaving only
                # "DESC-ONE". Stage 1 (base.py) has always joined with " | ";
                # PASS 2 was fixed the same way in P3.21. This is the same
                # defect in the same-label path.
                # `label` is the same defect one level up: the survivor always
                # has one, so a duplicate's differing label was destroyed with
                # no record. Under the unified stage a cross-label merge is an
                # ordinary outcome rather than a capped special case, so record
                # what was absorbed. Measured on a 3-surface cluster
                # (Person/Engineer/Author): both the legacy and unified paths
                # dropped "Engineer" and "Author" with nothing recoverable from
                # the properties. GraphNode.label is a single str, so widening
                # it would ripple through storage and query building; keeping
                # the evidence in a property is additive and loses nothing.
                merged_labels: list[str] = []
                for mi in members[1:]:
                    dup = valid_nodes[mi]
                    remap[dup.id] = survivor.id
                    if dup.label != survivor.label and dup.label not in merged_labels:
                        merged_labels.append(dup.label)
                    for lbl in str(dup.properties.get("merged_labels", "") or "").split(" | "):
                        if lbl and lbl != survivor.label and lbl not in merged_labels:
                            merged_labels.append(lbl)
                    for key, value in dup.properties.items():
                        if key not in survivor.properties:
                            survivor.properties[key] = value
                set_merged_descriptions(survivor, [valid_nodes[mi] for mi in members[1:]])
                if merged_labels:
                    # Same " | " convention as `description` above.
                    existing = str(survivor.properties.get("merged_labels", "") or "")
                    kept = [p for p in existing.split(" | ") if p]
                    for lbl in merged_labels:
                        if lbl not in kept:
                            kept.append(lbl)
                    survivor.properties["merged_labels"] = " | ".join(kept)
                    logger.debug(
                        "Merge absorbed label(s) %s into '%s' (%s)",
                        merged_labels,
                        survivor.properties.get("name", survivor.id),
                        survivor.label,
                    )

            label_merges = len(remap) - label_remap_before
            if label_merges:
                ctx.log(f"Label '{label}': {label_merges} total merges")

        return remap, total_hard, total_llm
