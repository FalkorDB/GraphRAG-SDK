# GraphRAG SDK 2.0 — Retrieval: Result Assembly
# Cosine reranking, question-type detection, and structured result assembly.

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from graphrag_sdk.core.models import RawSearchResult
from graphrag_sdk.core.providers import Embedder

logger = logging.getLogger(__name__)


def cosine_sim(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two float vectors."""
    va, vb = np.array(a), np.array(b)
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


async def rerank_chunks(
    embedder: Embedder,
    query_vector: list[float],
    candidate_chunks: dict[str, str],
    chunk_top_k: int = 15,
) -> list[str]:
    """Batch embed candidates and cosine-sort, take top_k.

    Returns:
        Top-k chunk texts ranked by cosine similarity to query.
    """
    if not candidate_chunks:
        return []

    chunk_texts = list(candidate_chunks.values())
    try:
        chunk_vectors = await embedder.aembed_documents(chunk_texts)
        scored = [
            (i, cosine_sim(query_vector, cvec))
            for i, cvec in enumerate(chunk_vectors)
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [chunk_texts[i] for i, _ in scored[:chunk_top_k]]
    except Exception:
        return chunk_texts[:chunk_top_k]


def detect_question_type(query: str) -> str:
    """Detect question type and return an answer-format hint."""
    q = query.strip().lower()
    if q.startswith(("is ", "are ", "was ", "were ", "did ", "does ",
                     "do ", "has ", "had ", "have ", "can ", "could ",
                     "will ", "would ", "should ")):
        return "Answer format: This is a yes/no question — start with Yes or No, then explain briefly."
    if q.startswith("who "):
        return "Answer format: Name the specific person(s) or character(s)."
    if q.startswith("where "):
        return "Answer format: Name the specific place or location."
    if q.startswith("when "):
        return "Answer format: Provide the specific time, date, or period."
    if q.startswith("how many") or q.startswith("how much"):
        return "Answer format: Provide a specific number or quantity."
    return ""


def assemble_raw_result(
    entity_list: list[tuple[str, dict]],
    relationship_strings: list[str],
    fact_strings: list[str],
    source_passages: list[str],
    q_type_hint: str = "",
) -> RawSearchResult:
    """Build structured RawSearchResult with section records."""
    records: list[dict[str, Any]] = []

    # Question-type hint (prepended so LLM sees it first)
    if q_type_hint:
        records.append({
            "section": "hint",
            "content": q_type_hint,
        })

    # Entity section
    seen_names: set[str] = set()
    entity_lines: list[str] = []
    for _, einfo in entity_list:
        name = einfo.get("name", "")
        if name and name.lower() not in seen_names:
            seen_names.add(name.lower())
            desc = einfo.get("description", "")
            entity_lines.append(f"- {name}: {desc}" if desc else f"- {name}")
    if entity_lines:
        records.append({
            "section": "entities",
            "content": "## Key Entities\n" + "\n".join(entity_lines[:25]),
        })

    # Relationship section
    if relationship_strings:
        records.append({
            "section": "relationships",
            "content": "## Entity Relationships\n"
            + "\n".join(f"- {r}" for r in relationship_strings[:20]),
        })

    # Knowledge Graph Facts section (from RELATES edge vector search)
    if fact_strings:
        records.append({
            "section": "facts",
            "content": "## Knowledge Graph Facts\n"
            + "\n".join(f"- {f}" for f in fact_strings[:15]),
        })

    # Passages section
    if source_passages:
        records.append({
            "section": "passages",
            "content": "## Source Document Passages\n"
            + "\n---\n".join(source_passages[:15]),
        })

    return RawSearchResult(
        records=records,
        metadata={"strategy": "multi_path"},
    )
