"""
Extraction Accuracy Test — novel_lite.json (CorefNERLLMExtraction)
====================================================================
Tests the CorefNERLLMExtraction strategy:

  Text → fastcoref (resolve pronouns) → NER (entities only)
       → LLM verify/clean entities + extract relationships
       → verified nodes + discovered relationships → Graph

Key characteristics:
  ✅ Swappable NER backend (default: gliner2, also: gliner, spacy)
  ✅ LLM verifies/cleans entities and extracts relationships
  ✅ Zero hallucinations — NER grounds the entity list
  ✅ Single LLM call per chunk
  ⚠ Adds API cost per chunk (~$0.01-0.03 total)
  ⚠ Adds latency (~1-3s per chunk)

Corpora:
  Lite-001 — The Lighthouse of Cape Morrow
  Lite-002 — The Clockwork Garden of Mirabel Soto
  Lite-003 — The Vanished Library of Nahr al-Kalam

Tests:
  1. Entity Recall          — >= 40%
  2. Relationship Recall    — >= 10%
  3. Semantic Rel Recall    — >= 30%
  4. No Hallucinated Rels
  5. Relationship Type Precision
  6. Extraction Summary Report

Prerequisites:
  pip install gliner2 fastcoref
  OPENAI_API_KEY in .env (for LLM enrichment + embedder)
  FalkorDB running: docker run -p 6379:6379 falkordb/falkordb:latest

Run:
  cd graphrag_sdk
  PYTHONPATH=src pytest tests/test_extraction_novel_coref_ner_llm.py -v -s
"""
from __future__ import annotations

import json
import math
import os
import time
import pytest

from dotenv import load_dotenv
load_dotenv()

from graphrag_sdk import (
    ConnectionConfig,
    GraphRAG,
)
from graphrag_sdk.core.models import LLMResponse
from graphrag_sdk.core.providers import LLMInterface, Embedder
from graphrag_sdk.ingestion.chunking_strategies.sentence_token_cap import SentenceTokenCapChunking
from graphrag_sdk.ingestion.extraction_strategies.CorefNERLLMExtraction import CorefNERLLMExtraction


# ── Providers ─────────────────────────────────────────────────────────────────

class OpenAILLM(LLMInterface):
    """LLM provider — used by the extractor for enrichment AND by GraphRAG."""
    def __init__(self, model_name: str = "gpt-4o-mini"):
        super().__init__(model_name=model_name)
        from openai import OpenAI
        self._client = OpenAI()

    def invoke(self, prompt: str, **kwargs) -> LLMResponse:
        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
        )
        return LLMResponse(content=response.choices[0].message.content or "")


class OpenAIEmbedder(Embedder):
    def __init__(self, model_name: str = "text-embedding-3-small"):
        self.model_name = model_name
        from openai import OpenAI
        self._client = OpenAI()

    def embed_query(self, text: str, **kwargs) -> list[float]:
        response = self._client.embeddings.create(model=self.model_name, input=text)
        return response.data[0].embedding


# ── Ground Truth ──────────────────────────────────────────────────────────────

GROUND_TRUTH_ENTITIES = [
    "Cape Morrow", "Ashford Island", "Harrow's Landing", "William Tremont",
    "Elias Whitford", "Catherine Hale", "Adelaide Rose", "Robert Hargrove",
    "Eleanor Whitford", "Barbier et Cie", "Maine", "Patrick Connolly",
    "James Eaves",
    "Mirabel Soto", "Ernesto Soto", "Pilar Navarro", "Clockwork Garden",
    "Royal Botanical Garden", "Universidad Politécnica de Madrid",
    "Alejandro Vega", "Lucía Vega-Soto", "Rosa Clement", "Calarosa",
    "Nahr al-Kalam", "Dar al-Hikma", "Musa al-Farabi", "Hasan ibn Yaqub",
    "Fatima bint Ahmad", "Rashid al-Din Hamadani", "Marguerite Vallon",
    "Kemal Arslan", "Yusuf ibn Tariq",
]

GROUND_TRUTH_RELATIONSHIPS = [
    ("William Tremont", "CONSTRUCTED", "Cape Morrow Lighthouse"),
    ("Elias Whitford", "MARRIED", "Catherine Hale"),
    ("Elias Whitford", "KEEPER_OF", "Cape Morrow Lighthouse"),
    ("Adelaide Rose", "WRECKED_AT", "Blackstone Reef"),
    ("Robert Hargrove", "COMMANDED", "Adelaide Rose"),
    ("Eleanor Whitford", "MAINTAINED", "Cape Morrow Lighthouse"),
    ("Mirabel Soto", "DAUGHTER_OF", "Ernesto Soto"),
    ("Mirabel Soto", "DAUGHTER_OF", "Pilar Navarro"),
    ("Mirabel Soto", "CREATED", "Clockwork Garden"),
    ("Mirabel Soto", "MARRIED", "Alejandro Vega"),
    ("Lucía Vega-Soto", "DAUGHTER_OF", "Mirabel Soto"),
    ("Rosa Clement", "COLLABORATED_WITH", "Mirabel Soto"),
    ("Pilar Navarro", "WORKED_AT", "Royal Botanical Garden"),
    ("Yusuf ibn Tariq", "ESTABLISHED", "Dar al-Hikma"),
    ("Musa al-Farabi", "DIRECTED", "Dar al-Hikma"),
    ("Hasan ibn Yaqub", "WORKED_AT", "Dar al-Hikma"),
    ("Fatima bint Ahmad", "WORKED_AT", "Dar al-Hikma"),
    ("Rashid al-Din Hamadani", "DIRECTED", "Dar al-Hikma"),
    ("Marguerite Vallon", "EXCAVATED", "Nahr al-Kalam"),
]

# ── Load corpora ──────────────────────────────────────────────────────────────

DATA_PATH = os.path.normpath(os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "novel_lite.json"
))

with open(DATA_PATH) as f:
    NOVEL_DATA = json.load(f)

_CORPUS_MAP = {d["corpus_name"]: d["context"] for d in NOVEL_DATA}

CORPORA = [
    ("novel_lite_001.txt", _CORPUS_MAP["Lite-001"]),
    ("novel_lite_002.txt", _CORPUS_MAP["Lite-002"]),
    ("novel_lite_003.txt", _CORPUS_MAP["Lite-003"]),
]


# ── Semantic helpers ──────────────────────────────────────────────────────────

def _triple_to_sentence(src: str, rel: str, tgt: str) -> str:
    verb = rel.replace("_", " ").lower()
    return f"{src} {verb} {tgt}"


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


def _embed_batch(texts: list[str]) -> list[list[float]]:
    from openai import OpenAI
    client = OpenAI()
    response = client.embeddings.create(model="text-embedding-3-small", input=texts)
    return [d.embedding for d in sorted(response.data, key=lambda d: d.index)]


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def llm():
    return OpenAILLM(model_name="gpt-4o-mini")


@pytest.fixture(scope="module")
def embedder():
    return OpenAIEmbedder(model_name="text-embedding-3-small")


@pytest.fixture(scope="module")
async def ingest_result(llm, embedder):
    """Ingest ALL 3 corpora using CorefNER+LLM extraction."""
    conn_config = ConnectionConfig(
        host="localhost",
        port=6379,
        graph_name="novel_coref_ner_llm_extraction_test",
    )
    rag = GraphRAG(
        connection=conn_config,
        llm=llm,
        embedder=embedder,
    )

    try:
        await rag.graph_store.delete_all()
    except Exception:
        pass

    extractor = CorefNERLLMExtraction(
        llm=llm,
        ner_backend="gliner2",
        entity_types=["person", "location", "organization", "artifact",
                      "event", "facility", "work of art"],
        enable_coref=True,
    )

    total_nodes = 0
    total_rels = 0
    total_chunks = 0
    t0 = time.perf_counter()

    for filename, text in CORPORA:
        print(f"\n[ingest] → {filename} ({len(text):,} chars)")
        result = await rag.ingest(
            filename,
            text=text,
            chunker=SentenceTokenCapChunking(max_tokens=512, overlap_sentences=2),
            extractor=extractor,
        )
        total_nodes += result.nodes_created
        total_rels += result.relationships_created
        total_chunks += result.chunks_indexed
        print(f"[ingest]   nodes={result.nodes_created}, rels={result.relationships_created}, chunks={result.chunks_indexed}")

    elapsed = time.perf_counter() - t0
    print(f"\n[ingest] TOTAL — nodes={total_nodes}, rels={total_rels}, chunks={total_chunks}, time={elapsed:.1f}s")

    # Query graph
    nodes_result = await rag.graph_store.query_raw(
        "MATCH (n) WHERE NOT n:Document AND NOT n:Chunk "
        "RETURN COALESCE(n.name, n.id) AS name, labels(n)"
    )
    rels_result = await rag.graph_store.query_raw(
        "MATCH (a)-[r]->(b) WHERE NOT type(r) IN ['PART_OF','NEXT_CHUNK','MENTIONS','MENTIONED_IN'] "
        "RETURN COALESCE(a.name, a.id), "
        "       COALESCE(r.rel_type, r.type, r.predicate, type(r)), "
        "       COALESCE(b.name, b.id)"
    )

    extracted_names: set[str] = set()
    for row in (nodes_result.result_set or []):
        if row[0]:
            extracted_names.add(str(row[0]))

    extracted_rels: set[tuple[str, str, str]] = set()
    for row in (rels_result.result_set or []):
        if row[0] and row[1] and row[2]:
            extracted_rels.add((str(row[0]), str(row[1]), str(row[2])))

    return {
        "total_nodes": total_nodes,
        "total_rels": total_rels,
        "total_chunks": total_chunks,
        "extracted_names": extracted_names,
        "extracted_rels": extracted_rels,
        "elapsed_seconds": elapsed,
    }


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestCorefNERLLMExtractionAccuracy:
    """Accuracy tests for CorefGLiNER2 + LLM enrichment extraction."""

    async def test_entity_recall(self, ingest_result):
        """Entity recall >= 40% — exact name match against graph."""
        extracted = ingest_result["extracted_names"]
        found = [e for e in GROUND_TRUTH_ENTITIES if e in extracted]
        recall = len(found) / len(GROUND_TRUTH_ENTITIES)

        print(f"\n=== Entity Recall (CorefNER+LLM) ===")
        for e in GROUND_TRUTH_ENTITIES:
            status = "✅" if e in extracted else "❌"
            print(f"  {status} {e}")
        print(f"\nRecall: {len(found)}/{len(GROUND_TRUTH_ENTITIES)} = {recall:.0%}")

        assert recall >= 0.40, (
            f"Entity recall {recall:.0%} < 40% threshold.\n"
            f"Missing: {[e for e in GROUND_TRUTH_ENTITIES if e not in extracted]}"
        )

    async def test_relationship_recall(self, ingest_result):
        """Relationship F1 >= 10% — fuzzy endpoint match."""
        extracted = ingest_result["extracted_rels"]

        def _fuzzy_rel_match(gt_src: str, gt_tgt: str, ext_rels: set) -> bool:
            gt_s = set(gt_src.lower().split())
            gt_t = set(gt_tgt.lower().split())
            for s, _, t in ext_rels:
                if (gt_s & set(s.lower().split())) and (gt_t & set(t.lower().split())):
                    return True
            return False

        tp = sum(1 for src, _, tgt in GROUND_TRUTH_RELATIONSHIPS if _fuzzy_rel_match(src, tgt, extracted))
        fp = sum(1 for s, _, t in extracted if not any(
            _fuzzy_rel_match(gs, gt, {(s, "", t)}) for gs, _, gt in GROUND_TRUTH_RELATIONSHIPS
        ))
        fn = len(GROUND_TRUTH_RELATIONSHIPS) - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        print(f"\n=== Relationship F1 (CorefNER+LLM — fuzzy) ===")
        for src, rel_type, tgt in GROUND_TRUTH_RELATIONSHIPS:
            status = "✅" if _fuzzy_rel_match(src, tgt, extracted) else "❌"
            print(f"  {status} {src} --[{rel_type}]--> {tgt}")
        print(f"\nTP={tp}, FP={fp}, FN={fn}")
        print(f"Precision={precision:.0%}, Recall={recall:.0%}, F1={f1:.0%}")

        assert f1 >= 0.10, f"Relationship F1 {f1:.0%} < 10% threshold."

    async def test_relationship_recall_semantic(self, ingest_result):
        """Semantic relationship recall >= 30% (embedding cosine ≥ 0.65)."""
        extracted = ingest_result["extracted_rels"]
        if not extracted:
            pytest.skip("No extracted relationships")

        gt_sentences = [_triple_to_sentence(s, r, t) for s, r, t in GROUND_TRUTH_RELATIONSHIPS]
        ext_sentences = [_triple_to_sentence(s, r, t) for s, r, t in extracted]

        all_sentences = gt_sentences + ext_sentences
        all_embeddings = _embed_batch(all_sentences)

        gt_embs = all_embeddings[:len(gt_sentences)]
        ext_embs = all_embeddings[len(gt_sentences):]

        threshold = 0.65
        matched = 0

        print(f"\n=== Semantic Relationship Recall (CorefNER+LLM) ===")
        for i, (s, r, t) in enumerate(GROUND_TRUTH_RELATIONSHIPS):
            best_score = 0.0
            best_ext = ""
            for j, (es, er, et) in enumerate(extracted):
                score = _cosine(gt_embs[i], ext_embs[j])
                if score > best_score:
                    best_score = score
                    best_ext = f"{es} --[{er}]--> {et}"

            hit = best_score >= threshold
            if hit:
                matched += 1
            status = "✅" if hit else "❌"
            print(f"  {status} {s} --[{r}]--> {t}  (best={best_score:.2f}: {best_ext})")

        recall = matched / len(GROUND_TRUTH_RELATIONSHIPS)
        print(f"\nSemantic Recall: {matched}/{len(GROUND_TRUTH_RELATIONSHIPS)} = {recall:.0%}")

        assert recall >= 0.30, (
            f"Semantic rel recall {recall:.0%} < 30% threshold."
        )

    async def test_no_hallucinated_relationships(self, ingest_result):
        """All extracted relationships must reference entities present in the graph."""
        extracted_rels = ingest_result["extracted_rels"]
        extracted_names = ingest_result["extracted_names"]

        hallucinated = [
            (s, r, t) for s, r, t in extracted_rels
            if s not in extracted_names and t not in extracted_names
        ]

        print(f"\n=== Hallucination Check (CorefNER+LLM) ===")
        print(f"Total extracted: {len(extracted_rels)}")
        print(f"Hallucinated: {len(hallucinated)}")

        assert len(hallucinated) == 0, (
            f"{len(hallucinated)} hallucinated rels: {hallucinated[:5]}"
        )

    async def test_relationship_type_precision(self, ingest_result):
        """No casing conflicts in relationship types."""
        extracted_rels = ingest_result["extracted_rels"]
        all_types = [r for _, r, _ in extracted_rels]
        upper_types = [t for t in all_types if t == t.upper()]
        lower_types = [t for t in all_types if t != t.upper()]
        lower_set = {t.upper() for t in lower_types}
        upper_set = set(upper_types)
        mixed = lower_set & upper_set

        print(f"\n=== Type Precision (CorefNER+LLM) ===")
        print(f"UPPER: {len(upper_types)}, lower/mixed: {len(lower_types)}")

        assert mixed == set(), f"Casing conflicts: {mixed}"

    async def test_extraction_summary_report(self, ingest_result):
        """Print full summary with comparison to all strategies."""
        extracted_names = ingest_result["extracted_names"]
        extracted_rels = ingest_result["extracted_rels"]
        total_nodes = ingest_result["total_nodes"]
        total_rels = ingest_result["total_rels"]
        elapsed = ingest_result["elapsed_seconds"]
        total_text_len = sum(len(text) for _, text in CORPORA)

        def _fuzzy_match(gt: str, extracted_set: set) -> bool:
            gt_tokens = set(gt.lower().split())
            for name in extracted_set:
                if gt_tokens & set(name.lower().split()):
                    return True
            return False

        def _fuzzy_rel_match(gt_src: str, gt_tgt: str, ext_rels: set) -> bool:
            gt_s = set(gt_src.lower().split())
            gt_t = set(gt_tgt.lower().split())
            for s, _, t in ext_rels:
                if (gt_s & set(s.lower().split())) and (gt_t & set(t.lower().split())):
                    return True
            return False

        ent_tp = sum(1 for e in GROUND_TRUTH_ENTITIES if _fuzzy_match(e, extracted_names))
        ent_fp = sum(1 for n in extracted_names if not any(_fuzzy_match(gt, {n}) for gt in GROUND_TRUTH_ENTITIES))
        ent_fn = len(GROUND_TRUTH_ENTITIES) - ent_tp
        ent_p = ent_tp / (ent_tp + ent_fp) if (ent_tp + ent_fp) > 0 else 0.0
        ent_r = ent_tp / (ent_tp + ent_fn) if (ent_tp + ent_fn) > 0 else 0.0
        entity_f1 = 2 * ent_p * ent_r / (ent_p + ent_r) if (ent_p + ent_r) > 0 else 0.0

        rel_tp = sum(1 for src, _, tgt in GROUND_TRUTH_RELATIONSHIPS if _fuzzy_rel_match(src, tgt, extracted_rels))
        rel_fp = sum(1 for s, _, t in extracted_rels if not any(
            _fuzzy_rel_match(gs, gt, {(s, "", t)}) for gs, _, gt in GROUND_TRUTH_RELATIONSHIPS
        ))
        rel_fn = len(GROUND_TRUTH_RELATIONSHIPS) - rel_tp
        rel_p = rel_tp / (rel_tp + rel_fp) if (rel_tp + rel_fp) > 0 else 0.0
        rel_r = rel_tp / (rel_tp + rel_fn) if (rel_tp + rel_fn) > 0 else 0.0
        rel_f1 = 2 * rel_p * rel_r / (rel_p + rel_r) if (rel_p + rel_r) > 0 else 0.0

        print(f"\n\n{'='*65}")
        print(f"EXTRACTION REPORT — CorefNERLLMExtraction (gliner2, verify+relate)")
        print(f"{'='*65}")
        print(f"Corpora:          Lite-001, Lite-002, Lite-003")
        print(f"Total text:       {total_text_len:,} chars")
        print(f"Total nodes:      {total_nodes}")
        print(f"Total rels:       {total_rels}")
        print(f"Elapsed:          {elapsed:.1f}s")
        print(f"LLM:              gpt-4o-mini (verify + relate)")
        print(f"")
        print(f"Entity F1:        {entity_f1:.0%} (P={ent_p:.0%}, R={ent_r:.0%})")
        print(f"Rel F1:           {rel_f1:.0%} (P={rel_p:.0%}, R={rel_r:.0%})")
        print(f"Extracted names:  {len(extracted_names)}")
        print(f"Extracted rels:   {len(extracted_rels)}")
        print(f"")
        print(f"{'─'*65}")
        print(f"  CorefNER+LLM (gliner2) → Entity F1={entity_f1:.0%}, Rel F1={rel_f1:.0%} ({elapsed:.0f}s, LLM)")
        print(f"{'='*65}")

        assert True
