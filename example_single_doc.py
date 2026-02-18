"""
GraphRAG SDK v2 — Single Document Debug Example
=================================================
Processes ONE document through the full pipeline and runs a few sample queries.
Use this for step-by-step debugging of ingestion and retrieval.

Usage:
    python example_single_doc.py                  # Index + query (uses smallest corpus doc)
    python example_single_doc.py --query-only     # Skip indexing, just query existing graph
    python example_single_doc.py --doc-index 5    # Use corpus document at index 5
    python example_single_doc.py --text "Alice works at Acme Corp."  # Use custom text

Set breakpoints anywhere to inspect intermediate state.
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── Paths ────────────────────────────────────────────────────────────────
SDK_PATH = Path(__file__).parent / "graphrag_sdk" / "src"
sys.path.insert(0, str(SDK_PATH))

BENCHMARK_DIR = Path(__file__).parent.parent / "neo4jvsfalkordb"
CORPUS_PATH = BENCHMARK_DIR / "Datasets" / "Corpus" / "novel.json"
QUESTIONS_PATH = BENCHMARK_DIR / "Datasets" / "Questions" / "novel_questions_sample_100.json"

# ── Azure OpenAI Config ──────────────────────────────────────────────────
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")

# ── FalkorDB Config ──────────────────────────────────────────────────────
FALKORDB_HOST = os.getenv("FALKORDB_HOST", "localhost")
FALKORDB_PORT = int(os.getenv("FALKORDB_PORT", 6379))
FALKORDB_PASSWORD = os.getenv("FALKORDB_PASSWORD", None)
GRAPH_NAME = "graphrag_debug"  # Separate graph for debugging

# ── Pipeline Config ──────────────────────────────────────────────────────
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200

# =============================================================================
# SDK Imports
# =============================================================================

from graphrag_sdk import GraphRAG, ConnectionConfig
from graphrag_sdk.core.context import Context
from graphrag_sdk.core.models import (
    EntityType,
    GraphSchema,
    RelationType,
    SchemaPattern,
)
from graphrag_sdk.core.providers import LiteLLM, LiteLLMEmbedder
from graphrag_sdk.ingestion.chunking_strategies.fixed_size import FixedSizeChunking
from graphrag_sdk.ingestion.extraction_strategies.merged_extraction import MergedExtraction
from graphrag_sdk.ingestion.resolution_strategies.description_merge import (
    DescriptionMergeResolution,
)


# =============================================================================
# Schema (same as benchmark)
# =============================================================================


def create_schema() -> GraphSchema:
    entities = [
        EntityType(label="Person", description="A human being or fictional character"),
        EntityType(label="Place", description="A geographic location"),
        EntityType(label="Location", description="A specific place or setting"),
        EntityType(label="Character", description="A character in a literary work"),
        EntityType(label="Event", description="A significant event or occurrence"),
        EntityType(label="Object", description="A physical or abstract object"),
        EntityType(label="Concept", description="An abstract idea or theme"),
        EntityType(label="Organization", description="A group, institution, or company"),
        EntityType(label="Work", description="A literary, artistic, or creative work"),
        EntityType(label="Time", description="A time period or temporal reference"),
    ]
    relations = [
        RelationType(label="LOCATED_IN", description="Is located in a place"),
        RelationType(label="RELATED_TO", description="Has a relationship with"),
        RelationType(label="PART_OF", description="Is part of an organization or group"),
        RelationType(label="MARRIED_TO", description="Is married to"),
        RelationType(label="PARENT_OF", description="Is parent of"),
        RelationType(label="CHILD_OF", description="Is child of"),
        RelationType(label="FRIEND_OF", description="Is a friend of"),
        RelationType(label="ENEMY_OF", description="Is an enemy of"),
        RelationType(label="CREATED", description="Created or authored something"),
        RelationType(label="VISITED", description="Visited a place"),
        RelationType(label="MENTIONED_IN", description="Is mentioned in a work"),
        RelationType(label="ASSOCIATED_WITH", description="Is associated with"),
        RelationType(label="WORKS_AT", description="Works at or employed by"),
        RelationType(label="KNOWS", description="Knows or is acquainted with"),
    ]
    patterns = [
        SchemaPattern(source="Person", relationship="LOCATED_IN", target="Place"),
        SchemaPattern(source="Character", relationship="LOCATED_IN", target="Location"),
        SchemaPattern(source="Person", relationship="RELATED_TO", target="Person"),
        SchemaPattern(source="Character", relationship="RELATED_TO", target="Character"),
        SchemaPattern(source="Person", relationship="PART_OF", target="Organization"),
        SchemaPattern(source="Person", relationship="MARRIED_TO", target="Person"),
        SchemaPattern(source="Character", relationship="MARRIED_TO", target="Character"),
        SchemaPattern(source="Person", relationship="PARENT_OF", target="Person"),
        SchemaPattern(source="Person", relationship="CHILD_OF", target="Person"),
        SchemaPattern(source="Person", relationship="FRIEND_OF", target="Person"),
        SchemaPattern(source="Character", relationship="FRIEND_OF", target="Character"),
        SchemaPattern(source="Person", relationship="ENEMY_OF", target="Person"),
        SchemaPattern(source="Person", relationship="CREATED", target="Work"),
        SchemaPattern(source="Person", relationship="VISITED", target="Place"),
        SchemaPattern(source="Person", relationship="MENTIONED_IN", target="Work"),
        SchemaPattern(source="Person", relationship="ASSOCIATED_WITH", target="Event"),
        SchemaPattern(source="Object", relationship="ASSOCIATED_WITH", target="Person"),
    ]
    return GraphSchema(entities=entities, relations=relations, patterns=patterns)


# =============================================================================
# Helpers
# =============================================================================


def load_corpus_doc(doc_index: int = 18) -> str:
    """Load a single document from the novel corpus. Default: smallest doc (index 18)."""
    with open(CORPUS_PATH) as f:
        corpus = json.load(f)
    text = corpus[doc_index].get("context", "").strip()

    # Strip Project Gutenberg boilerplate
    if "Project Gutenberg" in text:
        for marker in ["*** START", "***START"]:
            if marker in text:
                idx = text.find(marker)
                newline_idx = text.find("\n", idx)
                if newline_idx > idx:
                    text = text[newline_idx:].strip()
                break
        for marker in ["*** END", "***END", "End of Project Gutenberg"]:
            if marker in text:
                idx = text.find(marker)
                text = text[:idx].strip()
                break
    return text


def load_questions_for_doc(doc_index: int = 18) -> list[dict]:
    """Load questions from the benchmark that relate to any document."""
    with open(QUESTIONS_PATH) as f:
        return json.load(f)


# =============================================================================
# Evaluation — LLM-as-Judge
# =============================================================================

JUDGE_PROMPT = """You are an expert evaluator comparing a generated answer against the ground truth.

Question: {question}
Ground Truth Answer: {ground_truth}
Generated Answer: {generated_answer}

Score the generated answer from 0-10 based on:
- Factual correctness compared to ground truth
- Completeness of the answer
- Relevance to the question

Return ONLY the numeric score (0-10), nothing else."""


def evaluate_answer(llm, question: str, ground_truth: str, answer: str) -> int:
    try:
        response = llm.invoke(JUDGE_PROMPT.format(question=question, ground_truth=ground_truth, generated_answer=answer))
        return max(0, min(10, int(float(response.content.strip()))))
    except Exception as e:
        print(f"    Evaluation failed: {e}")
        return 0


# =============================================================================
# MAIN
# =============================================================================


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="GraphRAG SDK v2 — Single Doc Debug")
    parser.add_argument("--query-only", action="store_true", help="Skip indexing, query existing graph")
    parser.add_argument("--doc-index", type=int, default=18, help="Corpus document index (default: 18, smallest)")
    parser.add_argument("--text", type=str, default=None, help="Custom text to index (overrides --doc-index)")
    parser.add_argument("--num-questions", type=int, default=5, help="Number of questions to run (default: 5)")
    parser.add_argument("--graph-name", type=str, default=GRAPH_NAME, help=f"FalkorDB graph name (default: {GRAPH_NAME})")
    args = parser.parse_args()

    graph_name = args.graph_name

    print("=" * 60)
    print("GraphRAG SDK v2 — Single Document Debug")
    print("=" * 60)
    print(f"  Graph: {graph_name}")
    print(f"  Mode: {'query-only' if args.query_only else 'index + query'}")

    # ── Setup providers ──────────────────────────────────────────────
    llm = LiteLLM(
        model=f"azure/{AZURE_OPENAI_DEPLOYMENT}",
        api_key=AZURE_OPENAI_API_KEY,
        api_base=AZURE_OPENAI_ENDPOINT,
        api_version=AZURE_OPENAI_API_VERSION,
    )
    embedder = LiteLLMEmbedder(
        model=f"azure/{AZURE_OPENAI_EMBEDDING_DEPLOYMENT}",
        api_key=AZURE_OPENAI_API_KEY,
        api_base=AZURE_OPENAI_ENDPOINT,
        api_version=AZURE_OPENAI_API_VERSION,
    )

    # ── Create GraphRAG instance (auto-configures MultiPathRetrieval) ──
    rag = GraphRAG(
        connection=ConnectionConfig(
            host=FALKORDB_HOST,
            port=FALKORDB_PORT,
            password=FALKORDB_PASSWORD if FALKORDB_PASSWORD else None,
            graph_name=graph_name,
        ),
        llm=llm,
        embedder=embedder,
        schema=create_schema(),
    )

    # ── Ingestion ────────────────────────────────────────────────────
    if not args.query_only:
        if args.text:
            text = args.text
            print(f"  Using custom text ({len(text)} chars)")
        else:
            text = load_corpus_doc(args.doc_index)
            print(f"  Using corpus doc index {args.doc_index} ({len(text):,} chars)")

        # Clear existing graph
        print("  Clearing graph...")
        try:
            await rag.graph_store.delete_all()
        except Exception:
            pass

        print("\n  Running ingestion pipeline...")
        t0 = time.time()
        result = await rag.ingest(
            "debug_doc.txt",
            text=text,
            chunker=FixedSizeChunking(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP),
            extractor=MergedExtraction(llm=llm, embedder=embedder),
            resolver=DescriptionMergeResolution(llm=llm),
            ctx=Context(tenant_id="debug", latency_budget_ms=1800000.0),
        )
        elapsed = time.time() - t0

        print(f"\n  Pipeline completed in {elapsed:.1f}s")
        print(f"  Nodes created: {result.nodes_created}")
        print(f"  Relationships created: {result.relationships_created}")
        print(f"  Chunks indexed: {result.chunks_indexed}")
        print(f"  Entities backfilled: {result.metadata.get('entities_backfilled', 0)}")

        # Print graph stats
        stats = await rag.graph_store.get_statistics()
        print(f"\n  Graph stats:")
        print(f"    Nodes: {stats['node_count']}")
        print(f"    Edges: {stats['edge_count']}")
        print(f"    Fact nodes: {stats['fact_node_count']}")
        print(f"    Synonym edges: {stats['synonym_edge_count']}")
        print(f"    Mention edges: {stats['mention_edge_count']}")
    else:
        print("\n  Skipping ingestion (--query-only)")
        stats = await rag.graph_store.get_statistics()
        print(f"  Existing graph: {stats['node_count']} nodes, {stats['edge_count']} edges")

    # ── Queries ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("QUERIES — Running sample questions")
    print("=" * 60)

    questions = load_questions_for_doc(args.doc_index)
    num_q = min(args.num_questions, len(questions))
    print(f"  Running {num_q} questions\n")

    scores = []
    for i, q in enumerate(questions[:num_q]):
        print(f"  [{i+1}/{num_q}] ---")
        print(f"  Question: {q['question'][:100]}...")

        t0 = time.time()
        result = await rag.query(q["question"], return_context=True)
        elapsed = time.time() - t0

        print(f"    Answer ({elapsed:.1f}s): {result.answer[:200]}...")

        score = evaluate_answer(llm, q["question"], q["answer"], result.answer)
        scores.append(score)
        print(f"    Score: {score}/10")
        print(f"    Ground truth: {q['answer'][:150]}...")
        print()

    # ── Summary ──────────────────────────────────────────────────────
    if scores:
        print("=" * 60)
        print("SUMMARY")
        print("=" * 60)
        avg_score = sum(scores) / len(scores)
        print(f"  Questions: {len(scores)}")
        print(f"  Average score: {avg_score:.1f}/10 ({avg_score * 10:.0f}%)")
        print(f"  Scores: {scores}")
        print(f"  Min score: {min(scores)}, Max score: {max(scores)}")


if __name__ == "__main__":
    asyncio.run(main())
