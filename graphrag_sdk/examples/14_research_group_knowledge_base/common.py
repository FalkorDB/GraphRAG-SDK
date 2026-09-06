"""Shared setup for the scripts in this folder: paths, environment, one GraphRAG.

Environment (``.env`` in this folder or exported):

    OPENAI_API_KEY   required
    FALKORDB_HOST    default localhost
    FALKORDB_PORT    default 6379
    GRAPH_NAME       default research_group
    LLM_MODEL        default openai/gpt-4o-mini
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
TABLES = DATA / "tables"
DOCS = DATA / "docs"
NOTES = DATA / "notes"
JSON = DATA / "json"
DIRTY = DATA / "dirty"
RESULTS = HERE / "results"


def load_env() -> None:
    """Read ``.env`` next to the scripts if python-dotenv is installed."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    if (HERE / ".env").is_file():
        load_dotenv(HERE / ".env")


def add_connection_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--graph-name", default=os.environ.get("GRAPH_NAME", "research_group"))
    parser.add_argument("--host", default=os.environ.get("FALKORDB_HOST", "localhost"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("FALKORDB_PORT", "6379")))
    parser.add_argument("--model", default=os.environ.get("LLM_MODEL", "openai/gpt-4o-mini"))


def build_rag(args: argparse.Namespace, *, graph_name: str | None = None, ontology=None):
    """One GraphRAG on FalkorDB with the example's ontology and Cypher retrieval on."""
    from graphrag_sdk import ConnectionConfig, GraphRAG, LiteLLM, LiteLLMEmbedder

    if ontology is None:
        from ontology import ONTOLOGY

        ontology = ONTOLOGY

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is not set (export it or put it in .env)")

    return GraphRAG(
        connection=ConnectionConfig(
            host=args.host, port=args.port, graph_name=graph_name or args.graph_name
        ),
        llm=LiteLLM(model=args.model, api_key=api_key, temperature=0.0),
        embedder=LiteLLMEmbedder(
            model="openai/text-embedding-3-small", api_key=api_key, dimensions=256
        ),
        embedding_dimension=256,
        ontology=ontology,
        enable_cypher=True,  # text-to-Cypher over the declared columns
    )
