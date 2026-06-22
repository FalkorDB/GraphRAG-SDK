# GraphRAG SDK — MCP server CLI (Phase 3.2)
# Usage:
#   python -m graphrag_sdk.mcp --transport stdio
#   python -m graphrag_sdk.mcp --transport sse --host 0.0.0.0 --port 8080
#
# Connection and model are read from environment variables / CLI flags so
# the server can be launched by an MCP client without code.

from __future__ import annotations

import argparse
import asyncio
import logging
import os


def _build_rag() -> object:
    from graphrag_sdk.api.main import GraphRAG
    from graphrag_sdk.core.connection import ConnectionConfig
    from graphrag_sdk.core.providers import LiteLLM, LiteLLMEmbedder

    config = ConnectionConfig(
        host=os.getenv("FALKOR_HOST", "localhost"),
        port=int(os.getenv("FALKOR_PORT", "6379")),
        username=os.getenv("FALKOR_USERNAME") or None,
        password=os.getenv("FALKOR_PASSWORD") or None,
        graph_name=os.getenv("GRAPHRAG_GRAPH", "graphrag"),
    )
    llm = LiteLLM(model_name=os.getenv("GRAPHRAG_MODEL", "gpt-4o-mini"))
    embedder = LiteLLMEmbedder(
        model_name=os.getenv("GRAPHRAG_EMBED_MODEL", "text-embedding-3-small")
    )
    return GraphRAG(connection=config, llm=llm, embedder=embedder)


def main() -> None:
    parser = argparse.ArgumentParser(description="GraphRAG SDK MCP server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default="stdio",
        help="Transport to serve over (default: stdio).",
    )
    parser.add_argument("--host", default="127.0.0.1", help="SSE bind host.")
    parser.add_argument("--port", type=int, default=8080, help="SSE bind port.")
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ...).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    from graphrag_sdk.mcp.server import GraphRAGMCPServer

    rag = _build_rag()
    server = GraphRAGMCPServer(rag)

    async def _serve() -> None:
        async with rag:  # type: ignore[attr-defined]
            if args.transport == "sse":
                await server.run("sse", host=args.host, port=args.port)
            else:
                await server.run("stdio")

    asyncio.run(_serve())


if __name__ == "__main__":
    main()
