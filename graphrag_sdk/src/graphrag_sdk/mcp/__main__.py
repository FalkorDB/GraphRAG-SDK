# GraphRAG SDK — MCP server CLI
# Usage:
#   python -m graphrag_sdk.mcp --transport stdio
#   GRAPHRAG_MCP_TOKEN=... python -m graphrag_sdk.mcp --transport sse --host 0.0.0.0
#
# Connection and model are read from environment variables / CLI flags so
# the server can be launched by an MCP client without code:
#   FALKOR_HOST / FALKOR_PORT / FALKOR_USERNAME / FALKOR_PASSWORD, GRAPHRAG_GRAPH,
#   GRAPHRAG_MODEL, GRAPHRAG_EMBED_MODEL, GRAPHRAG_EMBED_DIM, GRAPHRAG_MCP_TOKEN,
#   GRAPHRAG_MCP_READ_ONLY.

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
    llm = LiteLLM(model=os.getenv("GRAPHRAG_MODEL", "gpt-4o-mini"))
    # The dimension must match the embedding model (1536 for
    # text-embedding-3-small); the facade's default (256) would not. A set
    # GRAPHRAG_EMBED_DIM is also asked of text-embedding-3 models, so the
    # vectors fit it; older models reject a "dimensions" argument.
    dim_env = os.getenv("GRAPHRAG_EMBED_DIM", "").strip()
    dimension = int(dim_env) if dim_env else 1536
    embed_model = os.getenv("GRAPHRAG_EMBED_MODEL", "text-embedding-3-small")
    accepts_dimensions = "text-embedding-3" in embed_model
    embed_kwargs = {"dimensions": dimension} if dim_env and accepts_dimensions else {}
    embedder = LiteLLMEmbedder(model=embed_model, **embed_kwargs)
    return GraphRAG(connection=config, llm=llm, embedder=embedder, embedding_dimension=dimension)


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
        "--token",
        default=os.getenv("GRAPHRAG_MCP_TOKEN") or None,
        help="SSE bearer token (default: $GRAPHRAG_MCP_TOKEN). Required off localhost.",
    )
    parser.add_argument(
        "--read-only",
        action=argparse.BooleanOptionalAction,
        default=os.getenv("GRAPHRAG_MCP_READ_ONLY", "").strip().lower()
        in {"1", "true", "t", "yes", "y", "on"},
        help=(
            "Serve only the read tools (no ingest / finalize). Default: "
            "$GRAPHRAG_MCP_READ_ONLY; --no-read-only overrides it."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ...).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    try:
        from graphrag_sdk.mcp.server import GraphRAGMCPServer
    except ImportError as exc:
        raise SystemExit(
            "The MCP server requires extra dependencies. "
            "Install them with: pip install 'graphrag-sdk[mcp]'"
        ) from exc

    if args.transport == "sse":
        from graphrag_sdk.mcp.server import check_sse_binding

        try:
            check_sse_binding(args.host, args.token)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc

    rag = _build_rag()
    server = GraphRAGMCPServer(rag, auth_token=args.token, read_only=args.read_only)

    async def _serve() -> None:
        async with rag:  # type: ignore[attr-defined]
            if args.transport == "sse":
                await server.run("sse", host=args.host, port=args.port)
            else:
                await server.run("stdio")

    try:
        asyncio.run(_serve())
    except ImportError as exc:
        raise SystemExit(
            "The MCP server requires extra dependencies. "
            "Install them with: pip install 'graphrag-sdk[mcp]'"
        ) from exc


if __name__ == "__main__":
    main()
