# GraphRAG SDK 2.0 — Storage: Vector Store
# Native vector index management + search for FalkorDB.
# Pattern: Repository — abstracts all vector operations.

from __future__ import annotations

import logging
from typing import Any

from graphrag_sdk.core.connection import FalkorDBConnection
from graphrag_sdk.core.exceptions import DatabaseError
from graphrag_sdk.core.models import TextChunks

logger = logging.getLogger(__name__)


class VectorStore:
    """Vector index management and search for FalkorDB.

    Provides:
    - Vector index creation/deletion
    - Fulltext index creation/deletion
    - Chunk embedding + indexing
    - Vector similarity search

    Args:
        connection: FalkorDB connection instance.
        embedder: Embedder for generating vectors (optional — required for indexing).
        index_name: Name of the vector index (default: "chunk_embeddings").
        embedding_dimension: Dimension of embedding vectors.
        similarity_function: Distance metric ("cosine", "euclidean", "ip").

    Example::

        store = VectorStore(connection, embedder=my_embedder, embedding_dimension=1536)
        await store.create_vector_index()
        await store.index_chunks(chunks)
        results = await store.search(query_vector, top_k=5)
    """

    def __init__(
        self,
        connection: FalkorDBConnection,
        embedder: Any | None = None,
        index_name: str = "chunk_embeddings",
        embedding_dimension: int = 1536,
        similarity_function: str = "cosine",
    ) -> None:
        self._conn = connection
        self._embedder = embedder
        self.index_name = index_name
        self.embedding_dimension = embedding_dimension
        self.similarity_function = similarity_function

    # ── Index Management ─────────────────────────────────────────

    async def create_vector_index(
        self,
        label: str = "Chunk",
        property: str = "embedding",
    ) -> None:
        """Create a vector index on a node label/property.

        Args:
            label: Node label to index (default: "Chunk").
            property: Property containing the vector (default: "embedding").
        """
        query = (
            f"CREATE VECTOR INDEX FOR (n:{label}) ON (n.{property}) "
            f"OPTIONS {{dimension:{self.embedding_dimension}, "
            f"similarityFunction:'{self.similarity_function}'}}"
        )
        try:
            await self._conn.query(query)
            logger.info(
                f"Created vector index on {label}.{property} "
                f"(dim={self.embedding_dimension})"
            )
        except Exception as exc:
            logger.warning(f"Could not create vector index: {exc}")

    async def create_entity_vector_index(self) -> None:
        """Create a vector index on __Entity__.embedding."""
        await self.create_vector_index(label="__Entity__", property="embedding")

    async def create_fact_vector_index(self) -> None:
        """Create a vector index on Fact.embedding."""
        await self.create_vector_index(label="Fact", property="embedding")

    async def create_fulltext_index(
        self,
        label: str = "Chunk",
        *properties: str,
    ) -> None:
        """Create a fulltext index on node properties.

        Args:
            label: Node label to index.
            properties: Property names to index.
        """
        if not properties:
            properties = ("text",)
        props = ", ".join(f"'{p}'" for p in properties)
        query = f"CALL db.idx.fulltext.createNodeIndex('{label}', {props})"
        try:
            await self._conn.query(query)
            logger.info(f"Created fulltext index on {label}({', '.join(properties)})")
        except Exception as exc:
            logger.warning(f"Could not create fulltext index: {exc}")

    async def drop_vector_index(self, label: str = "Chunk") -> None:
        """Drop a vector index."""
        try:
            await self._conn.query(f"CALL db.idx.vector.drop('{label}')")
            logger.info(f"Dropped vector index on {label}")
        except Exception as exc:
            logger.warning(f"Could not drop vector index: {exc}")

    # ── Indexing ─────────────────────────────────────────────────

    async def index_chunks(self, chunks: TextChunks) -> int:
        """Embed and store vectors for all chunks.

        Uses batch embedding (``aembed_documents``) for efficiency,
        then stores each vector on the corresponding Chunk node.

        Args:
            chunks: TextChunks collection to embed and index.

        Returns:
            Number of chunks indexed.
        """
        if not self._embedder:
            logger.warning("No embedder configured — skipping chunk indexing")
            return 0

        if not chunks.chunks:
            return 0

        # Batch embed all chunk texts in one API call
        texts = [chunk.text for chunk in chunks.chunks]
        try:
            vectors = await self._embedder.aembed_documents(texts)
        except Exception as exc:
            logger.warning(f"Batch embedding failed, falling back to sequential: {exc}")
            vectors = []
            for chunk in chunks.chunks:
                try:
                    vec = await self._embedder.aembed_query(chunk.text)
                    vectors.append(vec)
                except Exception:
                    vectors.append([])

        # Store each embedding on its Chunk node
        count = 0
        for chunk, vector in zip(chunks.chunks, vectors):
            if not vector:
                continue
            try:
                query = (
                    "MATCH (c:Chunk {id: $chunk_id}) "
                    "SET c.embedding = vecf32($vector)"
                )
                await self._conn.query(query, {
                    "chunk_id": chunk.uid,
                    "vector": vector,
                })
                count += 1
            except Exception as exc:
                logger.warning(f"Failed to index chunk {chunk.uid}: {exc}")

        logger.debug(f"Indexed {count}/{len(chunks.chunks)} chunks")
        return count

    # ── Search ───────────────────────────────────────────────────

    async def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        label: str = "Chunk",
        index_name: str | None = None,
    ) -> list[dict[str, Any]]:
        """Vector similarity search.

        Args:
            query_vector: The query embedding vector.
            top_k: Number of results to return.
            label: Node label to search (default: "Chunk").
            index_name: Optional custom index name.

        Returns:
            List of dicts with id, text, score, and other properties.
        """
        query = (
            f"CALL db.idx.vector.queryNodes('{label}', 'embedding', $top_k, vecf32($vector)) "
            f"YIELD node, score "
            f"RETURN node.id AS id, node.text AS text, score "
            f"ORDER BY score DESC"
        )
        try:
            result = await self._conn.query(query, {
                "top_k": top_k,
                "vector": query_vector,
            })
            results: list[dict[str, Any]] = []
            for row in result.result_set:
                results.append({
                    "id": row[0],
                    "text": row[1] if len(row) > 1 else "",
                    "score": row[2] if len(row) > 2 else 0.0,
                })
            return results
        except Exception as exc:
            logger.warning(f"Vector search failed: {exc}")
            return []

    async def search_entities(
        self,
        query_vector: list[float],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Vector similarity search on __Entity__ nodes."""
        query = (
            "CALL db.idx.vector.queryNodes('__Entity__', 'embedding', $top_k, vecf32($vector)) "
            "YIELD node, score "
            "RETURN node.id AS id, node.name AS name, node.description AS description, score "
            "ORDER BY score DESC"
        )
        try:
            result = await self._conn.query(query, {
                "top_k": top_k,
                "vector": query_vector,
            })
            results: list[dict[str, Any]] = []
            for row in result.result_set:
                results.append({
                    "id": row[0],
                    "name": row[1] if len(row) > 1 else "",
                    "description": row[2] if len(row) > 2 else "",
                    "score": row[3] if len(row) > 3 else 0.0,
                })
            return results
        except Exception as exc:
            logger.warning(f"Entity vector search failed: {exc}")
            return []

    async def search_facts(
        self,
        query_vector: list[float],
        top_k: int = 15,
    ) -> list[dict[str, Any]]:
        """Vector similarity search on Fact nodes."""
        query = (
            "CALL db.idx.vector.queryNodes('Fact', 'embedding', $top_k, vecf32($vector)) "
            "YIELD node, score "
            "RETURN node.id AS id, node.subject AS subject, node.predicate AS predicate, "
            "node.object AS object, score "
            "ORDER BY score DESC"
        )
        try:
            result = await self._conn.query(query, {
                "top_k": top_k,
                "vector": query_vector,
            })
            results: list[dict[str, Any]] = []
            for row in result.result_set:
                results.append({
                    "id": row[0],
                    "subject": row[1] if len(row) > 1 else "",
                    "predicate": row[2] if len(row) > 2 else "",
                    "object": row[3] if len(row) > 3 else "",
                    "score": row[4] if len(row) > 4 else 0.0,
                })
            return results
        except Exception as exc:
            logger.warning(f"Fact vector search failed: {exc}")
            return []

    async def fulltext_search(
        self,
        query_text: str,
        top_k: int = 5,
        label: str = "Chunk",
    ) -> list[dict[str, Any]]:
        """Fulltext search.

        Args:
            query_text: Text to search for.
            top_k: Number of results.
            label: Node label to search.

        Returns:
            List of dicts with id, text, score.
        """
        query = (
            f"CALL db.idx.fulltext.queryNodes('{label}', $query_text) "
            f"YIELD node, score "
            f"RETURN node.id AS id, node.text AS text, score "
            f"ORDER BY score DESC LIMIT $top_k"
        )
        try:
            result = await self._conn.query(query, {
                "query_text": query_text,
                "top_k": top_k,
            })
            results: list[dict[str, Any]] = []
            for row in result.result_set:
                results.append({
                    "id": row[0],
                    "text": row[1] if len(row) > 1 else "",
                    "score": row[2] if len(row) > 2 else 0.0,
                })
            return results
        except Exception as exc:
            logger.warning(f"Fulltext search failed: {exc}")
            return []
