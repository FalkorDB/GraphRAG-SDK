# GraphRAG SDK 2.0 — Storage: Vector Store
# Native vector index management + search for FalkorDB.
# Pattern: Repository — abstracts all vector operations.

from __future__ import annotations

import logging
from typing import Any

from graphrag_sdk.core.connection import FalkorDBConnection
from graphrag_sdk.core.exceptions import DatabaseError
from graphrag_sdk.core.models import GraphRelationship, TextChunks
from graphrag_sdk.utils.cypher import sanitize_cypher_label

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
        self._indices_ensured: bool = False

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
        safe_label = sanitize_cypher_label(label)
        query = (
            f"CREATE VECTOR INDEX FOR (n:{safe_label}) ON (n.{property}) "
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
            if "already indexed" in str(exc).lower():
                logger.debug(f"Vector index on {label}.{property} already exists")
            else:
                logger.warning(f"Could not create vector index: {exc}")

    async def create_entity_vector_index(self) -> None:
        """Create a vector index on __Entity__.embedding."""
        await self.create_vector_index(label="__Entity__", property="embedding")

    async def create_relationship_vector_index(self, rel_type: str) -> None:
        """Create a vector index on edges of the given relationship type.

        Idempotent — silently succeeds if the index already exists.
        """
        safe_rel_type = sanitize_cypher_label(rel_type)
        query = (
            f"CREATE VECTOR INDEX FOR ()-[e:`{safe_rel_type}`]->() ON (e.embedding) "
            f"OPTIONS {{dimension:{self.embedding_dimension}, "
            f"similarityFunction:'{self.similarity_function}'}}"
        )
        try:
            await self._conn.query(query)
            logger.info(
                f"Created relationship vector index on [{rel_type}].embedding "
                f"(dim={self.embedding_dimension})"
            )
        except Exception as exc:
            if "already indexed" in str(exc).lower():
                logger.debug(f"Relationship vector index on [{rel_type}] already exists")
            else:
                logger.warning(f"Could not create relationship vector index for [{rel_type}]: {exc}")

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
        safe_label = sanitize_cypher_label(label)
        props = ", ".join(f"'{p}'" for p in properties)
        query = f"CALL db.idx.fulltext.createNodeIndex('{safe_label}', {props})"
        try:
            await self._conn.query(query)
            logger.info(f"Created fulltext index on {label}({', '.join(properties)})")
        except Exception as exc:
            if "already indexed" in str(exc).lower():
                logger.debug(f"Fulltext index on {label} already exists")
            else:
                logger.warning(f"Could not create fulltext index: {exc}")

    async def drop_vector_index(self, label: str = "Chunk") -> None:
        """Drop a vector index."""
        safe_label = sanitize_cypher_label(label)
        try:
            await self._conn.query(f"CALL db.idx.vector.drop('{safe_label}')")
            logger.info(f"Dropped vector index on {safe_label}")
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

        # Build batch data for UNWIND write
        batch_data = []
        for chunk, vector in zip(chunks.chunks, vectors):
            if not vector:
                continue
            batch_data.append({"chunk_id": chunk.uid, "vector": vector})

        if not batch_data:
            return 0

        # Batch write using UNWIND (500 per batch)
        batch_size = 500
        count = 0
        query = (
            "UNWIND $batch AS item "
            "MATCH (c:Chunk {id: item.chunk_id}) "
            "SET c.embedding = vecf32(item.vector)"
        )
        for start in range(0, len(batch_data), batch_size):
            batch = batch_data[start : start + batch_size]
            try:
                await self._conn.query(query, {"batch": batch})
                count += len(batch)
            except Exception as exc:
                logger.warning(
                    f"Batch chunk index failed ({len(batch)} chunks), "
                    f"falling back to individual: {exc}"
                )
                for item in batch:
                    try:
                        await self._conn.query(
                            "MATCH (c:Chunk {id: $chunk_id}) "
                            "SET c.embedding = vecf32($vector)",
                            item,
                        )
                        count += 1
                    except Exception as inner_exc:
                        logger.warning(f"Failed to index chunk {item['chunk_id']}: {inner_exc}")

        logger.debug(f"Indexed {count}/{len(chunks.chunks)} chunks")
        return count

    async def embed_relationships(self, batch_size: int = 500) -> int:
        """Batch-embed all RELATES edges that are missing embeddings.

        Queries RELATES edges where ``embedding IS NULL``, batch-embeds
        the ``fact`` property text, and writes the embedding vectors
        directly onto the edges.

        Args:
            batch_size: Number of edges per batch (default: 500).

        Returns:
            Number of edges embedded.
        """
        if not self._embedder:
            logger.warning("No embedder configured — skipping relationship embedding")
            return 0

        total_embedded = 0
        offset = 0

        while True:
            try:
                result = await self._conn.query(
                    "MATCH (a:__Entity__)-[r:RELATES]->(b:__Entity__) "
                    "WHERE r.embedding IS NULL AND r.fact IS NOT NULL "
                    "RETURN id(a) AS aid, id(b) AS bid, id(r) AS rid, r.fact AS fact "
                    "SKIP $offset LIMIT $limit",
                    {"offset": offset, "limit": batch_size},
                )
            except Exception as exc:
                logger.warning(f"RELATES edge query failed at offset {offset}: {exc}")
                break

            if not result.result_set:
                break

            edge_ids: list[int] = []
            texts: list[str] = []
            for row in result.result_set:
                rid = row[2]
                fact = row[3] if len(row) > 3 and row[3] else ""
                if fact:
                    edge_ids.append(rid)
                    texts.append(fact)

            if not texts:
                offset += batch_size
                continue

            try:
                vectors = await self._embedder.aembed_documents(texts)
            except Exception as exc:
                logger.warning(f"Batch embedding failed at offset {offset}: {exc}")
                offset += batch_size
                continue

            # Write embeddings back to edges using internal edge IDs
            embed_data = [
                {"rid": rid, "vector": vector}
                for rid, vector in zip(edge_ids, vectors)
                if vector
            ]
            if embed_data:
                # FalkorDB doesn't support UNWIND SET on edges by internal ID easily,
                # so we batch with individual queries grouped for efficiency
                for item in embed_data:
                    try:
                        await self._conn.query(
                            "MATCH ()-[r:RELATES]->() "
                            "WHERE id(r) = $rid "
                            "SET r.embedding = vecf32($vector)",
                            item,
                        )
                        total_embedded += 1
                    except Exception as inner_exc:
                        logger.warning(f"Failed to embed edge {item['rid']}: {inner_exc}")

            offset += batch_size

        logger.info(f"Embedded {total_embedded} RELATES edges")
        return total_embedded

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
        safe_label = sanitize_cypher_label(label)
        query = (
            f"CALL db.idx.vector.queryNodes('{safe_label}', 'embedding', $top_k, vecf32($vector)) "
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

    async def search_relationships(
        self,
        query_vector: list[float],
        top_k: int = 15,
    ) -> list[dict[str, Any]]:
        """Vector similarity search on RELATES edges.

        Uses the RELATES edge vector index for retrieval. Falls back to
        a Cypher-based cosine distance scan if edge vector queries
        are not supported.

        Args:
            query_vector: The query embedding vector.
            top_k: Number of results to return.

        Returns:
            List of dicts with src_name, type, tgt_name, fact, score.
        """
        # Try edge vector index query first (FalkorDB >= 4.2)
        query = (
            "CALL db.idx.vector.queryRelationships('RELATES', 'embedding', $top_k, vecf32($vector)) "
            "YIELD relationship AS r, score "
            "RETURN r.src_name AS src, r.rel_type AS type, "
            "r.tgt_name AS tgt, r.fact AS fact, score "
            "ORDER BY score DESC"
        )
        try:
            result = await self._conn.query(query, {
                "top_k": top_k,
                "vector": query_vector,
            })
            return [
                {
                    "src_name": row[0] if row[0] else "",
                    "type": row[1] if len(row) > 1 and row[1] else "",
                    "tgt_name": row[2] if len(row) > 2 and row[2] else "",
                    "fact": row[3] if len(row) > 3 and row[3] else "",
                    "score": row[4] if len(row) > 4 else 0.0,
                }
                for row in result.result_set
            ]
        except Exception as exc:
            logger.debug(f"Edge vector query not available, falling back to Cypher scan: {exc}")

        # Fallback: Cypher-based cosine distance scan
        fallback_query = (
            "MATCH (a:__Entity__)-[r:RELATES]->(b:__Entity__) "
            "WHERE r.embedding IS NOT NULL "
            "WITH a, r, b, vecf32.distance.cosine(r.embedding, vecf32($vector)) AS dist "
            "RETURN r.src_name AS src, r.rel_type AS type, "
            "r.tgt_name AS tgt, r.fact AS fact, (1-dist) AS score "
            "ORDER BY dist ASC LIMIT $top_k"
        )
        try:
            result = await self._conn.query(fallback_query, {
                "top_k": top_k,
                "vector": query_vector,
            })
            return [
                {
                    "src_name": row[0] if row[0] else "",
                    "type": row[1] if len(row) > 1 and row[1] else "",
                    "tgt_name": row[2] if len(row) > 2 and row[2] else "",
                    "fact": row[3] if len(row) > 3 and row[3] else "",
                    "score": row[4] if len(row) > 4 else 0.0,
                }
                for row in result.result_set
            ]
        except Exception as exc:
            logger.warning(f"Relationship vector search failed: {exc}")
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
        safe_label = sanitize_cypher_label(label)
        query = (
            f"CALL db.idx.fulltext.queryNodes('{safe_label}', $query_text) "
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

    # ── Batch Operations ──────────────────────────────────────────

    async def ensure_indices(self) -> dict[str, bool]:
        """Create all standard vector and fulltext indices (idempotent).

        Creates:
        - Chunk vector index (embedding)
        - __Entity__ vector index (embedding)
        - RELATES edge vector index (embedding)
        - Chunk fulltext index (text)
        - __Entity__ fulltext index (name, description)

        Returns:
            Dict mapping index name to whether creation succeeded.
        """
        if self._indices_ensured:
            return {}

        results: dict[str, bool] = {}

        for label, prop in [
            ("Chunk", "embedding"),
            ("__Entity__", "embedding"),
        ]:
            key = f"vector_{label}"
            try:
                await self.create_vector_index(label=label, property=prop)
                results[key] = True
            except Exception:
                results[key] = False

        # RELATES edge vector index
        try:
            await self.create_relationship_vector_index("RELATES")
            results["vector_RELATES"] = True
        except Exception:
            results["vector_RELATES"] = False

        for label, props in [
            ("Chunk", ("text",)),
            ("__Entity__", ("name", "description")),
        ]:
            key = f"fulltext_{label}"
            try:
                await self.create_fulltext_index(label, *props)
                results[key] = True
            except Exception:
                results[key] = False

        self._indices_ensured = True
        logger.info(f"ensure_indices complete: {results}")
        return results

    async def backfill_entity_embeddings(self, batch_size: int = 500) -> int:
        """Embed ``__Entity__`` nodes that are missing embeddings.

        Queries entities where ``embedding IS NULL``, batch-embeds
        ``name + description``, and stores the vectors.  Safe for
        incremental runs — only processes nodes without embeddings.

        Args:
            batch_size: Number of entities per batch (default: 500).

        Returns:
            Number of entities backfilled.
        """
        if not self._embedder:
            logger.warning("No embedder configured — skipping entity backfill")
            return 0

        total_backfilled = 0
        offset = 0

        while True:
            try:
                result = await self._conn.query(
                    "MATCH (e:__Entity__) WHERE e.embedding IS NULL "
                    "RETURN e.id AS id, e.name AS name, e.description AS desc "
                    "SKIP $offset LIMIT $limit",
                    {"offset": offset, "limit": batch_size},
                )
            except Exception as exc:
                logger.warning(f"Entity query failed at offset {offset}: {exc}")
                break

            if not result.result_set:
                break

            ids: list[str] = []
            texts: list[str] = []
            for row in result.result_set:
                eid = row[0]
                name = row[1] if len(row) > 1 and row[1] else str(eid)
                desc = row[2] if len(row) > 2 and row[2] else ""
                ids.append(eid)
                texts.append(str(name))

            try:
                vectors = await self._embedder.aembed_documents(texts)
            except Exception as exc:
                logger.warning(f"Batch embedding failed at offset {offset}: {exc}")
                offset += batch_size
                continue

            # Batch write using UNWIND
            backfill_data = [
                {"eid": eid, "vector": vector}
                for eid, vector in zip(ids, vectors)
                if vector
            ]
            if backfill_data:
                try:
                    await self._conn.query(
                        "UNWIND $batch AS item "
                        "MATCH (e:__Entity__ {id: item.eid}) "
                        "SET e.embedding = vecf32(item.vector)",
                        {"batch": backfill_data},
                    )
                    total_backfilled += len(backfill_data)
                except Exception as batch_exc:
                    logger.warning(
                        f"Batch entity backfill failed ({len(backfill_data)} entities), "
                        f"falling back to individual: {batch_exc}"
                    )
                    for item in backfill_data:
                        try:
                            await self._conn.query(
                                "MATCH (e:__Entity__ {id: $eid}) "
                                "SET e.embedding = vecf32($vector)",
                                item,
                            )
                            total_backfilled += 1
                        except Exception:
                            pass

            offset += batch_size

        logger.info(f"Backfilled {total_backfilled} entity embeddings")
        return total_backfilled
