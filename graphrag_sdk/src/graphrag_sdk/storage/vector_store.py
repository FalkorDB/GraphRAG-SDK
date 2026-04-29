# GraphRAG SDK — Storage: Vector Store
# Native vector index management + search for FalkorDB.
# Pattern: Repository — abstracts all vector operations.

from __future__ import annotations

import logging
from typing import Any

from graphrag_sdk.core.connection import FalkorDBConnection
from graphrag_sdk.core.models import TextChunks

logger = logging.getLogger(__name__)

_MAX_EMBEDDING_DIMENSION = 8192

# Pagination safety net for IS-NULL streaming loops. At batch_size=500
# this covers 5M items — large enough for any realistic graph. Trips
# only on a pathological condition where mutations don't reduce the
# IS-NULL filter (e.g. server bug, persistent embedding failures).
_MAX_PAGINATION_ITERATIONS = 10_000

# Substrings FalkorDB returns when an index already exists. Mirrors
# ``FalkorDBConnection._NON_TRANSIENT_MARKERS`` for the index-creation
# subset — both markers indicate the operation is already done, so we
# treat them as idempotent success rather than a creation failure.
_INDEX_EXISTS_MARKERS = ("already indexed", "already exists")

# Characters that RediSearch treats as syntax in fulltext queries.
_REDISEARCH_SPECIAL = set(r""",./<>{}[]\"':;!@#$%^&*()-+=~|""")


def _escape_fulltext_query(text: str) -> str:
    """Escape special characters for RediSearch fulltext queries.

    Prepends a backslash to every character that RediSearch treats as
    a separator or operator so the term is searched literally.
    """
    escaped: list[str] = []
    for ch in text:
        if ch in _REDISEARCH_SPECIAL:
            escaped.append(f"\\{ch}")
        else:
            escaped.append(ch)
    return "".join(escaped)


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
        embedding_dimension: Dimension of embedding vectors (1..8192).

    Example::

        store = VectorStore(connection, embedder=my_embedder, embedding_dimension=256)
        await store.ensure_indices()
        await store.index_chunks(chunks)
        results = await store.search_chunks(query_vector, top_k=5)
    """

    def __init__(
        self,
        connection: FalkorDBConnection,
        embedder: Any | None = None,
        embedding_dimension: int = 256,
    ) -> None:
        if not 1 <= embedding_dimension <= _MAX_EMBEDDING_DIMENSION:
            raise ValueError(
                f"embedding_dimension must be between 1 and {_MAX_EMBEDDING_DIMENSION}, "
                f"got {embedding_dimension}"
            )
        self._conn = connection
        self._embedder = embedder
        self.embedding_dimension = embedding_dimension
        self._indices_ensured: bool = False

    # ── Index Management ─────────────────────────────────────────
    #
    # Index identifiers (labels, properties, similarity function) are
    # hardcoded in every Cypher statement below. No user-supplied string
    # is interpolated into a query — only ``embedding_dimension``, an int
    # bound-checked at construction.

    async def _try_create_index(self, query: str, descriptor: str, kind: str) -> bool:
        """Run a CREATE INDEX query idempotently.

        Returns ``True`` if the index now exists (created or already existed),
        ``False`` if creation failed for any other reason. The bool flows up
        to ``ensure_indices()`` so the result map honestly reflects whether
        each index is in place — failures are not papered over as success.
        """
        try:
            await self._conn.query(query)
            if kind == "vector":
                logger.info(
                    f"Created vector index on {descriptor} (dim={self.embedding_dimension})"
                )
            else:
                logger.info(f"Created fulltext index on {descriptor}")
            return True
        except Exception as exc:
            msg = str(exc).lower()
            if any(marker in msg for marker in _INDEX_EXISTS_MARKERS):
                logger.debug(f"{kind.capitalize()} index on {descriptor} already exists")
                return True
            logger.warning(f"Could not create {kind} index on {descriptor}: {exc}")
            return False

    async def create_chunk_vector_index(self) -> bool:
        """Create the vector index on ``Chunk.embedding``."""
        query = (
            f"CREATE VECTOR INDEX FOR (n:Chunk) ON (n.embedding) "
            f"OPTIONS {{dimension:{self.embedding_dimension}, similarityFunction:'cosine'}}"
        )
        return await self._try_create_index(query, "Chunk.embedding", "vector")

    async def create_entity_vector_index(self) -> bool:
        """Create the vector index on ``__Entity__.embedding``."""
        query = (
            f"CREATE VECTOR INDEX FOR (n:__Entity__) ON (n.embedding) "
            f"OPTIONS {{dimension:{self.embedding_dimension}, similarityFunction:'cosine'}}"
        )
        return await self._try_create_index(query, "__Entity__.embedding", "vector")

    async def create_relates_vector_index(self) -> bool:
        """Create the vector index on ``RELATES.embedding`` edges."""
        query = (
            f"CREATE VECTOR INDEX FOR ()-[e:RELATES]->() ON (e.embedding) "
            f"OPTIONS {{dimension:{self.embedding_dimension}, similarityFunction:'cosine'}}"
        )
        return await self._try_create_index(query, "[RELATES].embedding", "vector")

    async def create_chunk_fulltext_index(self) -> bool:
        """Create the fulltext index on ``Chunk.text``."""
        return await self._try_create_index(
            "CALL db.idx.fulltext.createNodeIndex('Chunk', 'text')",
            "Chunk(text)",
            "fulltext",
        )

    async def create_entity_fulltext_index(self) -> bool:
        """Create the fulltext index on ``__Entity__(name, description)``."""
        return await self._try_create_index(
            "CALL db.idx.fulltext.createNodeIndex('__Entity__', 'name', 'description')",
            "__Entity__(name, description)",
            "fulltext",
        )

    async def drop_chunk_vector_index(self) -> None:
        """Drop the ``Chunk`` vector index."""
        try:
            await self._conn.query("CALL db.idx.vector.drop('Chunk')")
            logger.info("Dropped vector index on Chunk")
        except Exception as exc:
            logger.warning(f"Could not drop vector index on Chunk: {exc}")

    async def drop_entity_vector_index(self) -> None:
        """Drop the ``__Entity__`` vector index."""
        try:
            await self._conn.query("CALL db.idx.vector.drop('__Entity__')")
            logger.info("Dropped vector index on __Entity__")
        except Exception as exc:
            logger.warning(f"Could not drop vector index on __Entity__: {exc}")

    async def drop_relates_vector_index(self) -> None:
        """Drop the ``RELATES`` edge vector index."""
        try:
            await self._conn.query("CALL db.idx.vector.drop('RELATES')")
            logger.info("Dropped vector index on RELATES")
        except Exception as exc:
            logger.warning(f"Could not drop vector index on RELATES: {exc}")

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
                    logger.debug("Single chunk embedding failed", exc_info=True)
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
                            "MATCH (c:Chunk {id: $chunk_id}) SET c.embedding = vecf32($vector)",
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

        for _ in range(_MAX_PAGINATION_ITERATIONS):
            try:
                # No SKIP — embedded edges drop out of the IS NULL filter,
                # so each query naturally returns the next un-embedded batch.
                result = await self._conn.query(
                    "MATCH (a:__Entity__)-[r:RELATES]->(b:__Entity__) "
                    "WHERE r.embedding IS NULL AND r.fact IS NOT NULL "
                    "RETURN id(a) AS aid, id(b) AS bid, id(r) AS rid, r.fact AS fact "
                    "LIMIT $limit",
                    {"limit": batch_size},
                )
            except Exception as exc:
                logger.warning(f"RELATES edge query failed: {exc}")
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
                break

            try:
                vectors = await self._embedder.aembed_documents(texts)
            except Exception as exc:
                logger.warning(f"Batch embedding failed: {exc}")
                break

            # Write embeddings back to edges using internal edge IDs
            embed_data = [
                {"rid": rid, "vector": vector} for rid, vector in zip(edge_ids, vectors) if vector
            ]
            if not embed_data:
                logger.warning("All vectors in batch were empty — stopping to avoid infinite loop")
                break

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
        else:
            logger.error(
                "Pagination exceeded %d iterations in embed_relationships — aborting",
                _MAX_PAGINATION_ITERATIONS,
            )

        logger.info(f"Embedded {total_embedded} RELATES edges")
        return total_embedded

    # ── Search ───────────────────────────────────────────────────

    async def search_chunks(
        self,
        query_vector: list[float],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Vector similarity search on ``Chunk`` nodes.

        Args:
            query_vector: The query embedding vector.
            top_k: Number of results to return.

        Returns:
            List of dicts with id, text, score.
        """
        query = (
            "CALL db.idx.vector.queryNodes('Chunk', 'embedding', $top_k, vecf32($vector)) "
            "YIELD node, score "
            "RETURN node.id AS id, node.text AS text, score "
            "ORDER BY score DESC"
        )
        try:
            result = await self._conn.query(
                query,
                {
                    "top_k": top_k,
                    "vector": query_vector,
                },
            )
            results: list[dict[str, Any]] = []
            for row in result.result_set:
                results.append(
                    {
                        "id": row[0],
                        "text": row[1] if len(row) > 1 else "",
                        "score": row[2] if len(row) > 2 else 0.0,
                    }
                )
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
            result = await self._conn.query(
                query,
                {
                    "top_k": top_k,
                    "vector": query_vector,
                },
            )
            results: list[dict[str, Any]] = []
            for row in result.result_set:
                results.append(
                    {
                        "id": row[0],
                        "name": row[1] if len(row) > 1 else "",
                        "description": row[2] if len(row) > 2 else "",
                        "score": row[3] if len(row) > 3 else 0.0,
                    }
                )
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
            "CALL db.idx.vector.queryRelationships("
            "'RELATES', 'embedding', $top_k, vecf32($vector)) "
            "YIELD relationship AS r, score "
            "RETURN r.src_name AS src, r.rel_type AS type, "
            "r.tgt_name AS tgt, r.fact AS fact, score "
            "ORDER BY score DESC"
        )
        try:
            result = await self._conn.query(
                query,
                {
                    "top_k": top_k,
                    "vector": query_vector,
                },
            )
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
            "WITH a, r, b, vec.cosineDistance(r.embedding, vecf32($vector)) AS dist "
            "RETURN r.src_name AS src, r.rel_type AS type, "
            "r.tgt_name AS tgt, r.fact AS fact, (1-dist) AS score "
            "ORDER BY dist ASC LIMIT $top_k"
        )
        try:
            result = await self._conn.query(
                fallback_query,
                {
                    "top_k": top_k,
                    "vector": query_vector,
                },
            )
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

    async def fulltext_search_chunks(
        self,
        query_text: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Fulltext search on ``Chunk`` nodes.

        Args:
            query_text: Text to search for.
            top_k: Number of results.

        Returns:
            List of dicts with id, text, score.
        """
        escaped_text = _escape_fulltext_query(query_text)
        query = (
            "CALL db.idx.fulltext.queryNodes('Chunk', $query_text) "
            "YIELD node, score "
            "RETURN node.id AS id, node.text AS text, score "
            "ORDER BY score DESC LIMIT $top_k"
        )
        try:
            result = await self._conn.query(
                query,
                {
                    "query_text": escaped_text,
                    "top_k": top_k,
                },
            )
            results: list[dict[str, Any]] = []
            for row in result.result_set:
                results.append(
                    {
                        "id": row[0],
                        "text": row[1] if len(row) > 1 else "",
                        "score": row[2] if len(row) > 2 else 0.0,
                    }
                )
            return results
        except Exception as exc:
            logger.warning(f"Fulltext search failed: {exc}")
            return []

    async def fulltext_search_entities(
        self,
        query_text: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Fulltext search on ``__Entity__`` nodes.

        Args:
            query_text: Text to search for.
            top_k: Number of results.

        Returns:
            List of dicts with id, text (name), score.
        """
        escaped_text = _escape_fulltext_query(query_text)
        query = (
            "CALL db.idx.fulltext.queryNodes('__Entity__', $query_text) "
            "YIELD node, score "
            "RETURN node.id AS id, node.name AS text, score "
            "ORDER BY score DESC LIMIT $top_k"
        )
        try:
            result = await self._conn.query(
                query,
                {
                    "query_text": escaped_text,
                    "top_k": top_k,
                },
            )
            results: list[dict[str, Any]] = []
            for row in result.result_set:
                results.append(
                    {
                        "id": row[0],
                        "text": row[1] if len(row) > 1 else "",
                        "score": row[2] if len(row) > 2 else 0.0,
                    }
                )
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
            Dict mapping each index name to whether creation succeeded
            **on this call**. If a previous call already established all
            indices (``self._indices_ensured`` is ``True``), this method
            short-circuits and returns ``{}`` to signal "no work was
            performed" — not "no indexes exist". Callers that need a
            point-in-time view of index health should reset the cache
            (see ``GraphRAG.delete_all``) or call ``ensure_indices()``
            on a fresh ``VectorStore`` instance.
        """
        if self._indices_ensured:
            return {}

        results: dict[str, bool] = {}

        creators: list[tuple[str, Any]] = [
            ("vector_Chunk", self.create_chunk_vector_index),
            ("vector_Entity", self.create_entity_vector_index),
            ("vector_RELATES", self.create_relates_vector_index),
            ("fulltext_Chunk", self.create_chunk_fulltext_index),
            ("fulltext_Entity", self.create_entity_fulltext_index),
        ]
        for key, fn in creators:
            try:
                results[key] = await fn()
            except Exception:
                logger.debug("Index creation failed for %s", key, exc_info=True)
                results[key] = False

        # Only mark indices as ensured if everything actually succeeded —
        # otherwise the next call to ensure_indices() would skip retrying
        # transient failures.
        if all(results.values()):
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

        for _ in range(_MAX_PAGINATION_ITERATIONS):
            try:
                # No SKIP — embedded entities drop out of the IS NULL filter,
                # so each query naturally returns the next un-embedded batch.
                result = await self._conn.query(
                    "MATCH (e:__Entity__) WHERE e.embedding IS NULL "
                    "RETURN e.id AS id, e.name AS name, e.description AS desc "
                    "LIMIT $limit",
                    {"limit": batch_size},
                )
            except Exception as exc:
                logger.warning(f"Entity query failed: {exc}")
                break

            if not result.result_set:
                break

            ids: list[str] = []
            texts: list[str] = []
            for row in result.result_set:
                eid = row[0]
                name = row[1] if len(row) > 1 and row[1] else str(eid)
                ids.append(eid)
                texts.append(str(name))

            try:
                vectors = await self._embedder.aembed_documents(texts)
            except Exception as exc:
                logger.warning(f"Batch embedding failed: {exc}")
                break

            # Batch write using UNWIND
            backfill_data = [
                {"eid": eid, "vector": vector} for eid, vector in zip(ids, vectors) if vector
            ]
            if not backfill_data:
                logger.warning("All vectors in batch were empty — stopping to avoid infinite loop")
                break

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
                            "MATCH (e:__Entity__ {id: $eid}) SET e.embedding = vecf32($vector)",
                            item,
                        )
                        total_backfilled += 1
                    except Exception:
                        logger.debug(
                            "Single entity backfill failed for %s", item.get("eid"), exc_info=True
                        )
        else:
            logger.error(
                "Pagination exceeded %d iterations in backfill_entity_embeddings — aborting",
                _MAX_PAGINATION_ITERATIONS,
            )

        logger.info(f"Backfilled {total_backfilled} entity embeddings")
        return total_backfilled
