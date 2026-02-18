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

    async def index_facts(self, fact_strings: list[str], facts: list) -> int:
        """Embed and store vectors for all facts.

        Uses batch embedding (``aembed_documents``) for efficiency,
        then creates/updates Fact nodes with their embeddings via UNWIND.

        Args:
            fact_strings: Human-readable fact strings like "(subject, predicate, object)".
            facts: List of FactTriple objects.

        Returns:
            Number of facts indexed.
        """
        if not self._embedder:
            logger.warning("No embedder configured — skipping fact indexing")
            return 0

        if not fact_strings:
            return 0

        # Batch embed all fact strings in one API call
        try:
            vectors = await self._embedder.aembed_documents(fact_strings)
        except Exception as exc:
            logger.warning(f"Batch embedding failed, falling back to sequential: {exc}")
            vectors = []
            for text in fact_strings:
                try:
                    vec = await self._embedder.aembed_query(text)
                    vectors.append(vec)
                except Exception:
                    vectors.append([])

        # Build batch data for UNWIND write
        batch_data = []
        for fact, text, vector in zip(facts, fact_strings, vectors):
            if not vector:
                continue
            fact_id = str(hash((fact.subject, fact.predicate, fact.object)) % (10**16))
            chunk_id = getattr(fact, "source_chunk_id", "") or getattr(fact, "chunk_id", "")
            batch_data.append({
                "id": fact_id,
                "subject": fact.subject,
                "predicate": fact.predicate,
                "object": fact.object,
                "chunk_id": chunk_id,
                "text": text,
                "vector": vector,
            })

        if not batch_data:
            return 0

        # Batch write using UNWIND (500 per batch)
        batch_size = 500
        count = 0
        query = (
            "UNWIND $batch AS item "
            "MERGE (f:Fact {id: item.id}) "
            "SET f.subject = item.subject, f.predicate = item.predicate, "
            "f.object = item.object, f.source_chunk_id = item.chunk_id, "
            "f.text = item.text, f.embedding = vecf32(item.vector)"
        )
        for start in range(0, len(batch_data), batch_size):
            batch = batch_data[start : start + batch_size]
            try:
                await self._conn.query(query, {"batch": batch})
                count += len(batch)
            except Exception as exc:
                logger.warning(
                    f"Batch fact index failed ({len(batch)} facts), "
                    f"falling back to individual: {exc}"
                )
                for item in batch:
                    try:
                        await self._conn.query(
                            "MERGE (f:Fact {id: $id}) "
                            "SET f.subject = $subject, f.predicate = $predicate, "
                            "f.object = $object, f.source_chunk_id = $chunk_id, "
                            "f.text = $text, f.embedding = vecf32($vector)",
                            item,
                        )
                        count += 1
                    except Exception as inner_exc:
                        logger.warning(f"Failed to index fact {item['id']}: {inner_exc}")

        logger.debug(f"Indexed {count}/{len(fact_strings)} facts")
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

    # ── Batch Operations ──────────────────────────────────────────

    async def ensure_indices(self) -> dict[str, bool]:
        """Create all standard vector and fulltext indices (idempotent).

        Creates:
        - Chunk vector index (embedding)
        - __Entity__ vector index (embedding)
        - Fact vector index (embedding)
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
            ("Fact", "embedding"),
        ]:
            key = f"vector_{label}"
            try:
                await self.create_vector_index(label=label, property=prop)
                results[key] = True
            except Exception:
                results[key] = False

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
                texts.append(f"{name}\n{desc}" if desc else str(name))

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
