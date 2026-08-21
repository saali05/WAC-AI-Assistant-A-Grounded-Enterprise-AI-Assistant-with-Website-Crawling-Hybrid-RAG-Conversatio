from datetime import datetime, UTC
from typing import Any, Optional

from bson import ObjectId
from motor.motor_asyncio import (
    AsyncIOMotorCollection,
    AsyncIOMotorDatabase,
)
import numpy as np

from app.core.database import get_database
from app.core.logging import logger
from app.rag.models import (
    RAGDocumentModel,
    RAGChunkModel,
    CrawlRunModel,
)


class RAGDocumentRepository:
    """Repository for CRUD operations on rag_documents collection."""

    def __init__(
        self,
        db: Optional[AsyncIOMotorDatabase] = None,
    ) -> None:
        self._db = db

    @property
    def collection(self) -> AsyncIOMotorCollection:
        database = (
            self._db
            if self._db is not None
            else get_database()
        )

        return database.rag_documents

    async def create(
        self,
        document: RAGDocumentModel,
    ) -> str:

        doc_dict = document.model_dump(
            by_alias=True,
            exclude={"id"},
        )

        result = await self.collection.insert_one(
            doc_dict
        )

        return str(result.inserted_id)

    async def get_by_canonical_url(
        self,
        canonical_url: str,
    ) -> Optional[dict[str, Any]]:

        try:

            doc = await self.collection.find_one(
                {
                    "canonical_url": canonical_url
                }
            )

            if doc:
                doc["id"] = str(
                    doc.pop("_id")
                )

            return doc

        except Exception as exc:

            logger.error(
                "Failed to get document by "
                f"canonical URL {canonical_url}: {exc}"
            )

            return None

    async def get_by_id(
        self,
        doc_id: str,
    ) -> Optional[dict[str, Any]]:

        try:

            doc = await self.collection.find_one(
                {
                    "_id": ObjectId(doc_id)
                }
            )

            if doc:
                doc["id"] = str(
                    doc.pop("_id")
                )

            return doc

        except Exception as exc:

            logger.error(
                f"Failed to get document {doc_id}: "
                f"{exc}"
            )

            return None

    async def update(
        self,
        doc_id: str,
        update_data: dict[str, Any],
    ) -> bool:

        try:

            update_data["last_crawled_at"] = (
                datetime.now(UTC)
            )

            result = await self.collection.update_one(
                {
                    "_id": ObjectId(doc_id)
                },
                {
                    "$set": update_data
                },
            )

            return result.modified_count > 0

        except Exception as exc:

            logger.error(
                f"Failed to update document "
                f"{doc_id}: {exc}"
            )

            return False

    async def delete(
        self,
        doc_id: str,
    ) -> bool:

        try:

            result = await self.collection.delete_one(
                {
                    "_id": ObjectId(doc_id)
                }
            )

            return result.deleted_count > 0

        except Exception as exc:

            logger.error(
                f"Failed to delete document "
                f"{doc_id}: {exc}"
            )

            return False

    async def get_all(
        self,
        skip: int = 0,
        limit: int = 100,
        status: Optional[str] = "active",
    ) -> list[dict[str, Any]]:

        try:

            query: dict[str, Any] = {}

            if status:
                query["status"] = status

            cursor = (
                self.collection
                .find(query)
                .skip(skip)
                .limit(limit)
                .sort(
                    "last_crawled_at",
                    -1,
                )
            )

            docs = await cursor.to_list(
                length=limit
            )

            for doc in docs:

                doc["id"] = str(
                    doc.pop("_id")
                )

            return docs

        except Exception as exc:

            logger.error(
                f"Failed to retrieve documents: "
                f"{exc}"
            )

            return []

    async def count(
        self,
        status: Optional[str] = None,
    ) -> int:

        try:

            query: dict[str, Any] = {}

            if status:
                query["status"] = status

            return await self.collection.count_documents(
                query
            )

        except Exception as exc:

            logger.error(
                f"Failed to count documents: {exc}"
            )

            return 0


class RAGChunkRepository:
    """
    Repository for CRUD, vector search,
    keyword search, and embedding re-indexing
    on rag_chunks collection.
    """

    def __init__(
        self,
        db: Optional[AsyncIOMotorDatabase] = None,
    ) -> None:

        self._db = db

    @property
    def collection(self) -> AsyncIOMotorCollection:

        database = (
            self._db
            if self._db is not None
            else get_database()
        )

        return database.rag_chunks

    async def create_many(
        self,
        chunks: list[RAGChunkModel],
    ) -> list[str]:

        if not chunks:
            return []

        try:

            docs = [
                chunk.model_dump(
                    by_alias=True,
                    exclude={"id"},
                )
                for chunk in chunks
            ]

            result = await self.collection.insert_many(
                docs
            )

            return [
                str(inserted_id)
                for inserted_id in result.inserted_ids
            ]

        except Exception as exc:

            logger.error(
                f"Failed to create chunks: {exc}"
            )

            return []

    async def delete_by_document_id(
        self,
        document_id: str,
    ) -> int:

        try:

            result = await self.collection.delete_many(
                {
                    "document_id": document_id
                }
            )

            return result.deleted_count

        except Exception as exc:

            logger.error(
                "Failed to delete chunks for "
                f"document {document_id}: {exc}"
            )

            return 0

    async def deactivate_by_document_id(
        self,
        document_id: str,
    ) -> int:

        try:

            result = await self.collection.update_many(
                {
                    "document_id": document_id,
                    "status": "active",
                },
                {
                    "$set": {
                        "status": "inactive",
                        "updated_at": datetime.now(UTC),
                    }
                },
            )

            return result.modified_count

        except Exception as exc:

            logger.error(
                "Failed to deactivate chunks for "
                f"document {document_id}: {exc}"
            )

            return 0

    # ==========================================================
    # REINDEX SUPPORT
    # ==========================================================

    async def get_active_chunks(
        self,
        batch_size: int = 50,
    ):
        """
        Return an async cursor containing active chunks.

        A cursor is intentionally returned instead of using
        skip/limit pagination.

        This is safer during re-indexing because embeddings
        are updated while the collection is being processed.
        """

        try:

            cursor = self.collection.find(
                {
                    "status": "active"
                }
            ).sort(
                "_id",
                1,
            )

            return cursor

        except Exception as exc:

            logger.error(
                f"Failed to create active chunk cursor: "
                f"{exc}"
            )

            raise

    async def update_embedding(
        self,
        chunk_id: str,
        embedding: list[float],
        model: str,
        dimensions: int,
    ) -> bool:
        """
        Replace the embedding of an existing chunk.
        """

        if not embedding:

            logger.warning(
                f"Empty embedding received for "
                f"chunk {chunk_id}"
            )

            return False

        if len(embedding) != dimensions:

            logger.error(
                "Embedding dimension mismatch for "
                f"chunk {chunk_id}: "
                f"expected={dimensions}, "
                f"actual={len(embedding)}"
            )

            return False

        try:

            result = await self.collection.update_one(
                {
                    "_id": ObjectId(chunk_id)
                },
                {
                    "$set": {
                        "embedding": embedding,
                        "embedding_model": model,
                        "embedding_dimensions": dimensions,
                        "embedding_updated_at": (
                            datetime.now(UTC)
                        ),
                    }
                },
            )

            return result.matched_count == 1

        except Exception as exc:

            logger.error(
                "Failed to update embedding for "
                f"chunk {chunk_id}: {exc}"
            )

            return False

    # ==========================================================
    # VECTOR SEARCH
    # ==========================================================

    async def vector_search(
        self,
        query_vector: list[float],
        top_k: int = 20,
        min_score: float = 0.0,
    ) -> list[dict[str, Any]]:

        """
        Execute Atlas Vector Search when available.

        If Atlas Vector Search isn't available, fall back
        to local NumPy cosine similarity.
        """

        try:

            pipeline = [
                {
                    "$vectorSearch": {
                        "index": "vector_index",
                        "path": "embedding",
                        "queryVector": query_vector,
                        "numCandidates": top_k * 10,
                        "limit": top_k,
                        "filter": {
                            "status": "active"
                        },
                    }
                },
                {
                    "$project": {
                        "_id": 1,
                        "document_id": 1,
                        "chunk_index": 1,
                        "content": 1,
                        "title": 1,
                        "heading_path": 1,
                        "url": 1,
                        "canonical_url": 1,
                        "embedding": 1,
                        "status": 1,
                        "score": {
                            "$meta": "vectorSearchScore"
                        },
                    }
                },
            ]

            cursor = self.collection.aggregate(
                pipeline
            )

            results = await cursor.to_list(
                length=top_k
            )

            if results:

                formatted = []

                for item in results:

                    item["id"] = str(
                        item.pop("_id")
                    )

                    if (
                        item.get("score", 0.0)
                        >= min_score
                    ):
                        formatted.append(item)

                return formatted

        except Exception as exc:

            logger.debug(
                "Atlas $vectorSearch unavailable or failed (%s); "
                "using LOCAL NumPy cosine similarity fallback.",
                exc,
                # "Atlas $vectorSearch unavailable "
                # f"or failed ({exc}); using local "
                # "cosine similarity."
            )

        # ------------------------------------------------------
        # LOCAL COSINE SIMILARITY FALLBACK
        # ------------------------------------------------------

        try:

            cursor = self.collection.find(
                {
                    "status": "active"
                }
            )

            chunks = await cursor.to_list(
                length=10000
            )

            if not chunks or not query_vector:
                return []

            query = np.array(
                query_vector,
                dtype=np.float32,
            )

            query_norm = np.linalg.norm(query)

            if query_norm == 0:
                return []

            scored_chunks = []

            for chunk in chunks:

                embedding = chunk.get(
                    "embedding"
                )

                if not embedding:
                    continue

                if len(embedding) != len(
                    query_vector
                ):
                    continue

                vector = np.array(
                    embedding,
                    dtype=np.float32,
                )

                vector_norm = np.linalg.norm(
                    vector
                )

                if vector_norm == 0:
                    continue

                similarity = float(
                    np.dot(
                        query,
                        vector,
                    )
                    / (
                        query_norm
                        * vector_norm
                    )
                )

                if similarity < min_score:
                    continue

                chunk_copy = dict(chunk)

                chunk_copy["id"] = str(
                    chunk_copy.pop("_id")
                )

                chunk_copy["score"] = similarity

                scored_chunks.append(
                    chunk_copy
                )

            scored_chunks.sort(
                key=lambda item: item["score"],
                reverse=True,
            )

            return scored_chunks[:top_k]

        except Exception as exc:

            logger.error(
                f"Local vector search failed: {exc}"
            )

            return []

    # ==========================================================
    # KEYWORD SEARCH
    # ==========================================================

    async def keyword_search(
        self,
        query_str: str,
        top_k: int = 20,
    ) -> list[dict[str, Any]]:

        """
        Execute MongoDB text search.

        Falls back to regex keyword matching when
        a MongoDB text index is unavailable.
        """

        if not query_str.strip():
            return []

        # ------------------------------------------------------
        # MONGODB TEXT SEARCH
        # ------------------------------------------------------

        try:

            cursor = self.collection.find(
                {
                    "$text": {
                        "$search": query_str
                    },
                    "status": "active",
                },
                {
                    "score": {
                        "$meta": "textScore"
                    }
                },
            ).sort(
                [
                    (
                        "score",
                        {
                            "$meta": "textScore"
                        },
                    )
                ]
            ).limit(top_k)

            results = await cursor.to_list(
                length=top_k
            )

            if results:

                formatted = []

                for item in results:

                    item["id"] = str(
                        item.pop("_id")
                    )

                    formatted.append(item)

                return formatted

        except Exception:

            pass

        # ------------------------------------------------------
        # REGEX FALLBACK
        # ------------------------------------------------------

        try:

            terms = [
                term.strip()
                for term in query_str.split()
                if len(term.strip()) > 2
            ]

            if not terms:
                terms = [
                    query_str.strip()
                ]

            regex_patterns = [
                {
                    "content": {
                        "$regex": term,
                        "$options": "i",
                    }
                }
                for term in terms
            ]

            query = {
                "status": "active",
                "$or": regex_patterns,
            }

            cursor = self.collection.find(
                query
            ).limit(top_k)

            results = await cursor.to_list(
                length=top_k
            )

            formatted = []

            for item in results:

                item["id"] = str(
                    item.pop("_id")
                )

                content_lower = (
                    item.get(
                        "content",
                        "",
                    ).lower()
                )

                score = (
                    sum(
                        1.0
                        for term in terms
                        if term.lower()
                        in content_lower
                    )
                    / len(terms)
                )

                item["score"] = score

                formatted.append(item)

            formatted.sort(
                key=lambda item: item["score"],
                reverse=True,
            )

            return formatted

        except Exception as exc:

            logger.error(
                f"Keyword search failed: {exc}"
            )

            return []

    async def get_active_chunks_count(
        self,
    ) -> int:

        try:

            return await self.collection.count_documents(
                {
                    "status": "active"
                }
            )

        except Exception as exc:

            logger.error(
                f"Failed to count active chunks: {exc}"
            )

            return 0


class CrawlRunRepository:
    """Repository for managing crawl runs."""

    def __init__(
        self,
        db: Optional[AsyncIOMotorDatabase] = None,
    ) -> None:

        self._db = db

    @property
    def collection(self) -> AsyncIOMotorCollection:

        database = (
            self._db
            if self._db is not None
            else get_database()
        )

        return database.crawl_runs

    async def create(
        self,
        crawl_run: Optional[CrawlRunModel] = None,
    ) -> str:

        model = (
            crawl_run
            or CrawlRunModel()
        )

        doc = model.model_dump(
            by_alias=True,
            exclude={"id"},
        )

        result = await self.collection.insert_one(
            doc
        )

        return str(result.inserted_id)

    async def update(
        self,
        run_id: str,
        update_data: dict[str, Any],
    ) -> bool:

        try:

            result = await self.collection.update_one(
                {
                    "_id": ObjectId(run_id)
                },
                {
                    "$set": update_data
                },
            )

            return result.modified_count > 0

        except Exception as exc:

            logger.error(
                f"Failed to update crawl run "
                f"{run_id}: {exc}"
            )

            return False

    async def get_by_id(
        self,
        run_id: str,
    ) -> Optional[dict[str, Any]]:

        try:

            doc = await self.collection.find_one(
                {
                    "_id": ObjectId(run_id)
                }
            )

            if doc:

                doc["id"] = str(
                    doc.pop("_id")
                )

            return doc

        except Exception as exc:

            logger.error(
                f"Failed to get crawl run "
                f"{run_id}: {exc}"
            )

            return None

    async def get_latest(
        self,
    ) -> Optional[dict[str, Any]]:

        cursor = (
            self.collection
            .find()
            .sort(
                "started_at",
                -1,
            )
            .limit(1)
        )

        runs = await cursor.to_list(
            length=1
        )

        if runs:

            doc = runs[0]

            doc["id"] = str(
                doc.pop("_id")
            )

            return doc

        return None


async def initialize_rag_indexes(
    db: Optional[AsyncIOMotorDatabase] = None,
) -> None:

    """
    Initialize MongoDB indexes for RAG collections.
    """

    database = (
        db
        if db is not None
        else get_database()
    )

    logger.info(
        "⚡ Initializing RAG MongoDB indexes..."
    )

    # ----------------------------------------------------------
    # DOCUMENT INDEXES
    # ----------------------------------------------------------

    doc_coll = database.rag_documents

    await doc_coll.create_index(
        "canonical_url",
        unique=True,
    )

    await doc_coll.create_index(
        "content_hash"
    )

    await doc_coll.create_index(
        "status"
    )

    await doc_coll.create_index(
        "last_crawled_at"
    )

    # ----------------------------------------------------------
    # CHUNK INDEXES
    # ----------------------------------------------------------

    chunk_coll = database.rag_chunks

    await chunk_coll.create_index(
        "document_id"
    )

    await chunk_coll.create_index(
        "canonical_url"
    )

    await chunk_coll.create_index(
        "status"
    )

    try:

        await chunk_coll.create_index(
            [
                ("content", "text"),
                ("title", "text"),
            ]
        )

    except Exception as exc:

        logger.warning(
            "Could not create text index on "
            f"rag_chunks: {exc}"
        )

    # ----------------------------------------------------------
    # CRAWL RUN INDEXES
    # ----------------------------------------------------------

    run_coll = database.crawl_runs

    await run_coll.create_index(
        "started_at"
    )

    await run_coll.create_index(
        "status"
    )

    logger.info(
        "✅ RAG MongoDB indexes created successfully"
    )