from datetime import datetime, UTC, timedelta
import re
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
    # REINDEX & MIGRATION SUPPORT
    # ==========================================================

    async def get_active_chunks(
        self,
        batch_size: int = 50,
    ):
        """
        Return an async cursor containing all active chunks.

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

    async def get_mismatched_chunks_count(
        self,
        model: Optional[str] = None,
        dimensions: Optional[int] = None,
    ) -> int:
        """
        Count active chunks whose stored embedding model or dimensions
        do not match the target configuration.
        """

        target_model = model or settings.RAG_EMBEDDING_MODEL
        target_dimensions = (
            dimensions
            if dimensions is not None
            else settings.RAG_EMBEDDING_DIMENSIONS
        )

        query = {
            "status": "active",
            "$or": [
                {"embedding_model": {"$ne": target_model}},
                {"embedding_dimensions": {"$ne": target_dimensions}},
                {"embedding_model": {"$exists": False}},
                {"embedding_dimensions": {"$exists": False}},
            ],
        }

        try:
            return await self.collection.count_documents(query)
        except Exception as exc:
            logger.error(
                f"Failed to count mismatched chunks: {exc}"
            )
            return 0

    async def get_mismatched_chunks(
        self,
        batch_size: int = 25,
        model: Optional[str] = None,
        dimensions: Optional[int] = None,
    ):
        """
        Return an async cursor for active chunks whose stored embedding model
        or dimensions do not match the target configuration.
        """

        target_model = model or settings.RAG_EMBEDDING_MODEL
        target_dimensions = (
            dimensions
            if dimensions is not None
            else settings.RAG_EMBEDDING_DIMENSIONS
        )

        query = {
            "status": "active",
            "$or": [
                {"embedding_model": {"$ne": target_model}},
                {"embedding_dimensions": {"$ne": target_dimensions}},
                {"embedding_model": {"$exists": False}},
                {"embedding_dimensions": {"$exists": False}},
            ],
        }

        try:
            cursor = self.collection.find(query).sort("_id", 1)
            return cursor
        except Exception as exc:
            logger.error(
                f"Failed to create mismatched chunk cursor: {exc}"
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
    # KEYWORD SEARCH HELPERS
    # ==========================================================

    @staticmethod
    def _extract_keyword_terms(query_str: str) -> list[str]:
        """
        Extract meaningful search terms from a query string.
        - Strips surrounding punctuation while preserving tech terms (e.g. Node.js, Next.js, .NET, C++, C#).
        - Filters out conversational stop words.
        - Deduplicates terms while preserving case and order.
        """
        if not query_str or not query_str.strip():
            return []

        stop_words = {
            "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
            "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
            "below", "between", "both", "but", "by", "can", "can't", "cannot", "could",
            "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down",
            "during", "each", "few", "for", "from", "further", "had", "hadn't", "has",
            "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her",
            "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's",
            "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it",
            "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my",
            "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or",
            "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same",
            "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so",
            "some", "such", "tell", "than", "that", "that's", "the", "their", "theirs",
            "them", "themselves", "then", "there", "there's", "these", "they", "they'd",
            "they'll", "they're", "they've", "this", "those", "through", "to", "too",
            "under", "until", "up", "us", "use", "uses", "used", "using", "very", "was",
            "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what",
            "what's", "when", "when's", "where", "where's", "which", "while", "who",
            "who's", "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you",
            "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"
        }

        raw_tokens = query_str.strip().split()
        cleaned_tokens: list[str] = []

        for token in raw_tokens:
            # Strip edge punctuation but preserve tech symbols (#, +, ., -)
            t = re.sub(r"^[^\w#.+]+|[^\w#.+]+$", "", token)
            while t.endswith(".") and not re.search(r"\.[a-zA-Z0-9]+$", t):
                t = t.rstrip(".")
            t = re.sub(r"[^\w#.+]+$", "", t)
            while t.endswith(".") and not re.search(r"\.[a-zA-Z0-9]+$", t):
                t = t.rstrip(".")
            if t:
                cleaned_tokens.append(t)

        seen: set[str] = set()
        terms: list[str] = []

        for t in cleaned_tokens:
            lower = t.lower()
            if lower not in stop_words and lower not in seen and len(t) >= 2:
                seen.add(lower)
                terms.append(t)

        # Fallback if all tokens were stop words (e.g. "what is it")
        if not terms and cleaned_tokens:
            for t in cleaned_tokens:
                lower = t.lower()
                if lower not in seen and len(t) >= 2:
                    seen.add(lower)
                    terms.append(t)

        return terms

    @staticmethod
    def _build_term_pattern(term: str) -> re.Pattern:
        """Build regex pattern respecting word boundaries for alphanumeric and symbol terms."""
        if re.match(r"^\w+$", term):
            return re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE)
        else:
            return re.compile(rf"(?<![\w#+.-]){re.escape(term)}(?![\w#+.-])", re.IGNORECASE)

    @classmethod
    def _calculate_keyword_score(
        cls,
        item: dict[str, Any],
        terms: list[str],
        patterns: Optional[dict[str, re.Pattern]] = None,
    ) -> float:
        """
        Calculate a normalized keyword relevance score in the range 0.0 - 1.0.
        Evaluates matches across title, heading_path, and content.
        """
        if not terms:
            return 0.0

        if patterns is None:
            patterns = {term: cls._build_term_pattern(term) for term in terms}

        title = str(item.get("title") or "")
        heading_path_val = item.get("heading_path") or []
        if isinstance(heading_path_val, list):
            heading_text = " ".join(str(h) for h in heading_path_val)
        else:
            heading_text = str(heading_path_val)
        content = str(item.get("content") or "")

        matched_terms: set[str] = set()
        title_matches = 0
        heading_matches = 0
        content_matches = 0

        for term in terms:
            pattern = patterns[term]
            in_title = bool(pattern.search(title))
            in_heading = bool(pattern.search(heading_text))
            in_content = bool(pattern.search(content))

            if in_title:
                title_matches += 1
            if in_heading:
                heading_matches += 1
            if in_content:
                content_matches += 1

            if in_title or in_heading or in_content:
                matched_terms.add(term)

        if not matched_terms:
            return 0.0

        num_terms = len(terms)
        term_coverage = len(matched_terms) / num_terms
        effective_target = min(num_terms, 4)
        density = min(1.0, len(matched_terms) / max(1, effective_target))

        title_boost = 0.15 if title_matches > 0 else 0.0
        heading_boost = 0.10 if heading_matches > 0 else 0.0
        content_boost = 0.10 if content_matches > 0 else 0.0

        raw_score = (
            (0.55 * density)
            + (0.15 * term_coverage)
            + title_boost
            + heading_boost
            + content_boost
        )

        return round(min(1.0, max(0.0, raw_score)), 4)

    # ==========================================================
    # KEYWORD SEARCH
    # ==========================================================

    async def keyword_search(
        self,
        query_str: str,
        top_k: int = 20,
    ) -> list[dict[str, Any]]:
        """
        Execute MongoDB keyword/text search with regex fallback.

        Searches across content, title, and heading_path, and returns
        relevance-ranked chunks with normalized scores (0.0 - 1.0).
        """
        if not query_str or not query_str.strip():
            return []

        extracted_terms = self._extract_keyword_terms(query_str)
        if not extracted_terms:
            logger.info(f"Keyword search: no meaningful terms extracted from '{query_str}'")
            return []

        logger.info(
            f"Keyword search extracted terms={extracted_terms} query='{query_str}'"
        )

        patterns = {term: self._build_term_pattern(term) for term in extracted_terms}
        results_list: list[dict[str, Any]] = []
        text_search_succeeded = False

        # ------------------------------------------------------
        # 1. MONGODB TEXT SEARCH
        # ------------------------------------------------------
        try:
            text_query = " ".join(extracted_terms)
            cursor = self.collection.find(
                {
                    "$text": {"$search": text_query},
                    "status": "active",
                },
                {
                    "score": {"$meta": "textScore"}
                },
            ).sort(
                [("score", {"$meta": "textScore"})]
            ).limit(max(top_k * 2, 50))

            text_results = await cursor.to_list(length=max(top_k * 2, 50))
            logger.info(
                f"MongoDB $text search returned {len(text_results)} results"
            )

            if text_results:
                seen_ids: set[str] = set()
                for item in text_results:
                    chunk_id = str(item.pop("_id"))
                    if chunk_id in seen_ids:
                        continue
                    seen_ids.add(chunk_id)

                    item["id"] = chunk_id
                    item.setdefault("title", "")
                    item.setdefault("content", "")
                    item.setdefault("url", "")
                    item.setdefault("heading_path", [])

                    score = self._calculate_keyword_score(item, extracted_terms, patterns)
                    if score > 0.0:
                        item["score"] = score
                        results_list.append(item)

                if results_list:
                    text_search_succeeded = True

        except Exception as exc:
            logger.warning(
                f"MongoDB $text search failed or unavailable ({exc}). Falling back to regex keyword search."
            )

        # ------------------------------------------------------
        # 2. REGEX KEYWORD FALLBACK
        # ------------------------------------------------------
        if not text_search_succeeded:
            try:
                or_clauses = []
                for term in extracted_terms:
                    escaped_term = re.escape(term)
                    or_clauses.append({"content": {"$regex": escaped_term, "$options": "i"}})
                    or_clauses.append({"title": {"$regex": escaped_term, "$options": "i"}})
                    or_clauses.append({"heading_path": {"$regex": escaped_term, "$options": "i"}})

                query = {
                    "status": "active",
                    "$or": or_clauses,
                }

                candidate_limit = max(top_k * 10, 200)
                cursor = self.collection.find(query).limit(candidate_limit)
                raw_fallback_results = await cursor.to_list(length=candidate_limit)

                logger.info(
                    f"MongoDB regex fallback returned {len(raw_fallback_results)} raw candidates for {len(extracted_terms)} terms"
                )

                seen_ids = set()
                for item in raw_fallback_results:
                    chunk_id = str(item.pop("_id"))
                    if chunk_id in seen_ids:
                        continue
                    seen_ids.add(chunk_id)

                    item["id"] = chunk_id
                    item.setdefault("title", "")
                    item.setdefault("content", "")
                    item.setdefault("url", "")
                    item.setdefault("heading_path", [])

                    score = self._calculate_keyword_score(item, extracted_terms, patterns)
                    if score > 0.0:
                        item["score"] = score
                        results_list.append(item)

            except Exception as exc:
                logger.error(
                    f"Regex keyword fallback search failed: {exc}"
                )
                return []

        # ------------------------------------------------------
        # 3. RANK & LOG RESULTS
        # ------------------------------------------------------
        results_list.sort(key=lambda x: x["score"], reverse=True)
        final_results = results_list[:top_k]

        top_title = final_results[0].get("title", "N/A") if final_results else "None"
        top_score = final_results[0].get("score", 0.0) if final_results else 0.0

        logger.info(
            f"Keyword search completed | query='{query_str}' | extracted_terms={extracted_terms} | "
            f"method={'$text' if text_search_succeeded else 'regex_fallback'} | total_candidates={len(results_list)} | final_count={len(final_results)} | "
            f"top_title='{top_title}' | top_score={top_score:.4f}"
        )

        return final_results

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

    async def mark_stale_runs(
        self,
        stale_threshold_seconds: float = 86400.0,
    ) -> int:
        """
        Mark runs stuck in 'running' status older than threshold as 'failed'.
        """
        cutoff = datetime.now(UTC) - timedelta(seconds=stale_threshold_seconds)
        try:
            result = await self.collection.update_many(
                {
                    "status": "running",
                    "started_at": {"$lt": cutoff},
                },
                {
                    "$set": {
                        "status": "failed",
                        "finished_at": datetime.now(UTC),
                        "errors": [
                            {
                                "url": "system",
                                "error": "Crawl run timed out or interrupted (stale run cleanup)",
                                "timestamp": datetime.now(UTC),
                            }
                        ],
                    }
                },
            )
            if result.modified_count > 0:
                logger.warning(
                    f"Marked {result.modified_count} stale crawl runs as failed."
                )
            return result.modified_count
        except Exception as exc:
            logger.error(f"Failed to cleanup stale crawl runs: {exc}")
            return 0


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

    await chunk_coll.create_index(
        [
            ("status", 1),
            ("embedding_model", 1),
            ("embedding_dimensions", 1),
        ]
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