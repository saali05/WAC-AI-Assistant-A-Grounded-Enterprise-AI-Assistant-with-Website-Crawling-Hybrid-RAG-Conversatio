from typing import Optional

from app.core.config import settings
from app.core.logging import logger
from app.rag.models import RetrievedChunk
from app.rag.retrieval.keyword_search import KeywordSearch
from app.rag.retrieval.vector_search import VectorSearch


class HybridSearch:
    """
    Hybrid retrieval combining:

        Vector Search
              +
        Keyword Search
              ↓
        Reciprocal Rank Fusion (RRF)

    RRF combines the ranking positions of the two
    retrieval systems instead of directly comparing
    their raw scores.

    This is important because vector similarity and
    keyword relevance scores have different scales.
    """

    def __init__(
        self,
        vector_search: Optional[VectorSearch] = None,
        keyword_search: Optional[KeywordSearch] = None,
        vector_weight: Optional[float] = None,
        keyword_weight: Optional[float] = None,
        rrf_k: int = 60,
    ) -> None:

        self.vector_search = (
            vector_search or VectorSearch()
        )

        self.keyword_search = (
            keyword_search or KeywordSearch()
        )

        self.vector_weight = (
            vector_weight
            if vector_weight is not None
            else settings.RAG_VECTOR_WEIGHT
        )

        self.keyword_weight = (
            keyword_weight
            if keyword_weight is not None
            else settings.RAG_KEYWORD_WEIGHT
        )

        self.rrf_k = rrf_k

    async def search(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> list[RetrievedChunk]:
        # """
        # Perform vector + keyword retrieval and combine
        # their rankings using weighted Reciprocal Rank Fusion.

        # Pipeline:

        #     Query
        #       ↓
        # ┌───────────────┐
        # │               │
        # Vector       Keyword
        # Search        Search
        # │               │
        # └───────┬───────┘
        #         ↓
        #       RRF
        #         ↓
        #  Combined ranking
        #         ↓
        #   RetrievedChunk
        # """

        vector_k = (
            top_k
            if top_k is not None
            else settings.RAG_TOP_K_VECTOR
        )
        keyword_k = settings.RAG_TOP_K_KEYWORD
        final_k = settings.RAG_TOP_K_FINAL


        if not query or not query.strip():
            return []

        logger.info(
            f"HybridSearch started | "
            f"query={query} | "
            f"vector_k={vector_k} | "
            f"keyword_k={keyword_k} | "
            f"final_k={final_k} | "
            f"vector_weight={self.vector_weight} | "
            f"keyword_weight={self.keyword_weight}"
        )

        # ======================================================
        # 1. VECTOR SEARCH
        # ======================================================

        vector_results = await self.vector_search.search(
            query=query,
            top_k=vector_k,
        )

        logger.info(
            f"HybridSearch vector results: "
            f"{len(vector_results)}"
        )

        # ======================================================
        # 2. KEYWORD SEARCH
        # ======================================================

        keyword_results = await self.keyword_search.search(
            query=query,
            top_k=keyword_k,
        )

        logger.info(
            f"HybridSearch keyword results: "
            f"{len(keyword_results)}"
        )

        # ======================================================
        # 3. STORE ALL UNIQUE CHUNKS
        # ======================================================

        chunk_dict: dict[str, dict] = {}

        vector_scores: dict[str, float] = {}
        keyword_scores: dict[str, float] = {}

        rrf_scores: dict[str, float] = {}

        # ======================================================
        # 4. PROCESS VECTOR RANKING
        # ======================================================

        for rank, item in enumerate(
            vector_results,
            start=1,
        ):

            chunk_id = item.get("id")

            if not chunk_id:
                continue

            chunk_dict[chunk_id] = item

            vector_scores[chunk_id] = (
                float(item.get("score", 0.0))
            )

            rrf_contribution = (
                self.vector_weight
                / (self.rrf_k + rank)
            )

            rrf_scores[chunk_id] = (
                rrf_scores.get(chunk_id, 0.0)
                + rrf_contribution
            )

        # ======================================================
        # 5. PROCESS KEYWORD RANKING
        # ======================================================

        for rank, item in enumerate(
            keyword_results,
            start=1,
        ):

            chunk_id = item.get("id")

            if not chunk_id:
                continue

            # Keep the existing chunk object if it was
            # already found by vector search.
            if chunk_id not in chunk_dict:
                chunk_dict[chunk_id] = item

            keyword_scores[chunk_id] = (
                float(item.get("score", 0.0))
            )

            rrf_contribution = (
                self.keyword_weight
                / (self.rrf_k + rank)
            )

            rrf_scores[chunk_id] = (
                rrf_scores.get(chunk_id, 0.0)
                + rrf_contribution
            )

        # ======================================================
        # 6. BUILD FUSED RESULTS
        # ======================================================

        fused_chunks: list[RetrievedChunk] = []

        for chunk_id, item in chunk_dict.items():

            vector_score = vector_scores.get(
                chunk_id,
                0.0,
            )

            keyword_score = keyword_scores.get(
                chunk_id,
                0.0,
            )

            fusion_score = rrf_scores.get(
                chunk_id,
                0.0,
            )

            fused_chunks.append(
                RetrievedChunk(
                    chunk_id=chunk_id,

                    document_id=item.get(
                        "document_id",
                        "",
                    ),

                    content=item.get(
                        "content",
                        "",
                    ),

                    title=item.get(
                        "title",
                        "",
                    ),

                    heading_path=item.get(
                        "heading_path",
                        [],
                    ),

                    url=item.get(
                        "url",
                        "",
                    ),

                    canonical_url=item.get(
                        "canonical_url",
                        "",
                    ),

                    score=fusion_score,

                    vector_score=vector_score,

                    keyword_score=keyword_score,

                    fusion_score=fusion_score,

                    created_at=item.get("created_at"),

                    updated_at=item.get("updated_at") or item.get("last_crawled_at"),
                )
            )

        # ======================================================
        # 7. SORT BY RRF SCORE
        # ======================================================

        fused_chunks.sort(
            key=lambda x: x.score,
            reverse=True,
        )

        final_results = fused_chunks[:final_k]

        logger.info(
            f"HybridSearch completed | "
            f"vector={len(vector_results)} | "
            f"keyword={len(keyword_results)} | "
            f"unique={len(fused_chunks)} | "
            f"final={len(final_results)}"
        )

        return final_results