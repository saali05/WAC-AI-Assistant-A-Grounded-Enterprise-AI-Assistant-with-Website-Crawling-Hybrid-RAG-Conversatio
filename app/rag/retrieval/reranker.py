from abc import ABC, abstractmethod
from typing import Optional

from app.core.config import settings
from app.rag.models import RetrievedChunk


class BaseReranker(ABC):
    """Abstract reranker interface for modular reranking implementations."""

    @abstractmethod
    async def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: Optional[int] = None,
    ) -> list[RetrievedChunk]:
        """Rerank retrieved chunks for given query."""
        pass


class FusionReranker(BaseReranker):
    """
    Lightweight reranker that combines:

    1. RRF ranking score
    2. Title matching
    3. Heading matching
    4. Keyword matching in content
    """

    async def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: Optional[int] = None,
    ) -> list[RetrievedChunk]:

        k = top_k or settings.RAG_TOP_K_FINAL

        if not chunks:
            return []

        query_terms = [
            term.lower()
            for term in query.split()
            if len(term.strip()) > 2
        ]

        reranked: list[RetrievedChunk] = []

        for chunk in chunks:
            # Absolute base score from vector similarity if available, else normalized RRF score
            if chunk.vector_score is not None and chunk.vector_score > 0:
                base_score = chunk.vector_score
            else:
                base_score = min(1.0, (chunk.fusion_score or chunk.score or 0.0) * 61.0)

            title_lower = chunk.title.lower()

            heading_str = (
                " ".join(chunk.heading_path)
                .lower()
            )

            content_lower = chunk.content.lower()

            title_matches = 0
            heading_matches = 0
            content_matches = 0

            for term in query_terms:

                if term in title_lower:
                    title_matches += 1

                if term in heading_str:
                    heading_matches += 1

                if term in content_lower:
                    content_matches += 1

            # -----------------------------------------
            # Calculate structural/content boosts
            # -----------------------------------------

            title_boost = min(
                title_matches * 0.08,
                0.25,
            )

            heading_boost = min(
                heading_matches * 0.05,
                0.15,
            )

            content_boost = min(
                content_matches * 0.02,
                0.20,
            )

            # -----------------------------------------
            # Final reranking score
            # -----------------------------------------

            weighted_score = (
                (base_score * 0.70)
                + title_boost
                + heading_boost
                + content_boost
            )
            final_score = max(base_score, weighted_score)

            reranked_val = round(
                min(final_score, 1.0),
                4,
            )
            chunk.reranked_score = reranked_val
            chunk.score = reranked_val

            reranked.append(chunk)

        # ---------------------------------------------
        # Highest relevance first, with freshness tie-breaking
        # ---------------------------------------------

        reranked.sort(
            key=lambda x: (
                x.score,
                x.updated_at.timestamp() if x.updated_at else (x.created_at.timestamp() if x.created_at else 0)
            ),
            reverse=True,
        )

        return reranked[:k]