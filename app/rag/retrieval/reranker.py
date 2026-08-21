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
        top_k: Optional[int] = None
    ) -> list[RetrievedChunk]:
        """Rerank retrieved chunks for given query."""
        pass


class FusionReranker(BaseReranker):
    """Score fusion and structural title/heading boosting reranker."""

    async def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: Optional[int] = None
    ) -> list[RetrievedChunk]:
        k = top_k or settings.RAG_TOP_K_FINAL
        if not chunks:
            return []

        query_terms = [t.lower() for t in query.split() if len(t.strip()) > 2]

        reranked: list[RetrievedChunk] = []
        for chunk in chunks:
            boost = 1.0

            # Title & heading match boost
            title_lower = chunk.title.lower()
            heading_str = " ".join(chunk.heading_path).lower()

            for term in query_terms:
                if term in title_lower:
                    boost += 0.15
                if term in heading_str:
                    boost += 0.10

            chunk.score = min(round(chunk.score * boost, 4), 1.0)
            reranked.append(chunk)

        reranked.sort(key=lambda x: x.score, reverse=True)
        return reranked[:k]
