from typing import Optional
from app.core.config import settings
from app.core.logging import logger
from app.rag.models import RAGResult
from app.rag.retrieval.context_builder import ContextBuilder
from app.rag.retrieval.hybrid_search import HybridSearch
from app.rag.retrieval.query_rewriter import QueryRewriter
from app.rag.retrieval.reranker import BaseReranker, FusionReranker
from app.rag.validation.relevance import WACRelevanceGate


class RAGService:
    """High-level RAG retrieval service orchestrating relevance gating, search, reranking, and context building."""

    def __init__(
        self,
        hybrid_search: Optional[HybridSearch] = None,
        reranker: Optional[BaseReranker] = None,
        min_relevance_score: Optional[float] = None
    ) -> None:
        self.hybrid_search = hybrid_search or HybridSearch()
        self.reranker = reranker or FusionReranker()
        self.min_relevance_score = min_relevance_score if min_relevance_score is not None else settings.RAG_MIN_RELEVANCE_SCORE

    async def get_grounded_context(
        self,
        user_message: str,
        conversation_history: str = ""
    ) -> RAGResult:
        """
        Execute full RAG pipeline:
        1. Evaluate WAC Relevance Gate
        2. Rewrite conversational query
        3. Perform hybrid vector + keyword search
        4. Rerank retrieved chunks
        5. Apply min relevance threshold
        6. Build context block and source citations
        """
        if not settings.RAG_ENABLED:
            return RAGResult(is_relevant=True, has_context=False, context="", sources=[], retrieval_score=0.0)

        # 1. WAC Relevance Gate
        is_wac_related, refusal = WACRelevanceGate.evaluate(user_message)
        if not is_wac_related:
            logger.info(f"RAG Relevance Gate: Query rejected as out-of-domain ('{user_message}')")
            return RAGResult(
                is_relevant=False,
                has_context=False,
                context="",
                sources=[],
                retrieval_score=0.0,
                refusal_reason=refusal
            )

        # 2. Query Rewriting
        rewritten_query = QueryRewriter.rewrite(user_message, conversation_history)

        # 3. Hybrid Search
        retrieved_chunks = await self.hybrid_search.search(rewritten_query, top_k=settings.RAG_TOP_K_VECTOR)

        if not retrieved_chunks:
            logger.info(f"RAG Retrieval: No chunks retrieved for query '{rewritten_query}'")
            return RAGResult(
                is_relevant=True,
                has_context=False,
                context="",
                sources=[],
                retrieval_score=0.0,
                refusal_reason="I couldn't find reliable information about that in WAC's current knowledge base."
            )

        # 4. Reranking
        reranked_chunks = await self.reranker.rerank(rewritten_query, retrieved_chunks, top_k=settings.RAG_TOP_K_FINAL)
        top_score = reranked_chunks[0].score if reranked_chunks else 0.0

        # 5. Relevance Threshold Check
        if top_score < self.min_relevance_score:
            logger.info(f"RAG Threshold: Top retrieval score ({top_score:.2f}) below min threshold ({self.min_relevance_score})")
            return RAGResult(
                is_relevant=True,
                has_context=False,
                context="",
                sources=[],
                retrieval_score=top_score,
                refusal_reason="I couldn't find reliable information about that in WAC's current knowledge base."
            )

        # 6. Context Building
        context_str, sources = ContextBuilder.build_context_and_sources(reranked_chunks)

        logger.info(f"RAG Context Built: {len(reranked_chunks)} chunks, top score={top_score:.2f}")

        return RAGResult(
            is_relevant=True,
            has_context=True,
            context=context_str,
            sources=sources,
            retrieval_score=top_score
        )
