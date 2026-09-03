from typing import Optional

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from app.core.config import settings
from app.core.logging import logger
from app.rag.retrieval.hybrid_search import HybridSearch


class WACRetriever(BaseRetriever):
    """
    LangChain adapter for the existing WAC Hybrid RAG system.

    LangChain sees this class as a Retriever.

    Internally it continues using the existing WAC RAG pipeline:

        WACRetriever
             ↓
        HybridSearch
             ↓
        ┌───────────────┐
        │               │
    VectorSearch   KeywordSearch
        │               │
        └───────┬───────┘
                ↓
        Reciprocal Rank Fusion
                ↓
        RetrievedChunk
                ↓
        LangChain Document
    """

    top_k: int = settings.RAG_TOP_K_VECTOR

    def __init__(
        self,
        hybrid_search: Optional[HybridSearch] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        object.__setattr__(
            self,
            "hybrid_search",
            hybrid_search or HybridSearch(),
        )

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager=None,
    ) -> list[Document]:
        """
        Synchronous retrieval.

        The existing WAC RAG pipeline is asynchronous, so this
        adapter intentionally requires the async interface.
        """

        raise NotImplementedError(
            "WACRetriever is asynchronous. "
            "Use retriever.ainvoke() instead."
        )

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager=None,
    ) -> list[Document]:
        """
        Execute the existing WAC HybridSearch and convert
        RetrievedChunk objects into LangChain Documents.
        """

        if not query or not query.strip():
            logger.warning(
                "LangChain WACRetriever received an empty query."
            )
            return []

        logger.info(
            "LangChain WACRetriever search started | query=%s",
            query,
        )

        try:

            # --------------------------------------------------
            # Existing WAC Hybrid RAG
            # --------------------------------------------------

            results = await self.hybrid_search.search(
                query=query,
                top_k=self.top_k,
            )

            documents: list[Document] = []

            # --------------------------------------------------
            # Convert RetrievedChunk → LangChain Document
            # --------------------------------------------------

            for result in results:

                if not result.content:
                    continue

                metadata = {
                    "chunk_id": result.chunk_id,
                    "document_id": result.document_id,
                    "title": result.title,
                    "heading_path": result.heading_path,
                    "url": result.url,
                    "canonical_url": result.canonical_url,
                    "score": result.score,
                    "vector_score": result.vector_score,
                    "keyword_score": result.keyword_score,
                    "fusion_score": result.fusion_score,
                    "reranked_score": getattr(result, "reranked_score", None),
                    "retrieval_confidence": getattr(result, "retrieval_confidence", None),
                    "created_at": getattr(result, "created_at", None),
                    "updated_at": getattr(result, "updated_at", None),
                }

                documents.append(
                    Document(
                        page_content=result.content,
                        metadata=metadata,
                    )
                )

            logger.info(
                f"LangChain WACRetriever search completed | "
                f"query={query} | documents={len(documents)}",
            )

            return documents

        except Exception as exc:

            logger.exception(
                "LangChain WACRetriever failed | query=%s | error=%s",
                query,
                exc,
            )

            raise