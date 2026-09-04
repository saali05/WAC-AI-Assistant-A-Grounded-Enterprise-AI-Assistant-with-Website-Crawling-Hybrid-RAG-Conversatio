from typing import Any, List, Optional
from pydantic import ConfigDict, Field
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from app.core.config import settings
from app.core.logging import logger
from app.rag.retrieval.hybrid_search import HybridSearch


class WACRetriever(BaseRetriever):
    """
    LangChain adapter for the existing WAC Hybrid RAG system.

    LangChain sees this class as a Retriever.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    hybrid_search: Any = Field(default_factory=HybridSearch)
    top_k: int = Field(default_factory=lambda: settings.RAG_TOP_K_VECTOR)

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager=None,
    ) -> List[Document]:
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
    ) -> List[Document]:
        """
        Execute the existing WAC HybridSearch and convert
        RetrievedChunk objects into LangChain Documents.
        """
        if not query or not query.strip():
            logger.warning("LangChain WACRetriever received an empty query.")
            return []

        logger.info("LangChain WACRetriever search started | query=%s", query)

        try:
            results = await self.hybrid_search.search(
                query=query,
                top_k=self.top_k,
            )

            documents: List[Document] = []
            for result in results:
                if not result.content:
                    continue

                heading_val = (
                    " > ".join(result.heading_path)
                    if isinstance(result.heading_path, list)
                    else str(result.heading_path or "")
                )

                metadata = {
                    "chunk_id": result.chunk_id,
                    "document_id": result.document_id,
                    "title": result.title,
                    "heading": heading_val,
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
                "LangChain WACRetriever search completed | query=%s | documents=%d",
                query,
                len(documents),
            )
            return documents

        except Exception as exc:
            logger.exception(
                "LangChain WACRetriever failed | query=%s | error=%s",
                query,
                exc,
            )
            raise