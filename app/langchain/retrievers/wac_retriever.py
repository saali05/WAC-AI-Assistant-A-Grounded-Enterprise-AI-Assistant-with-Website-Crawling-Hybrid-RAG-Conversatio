from typing import Any, List, Optional, Tuple
from pydantic import ConfigDict, Field
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from app.core.config import settings
from app.core.logging import logger
from app.rag.models import RAGResult, RetrievedChunk
from app.services.rag_service import RAGService


class WACRetriever(BaseRetriever):
    """
    Canonical LangChain Retriever adapter for the WAC Hybrid RAG system.

    Connects LangChain directly to the authoritative WAC RAGService,
    preserving query rewriting, retrieval expansion, hybrid vector + keyword search,
    reciprocal rank fusion, intent-aware reranking, and evidence sufficiency.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    rag_service: Any = Field(default=None)
    top_k: int = Field(default_factory=lambda: settings.RAG_TOP_K_FINAL)

    def __init__(
        self,
        rag_service: Optional[RAGService] = None,
        hybrid_search: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        if rag_service is None:
            if hybrid_search is not None:
                rag_service = RAGService(hybrid_search=hybrid_search)
            else:
                rag_service = RAGService()
        super().__init__(rag_service=rag_service, **kwargs)

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager=None,
    ) -> List[Document]:
        """Synchronous retrieval is not supported because WAC RAG is async."""
        raise NotImplementedError(
            "WACRetriever is asynchronous. Use retriever.ainvoke() instead."
        )

    def _convert_chunk_to_document(self, chunk: RetrievedChunk) -> Document:
        """Convert a canonical RetrievedChunk into a LangChain Document with full metadata."""
        heading_val = (
            " > ".join(chunk.heading_path)
            if isinstance(chunk.heading_path, list)
            else str(chunk.heading_path or "")
        )

        metadata = {
            "id": chunk.chunk_id,
            "chunk_id": chunk.chunk_id,
            "document_id": chunk.document_id,
            "title": chunk.title,
            "heading": heading_val,
            "heading_path": chunk.heading_path,
            "url": chunk.url,
            "canonical_url": chunk.canonical_url,
            "score": chunk.score,
            "vector_score": chunk.vector_score,
            "keyword_score": chunk.keyword_score,
            "fusion_score": chunk.fusion_score,
            "reranked_score": getattr(chunk, "reranked_score", None),
            "retrieval_confidence": getattr(chunk, "retrieval_confidence", None),
            "created_at": getattr(chunk, "created_at", None),
            "updated_at": getattr(chunk, "updated_at", None),
        }

        return Document(
            page_content=chunk.content or "",
            metadata=metadata,
        )

    async def retrieve_with_rag_result(
        self,
        query: str,
        conversation_history: str = "",
    ) -> Tuple[List[Document], RAGResult]:
        """
        Execute full canonical RAG pipeline and return both LangChain Documents
        and the full RAGResult object (containing sufficiency and score metadata).
        """
        if not query or not query.strip():
            logger.warning("LangChain WACRetriever received an empty query.")
            empty_result = RAGResult(
                is_relevant=True,
                has_context=False,
                evidence_sufficient=False,
                context="",
                sources=[],
                retrieval_score=0.0,
                refusal_reason="No search query was provided.",
            )
            return [], empty_result

        logger.info(f"LangChain WACRetriever search started | query='{query}'")

        rag_result: RAGResult = await self.rag_service.get_grounded_context(
            user_message=query,
            conversation_history=conversation_history,
        )

        documents: List[Document] = []
        if rag_result.retrieved_chunks:
            for chunk in rag_result.retrieved_chunks:
                documents.append(self._convert_chunk_to_document(chunk))
        elif rag_result.sources:
            # Fallback if only source citations are present
            for source in rag_result.sources:
                doc = Document(
                    page_content=source.title or "",
                    metadata={
                        "id": source.url,
                        "title": source.title,
                        "url": source.url,
                        "heading": source.heading or "",
                        "score": source.score,
                        "canonical_url": source.canonical_url,
                    }
                )
                documents.append(doc)

        logger.info(
            f"LangChain WACRetriever search completed | query='{query}' | "
            f"has_context={rag_result.has_context} | evidence_sufficient={rag_result.evidence_sufficient} | "
            f"confidence={rag_result.retrieval_score:.4f} | documents={len(documents)} | sources={len(rag_result.sources)}"
        )

        return documents, rag_result

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager=None,
    ) -> List[Document]:
        """LangChain LCEL standard async retrieval interface."""
        documents, _ = await self.retrieve_with_rag_result(query=query)
        return documents