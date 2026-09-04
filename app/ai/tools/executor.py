from typing import Any

from app.core.logging import logger
from app.rag.models import RAGResult
from app.langchain.retrievers.wac_retriever import WACRetriever


class ToolExecutor:
    """
    Executes functions requested by Gemini.

    The WAC knowledge tool now uses the LangChain
    WACRetriever, which internally connects to the
    existing WAC HybridSearch system.
    """

    def __init__(
        self,
        retriever: WACRetriever | None = None,
    ) -> None:

        self.retriever = retriever or WACRetriever()

    async def search_wac_knowledge(
        self,
        query: str,
        conversation_history: str = "",
    ) -> tuple[dict[str, Any], RAGResult]:

        logger.info(
            "Tool execution started | "
            "tool=search_wac_knowledge | query=%s",
            query,
        )

        # --------------------------------------------------
        # Validate query
        # --------------------------------------------------

        if not query or not query.strip():

            rag_result = RAGResult(
                is_relevant=True,
                has_context=False,
                context="",
                sources=[],
                retrieval_score=0.0,
                refusal_reason=(
                    "No search query was provided."
                ),
            )

            return (
                {
                    "query": query,
                    "is_relevant": True,
                    "has_context": False,
                    "context": "",
                    "retrieval_score": 0.0,
                    "sources": [],
                    "message": "No search query was provided.",
                },
                rag_result,
            )

        # --------------------------------------------------
        # LangChain Retriever
        # --------------------------------------------------

        documents = await self.retriever.ainvoke(
            query.strip()
        )

        logger.info(
            "LangChain WACRetriever returned %s documents",
            len(documents),
        )

        # --------------------------------------------------
        # No documents
        # --------------------------------------------------

        if not documents:

            rag_result = RAGResult(
                is_relevant=True,
                has_context=False,
                context="",
                sources=[],
                retrieval_score=0.0,
                refusal_reason=(
                    "I couldn't find reliable information "
                    "about that in WAC's current knowledge base."
                ),
            )

            return (
                {
                    "query": query,
                    "is_relevant": True,
                    "has_context": False,
                    "context": "",
                    "retrieval_score": 0.0,
                    "sources": [],
                    "message": rag_result.refusal_reason,
                },
                rag_result,
            )

        # --------------------------------------------------
        # Build context from LangChain Documents
        # --------------------------------------------------

        context_parts: list[str] = []
        sources: list[dict[str, Any]] = []

        scores: list[float] = []

        for index, document in enumerate(
            documents,
            start=1,
        ):

            metadata = document.metadata

            title = metadata.get(
                "title",
                "",
            )

            url = metadata.get(
                "url",
                "",
            )

            score = metadata.get(
                "score",
                0.0,
            )

            if isinstance(score, (int, float)):
                scores.append(float(score))

            # ----------------------------------------------
            # Context
            # ----------------------------------------------

            context_parts.append(
                f"""
                Source {index}
                Title: {title}
                URL: {url}

                Content:
                {document.page_content}
                """.strip()
                            )

            # ----------------------------------------------
            # Source information
            # ----------------------------------------------

            sources.append(
                {
                    "title": title,
                    "url": url,
                    "score": score,
                }
            )

        context = "\n\n".join(
            context_parts
        )

        top_score = (
            max(scores)
            if scores
            else 0.0
        )

        # --------------------------------------------------
        # Build RAGResult
        # --------------------------------------------------

        rag_result = RAGResult(
            is_relevant=True,
            has_context=True,
            context=context,
            sources=sources,
            retrieval_score=top_score,
        )

        # --------------------------------------------------
        # Tool response sent back to Gemini
        # --------------------------------------------------

        result = {
            "query": query,
            "is_relevant": True,
            "has_context": True,
            "context": context,
            "retrieval_score": top_score,
            "sources": sources,
            "message": (
                "Reliable WAC knowledge was retrieved."
            ),
        }

        logger.info(
            "Tool execution completed | "
            "tool=search_wac_knowledge | "
            "documents=%s | has_context=%s | score=%s",
            len(documents),
            rag_result.has_context,
            rag_result.retrieval_score,
        )

        return result, rag_result

    async def execute(
        self,
        name: str,
        arguments: dict[str, Any],
        conversation_history: str = "",
    ) -> tuple[dict[str, Any], RAGResult | None]:

        # --------------------------------------------------
        # Gemini Function Call
        # --------------------------------------------------

        if name == "search_wac_knowledge":

            return await self.search_wac_knowledge(
                query=str(
                    arguments.get(
                        "query",
                        "",
                    )
                ),
                conversation_history=conversation_history,
            )

        raise ValueError(
            f"Unknown AI tool: {name}"
        )