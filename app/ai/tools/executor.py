from typing import Any

from app.core.logging import logger
from app.rag.models import RAGResult
from app.services.rag_service import RAGService


class ToolExecutor:
    """
    Executes functions requested by Gemini.
    """

    def __init__(
        self,
        rag_service: RAGService | None = None,
    ) -> None:
        self.rag_service = rag_service or RAGService()

    async def search_wac_knowledge(
        self,
        query: str,
        conversation_history: str = "",
    ) -> tuple[dict[str, Any], RAGResult]:

        logger.info(
            "Tool execution started | tool=search_wac_knowledge | query=%s",
            query,
        )

        rag_result = await self.rag_service.get_grounded_context(
            user_message=query,
            conversation_history=conversation_history,
        )

        result = {
            "query": query,
            "is_relevant": rag_result.is_relevant,
            "has_context": rag_result.has_context,
            "context": rag_result.context,
            "retrieval_score": rag_result.retrieval_score,
            "sources": rag_result.sources,
            "message": (
                "Reliable WAC knowledge was retrieved."
                if rag_result.has_context
                else (
                    rag_result.refusal_reason
                    or "No reliable WAC information was found."
                )
            ),
        }

        logger.info(
            "Tool execution completed | tool=search_wac_knowledge | "
            "has_context=%s | score=%s",
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

        if name == "search_wac_knowledge":
            return await self.search_wac_knowledge(
                query=str(arguments.get("query", "")),
                conversation_history=conversation_history,
            )

        raise ValueError(f"Unknown AI tool: {name}")