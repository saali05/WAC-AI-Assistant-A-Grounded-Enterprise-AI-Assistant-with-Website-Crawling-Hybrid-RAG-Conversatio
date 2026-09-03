from typing import Optional

from app.ai.factory import ProviderFactory
from app.ai.schemas import AIRequest, AIResponse
from app.core.config import settings
from app.core.logging import logger
from app.prompts.prompt_builder import PromptBuilder
from app.prompts.system_prompt import SYSTEM_PROMPT
from app.rag.models import RAGResult
from app.services.company_service import CompanyService
from app.services.rag_service import RAGService


class AIService:
    """
    Main AI orchestration service enforcing Mandatory Grounded WAC RAG.

    Flow:
    1. Mandatory WAC Relevance & RAG Retrieval
    2. If Out-of-Domain -> Grounded Refusal
    3. If Weak / Missing Context -> Grounded Refusal (No Gemini guess)
    4. If Grounded Context Available -> Grounded Answer Generation (Gemini / Groq)
    """

    @property
    def company_service(self) -> CompanyService:
        return CompanyService()

    @property
    def rag_service(self) -> RAGService:
        return RAGService()

    async def chat(
        self,
        message: str,
        history: str = "",
        provider: Optional[str] = None,
    ) -> tuple[AIResponse, RAGResult]:

        selected_provider = (
            provider or settings.DEFAULT_PROVIDER
        )

        logger.info(
            f"AIService chat started | "
            f"provider={selected_provider} | "
            f"query='{message}'"
        )

        # --------------------------------------------------
        # 1. MANDATORY RAG RETRIEVAL & DOMAIN CHECK
        # --------------------------------------------------

        rag_result = await self.rag_service.get_grounded_context(
            user_message=message,
            conversation_history=history,
        )

        # --------------------------------------------------
        # 2. OUT-OF-DOMAIN REFUSAL
        # --------------------------------------------------

        if not rag_result.is_relevant:
            logger.info("Mandatory RAG: Out-of-domain query rejected")
            refusal_text = (
                rag_result.refusal_reason
                or (
                    "I'm the WAC AI Assistant, specifically designed to help "
                    "with Web and Craft's services, technologies, solutions, "
                    "company information, and career opportunities."
                )
            )
            return (
                AIResponse(content=refusal_text),
                rag_result,
            )

        # --------------------------------------------------
        # 3. MISSING / WEAK EVIDENCE REFUSAL
        # --------------------------------------------------

        if not rag_result.has_context:
            logger.info("Mandatory RAG: Evidence unavailable or low confidence; returning grounded refusal")
            refusal_text = (
                rag_result.refusal_reason
                or (
                    "I couldn't find reliable information about that "
                    "in WAC's current knowledge base."
                )
            )
            return (
                AIResponse(content=refusal_text),
                rag_result,
            )

        # --------------------------------------------------
        # 4. GROUNDED ANSWER GENERATION (Gemini / Groq)
        # --------------------------------------------------

        logger.info(
            f"Mandatory RAG: Reliable evidence retrieved (confidence={rag_result.retrieval_score:.4f}). Generating grounded response."
        )

        request = AIRequest(
            user_message=message,
            conversation_history=history,
            company_context=rag_result.context,
            system_prompt=SYSTEM_PROMPT,
        )

        prompt = PromptBuilder.build(request)

        ai_provider = ProviderFactory.get_provider(
            selected_provider
        )

        response = await ai_provider.generate(
            prompt
        )

        return response, rag_result