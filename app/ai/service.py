from typing import Optional
from app.ai.factory import ProviderFactory
from app.ai.schemas import AIRequest, AIResponse, AIUsage
from app.core.config import settings
from app.prompts.prompt_builder import PromptBuilder
from app.prompts.system_prompt import SYSTEM_PROMPT
from app.rag.models import RAGResult
from app.services.company_service import CompanyService
from app.services.rag_service import RAGService


class AIService:

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

        selected_provider = provider or settings.DEFAULT_PROVIDER

        # Execute RAG Retrieval Pipeline
        rag_result = await self.rag_service.get_grounded_context(
            user_message=message,
            conversation_history=history,
        )

        # 1. Check out-of-domain refusal from relevance gate
        if not rag_result.is_relevant and rag_result.refusal_reason:
            usage = AIUsage(provider=selected_provider, model="none", request_type="text")
            return AIResponse(content=rag_result.refusal_reason, usage=usage), rag_result

        # 2. Dynamic context selection: RAG context if available, otherwise CompanyService context fallback
        company_context = rag_result.context if rag_result.has_context else self.company_service.get_context(message)

        # 3. Build prompt using existing PromptBuilder architecture
        request = AIRequest(
            user_message=message,
            conversation_history=history,
            company_context=company_context,
            system_prompt=SYSTEM_PROMPT,
        )

        prompt = PromptBuilder.build(request)

        # 4. Generate AI response from selected provider (Gemini / Groq)
        ai_provider = ProviderFactory.get_provider(selected_provider)
        response = await ai_provider.generate(prompt)

        return response, rag_result