from app.ai.factory import ProviderFactory
from app.core.config import settings
from app.prompts.prompt_builder import PromptBuilder
from app.prompts.system_prompt import SYSTEM_PROMPT
from app.services.company_service import CompanyService
from app.ai.schemas import AIRequest, AIResponse


class AIService:

    @property
    def company_service(self):
        return CompanyService()

    async def chat(
        self,
        message: str,
        history: str = "",
        provider: str | None = None,
    ) -> AIResponse:

        selected_provider = provider or settings.DEFAULT_PROVIDER

        company_context = self.company_service.get_context(
            message
        )

        request = AIRequest(
            user_message=message,
            conversation_history=history,
            company_context=company_context,
            system_prompt=SYSTEM_PROMPT,
        )

        prompt = PromptBuilder.build(request)

        ai_provider = ProviderFactory.get_provider(selected_provider)

        return await ai_provider.generate(prompt)