from app.ai.factory import ProviderFactory
from app.core.config import settings


class AIService:
    """
    Main service responsible for interacting with AI providers.
    """

    async def chat(
        self,
        message: str,
        provider: str | None = None,
    ) -> str:

        selected_provider = provider or settings.DEFAULT_PROVIDER

        ai_provider = ProviderFactory.get_provider(selected_provider)

        return await ai_provider.generate(message)