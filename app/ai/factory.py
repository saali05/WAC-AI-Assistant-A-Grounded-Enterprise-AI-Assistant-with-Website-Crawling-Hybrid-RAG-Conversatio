from app.ai.providers.base import BaseAIProvider
from app.ai.providers.gemini import GeminiProvider
from app.ai.providers.groq import GroqProvider


class ProviderFactory:
    """
    Factory responsible for creating AI provider instances.
    """

    _providers = {
        "gemini": GeminiProvider,
        "groq": GroqProvider,
    }

    @classmethod
    def get_provider(cls, provider_name: str) -> BaseAIProvider:
        provider_class = cls._providers.get(provider_name.lower())

        if provider_class is None:
            raise ValueError(f"Unsupported provider: {provider_name}")

        return provider_class()