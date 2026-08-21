from abc import ABC, abstractmethod
from app.ai.schemas import AIResponse


class BaseAIProvider(ABC):
    """
    Abstract base class for all AI providers.
    """

    @abstractmethod
    async def generate(self, message: str) -> AIResponse:
        """
        Generate a normalized AIResponse from the AI provider.
        """
        pass