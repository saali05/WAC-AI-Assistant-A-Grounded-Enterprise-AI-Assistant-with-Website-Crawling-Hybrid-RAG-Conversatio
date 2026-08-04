from abc import ABC, abstractmethod


class BaseAIProvider(ABC):
    """
    Abstract base class for all AI providers.
    """

    @abstractmethod
    async def generate(self, message: str) -> str:
        """
        Generate a response from the AI provider.
        """
        pass