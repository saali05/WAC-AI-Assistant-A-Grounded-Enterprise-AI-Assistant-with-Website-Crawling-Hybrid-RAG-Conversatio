from typing import Optional, Any

from app.core.config import settings
from app.core.logging import logger
from app.rag.embeddings.providers import (
    BaseEmbeddingProvider,
    GeminiEmbeddingProvider,
    LocalEmbeddingProvider,
    EmbeddingProviderFactory,
)


class EmbeddingService:
    """
    Unified Embedding Service supporting both Gemini and Local BGE providers.

    Independent from AI generation provider settings.
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        dimensions: Optional[int] = None,
        api_key: Optional[str] = None,
        max_retry_delay: Optional[float] = None,
        default_retries: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        self.provider_instance: BaseEmbeddingProvider = EmbeddingProviderFactory.create(
            provider=provider,
            model=model,
            dimensions=dimensions,
            api_key=api_key,
            max_retry_delay=max_retry_delay,
            default_retries=default_retries,
            **kwargs,
        )

        self.model = self.provider_instance.model
        self.dimensions = self.provider_instance.dimensions
        self.provider_name = self.provider_instance.provider_name

        logger.info(
            f"EmbeddingService initialized | "
            f"provider={self.provider_name} | "
            f"model={self.model} | "
            f"dimensions={self.dimensions}"
        )

    @property
    def client(self) -> Any:
        """Access underlying client for Gemini provider when available."""
        if hasattr(self.provider_instance, "client"):
            return self.provider_instance.client
        return None

    @property
    def _client(self) -> Any:
        """Access internal client instance."""
        if hasattr(self.provider_instance, "_client"):
            return self.provider_instance._client
        return None

    @_client.setter
    def _client(self, value: Any) -> None:
        """Set internal client instance (used in tests/mocking)."""
        if hasattr(self.provider_instance, "_client"):
            self.provider_instance._client = value

    @property
    def max_retry_delay(self) -> float:
        return getattr(self.provider_instance, "max_retry_delay", 60.0)

    @property
    def default_retries(self) -> int:
        return getattr(self.provider_instance, "default_retries", 3)

    # Static helpers preserved for backward compatibility
    is_rate_limit_error = staticmethod(GeminiEmbeddingProvider.is_rate_limit_error)
    extract_retry_delay = staticmethod(GeminiEmbeddingProvider.extract_retry_delay)

    async def get_embedding(
        self,
        text: str,
        retries: Optional[int] = None,
    ) -> list[float]:
        """Generate embedding vector for a single text."""
        return await self.provider_instance.get_embedding(text, retries=retries)

    async def get_batch_embeddings(
        self,
        texts: list[str],
        retries: Optional[int] = None,
    ) -> list[list[float]]:
        """Generate embedding vectors for a batch of texts."""
        return await self.provider_instance.get_batch_embeddings(texts, retries=retries)

    def _validate_embeddings(
        self,
        embeddings: list[list[float]],
    ) -> None:
        """Validate embedding dimensions."""
        self.provider_instance._validate_embeddings(embeddings)