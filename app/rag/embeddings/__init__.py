"""
Vector embedding providers and unified embedding service.
"""

from app.rag.embeddings.embedding_service import EmbeddingService
from app.rag.embeddings.providers import (
    BaseEmbeddingProvider,
    GeminiEmbeddingProvider,
    LocalEmbeddingProvider,
    EmbeddingProviderFactory,
)

__all__ = [
    "EmbeddingService",
    "BaseEmbeddingProvider",
    "GeminiEmbeddingProvider",
    "LocalEmbeddingProvider",
    "EmbeddingProviderFactory",
]
