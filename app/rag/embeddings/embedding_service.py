import asyncio
from typing import Optional

from google import genai

from app.core.config import settings
from app.core.logging import logger
from app.rag.exceptions import EmbeddingException


class EmbeddingService:
    """Gemini Embedding Service for document chunks and user queries."""

    def __init__(
        self,
        model: Optional[str] = None,
        dimensions: Optional[int] = None,
        api_key: Optional[str] = None,
    ) -> None:
        self.model = model or settings.RAG_EMBEDDING_MODEL
        self.dimensions = dimensions or settings.RAG_EMBEDDING_DIMENSIONS
        self.api_key = api_key or settings.GEMINI_API_KEY
        self._client: Optional[genai.Client] = None

    @property
    def client(self) -> genai.Client:
        if self._client is None:
            if not self.api_key:
                raise EmbeddingException(
                    "GEMINI_API_KEY is not configured for embedding service."
                )

            self._client = genai.Client(
                api_key=self.api_key
            )

        return self._client

    async def get_embedding(
        self,
        text: str,
        retries: int = 2,
    ) -> list[float]:

        embeddings = await self.get_batch_embeddings(
            [text],
            retries=retries,
        )

        if embeddings:
            return embeddings[0]

        return [0.0] * self.dimensions

    async def get_batch_embeddings(
        self,
        texts: list[str],
        retries: int = 2,
    ) -> list[list[float]]:

        if not texts:
            return []

        clean_texts = [
            t.strip() if t and t.strip() else " "
            for t in texts
        ]

        for attempt in range(retries + 1):

            try:
                response = self.client.models.embed_content(
                    model=self.model,
                    contents=clean_texts,
                )

                if (
                    hasattr(response, "embeddings")
                    and response.embeddings
                ):
                    return [
                        list(embedding.values)
                        for embedding in response.embeddings
                    ]

                if (
                    hasattr(response, "embedding")
                    and response.embedding
                ):
                    return [
                        list(response.embedding.values)
                    ]

            except Exception as exc:

                if attempt < retries:
                    await asyncio.sleep(
                        2 ** attempt
                    )
                    continue

                logger.warning(
                    f"Gemini embedding API call failed "
                    f"({exc}). Using fallback vector."
                )

                return [
                    self._create_fallback_vector(text)
                    for text in clean_texts
                ]

        return [
            self._create_fallback_vector(text)
            for text in clean_texts
        ]

    def _create_fallback_vector(
        self,
        text: str,
    ) -> list[float]:

        import hashlib
        import numpy as np

        hash_bytes = hashlib.sha256(
            text.encode("utf-8")
        ).digest()

        seed = int.from_bytes(
            hash_bytes[:4],
            "big",
        )

        rng = np.random.RandomState(seed)

        vec = rng.randn(
            self.dimensions
        ).astype(np.float32)

        norm = np.linalg.norm(vec)

        if norm > 0:
            vec = vec / norm

        return vec.tolist()