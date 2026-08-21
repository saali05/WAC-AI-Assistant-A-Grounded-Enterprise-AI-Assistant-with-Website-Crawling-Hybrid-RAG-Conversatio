import asyncio
import hashlib
from typing import Optional

import numpy as np
from google import genai
from google.genai import types

from app.core.config import settings
from app.core.logging import logger
from app.rag.exceptions import EmbeddingException


class EmbeddingService:
    """
    Gemini Embedding Service.

    Used for:
    - WAC website document chunks
    - User search queries
    - RAG semantic retrieval

    Current model:
        gemini-embedding-001

    Default output dimension:
        768
    """

    def __init__(
        self,
        model: Optional[str] = None,
        dimensions: Optional[int] = None,
        api_key: Optional[str] = None,
    ) -> None:

        self.model = (
            model
            or settings.RAG_EMBEDDING_MODEL
        )

        self.dimensions = (
            dimensions
            or settings.RAG_EMBEDDING_DIMENSIONS
        )

        self.api_key = (
            api_key
            or settings.GEMINI_API_KEY
        )

        self._client: Optional[genai.Client] = None

        logger.info(
            "Embedding service initialized | "
            f"model={self.model} | "
            f"dimensions={self.dimensions}"
        )

    @property
    def client(self) -> genai.Client:
        """
        Lazily create Gemini client.
        """

        if self._client is None:

            if not self.api_key:
                raise EmbeddingException(
                    "GEMINI_API_KEY is not configured "
                    "for embedding service."
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

        """
        Generate a single embedding.
        """

        embeddings = await self.get_batch_embeddings(
            [text],
            retries=retries,
        )

        if not embeddings:
            raise EmbeddingException(
                "Gemini returned no embedding."
            )

        return embeddings[0]

    async def get_batch_embeddings(
        self,
        texts: list[str],
        retries: int = 2,
    ) -> list[list[float]]:

        """
        Generate embeddings for multiple texts.

        Uses Gemini's embed_content API.

        For gemini-embedding-001, 768 dimensions
        are explicitly requested.
        """

        if not texts:
            return []

        clean_texts = [
            text.strip()
            if text and text.strip()
            else " "
            for text in texts
        ]

        for attempt in range(
            retries + 1
        ):

            try:

                response = (
                    self.client.models.embed_content(
                        model=self.model,
                        contents=clean_texts,
                        config=types.EmbedContentConfig(
                            output_dimensionality=self.dimensions,
                        ),
                    )
                )

                embeddings = getattr(
                    response,
                    "embeddings",
                    None,
                )

                if embeddings:

                    result = [
                        list(
                            embedding.values
                        )
                        for embedding in embeddings
                    ]

                    self._validate_dimensions(
                        result
                    )

                    return result

                # Some SDK responses may expose
                # a single embedding.
                single_embedding = getattr(
                    response,
                    "embedding",
                    None,
                )

                if single_embedding:

                    result = [
                        list(
                            single_embedding.values
                        )
                    ]

                    self._validate_dimensions(
                        result
                    )

                    return result

                raise EmbeddingException(
                    "Gemini returned an empty embedding response."
                )

            except EmbeddingException:

                if attempt >= retries:
                    raise

                await asyncio.sleep(
                    2 ** attempt
                )

            except Exception as exc:

                logger.warning(
                    "Gemini embedding attempt "
                    f"{attempt + 1}/{retries + 1} failed: "
                    f"{exc}"
                )

                if attempt < retries:

                    await asyncio.sleep(
                        2 ** attempt
                    )

                    continue

                logger.exception(
                    "Gemini embedding generation failed."
                )

                # IMPORTANT:
                # Do NOT silently create fallback
                # vectors in production.
                raise EmbeddingException(
                    "Gemini embedding generation failed: "
                    f"{exc}"
                ) from exc

        raise EmbeddingException(
            "Unable to generate embeddings."
        )

    def _validate_dimensions(
        self,
        embeddings: list[list[float]],
    ) -> None:

        """
        Make sure Gemini returned exactly the
        vector size expected by MongoDB.
        """

        for index, embedding in enumerate(
            embeddings
        ):

            actual_dimensions = len(
                embedding
            )

            if (
                actual_dimensions
                != self.dimensions
            ):

                raise EmbeddingException(
                    "Embedding dimension mismatch: "
                    f"expected {self.dimensions}, "
                    f"received {actual_dimensions} "
                    f"for item {index}."
                )

    def _create_fallback_vector(
        self,
        text: str,
    ) -> list[float]:

        """
        Deterministic fallback vector.

        This method is intentionally retained for
        offline/unit-test compatibility.

        It must NOT be used for production RAG
        indexing or retrieval.
        """

        hash_bytes = hashlib.sha256(
            text.encode("utf-8")
        ).digest()

        seed = int.from_bytes(
            hash_bytes[:4],
            "big",
        )

        rng = np.random.RandomState(seed)

        vector = rng.randn(
            self.dimensions
        ).astype(np.float32)

        norm = np.linalg.norm(vector)

        if norm > 0:
            vector = vector / norm

        return vector.tolist()