import asyncio
from typing import Optional

from google import genai
from google.genai import types

from app.core.config import settings
from app.core.logging import logger
from app.rag.exceptions import EmbeddingException


class EmbeddingService:
    """
    Gemini Embedding Service for document chunks and user queries.

    Important:
    The same embedding dimensionality must be used for:

        Documents
             ↓
        Gemini Embedding
             ↓
        768 dimensions
             ↓
        MongoDB vector index

    and:

        User Query
             ↓
        Gemini Embedding
             ↓
        768 dimensions
             ↓
        MongoDB vector search
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

        self._client: Optional[
            genai.Client
        ] = None

        logger.info(
            f"EmbeddingService initialized | "
            f"model={self.model} | "
            f"dimensions={self.dimensions}"
        )

    @property
    def client(self) -> genai.Client:

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

    # ==========================================================
    # SINGLE EMBEDDING
    # ==========================================================

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

        raise EmbeddingException(
            "Gemini returned no embedding."
        )

    # ==========================================================
    # BATCH EMBEDDINGS
    # ==========================================================

    async def get_batch_embeddings(
        self,
        texts: list[str],
        retries: int = 2,
    ) -> list[list[float]]:

        if not texts:
            return []

        clean_texts = [
            t.strip()
            if t and t.strip()
            else " "
            for t in texts
        ]

        for attempt in range(
            retries + 1
        ):

            try:

                # --------------------------------------------------
                # IMPORTANT:
                #
                # Explicitly request the configured embedding
                # dimensionality.
                #
                # Without this configuration Gemini returns
                # the default 3072-dimensional embedding.
                # --------------------------------------------------

                response = (
                    self.client.models.embed_content(
                        model=self.model,
                        contents=clean_texts,
                        config=types.EmbedContentConfig(
                            output_dimensionality=self.dimensions,
                        ),
                    )
                )

                # --------------------------------------------------
                # Batch response
                # --------------------------------------------------

                if (
                    hasattr(response, "embeddings")
                    and response.embeddings
                ):

                    embeddings = [
                        list(embedding.values)
                        for embedding
                        in response.embeddings
                    ]

                    self._validate_embeddings(
                        embeddings
                    )

                    return embeddings

                # --------------------------------------------------
                # Single response
                # --------------------------------------------------

                if (
                    hasattr(response, "embedding")
                    and response.embedding
                ):

                    embeddings = [
                        list(
                            response.embedding.values
                        )
                    ]

                    self._validate_embeddings(
                        embeddings
                    )

                    return embeddings

                raise EmbeddingException(
                    "Gemini returned an empty embedding response."
                )

            except EmbeddingException:

                raise

            except Exception as exc:

                is_rate_limit = "429" in str(exc) or "resource_exhausted" in str(exc).lower() or "quota" in str(exc).lower()
                backoff = (2 ** (attempt + 1)) * (3 if is_rate_limit else 1)

                logger.warning(
                    f"Gemini embedding attempt "
                    f"{attempt + 1}/{retries + 1} failed: "
                    f"{exc}. Retrying in {backoff}s..."
                )

                if attempt < retries:

                    await asyncio.sleep(backoff)

                    continue

                logger.exception(
                    "Gemini embedding generation failed."
                )

                raise EmbeddingException(
                    "Gemini embedding generation failed."
                ) from exc

        raise EmbeddingException(
            "Unable to generate Gemini embedding."
        )

    # ==========================================================
    # VALIDATION
    # ==========================================================

    def _validate_embeddings(
        self,
        embeddings: list[list[float]],
    ) -> None:

        if not embeddings:

            raise EmbeddingException(
                "No embeddings were generated."
            )

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
                    f"expected={self.dimensions}, "
                    f"actual={actual_dimensions}, "
                    f"index={index}"
                )

        logger.debug(
            f"Embedding validation successful | "
            f"count={len(embeddings)} | "
            f"dimensions={self.dimensions}"
        )

    # ==========================================================
    # NO RANDOM FALLBACK
    # ==========================================================

    def _create_fallback_vector(
        self,
        text: str,
    ) -> list[float]:

        """
        Deprecated.

        Do NOT generate random vectors for production RAG.

        A random vector has no semantic relationship to the
        input text and can produce meaningless retrieval results.
        """

        raise EmbeddingException(
            "Embedding generation failed. "
            "Random fallback vectors are disabled."
        )