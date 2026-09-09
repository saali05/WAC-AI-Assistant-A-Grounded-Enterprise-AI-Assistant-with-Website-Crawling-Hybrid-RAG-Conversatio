import asyncio
import re
from typing import Optional, Any

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
        max_retry_delay: Optional[float] = None,
        default_retries: Optional[int] = None,
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

        self.max_retry_delay = (
            max_retry_delay
            if max_retry_delay is not None
            else getattr(settings, "EMBEDDING_MAX_RETRY_DELAY_SECONDS", 60.0)
        )

        self.default_retries = (
            default_retries
            if default_retries is not None
            else getattr(settings, "RAG_EMBEDDING_RETRIES", 3)
        )

        self._client: Optional[
            genai.Client
        ] = None

        logger.info(
            f"EmbeddingService initialized | "
            f"model={self.model} | "
            f"dimensions={self.dimensions} | "
            f"max_retry_delay={self.max_retry_delay}s | "
            f"default_retries={self.default_retries}"
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
    # RETRY / QUOTA HELPERS
    # ==========================================================

    @staticmethod
    def is_rate_limit_error(exc: Exception) -> bool:
        """Check if exception is a 429 RESOURCE_EXHAUSTED or quota error."""
        code = getattr(exc, "code", None)
        if code == 429:
            return True

        status = getattr(exc, "status", None)
        if status in ("RESOURCE_EXHAUSTED", "429"):
            return True

        exc_str = f"{getattr(exc, 'message', '')} {str(exc)}".lower()
        return (
            "429" in exc_str
            or "resource_exhausted" in exc_str
            or "resource exhausted" in exc_str
            or "quota" in exc_str
            or "rate_limit" in exc_str
            or "rate limit" in exc_str
        )


    @staticmethod
    def extract_retry_delay(exc: Exception) -> Optional[float]:
        """
        Extract server-requested retry delay in seconds from Google GenAI / HTTP errors.

        Inspects structured Google RPC RetryInfo in details, HTTP headers, and message text.
        """
        # 1. Structured RetryInfo in details
        details = getattr(exc, "details", None)
        if details:
            candidate_lists = []
            if isinstance(details, list):
                candidate_lists.append(details)
            elif isinstance(details, dict):
                if "details" in details and isinstance(details["details"], list):
                    candidate_lists.append(details["details"])
                if "error" in details and isinstance(details["error"], dict):
                    err_dict = details["error"]
                    if "details" in err_dict and isinstance(err_dict["details"], list):
                        candidate_lists.append(err_dict["details"])
                candidate_lists.append([details])

            for item_list in candidate_lists:
                for item in item_list:
                    if isinstance(item, dict):
                        item_type = str(item.get("@type", ""))
                        if "RetryInfo" in item_type or "retryDelay" in item or "retry_delay" in item:
                            delay_val = item.get("retryDelay") or item.get("retry_delay")
                            if delay_val is not None:
                                try:
                                    if isinstance(delay_val, (int, float)):
                                        return float(delay_val)
                                    if isinstance(delay_val, str):
                                        cleaned = delay_val.rstrip("s").strip()
                                        return float(cleaned)
                                except (ValueError, TypeError):
                                    pass

        # 2. HTTP response headers (Retry-After)
        response = getattr(exc, "response", None)
        if response is not None and hasattr(response, "headers"):
            headers = response.headers
            if headers:
                retry_after = headers.get("retry-after") or headers.get("Retry-After")
                if retry_after:
                    try:
                        return float(retry_after)
                    except (ValueError, TypeError):
                        pass

        # 3. String / Message regex extraction
        full_text = f"{getattr(exc, 'message', '')} {str(exc)}"
        match = re.search(r"retry(?:\s+after|\s+in)\s+(\d+(?:\.\d+)?)\s*s?", full_text, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except (ValueError, TypeError):
                pass

        return None

    # ==========================================================
    # SINGLE EMBEDDING
    # ==========================================================

    async def get_embedding(
        self,
        text: str,
        retries: Optional[int] = None,
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
        retries: Optional[int] = None,
    ) -> list[list[float]]:

        if not texts:
            return []

        clean_texts = [
            t.strip()
            if t and t.strip()
            else " "
            for t in texts
        ]

        retries_to_use = (
            retries
            if retries is not None
            else self.default_retries
        )

        for attempt in range(
            retries_to_use + 1
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

                is_rate_limit = self.is_rate_limit_error(exc)

                if is_rate_limit:
                    server_delay = self.extract_retry_delay(exc)
                    if server_delay is not None and server_delay > 0:
                        backoff = min(server_delay + 0.5, self.max_retry_delay)
                        logger.warning(
                            f"[QUOTA] Gemini rate limit reached (attempt {attempt + 1}/{retries_to_use + 1}). "
                            f"Server requested retry after {server_delay:.1f}s. "
                            f"Waiting {backoff:.1f}s before retry..."
                        )
                    else:
                        backoff = min(float((2 ** (attempt + 1)) * 5), self.max_retry_delay)
                        logger.warning(
                            f"[QUOTA] Gemini rate limit reached (attempt {attempt + 1}/{retries_to_use + 1}). "
                            f"Retrying in {backoff:.1f}s..."
                        )
                else:
                    backoff = min(float(2 ** (attempt + 1)), self.max_retry_delay)
                    logger.warning(
                        f"Gemini embedding attempt {attempt + 1}/{retries_to_use + 1} failed: "
                        f"{exc}. Retrying in {backoff:.1f}s..."
                    )

                if attempt < retries_to_use:

                    await asyncio.sleep(backoff)

                    continue

                logger.exception(
                    "Gemini embedding generation failed."
                )

                raise EmbeddingException(
                    f"Gemini embedding generation failed after {retries_to_use + 1} attempts."
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