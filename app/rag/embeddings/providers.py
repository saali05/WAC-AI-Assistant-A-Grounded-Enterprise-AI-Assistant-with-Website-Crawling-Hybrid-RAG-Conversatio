import asyncio
import re
import threading
from abc import ABC, abstractmethod
from typing import Optional, Any

from app.core.config import settings
from app.core.logging import logger
from app.rag.exceptions import EmbeddingException, RAGConfigurationException


class BaseEmbeddingProvider(ABC):
    """
    Abstract base class for RAG embedding providers.
    """

    provider_name: str
    model: str
    dimensions: int

    @abstractmethod
    async def get_embedding(
        self,
        text: str,
        retries: Optional[int] = None,
    ) -> list[float]:
        """Generate embedding vector for a single text."""
        pass

    @abstractmethod
    async def get_batch_embeddings(
        self,
        texts: list[str],
        retries: Optional[int] = None,
    ) -> list[list[float]]:
        """Generate embedding vectors for a batch of texts."""
        pass

    def _validate_embeddings(
        self,
        embeddings: list[list[float]],
    ) -> None:
        """Validate generated embeddings against expected dimensions."""
        if not embeddings:
            raise EmbeddingException("No embeddings were generated.")

        for index, embedding in enumerate(embeddings):
            actual_dimensions = len(embedding)
            if actual_dimensions != self.dimensions:
                raise EmbeddingException(
                    f"Embedding dimension mismatch: "
                    f"expected={self.dimensions}, "
                    f"actual={actual_dimensions}, "
                    f"index={index}"
                )

        logger.debug(
            f"Embedding validation successful | "
            f"provider={self.provider_name} | "
            f"count={len(embeddings)} | "
            f"dimensions={self.dimensions}"
        )


class GeminiEmbeddingProvider(BaseEmbeddingProvider):
    """
    Hosted Gemini Embedding Provider using google-genai SDK.
    """

    provider_name = "gemini"

    def __init__(
        self,
        model: Optional[str] = None,
        dimensions: Optional[int] = None,
        api_key: Optional[str] = None,
        max_retry_delay: Optional[float] = None,
        default_retries: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        self.model = model or settings.RAG_EMBEDDING_MODEL
        self.dimensions = dimensions or settings.RAG_EMBEDDING_DIMENSIONS
        self.api_key = api_key or settings.GEMINI_API_KEY
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

        self._client: Optional[Any] = None

        logger.info(
            f"Embedding provider selected | "
            f"provider=gemini | "
            f"model={self.model} | "
            f"dimensions={self.dimensions} | "
            f"max_retry_delay={self.max_retry_delay}s | "
            f"default_retries={self.default_retries}"
        )

    @property
    def client(self) -> Any:
        if self._client is None:
            if not self.api_key:
                raise EmbeddingException(
                    "GEMINI_API_KEY is not configured for Gemini embedding provider."
                )
            from google import genai
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    # ==========================================================
    # RETRY / QUOTA HELPERS
    # ==========================================================

    @staticmethod
    def extract_retry_delay(exc: Exception) -> Optional[float]:
        """
        Extract server-requested retry delay in seconds from Google GenAI / HTTP errors.
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

    @staticmethod
    def is_hard_quota_exhausted_error(exc: Exception) -> bool:
        """
        Check if exception indicates daily or permanent quota exhaustion without a temporary retry delay.
        """
        if GeminiEmbeddingProvider.extract_retry_delay(exc) is not None:
            return False

        exc_str = f"{getattr(exc, 'message', '')} {str(exc)}".lower()
        return (
            "per_day" in exc_str
            or "per day" in exc_str
            or "daily limit" in exc_str
            or "daily quota" in exc_str
            or "daily_requests" in exc_str
            or ("quota" in exc_str and "exhausted" in exc_str and "limit" in exc_str)
        )

    @staticmethod
    def is_non_retryable_error(exc: Exception) -> bool:
        """
        Check for non-retryable client, auth, or configuration errors (400, 401, 403, 404).
        """
        code = getattr(exc, "code", None)
        if code in (400, 401, 403, 404):
            return True
        status = str(getattr(exc, "status", "")).upper()
        if status in ("INVALID_ARGUMENT", "UNAUTHENTICATED", "PERMISSION_DENIED", "NOT_FOUND"):
            return True
        exc_str = f"{getattr(exc, 'message', '')} {str(exc)}".lower()
        if "api_key_invalid" in exc_str or "api key not valid" in exc_str:
            return True
        return False

    @staticmethod
    def is_rate_limit_error(exc: Exception) -> bool:
        """
        Check if exception is a temporary 429 RESOURCE_EXHAUSTED rate-limit condition.
        Excludes permanent daily quota exhaustion and non-retryable errors.
        """
        if GeminiEmbeddingProvider.is_hard_quota_exhausted_error(exc):
            return False
        if GeminiEmbeddingProvider.is_non_retryable_error(exc):
            return False

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
            or "rate_limit" in exc_str
            or "rate limit" in exc_str
            or "rate_limit_exceeded" in exc_str
        )

    # ==========================================================
    # EMBEDDING GENERATION
    # ==========================================================

    async def get_embedding(
        self,
        text: str,
        retries: Optional[int] = None,
    ) -> list[float]:
        embeddings = await self.get_batch_embeddings([text], retries=retries)
        if embeddings:
            return embeddings[0]
        raise EmbeddingException("Gemini returned no embedding.")

    async def get_batch_embeddings(
        self,
        texts: list[str],
        retries: Optional[int] = None,
    ) -> list[list[float]]:
        if not texts:
            return []

        clean_texts = [
            t.strip() if t and t.strip() else " "
            for t in texts
        ]

        retries_to_use = (
            retries if retries is not None else self.default_retries
        )

        from google.genai import types

        for attempt in range(retries_to_use + 1):
            try:
                response = self.client.models.embed_content(
                    model=self.model,
                    contents=clean_texts,
                    config=types.EmbedContentConfig(
                        output_dimensionality=self.dimensions,
                    ),
                )

                if hasattr(response, "embeddings") and response.embeddings:
                    embeddings = [
                        list(embedding.values)
                        for embedding in response.embeddings
                    ]
                    self._validate_embeddings(embeddings)
                    return embeddings

                if hasattr(response, "embedding") and response.embedding:
                    embeddings = [list(response.embedding.values)]
                    self._validate_embeddings(embeddings)
                    return embeddings

                raise EmbeddingException("Gemini returned an empty embedding response.")

            except EmbeddingException:
                raise

            except Exception as exc:
                # 1. Permanent/Daily quota exhaustion -> Fail immediately
                if self.is_hard_quota_exhausted_error(exc):
                    logger.error(
                        f"[QUOTA EXHAUSTION] Gemini embedding daily/free-tier quota exhausted: {exc}. "
                        "Failing immediately without wasting retry attempts."
                    )
                    raise EmbeddingException(
                        f"Gemini embedding quota exhausted: {exc}"
                    ) from exc

                # 2. Non-retryable auth/client error -> Fail immediately
                if self.is_non_retryable_error(exc):
                    logger.error(f"Non-retryable Gemini embedding error: {exc}")
                    raise EmbeddingException(
                        f"Non-retryable Gemini embedding error: {exc}"
                    ) from exc

                # 3. Temporary rate limit or transient error -> Retry with backoff
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

                logger.exception("Gemini embedding generation failed.")
                raise EmbeddingException(
                    f"Gemini embedding generation failed after {retries_to_use + 1} attempts."
                ) from exc

        raise EmbeddingException("Unable to generate Gemini embedding.")


class LocalEmbeddingProvider(BaseEmbeddingProvider):
    """
    Local BGE Embedding Provider using sentence-transformers (BAAI/bge-base-en-v1.5).

    Generates 768-dimensional embeddings locally on CPU/local hardware
    without external API calls or quota consumption. Thread-safe for
    concurrent asynchronous requests via asyncio.to_thread and torch.inference_mode.
    """

    provider_name = "local"
    _model_cache: dict[str, Any] = {}
    _lock = threading.Lock()

    def __init__(
        self,
        model: Optional[str] = None,
        dimensions: Optional[int] = None,
        device: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self.model = model or getattr(settings, "LOCAL_EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
        self.dimensions = dimensions or getattr(settings, "LOCAL_EMBEDDING_DIMENSIONS", 768)
        self.device = device or "cpu"

        logger.info(
            f"Embedding provider selected | "
            f"provider=local | "
            f"model={self.model} | "
            f"dimensions={self.dimensions} | "
            f"device={self.device}"
        )

    def _get_model(self) -> Any:
        """Load and cache SentenceTransformer model instance (thread-safe singleton)."""
        cache_key = f"{self.model}::{self.device}"
        if cache_key not in self._model_cache:
            with self._lock:
                if cache_key not in self._model_cache:
                    try:
                        from sentence_transformers import SentenceTransformer
                        logger.info(f"Loading local embedding model: {self.model} on device: {self.device}")
                        model_instance = SentenceTransformer(
                            self.model,
                            device=self.device,
                        )
                        self._model_cache[cache_key] = model_instance
                    except ImportError as exc:
                        logger.exception("sentence-transformers is not installed.")
                        raise EmbeddingException(
                            "Local embedding provider requires sentence-transformers. "
                            "Please install sentence-transformers: pip install sentence-transformers"
                        ) from exc
                    except Exception as exc:
                        logger.exception(f"Failed to load local embedding model '{self.model}': {exc}")
                        raise EmbeddingException(
                            f"Failed to initialize local embedding model '{self.model}': {exc}"
                        ) from exc
        return self._model_cache[cache_key]

    def _encode_sync(self, clean_texts: list[str]) -> list[list[float]]:
        """Synchronous encoding using sentence-transformers under inference mode."""
        try:
            import torch
            model = self._get_model()
            with torch.inference_mode():
                embeddings = model.encode(
                    clean_texts,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
            if hasattr(embeddings, "tolist"):
                return embeddings.tolist()
            return [list(map(float, vec)) for vec in embeddings]
        except EmbeddingException:
            raise
        except Exception as exc:
            logger.exception(f"Local embedding generation failed: {exc}")
            raise EmbeddingException(
                f"Local embedding generation failed: {exc}"
            ) from exc

    async def get_embedding(
        self,
        text: str,
        retries: Optional[int] = None,
    ) -> list[float]:
        embeddings = await self.get_batch_embeddings([text], retries=retries)
        if embeddings:
            return embeddings[0]
        raise EmbeddingException("Local embedding model returned no vector.")

    async def get_batch_embeddings(
        self,
        texts: list[str],
        retries: Optional[int] = None,
    ) -> list[list[float]]:
        if not texts:
            return []

        clean_texts = [
            t.strip() if t and t.strip() else " "
            for t in texts
        ]

        embeddings = await asyncio.to_thread(self._encode_sync, clean_texts)
        self._validate_embeddings(embeddings)
        return embeddings


class EmbeddingProviderFactory:
    """
    Factory creating embedding provider instances.
    """

    _providers = {
        "gemini": GeminiEmbeddingProvider,
        "local": LocalEmbeddingProvider,
    }

    @classmethod
    def create(
        cls,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        dimensions: Optional[int] = None,
        **kwargs: Any,
    ) -> BaseEmbeddingProvider:
        provider_name = (
            provider
            or getattr(settings, "RAG_EMBEDDING_PROVIDER", "gemini")
        ).strip().lower()

        provider_cls = cls._providers.get(provider_name)
        if provider_cls is None:
            supported = ", ".join(f"'{k}'" for k in cls._providers.keys())
            raise RAGConfigurationException(
                f"Unsupported embedding provider '{provider_name}'. "
                f"Supported providers are: {supported}."
            )

        return provider_cls(
            model=model,
            dimensions=dimensions,
            **kwargs,
        )
