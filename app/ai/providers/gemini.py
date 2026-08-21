import asyncio
import time
import re
from google import genai

from app.ai.providers.base import BaseAIProvider
from app.ai.schemas import AIResponse, AIUsage
from app.ai.pricing import calculate_cost, MODEL_PRICING
from app.core.config import settings
from app.core.logging import logger

from app.ai.exceptions import (
    AIResponseException,
    InvalidAPIKeyException,
    ProviderUnavailableException,
    RateLimitException,
)


class GeminiProvider(BaseAIProvider):
    """
    Google Gemini AI Provider.
    """

    MAX_RETRIES = 3

    def __init__(self):
        self.client = genai.Client(
            api_key=settings.GEMINI_API_KEY
        )

        self.model = settings.GEMINI_MODEL

        logger.info(
            f"Gemini Provider initialized with model: {self.model}"
        )

    async def generate(
        self,
        prompt: str,
    ) -> AIResponse:

        logger.info(
            f"Sending prompt to Gemini ({len(prompt)} characters)"
        )

        for attempt in range(1, self.MAX_RETRIES + 1):

            try:
                start_time = time.perf_counter()

                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                )

                end_time = time.perf_counter()
                duration_sec = max(end_time - start_time, 0.0001)
                latency_ms = round(duration_sec * 1000, 2)

                logger.info("Gemini response generated successfully.")

                # Extract usage metadata
                usage_metadata = getattr(response, "usage_metadata", None)
                prompt_tokens = getattr(usage_metadata, "prompt_token_count", None) if usage_metadata else None
                output_tokens = getattr(usage_metadata, "candidates_token_count", None) if usage_metadata else None
                total_tokens = getattr(usage_metadata, "total_token_count", None) if usage_metadata else None
                cached_tokens = getattr(usage_metadata, "cached_content_token_count", None) if usage_metadata else None
                thinking_tokens = getattr(usage_metadata, "thoughts_token_count", None) if usage_metadata else None

                tokens_per_sec = None
                if output_tokens is not None and duration_sec > 0:
                    tokens_per_sec = round(output_tokens / duration_sec, 2)

                model_spec = MODEL_PRICING.get(self.model, {})
                context_limit = model_spec.get("context_limit", 1048576)

                context_remaining = None
                if prompt_tokens is not None:
                    context_remaining = max(context_limit - prompt_tokens, 0)

                estimated_cost = calculate_cost(
                    model=self.model,
                    input_tokens=prompt_tokens,
                    output_tokens=output_tokens,
                )

                usage = AIUsage(
                    provider="gemini",
                    model=self.model,
                    request_type="text",
                    input_tokens=prompt_tokens,
                    output_tokens=output_tokens,
                    total_tokens=total_tokens,
                    cached_tokens=cached_tokens,
                    thinking_tokens=thinking_tokens,
                    estimated_cost=estimated_cost,
                    currency="USD",
                    latency_ms=latency_ms,
                    tokens_per_second=tokens_per_sec,
                    time_to_first_token_ms=None,
                    context_limit=context_limit,
                    context_remaining=context_remaining,
                    provider_limit_requests=None,
                    provider_remaining_requests=None,
                    provider_limit_tokens=None,
                    provider_remaining_tokens=None,
                    quota_reset_time=None,
                    usage_source="provider_metadata" if usage_metadata else "unavailable",
                    quota_scope="unknown",
                )

                content_text = getattr(response, "text", "") or ""
                return AIResponse(content=content_text, usage=usage)

            except Exception as exc:

                message = str(exc).lower()

                logger.warning(
                    f"Gemini attempt {attempt}/{self.MAX_RETRIES} failed: {exc}"
                )

                # -----------------------------
                # Rate Limit / Quota
                # -----------------------------
                if (
                    "resource_exhausted" in message
                    or "quota" in message
                    or "429" in message
                ):
                    retry_match = re.search(r"retry after (\d+)", message)
                    retry_after = int(retry_match.group(1)) if retry_match else None
                    raise RateLimitException(
                        message="The Gemini AI service has reached its current usage limit. Please try again later.",
                        provider="gemini",
                        model=self.model,
                        retry_after_seconds=retry_after,
                        limit_type="requests_per_day" if "day" in message else "rate_limit",
                    )

                # -----------------------------
                # Invalid API Key
                # -----------------------------
                if (
                    "401" in message
                    or "permission_denied" in message
                    or "api key" in message
                    or "unauthenticated" in message
                ):
                    raise InvalidAPIKeyException(
                        "The configured Gemini API key is invalid."
                    )

                # -----------------------------
                # Gemini Unavailable
                # -----------------------------
                if (
                    "503" in message
                    or "unavailable" in message
                    or "deadline exceeded" in message
                    or "timeout" in message
                ):

                    if attempt < self.MAX_RETRIES:

                        logger.info(
                            f"Retrying Gemini in {2 ** attempt} seconds..."
                        )

                        await asyncio.sleep(2 ** attempt)

                        continue

                    raise ProviderUnavailableException(
                        "The AI service is temporarily unavailable. Please try again in a few minutes."
                    )

                # -----------------------------
                # Unknown Errors
                # -----------------------------
                if attempt < self.MAX_RETRIES:

                    logger.info(
                        f"Retrying Gemini in {2 ** attempt} seconds..."
                    )

                    await asyncio.sleep(2 ** attempt)

                    continue

                raise AIResponseException(
                    f"Gemini failed after {self.MAX_RETRIES} attempts."
                )

        raise AIResponseException(
            "Unable to generate a response."
        )
