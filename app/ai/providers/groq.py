import time

from groq import Groq

from app.ai.providers.base import BaseAIProvider
from app.ai.schemas import AIResponse, AIUsage
from app.ai.pricing import calculate_cost, MODEL_PRICING
from app.core.config import settings
from app.core.logging import logger

from app.ai.exceptions import (
    AIResponseException,
    InvalidAPIKeyException,
    RateLimitException,
    ProviderUnavailableException,
)


class GroqProvider(BaseAIProvider):
    """
    Groq AI Provider.

    Responsibilities:
    - Generate text responses using Groq
    - Extract token usage
    - Extract provider rate-limit headers
    - Calculate estimated API cost
    - Calculate performance metrics
    - Return normalized AIResponse
    """

    def __init__(self):
        self.client = Groq(
            api_key=settings.GROQ_API_KEY
        )

        self.model = settings.GROQ_MODEL

        logger.info(
            f"Groq Provider initialized | model={self.model}"
        )

    async def generate(
        self,
        prompt: str,
    ) -> AIResponse:

        start_time = time.perf_counter()

        logger.info(
            f"Sending prompt to Groq "
            f"(model={self.model}, chars={len(prompt)})"
        )

        try:
            # ==================================================
            # GROQ REQUEST
            # ==================================================
            #
            # with_raw_response gives us:
            #
            # 1. Parsed Groq completion
            # 2. HTTP response headers
            #
            # We need the headers for the rate-limit dashboard.
            #
            # ==================================================

            raw_response = (
                self.client
                .chat
                .completions
                .with_raw_response
                .create(
                    model=self.model,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                )
            )

            # Parsed Groq ChatCompletion object
            completion = raw_response.parse()

            # ==================================================
            # LATENCY
            # ==================================================

            duration_sec = max(
                time.perf_counter() - start_time,
                0.0001,
            )

            latency_ms = round(
                duration_sec * 1000,
                2,
            )

            # ==================================================
            # RESPONSE CONTENT
            # ==================================================

            content = ""

            if completion.choices:
                content = (
                    completion
                    .choices[0]
                    .message
                    .content
                    or ""
                )

            # ==================================================
            # GROQ TOKEN USAGE
            # ==================================================

            usage_data = getattr(
                completion,
                "usage",
                None,
            )

            input_tokens = (
                getattr(
                    usage_data,
                    "prompt_tokens",
                    0,
                )
                or 0
            )

            output_tokens = (
                getattr(
                    usage_data,
                    "completion_tokens",
                    0,
                )
                or 0
            )

            total_tokens = (
                getattr(
                    usage_data,
                    "total_tokens",
                    input_tokens + output_tokens,
                )
                or 0
            )

            # ==================================================
            # MODEL PRICING
            # ==================================================

            model_spec = MODEL_PRICING.get(
                self.model,
                {},
            )

            context_limit = model_spec.get(
                "context_limit",
                131_072,
            )

            # ==================================================
            # CONTEXT REMAINING
            # ==================================================
            #
            # This represents the remaining context capacity
            # for the current request.
            #
            # It is NOT the same as provider quota.
            #
            # ==================================================

            context_remaining = max(
                context_limit - input_tokens,
                0,
            )

            # ==================================================
            # COST
            # ==================================================

            estimated_cost = calculate_cost(
                model=self.model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

            # ==================================================
            # OUTPUT SPEED
            # ==================================================

            tokens_per_second = None

            if output_tokens > 0:
                tokens_per_second = round(
                    output_tokens / duration_sec,
                    2,
                )

            # ==================================================
            # RATE LIMIT HEADERS
            # ==================================================
            #
            # Groq exposes rate-limit information through
            # HTTP headers.
            #
            # Examples:
            #
            # x-ratelimit-limit-requests
            # x-ratelimit-remaining-requests
            # x-ratelimit-limit-tokens
            # x-ratelimit-remaining-tokens
            # x-ratelimit-reset-requests
            # x-ratelimit-reset-tokens
            #
            # ==================================================

            headers = raw_response.headers

            def parse_header_int(
                key: str,
            ) -> int | None:

                value = headers.get(key)

                if value is None:
                    return None

                try:
                    return int(value)
                except (
                    TypeError,
                    ValueError,
                ):
                    return None

            limit_requests = parse_header_int(
                "x-ratelimit-limit-requests"
            )

            remaining_requests = parse_header_int(
                "x-ratelimit-remaining-requests"
            )

            limit_tokens = parse_header_int(
                "x-ratelimit-limit-tokens"
            )

            remaining_tokens = parse_header_int(
                "x-ratelimit-remaining-tokens"
            )

            quota_reset = (
                headers.get(
                    "x-ratelimit-reset-tokens"
                )
                or headers.get(
                    "x-ratelimit-reset-requests"
                )
            )

            # ==================================================
            # QUOTA SOURCE
            # ==================================================

            has_provider_headers = (
                remaining_requests is not None
                or remaining_tokens is not None
            )

            if has_provider_headers:
                usage_source = "provider_headers"
            elif usage_data is not None:
                usage_source = "provider_metadata"
            else:
                usage_source = "unavailable"

            # ==================================================
            # QUOTA SCOPE
            # ==================================================
            #
            # According to Groq:
            #
            # Requests headers -> RPD
            # Token headers    -> TPM
            #
            # ==================================================

            quota_scope = "unknown"

            if has_provider_headers:

                if (
                    remaining_requests is not None
                    and remaining_tokens is not None
                ):
                    quota_scope = "mixed"

                elif remaining_tokens is not None:
                    quota_scope = "minute"

                elif remaining_requests is not None:
                    quota_scope = "day"

            # ==================================================
            # NORMALIZED AI USAGE
            # ==================================================

            ai_usage = AIUsage(

                provider="groq",

                model=self.model,

                request_type="text",

                input_tokens=input_tokens,

                output_tokens=output_tokens,

                total_tokens=total_tokens,

                cached_tokens=None,

                thinking_tokens=None,

                estimated_cost=estimated_cost,

                currency="USD",

                latency_ms=latency_ms,

                tokens_per_second=tokens_per_second,

                time_to_first_token_ms=None,

                audio_input_seconds=None,

                audio_output_seconds=None,

                live_session_id=None,

                context_limit=context_limit,

                context_remaining=context_remaining,

                provider_limit_requests=(
                    limit_requests
                ),

                provider_remaining_requests=(
                    remaining_requests
                ),

                provider_limit_tokens=(
                    limit_tokens
                ),

                provider_remaining_tokens=(
                    remaining_tokens
                ),

                quota_reset_time=quota_reset,

                usage_source=usage_source,

                quota_scope=quota_scope,
            )

            # ==================================================
            # LOGGING
            # ==================================================

            logger.info(
                "Groq response generated successfully | "
                f"model={self.model} | "
                f"input_tokens={input_tokens} | "
                f"output_tokens={output_tokens} | "
                f"total_tokens={total_tokens} | "
                f"cost=${estimated_cost:.10f} | "
                f"latency={latency_ms}ms | "
                f"speed={tokens_per_second} tok/s"
            )

            # ==================================================
            # RETURN NORMALIZED RESPONSE
            # ==================================================

            return AIResponse(
                content=content,
                usage=ai_usage,
            )

        # ======================================================
        # API KEY
        # ======================================================

        except Exception as exc:

            message = str(exc).lower()

            logger.exception(
                "Groq provider failed"
            )

            # --------------------------------------------------
            # RATE LIMIT
            # --------------------------------------------------

            if (
                "429" in message
                or "rate limit" in message
                or "rate_limit" in message
            ):
                raise RateLimitException(
                    "Groq API rate limit reached. "
                    "Please try again later."
                ) from exc

            # --------------------------------------------------
            # INVALID API KEY
            # --------------------------------------------------

            if (
                "401" in message
                or "api key" in message
                or "authentication" in message
                or "unauthorized" in message
            ):
                raise InvalidAPIKeyException(
                    "Invalid Groq API key."
                ) from exc

            # --------------------------------------------------
            # PROVIDER UNAVAILABLE
            # --------------------------------------------------

            if (
                "503" in message
                or "502" in message
                or "504" in message
                or "unavailable" in message
            ):
                raise ProviderUnavailableException(
                    "Groq is currently unavailable. "
                    "Please try again later."
                ) from exc

            # --------------------------------------------------
            # GENERIC AI ERROR
            # --------------------------------------------------

            raise AIResponseException(
                f"Groq provider failed: {exc}"
            ) from exc