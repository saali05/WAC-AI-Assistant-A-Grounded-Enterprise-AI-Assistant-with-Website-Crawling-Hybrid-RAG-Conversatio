import asyncio

from google import genai

from app.ai.providers.base import BaseAIProvider
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
    ) -> str:

        logger.info(
            f"Sending prompt to Gemini ({len(prompt)} characters)"
        )

        for attempt in range(1, self.MAX_RETRIES + 1):

            try:

                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                )

                logger.info(
                    "Gemini response generated successfully."
                )

                return response.text

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
                    raise RateLimitException(
                        "The AI service has reached its current usage limit. Please try again later or switch to another provider."
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