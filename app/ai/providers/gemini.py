from google import genai

from app.ai.providers.base import BaseAIProvider
from app.core.config import settings
from app.core.logging import logger

class GeminiProvider(BaseAIProvider):
    """
    Google Gemini AI Provider.
    """

    def __init__(self):
        self.client = genai.Client(
            api_key=settings.GEMINI_API_KEY
        )

        self.model = settings.GEMINI_MODEL

        logger.info("Gemini Provider initialized")

    async def generate(self, message: str) -> str:
        """
        Generate a response using Gemini.
        """
        try:

            response = self.client.models.generate_content(
                model=self.model,
                contents=message,
            )

            logger.info("Gemini response generated")

            return response.text

        except Exception as e:
            logger.exception("Gemini provider failed")
            raise