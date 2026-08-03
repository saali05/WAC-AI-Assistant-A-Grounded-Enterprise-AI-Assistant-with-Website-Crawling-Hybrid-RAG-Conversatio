from google import genai
from google.genai import types

from app.core.config import settings
from app.core.logging import logger

class GeminiClient:
    """
    Wrapper around the Google Gemini API.
    """

    def __init__(self):
        self.client = genai.Client(
            api_key=settings.GEMINI_API_KEY
        )

        self.model = settings.MODEL_NAME

        logger.info("Gemini Client initialized")

    async def generate(self, prompt: str) -> str:
        """
         Generate a response from Gemini.
        """

        try:

            response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            )

            logger.info("Gemini response generated successfully")

            return response.text

        except Exception as e:

            logger.error(f"Gemini Error: {e}")

            raise