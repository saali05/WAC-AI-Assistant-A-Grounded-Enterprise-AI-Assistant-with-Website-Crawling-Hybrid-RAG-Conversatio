from groq import Groq

from app.ai.providers.base import BaseAIProvider
from app.core.config import settings
from app.core.logging import logger


class GroqProvider(BaseAIProvider):
    """
    Groq AI Provider.
    """

    def __init__(self):
        self.client = Groq(
            api_key=settings.GROQ_API_KEY
        )

        self.model = settings.GROQ_MODEL

        logger.info("Groq Provider initialized")

    async def generate(self, message: str) -> str:
        """
        Generate a response using Groq.
        """
        try:

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": message,
                    }
                ],
            )

            logger.info("Groq response generated")

            return response.choices[0].message.content

        except Exception:
            logger.exception("Groq provider failed")
            raise