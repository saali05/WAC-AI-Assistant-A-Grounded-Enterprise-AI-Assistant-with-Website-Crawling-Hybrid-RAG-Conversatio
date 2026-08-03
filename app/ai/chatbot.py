from app.ai.gemini_client import GeminiClient


class ChatBotService:
    """
    Service responsible for chatbot interactions.
    """

    def __init__(self):
        self.gemini = GeminiClient()

    async def chat(self, message: str) -> str:
        """
        Generate a chatbot response.
        """
        return await self.gemini.generate(message)