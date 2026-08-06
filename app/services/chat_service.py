from app.ai.service import AIService

from app.repositories.message_repository import MessageRepository

from app.services.response_formatter import ResponseFormatter
from app.services.conversation_service import ConversationService
from app.services.session_memory_service import SessionMemoryService
from app.services.response_formatter import ResponseFormatter

class ChatService:
    @property
    def session_memory_service(self):
        return SessionMemoryService()

    @property
    def ai_service(self) -> AIService:
        return AIService()

    @property
    def conversation_service(self) -> ConversationService:
        return ConversationService()

    @property
    def message_repository(self) -> MessageRepository:
        return MessageRepository()

    async def send_message(
        self,
        provider: str,
        message: str,
        conversation_id: str | None = None,
    ) -> dict:
        """
        Complete chat flow.

        1. Get or create conversation
        2. Save user message
        3. Generate AI response
        4. Save assistant message
        5. Return response
        """

        conversation = await self.conversation_service.get_or_create(
            conversation_id=conversation_id,
            first_message=message,
        )

        conversation_id = conversation["id"]
        # print("1. Conversation created")
        # Save user message
        await self.message_repository.create(
            conversation_id=conversation_id,
            role="user",
            provider=provider,
            content=message,
        )
        # print("2. User message saved")
        # Generate AI response
        history = await self.session_memory_service.build_history(
            conversation_id
  )
        # print("3. History loaded")
        response = await self.ai_service.chat(
            provider=provider,
            message=message,
            history=history,
            )
        response = ResponseFormatter.format(response)
        # print("4. AI response received")
        # print(response)


        # Save assistant response
        await self.message_repository.create(
            conversation_id=conversation_id,
            role="assistant",
            provider=provider,
            content=response,
        )
        # print("5. Assistant message saved")
        result = {
        "conversation_id": conversation_id,
        "title": conversation["title"],
        "response": response,
        }

        # print("6. Returning response")
        # print(result)

        return {
            "conversation_id": conversation_id,
            "title": conversation["title"],
            "response": response,
        }