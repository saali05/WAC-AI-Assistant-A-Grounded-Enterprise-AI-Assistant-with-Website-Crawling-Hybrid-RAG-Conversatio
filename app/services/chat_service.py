from app.ai.service import AIService
from app.repositories.message_repository import MessageRepository
from app.services.response_formatter import ResponseFormatter
from app.services.conversation_service import ConversationService
from app.services.session_memory_service import SessionMemoryService
from app.services.usage_service import UsageService


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

    @property
    def usage_service(self) -> UsageService:
        return UsageService()

    async def send_message(
        self,
        provider: str,
        message: str,
        conversation_id: str | None = None,
    ) -> dict:
        """
        Complete chat flow with RAG grounded context & source citations.

        1. Get or create conversation
        2. Save user message
        3. Generate AI response + RAG context
        4. Save assistant message
        5. Record AI usage
        6. Return response + sources + rag_used metadata
        """

        conversation = await self.conversation_service.get_or_create(
            conversation_id=conversation_id,
            first_message=message,
        )

        conversation_id = conversation["id"]

        # Save user message
        await self.message_repository.create(
            conversation_id=conversation_id,
            role="user",
            provider=provider,
            content=message,
        )

        # Generate AI response with RAG
        history = await self.session_memory_service.build_history(conversation_id)
        ai_response, rag_result = await self.ai_service.chat(
            provider=provider,
            message=message,
            history=history,
        )

        formatted_content = ResponseFormatter.format(ai_response.content)

        # Save assistant response message
        msg_id = await self.message_repository.create(
            conversation_id=conversation_id,
            role="assistant",
            provider=provider,
            content=formatted_content,
        )

        # Record usage
        if hasattr(ai_response, "usage") and ai_response.usage:
            try:
                await self.usage_service.record_usage(
                    conversation_id=conversation_id,
                    usage=ai_response.usage,
                    message_id=msg_id,
                )
            except Exception:
                pass

        # Prepare source citation response objects
        sources_list = [
            {
                "title": s.title,
                "url": s.url,
                "heading": s.heading,
                "score": s.score
            }
            for s in rag_result.sources
        ] if rag_result.sources else []

        return {
            "conversation_id": conversation_id,
            "title": conversation["title"],
            "response": formatted_content,
            "sources": sources_list,
            "rag_used": rag_result.has_context,
            "retrieval_score": rag_result.retrieval_score if rag_result.has_context else None
        }