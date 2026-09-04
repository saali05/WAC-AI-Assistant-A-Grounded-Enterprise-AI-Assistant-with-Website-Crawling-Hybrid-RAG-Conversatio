from app.ai.service import AIService
from app.core.config import settings
from app.rag.validation.relevance import WACRelevanceGate, WAC_GREETING_RESPONSE
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
        3. Handle greeting intent OR Generate AI response + RAG context
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

        # 0. Fast-path Greeting check (bypasses RAG vector search & refusal)
        if WACRelevanceGate.is_greeting(message):
            formatted_content = ResponseFormatter.format(WAC_GREETING_RESPONSE)
            await self.message_repository.create(
                conversation_id=conversation_id,
                role="assistant",
                provider=provider,
                content=formatted_content,
            )
            return {
                "conversation_id": conversation_id,
                "title": conversation["title"],
                "response": formatted_content,
                "sources": [],
                "rag_used": False,
                "retrieval_score": None,
            }

        # Toggle between standard pipeline and LangChain LCEL pipeline
        if getattr(settings, "USE_LANGCHAIN_PIPELINE", False):
            from app.langchain.chain import WACLangChainPipeline
            from langchain_core.messages import HumanMessage, AIMessage
            from app.ai.schemas import AIUsage

            # Build chat history for LangChain
            messages = await self.message_repository.get_by_conversation(conversation_id)
            # Exclude current user message (last message) from historical context
            past_messages = messages[:-1] if len(messages) > 1 else []
            recent_past = past_messages[-SessionMemoryService.MAX_MESSAGES:]

            lc_history = []
            for msg in recent_past:
                if msg.get("role") == "user":
                    lc_history.append(HumanMessage(content=msg.get("content", "")))
                elif msg.get("role") == "assistant":
                    lc_history.append(AIMessage(content=msg.get("content", "")))

            pipeline = WACLangChainPipeline(provider=provider)
            lc_response = await pipeline.ainvoke(
                input_text=message,
                chat_history=lc_history,
            )

            formatted_content = ResponseFormatter.format(lc_response.answer)

            # Save assistant response message
            msg_id = await self.message_repository.create(
                conversation_id=conversation_id,
                role="assistant",
                provider=provider,
                content=formatted_content,
            )

            # Record usage
            if lc_response.usage:
                ai_usage = AIUsage(
                    provider=provider,
                    model=lc_response.usage.model_name or provider,
                    prompt_tokens=lc_response.usage.prompt_tokens,
                    completion_tokens=lc_response.usage.completion_tokens,
                    total_tokens=lc_response.usage.total_tokens,
                )
                try:
                    await self.usage_service.record_usage(
                        conversation_id=conversation_id,
                        usage=ai_usage,
                        message_id=msg_id,
                    )
                except Exception:
                    pass

            sources_list = [
                {
                    "title": s.get("title", ""),
                    "url": s.get("url", ""),
                    "heading": s.get("heading", ""),
                    "score": s.get("score", 0.0),
                }
                for s in lc_response.sources
            ]

            top_score = sources_list[0]["score"] if sources_list else None

            return {
                "conversation_id": conversation_id,
                "title": conversation["title"],
                "response": formatted_content,
                "sources": sources_list,
                "rag_used": len(sources_list) > 0,
                "retrieval_score": top_score,
            }

        # Legacy / Native Chat Pipeline
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
                "score": s.score,
            }
            for s in rag_result.sources
        ] if rag_result.sources else []

        return {
            "conversation_id": conversation_id,
            "title": conversation["title"],
            "response": formatted_content,
            "sources": sources_list,
            "rag_used": rag_result.has_context,
            "retrieval_score": rag_result.retrieval_score if rag_result.has_context else None,
        }